# src/rag_system.py
# -*- coding: utf-8 -*-
"""
RAG 시스템 메인 클래스
- 벡터 검색(Chroma) + LLM 결합
- 날짜/주차 강의 요약 (단일 날짜/주차 모두 인식)
- 회고록(블로그 스타일) 생성
- 코드 검색(자연어 의도 자동 인식) / 코드 설명
- 다중 쿼리 확장 + 수동 앙상블 + 의도 기반 스코어링(모델+주제: CNN/RAG/LangChain/Retriever 등)
"""

import os
import re
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import dateparser

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --- Re-ranking 모델 임포트 ---
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

# --- BM25 / Ensemble 호환 임포트 (버전별 안전처리) ---
BM25Retriever = None
EnsembleRetriever = None
try:
    from langchain.retrievers import BM25Retriever as _BM25A, EnsembleRetriever as _ENS_A
    BM25Retriever = _BM25A
    EnsembleRetriever = _ENS_A
except Exception:
    try:
        from langchain_community.retrievers import BM25Retriever as _BM25B
        BM25Retriever = _BM25B
    except Exception:
        BM25Retriever = None
    try:
        from langchain.retrievers import EnsembleRetriever as _ENS_B
        EnsembleRetriever = _ENS_B
    except Exception:
        EnsembleRetriever = None


class RAGSystem:
    """RAG 시스템 메인 클래스"""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_model: str = "google/embeddinggemma-300m",
        llm_model: str = "gpt-4o-mini",
        k: int = 5,
        use_reranking: bool = True,
        rerank_model: str = "dragonkue/bge-reranker-v2-m3-ko",
        rerank_top_k: int = 10
    ):
        self.db_path = db_path
        self.k = k
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k

        # Re-ranking 모델 초기화
        self.reranker = None
        if self.use_reranking and RERANKER_AVAILABLE:
            try:
                print(f"Re-ranking 모델 로딩 중: {rerank_model}")
                self.reranker = CrossEncoder(rerank_model)
                print("Re-ranking 모델이 성공적으로 로드되었습니다.")
            except Exception as e:
                print(f"Re-ranking 모델 로드 실패: {e}")
                self.reranker = None
        elif self.use_reranking and not RERANKER_AVAILABLE:
            print("Warning: sentence-transformers가 설치되지 않아 re-ranking을 사용할 수 없습니다.")

        # 임베딩
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # LLM
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0.1,
            max_tokens=2000
        )

        # 스토어 & 리트리버
        self.vectorstore: Optional[Chroma] = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._all_documents: List[Document] = []
        self._load_vectorstore()

        # 일반 Q/A 프롬프트
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "다음은 LLM 강의 자료에서 검색된 관련 정보입니다:\n\n"
                "{context}\n\n"
                "위 정보를 바탕으로 다음 질문에 답해주세요:\n"
                "질문: {question}\n\n"
                "답변 시 다음 사항을 고려해주세요:\n"
                "1. 제공된 강의 자료의 내용만을 기반으로 답변하세요\n"
                "2. 소스 코드가 포함된 경우, 해당 파일명과 셀 번호를 명시하세요\n"
                "3. 강의 날짜가 언급된 경우, 해당 날짜를 포함하여 답변하세요\n"
                "4. 확실하지 않은 내용은 추측하지 말고 \"제공된 자료에서 확인할 수 없습니다\"라고 말하세요\n\n"
                "답변:"
            )
        )

    # -------------------------------
    # Vectorstore & Retriever
    # -------------------------------
    def _load_vectorstore(self):
        if not os.path.exists(self.db_path):
            print(f"벡터 DB가 존재하지 않습니다: {self.db_path}")
            return
        try:
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            raw = self.vectorstore.get()
            self._all_documents = [
                Document(page_content=c, metadata=m)
                for c, m in zip(raw.get("documents", []), raw.get("metadatas", []))
            ]

            self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": max(self.k, 20)})

            if BM25Retriever and self._all_documents:
                try:
                    self.bm25_retriever = BM25Retriever.from_documents(self._all_documents)
                    self.bm25_retriever.k = max(self.k, 20)
                except Exception as e:
                    print(f"BM25 초기화 실패: {e}")
                    self.bm25_retriever = None

            if EnsembleRetriever and self.vector_retriever and self.bm25_retriever:
                try:
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[self.vector_retriever, self.bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                except Exception as e:
                    print(f"EnsembleRetriever 초기화 실패: {e}")
                    self.ensemble_retriever = None

            print("벡터 스토어가 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"벡터 스토어 로드 중 오류: {e}")

    # -------------------------------
    # Re-ranking 유틸
    # -------------------------------
    def _rerank_documents(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Re-rank documents using CrossEncoder model"""
        if not self.reranker or not documents:
            return documents

        top_k = top_k or self.k

        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = [(query, doc.page_content) for doc in documents]

            # Re-ranking 점수 계산
            scores = self.reranker.predict(query_doc_pairs)

            # 점수와 문서를 쌍으로 만들어 정렬
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # 상위 k개 문서 반환
            reranked_docs = [doc for score, doc in scored_docs[:top_k]]

            print(f"Re-ranking 완료: {len(documents)} -> {len(reranked_docs)} 문서")
            return reranked_docs

        except Exception as e:
            print(f"Re-ranking 중 오류 발생: {e}")
            return documents[:top_k]

    # -------------------------------
    # 검색 유틸 (수동 앙상블 + 스코어링)
    # -------------------------------
    def _rank_combine(self, vdocs: List[Document], bdocs: List[Document], k: int) -> List[Document]:
        def key_of(d: Document) -> Tuple:
            m = d.metadata or {}
            return (
                m.get("source"),
                m.get("filename"),
                m.get("lecture_date"),
                m.get("cell_index"),
                m.get("chunk_index"),
                hash(d.page_content.strip()),
            )

        scores: Dict[Tuple, float] = {}
        where: Dict[Tuple, Document] = {}

        for rank, d in enumerate(vdocs):
            kkey = key_of(d)
            scores[kkey] = scores.get(kkey, 0.0) + 0.7 / (rank + 1)
            where[kkey] = d
        for rank, d in enumerate(bdocs):
            kkey = key_of(d)
            scores[kkey] = scores.get(kkey, 0.0) + 0.3 / (rank + 1)
            where[kkey] = d

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        merged = [where[k] for k, _ in ordered][:k]
        return merged

    def _dedup_docs(self, docs: List[Document], limit: int) -> List[Document]:
        seen = set()
        unique = []
        for d in docs:
            sig = (d.metadata.get("filename"), d.metadata.get("lecture_date"), d.metadata.get("cell_index"), hash(d.page_content.strip()))
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(d)
            if len(unique) >= limit:
                break
        return unique

    def _search_once(self, query: str, k: int) -> List[Document]:
        try:
            # 더 많은 문서를 가져와서 re-ranking할 준비
            retrieve_k = self.rerank_top_k if self.use_reranking and self.reranker else k

            if self.ensemble_retriever:
                docs = self.ensemble_retriever.invoke(query)[:retrieve_k]
            else:
                v = self.vector_retriever.invoke(query)[:retrieve_k] if self.vector_retriever else []
                b = self.bm25_retriever.invoke(query)[:retrieve_k] if self.bm25_retriever else []
                if v and b:
                    docs = self._rank_combine(v, b, retrieve_k)
                else:
                    docs = (v or b)[:retrieve_k]

            # Re-ranking 적용
            if self.use_reranking and self.reranker and docs:
                docs = self._rerank_documents(query, docs, k)
            else:
                docs = docs[:k]

            return docs
        except Exception as e:
            print(f"검색 중 오류: {e}")
            return []

    def _multi_query_search(self, queries: List[str], k_each: int = 20, k_final: int = 20) -> List[Document]:
        bag: List[Document] = []
        for q in queries:
            bag.extend(self._search_once(q, k_each))
        return self._dedup_docs(bag, k_final)

    # -------------------------------
    # 의도(모델/주제) 인식 & 매칭
    # -------------------------------
    def _intent_models(self, text: str) -> List[str]:
        """모델/주제 의도 모두 감지(CNN, RAG, LangChain, Retriever 등)"""
        t = (text or "").lower()
        intents = []

        # 모델군
        if any(k in t for k in ["cnn", "convolution", "conv2d"]): intents.append("cnn")
        if "rnn" in t: intents.append("rnn")
        if "lstm" in t: intents.append("lstm")
        if "gru" in t: intents.append("gru")
        if any(k in t for k in ["transformer", "attention"]): intents.append("transformer")
        if "bert" in t: intents.append("bert")
        if "gpt" in t: intents.append("gpt")
        if any(k in t for k in ["knn", "k-nearest"]): intents.append("knn")
        if any(k in t for k in ["svm", "svc", "svr", "support vector"]): intents.append("svm")

        # 주제군
        if "langchain" in t: intents.append("langchain")
        if "retriever" in t or "bm25" in t: intents.append("retriever")
        if any(k in t for k in ["rag", "retrieval augmented generation"]): intents.append("rag")
        if any(k in t for k in ["vector store", "vectorstore", "chroma", "faiss"]): intents.append("vectorstore")
        if "langgraph" in t: intents.append("langgraph")
        if "openai" in t: intents.append("openai")

        # RNN 묶음
        if "lstm" in intents and "rnn" not in intents: intents.append("rnn")
        if "gru" in intents and "rnn" not in intents: intents.append("rnn")

        return list(dict.fromkeys(intents))

    def _doc_matches_intent(self, doc: Document, intents: List[str]) -> bool:
        if not intents:
            return True
        text = (doc.page_content or "").lower()
        meta = (doc.metadata or {})
        mtypes = (meta.get("model_types") or "").lower()
        fname = (meta.get("filename") or "").lower()

        def has_any(keys: List[str]) -> bool:
            return any(k in text or k in mtypes or k in fname for k in keys)

        SYNS = {
            # 모델
            "cnn": ["cnn", "conv", "convolution", "conv2d", "maxpool", "resnet", "vgg", "alexnet"],
            "rnn": ["rnn", "recurrent"],
            "lstm": ["lstm"],
            "gru": ["gru"],
            "transformer": ["transformer", "self-attention", "multiheadattention", "encoderlayer", "decoderlayer", "bert", "gpt"],
            "bert": ["bert"],
            "gpt": ["gpt"],
            "knn": ["knn", "k-nearest", "kneighborsclassifier"],
            "svm": ["svm", "svc", "svr", "support vector"],
            # 주제
            "langchain": ["langchain", "lcel", "runnable", "langchain_community", "langchain_openai", "as_retriever", "similarity_search", "invoke", "get_relevant_documents"],
            "retriever": ["retriever", "bm25", "ensemble", "as_retriever", "get_relevant_documents", "similarity_search"],
            "rag": ["rag", "retrieval augmented generation", "retrieval", "context", "vector store", "vectorstore"],
            "vectorstore": ["vectorstore", "vector store", "chroma", "faiss", "persist_directory", "embedding_function"],
            "langgraph": ["langgraph", "stategraph", "node", "edge", "start", "end", "runnable"],
            "openai": ["openai", "chatopenai", "gpt", "openai api"],
        }

        for it in intents:
            if it in SYNS and has_any(SYNS[it]):
                return True
        return False

    def _filter_by_intent(self, docs: List[Document], intents: List[str], min_keep: int = 6) -> List[Document]:
        if not intents:
            return docs
        kept = [d for d in docs if self._doc_matches_intent(d, intents)]
        if len(kept) >= min_keep:
            return kept
        extras = [d for d in docs if d not in kept]
        return kept + extras[: max(0, min_keep - len(kept))]

    # -------------------------------
    # 스코어링(문서/코드 공용)
    # -------------------------------
    def _custom_doc_score(self, query: str, doc: Document, intents: Optional[List[str]] = None, code_mode: bool = False) -> float:
        q = (query or "").lower()
        text = (doc.page_content or "").lower()
        meta = (doc.metadata or {})
        libs = (meta.get("libraries") or "").lower()
        fname = (meta.get("filename") or "").lower()
        mtypes = (meta.get("model_types") or "").lower()

        score = 0.0

        # 공통 가산: 대표 라이브러리/키워드
        if any(k in text or k in libs for k in [
            "torch", "pytorch", "tensorflow", "keras", "sklearn",
            "kneighborsclassifier", "svm", "conv2d", "lstm", "gru", "transformer",
            "langchain", "lcel", "as_retriever", "similarity_search", "invoke",
            "vectorstore", "chroma", "faiss", "bm25", "ensemble"
        ]):
            score += 1.5

        # 파일명 매칭 보너스 (예: rag_*.ipynb, langchain_*.ipynb)
        if any(k in fname for k in ["rag", "langchain", "retriever", "bm25", "chroma", "faiss"]):
            score += 1.0

        # 코드 셀 우선
        if code_mode and (meta.get("content_type") == "code"):
            score += 0.7

        # 딥러닝 쿼리에서 지도/DB 감점
        if any(k in q for k in ["딥러닝", "deep learning", "cnn", "rnn", "lstm", "gru", "transformer"]):
            if any(k in text for k in ["folium", "heatmap", "map("]):
                score -= 2.0
            if any(k in text for k in ["sqlite", "sqlite3", " select ", " create table "]):
                score -= 0.5

        # 설치/환경 커맨드 감점 소폭
        if any(sym in text for sym in ["!pip install", "!apt-get", "%pip install"]):
            score -= 0.4

        # 의도 일치/불일치 보정
        if intents:
            if self._doc_matches_intent(doc, intents):
                score += 3.5
            else:
                # 다른 신호 강하면 감점
                other_by = {
                    "cnn": ["lstm", "gru", "transformer", "bert", "gpt", "svm", "knn"],
                    "rnn": ["cnn", "transformer", "bert", "gpt", "svm", "knn"],
                    "lstm": ["cnn", "transformer", "bert", "gpt", "svm", "knn"],
                    "gru": ["cnn", "transformer", "bert", "gpt", "svm", "knn"],
                    "transformer": ["cnn", "lstm", "gru", "svm", "knn"],
                    "bert": ["cnn", "lstm", "gru", "svm", "knn"],
                    "gpt": ["cnn", "lstm", "gru", "svm", "knn"],
                    "knn": ["cnn", "transformer", "bert", "gpt", "lstm", "gru", "svm"],
                    "svm": ["cnn", "transformer", "bert", "gpt", "lstm", "gru", "knn"],
                    "langchain": ["pandas", "folium", "sqlite", "matplotlib"],  # 비핵심
                    "retriever": ["folium", "sqlite"],
                    "rag": ["folium", "sqlite"],
                    "vectorstore": ["folium", "sqlite"],
                    "langgraph": ["folium", "sqlite"],
                }
                flat_others = set()
                for it in intents:
                    flat_others.update(other_by.get(it, []))
                if any(k in text for k in flat_others) and not any(k in text for k in intents):
                    score -= 2.0
                else:
                    score -= 0.8

        # 모델타입 일치 보너스
        if mtypes and intents:
            if any(it in mtypes for it in intents):
                score += 1.0

        return score

    # -------------------------------
    # 질의 확장 (의도 기반)
    # -------------------------------
    def _expand_queries(self, text: str) -> List[str]:
        t = text.lower().strip()
        intents = self._intent_models(t)

        exp_map: Dict[str, List[str]] = {
            # 모델
            "cnn": ["convolution", "conv2d", "convolutional", "resnet", "vgg", "cnn 코드", "cnn 예제"],
            "rnn": ["recurrent", "sequence model", "rnn 코드", "rnn 예제"],
            "lstm": ["lstm 코드", "lstm 예제"],
            "gru": ["gru 코드", "gru 예제"],
            "transformer": ["attention", "encoder layer", "decoder layer", "transformer 코드", "transformer 예제"],
            "bert": ["bert 코드", "bert fine-tuning"],
            "gpt": ["gpt 코드", "gpt fine-tuning"],
            "knn": ["k-nearest neighbors", "kneighborsclassifier", "knn 코드", "knn 예제"],
            "svm": ["support vector machine", "svc", "svr", "svm 코드", "svm 예제"],
            # 주제
            "langchain": ["lcel", "runnable", "langchain_community", "langchain_openai", "as_retriever", "similarity_search", "invoke", "get_relevant_documents"],
            "retriever": ["as_retriever", "get_relevant_documents", "similarity_search", "bm25", "ensemble"],
            "rag": ["retrieval augmented generation", "retriever", "vectorstore", "context", "prompt", "rag 코드", "rag 예제"],
            "vectorstore": ["chroma", "faiss", "persist_directory", "embedding_function", "vectorstore 코드"],
            "langgraph": ["stategraph", "node", "edge", "start", "end", "runnable", "langgraph 코드"],
            "openai": ["chatopenai", "openai api", "openai 코드"],
        }

        qs = [text]
        if intents:
            for it in intents:
                qs += exp_map.get(it, [])
        else:
            # 의도 없을 때만 일반 확장
            generic = {"딥러닝": ["deep learning", "neural network", "신경망", "torch", "tensorflow", "keras"]}
            for key, adds in generic.items():
                if key in t:
                    qs += adds

        lone_terms = ["cnn", "rnn", "lstm", "gru", "transformer", "knn", "svm", "딥러닝", "rag", "langchain", "retriever"]
        if t in lone_terms:
            qs += [f"{t} 코드", f"{t} 예제", f"{t} 구현", f"{t} 모델", f"{t} 학습"]

        qs = list(dict.fromkeys([q for q in qs if q and q.strip()]))[1:2]
        return qs

    # -------------------------------
    # 문서 검색 (외부 호출)
    # -------------------------------
    def search_documents(self, query: str, filter_metadata: Optional[Dict] = None, k: Optional[int] = None) -> List[Document]:
        if not self.vectorstore:
            return []
        kk = k or self.k
        try:
            if filter_metadata:
                # 필터가 있는 경우, re-ranking을 위해 더 많은 문서를 가져온 후 re-rank
                retrieve_k = self.rerank_top_k if self.use_reranking and self.reranker else kk
                docs = self.vectorstore.similarity_search(query, k=retrieve_k, filter=filter_metadata)

                # Re-ranking 적용
                if self.use_reranking and self.reranker and docs:
                    docs = self._rerank_documents(query, docs, kk)
                else:
                    docs = docs[:kk]

                return docs
            else:
                return self._search_once(query, kk)
        except Exception as e:
            print(f"문서 검색 중 오류: {e}")
            return []

    # -------------------------------
    # 날짜/주차 유틸 & 강의 요약/회고
    # -------------------------------
    def _normalize_week_phrase(self, s: str) -> str:
        ord_map = {"첫째":"1","첫번째":"1","둘째":"2","두번째":"2","셋째":"3","세번째":"3","넷째":"4","네번째":"4","다섯째":"5","다섯번째":"5"}
        for k, v in ord_map.items():
            s = s.replace(f"{k} 주", f"{v}주").replace(f"{k}주", f"{v}주")
        s = re.sub(r"(\d+)\s*번째\s*주", r"\1주", s)
        s = re.sub(r"(\d+)\s*주(?!차)", r"\1주차", s)
        return " ".join(s.split()).strip()

    def _extract_date_from_text(self, text: str) -> Optional[str]:
        s = text.strip()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
        if m: return m.group(1)
        m = re.search(r"\b(\d{8})\b", s)
        if m:
            d = m.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
        if m:
            year = dt.datetime.now().year
            month = int(m.group(1)); day = int(m.group(2))
            try:
                return dt.date(year, month, day).strftime("%Y-%m-%d")
            except Exception:
                pass
        parsed = dateparser.parse(s, languages=["ko", "en"])
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        return None

    def _week_range(self, year: int, month: int, week: int) -> Tuple[dt.date, dt.date, List[str]]:
        first = dt.date(year, month, 1)
        first_monday = first + dt.timedelta(days=(7 - first.weekday()) % 7)  # 0=월
        start = first_monday + dt.timedelta(weeks=week - 1)
        end = start + dt.timedelta(days=4)
        days = [(start + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
        return start, end, days

    def _get_docs_for_dates(self, days: List[str], k_per_day: int = 30) -> Dict[str, List[Document]]:
        grouped: Dict[str, List[Document]] = {d: [] for d in days}
        seen = set()
        for d in days:
            docs = self.search_documents("강의 내용 요약", {"lecture_date": d}, k=k_per_day)
            for doc in docs:
                if doc.metadata.get("lecture_date") != d:
                    continue
                h = hash(doc.page_content.strip())
                if h in seen:
                    continue
                seen.add(h)
                grouped[d].append(doc)
        return grouped

    def _summarize_days(self, label: str, days: List[str], period_str: str = "", k_per_day: int = 30) -> str:
        grouped = self._get_docs_for_dates(days, k_per_day=k_per_day)
        any_docs = any(len(v) > 0 for v in grouped.values())
        if not any_docs:
            return f"{label} 강의 자료를 찾을 수 없습니다."

        parts = []
        for d in days:
            docs = grouped.get(d, [])
            if not docs:
                parts.append(f"### {d}\n- 자료 없음")
                continue
            parts.append(f"### {d}")
            for doc in docs:
                fn = doc.metadata.get("filename", "Unknown")
                cell = doc.metadata.get("cell_index", "")
                snippet = doc.page_content.strip().replace("\n", " ")
                if len(snippet) > 500:
                    snippet = snippet[:500] + " ..."
                parts.append(f"- 파일: {fn}  셀:{cell}\n  내용: {snippet}")

        allowed_dates_str = ", ".join(days)
        context = "\n".join(parts)

        prompt = (
            f"{label}{(' ' + period_str) if period_str else ''} 강의 자료 요약을 작성하세요.\n"
            f"반드시 아래 '허용 날짜'에 포함된 날짜만 언급하세요. 다른 날짜는 절대 언급하지 마세요.\n"
            f"허용 날짜: {allowed_dates_str}\n\n"
            f"{context}\n\n"
            "요약 지침:\n"
            "- 날짜별(YYYY-MM-DD) 소제목으로 정리 (허용 날짜만)\n"
            "- 각 날짜에 대해: 다룬 주제 / 실습 코드·모델 / 핵심 개념을 간결히\n"
            "- 문서가 없는 날짜는 '자료 없음' 명시\n"
            "출력 형식:\n"
            "1) 날짜별 항목 나열\n"
            "2) 마지막에 '참고 자료'로 파일명 목록 (있으면)\n"
        )
        try:
            summary = self.llm.invoke(prompt).content
        except Exception as e:
            return f"요약 생성 중 오류가 발생했습니다: {e}"

        refs = []
        for d in days:
            for doc in grouped.get(d, []):
                fn = doc.metadata.get("filename", "Unknown")
                cell = doc.metadata.get("cell_index", "")
                refs.append((d, fn, str(cell)))
        refs = sorted(list({(d, fn, cell) for (d, fn, cell) in refs}), key=lambda x: (x[0], x[1], x[2]))
        if refs:
            summary += "\n\n참고 자료\n" + "\n".join([f"{d} {fn}{' (셀:'+cell+')' if cell else ''}" for d, fn, cell in refs])
        return summary

    def get_lecture_summary(self, date: str) -> str:
        return self._summarize_days(date, [date], k_per_day=30)

    def get_lecture_summary_any(self, date_text: str) -> str:
        try:
            txt = self._normalize_week_phrase(date_text)
            m = re.search(r'(\d{1,2})\s*월\s*(\d{1,2})\s*주차', txt)
            if m:
                month, week = int(m.group(1)), int(m.group(2))
                year = dt.datetime.now().year
                start, end, days = self._week_range(year, month, week)
                return self._summarize_days(f"{month}월 {week}주차", days, f"({start}~{end})", k_per_day=30)

            day = self._extract_date_from_text(txt)
            if not day:
                return f"날짜를 인식할 수 없습니다: {date_text}"
            return self._summarize_days(day, [day], k_per_day=30)
        except Exception as e:
            return f"강의 요약 처리 중 오류: {e}"

    # -------------------------------
    # 회고록(블로그 스타일)
    # -------------------------------
    def _reflect_days(self, label: str, days: List[str], period_str: str = "", k_per_day: int = 30) -> str:
        grouped = self._get_docs_for_dates(days, k_per_day=k_per_day)
        any_docs = any(len(v) > 0 for v in grouped.values())
        if not any_docs:
            return f"{label} 자료가 없어 회고록을 작성할 수 없습니다."

        parts = []
        for d in days:
            docs = grouped.get(d, [])
            if not docs:
                continue
            parts.append(f"### {d}")
            for doc in docs:
                fn = doc.metadata.get("filename", "Unknown")
                cell = doc.metadata.get("cell_index", "")
                snippet = doc.page_content.strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400] + " ..."
                parts.append(f"- 파일: {fn}  셀:{cell}  내용: {snippet}")
        context = "\n".join(parts)
        allowed = ", ".join(days)

        base_env = os.environ.get("CAMP_START_DATE")
        day_range = ""
        if base_env:
            try:
                base = dt.datetime.strptime(base_env, "%Y-%m-%d").date()
                nums = []
                for d in days:
                    di = dt.datetime.strptime(d, "%Y-%m-%d").date()
                    nums.append((di - base).days + 1)
                nums.sort()
                if nums:
                    day_range = f"DAY{nums[0]}" if len(nums) == 1 else f"DAY{nums[0]}~DAY{nums[-1]}"
            except Exception:
                pass

        title = f"[SKN_AI_CAMP16] {label} 회고"
        if period_str:
            title += f" {period_str}"
        if day_range:
            title = f"{title}\n\n{day_range}"

        prompt = f"""
아래 컨텍스트를 바탕으로 '{label}{(' ' + period_str) if period_str else ''}'에 대한 '회고록'을 작성하세요.
반드시 '허용 날짜'에 포함된 날짜만 언급하고, 제공 자료에서 확인 가능한 사실만 활용하세요.

허용 날짜: {allowed}

컨텍스트:
{context}

출력 형식(마크다운, 블로그 스타일):
# {title}

## 개요
- 한 주(혹은 하루) 전체를 2~3문장으로 요약

## 날짜별 하이라이트
- 각 날짜(YYYY-MM-DD)별로:
  - ✔️ 잘 된 점(Wins)
  - ❗ 어려웠던 점(Challenges)
  - 💡 배운 점(Lessons)
  - 🔧 개선 아이디어
  - ▶ 다음 액션(Action Items, SMART 1~2개)

## 전반적 회고
- 성취
- 개선할 점
- 리스크/의존성(있으면)
- 다음 주 계획(SMART 2~3개)

규칙:
- 허용 날짜 외의 날짜는 절대 언급하지 말 것
- 추측 금지, 자료 없으면 '자료 없음'이라고 간단히 표기
"""
        try:
            return self.llm.invoke(prompt).content
        except Exception as e:
            return f"회고록 생성 중 오류가 발생했습니다: {e}"

    def get_reflection_any(self, date_text: str) -> str:
        try:
            txt = self._normalize_week_phrase(date_text)
            m = re.search(r'(\d{1,2})\s*월\s*(\d{1,2})\s*주차', txt)
            if m:
                month, week = int(m.group(1)), int(m.group(2))
                year = dt.datetime.now().year
                start, end, days = self._week_range(year, month, week)
                return self._reflect_days(f"{month}월 {week}주차", days, f"({start}~{end})", k_per_day=30)

            day = self._extract_date_from_text(txt)
            if not day:
                return f"날짜를 인식할 수 없습니다: {date_text}"
            return self._reflect_days(day, [day], k_per_day=20)
        except Exception as e:
            return f"회고록 처리 중 오류: {e}"

    # -------------------------------
    # 코드 검색/설명
    # -------------------------------
    def _looks_like_code_intent(self, text: str) -> bool:
        t = text.lower().strip()
        model_terms = ["cnn", "rnn", "lstm", "gru", "knn", "svm", "bert", "gpt", "transformer", "딥러닝",
                       "rag", "langchain", "retriever", "chroma", "faiss", "bm25", "vectorstore"]
        return (
            ("코드" in text) or ("예제" in text) or ("구현" in text) or ("소스" in text) or ("작성" in text)
            or ("샘플" in text) or ("sample" in t) or ("code" in t)
            or t in model_terms
        )

    def _strong_code_search(self, query: str, k: int = 20) -> List[Document]:
        intents = self._intent_models(query)

        # Re-ranking을 고려한 검색 수 조정
        search_k = max(40, k * 2) if self.use_reranking and self.reranker else max(20, k)

        # 1) 코드 메타 우선
        code_first = self.search_documents(query, filter_metadata={"content_type": "code"}, k=search_k)

        # 2) 의도 확장
        exp_qs = self._expand_queries(query)
        broad = self._multi_query_search(exp_qs, k_each=search_k, k_final=120)

        # 3) 코드만 + 합치기
        pool = []
        pool.extend(code_first)
        pool.extend([d for d in broad if (d.metadata or {}).get("content_type") == "code"])
        pool = self._dedup_docs(pool, limit=250)

        # 4) 의도 필터
        pool = self._filter_by_intent(pool, intents, min_keep=6)

        # 5) 스코어링
        scored = sorted(pool, key=lambda d: self._custom_doc_score(query, d, intents=intents, code_mode=True), reverse=True)

        # 6) Re-ranking 적용 (스코어링 후)
        if self.use_reranking and self.reranker and scored:
            scored = self._rerank_documents(query, scored, k)
        else:
            scored = scored[:k]

        return scored

    def get_code_snippets(self, query: str) -> List[Dict[str, Any]]:
        code_docs = self._strong_code_search(query, k=20)
        return [
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "cell_index": doc.metadata.get("cell_index", "Unknown"),
                "lecture_date": doc.metadata.get("lecture_date", "Unknown"),
                "libraries": doc.metadata.get("libraries", "").split(", ") if doc.metadata.get("libraries") else [],
                "model_types": doc.metadata.get("model_types", "").split(", ") if doc.metadata.get("model_types") else []
            }
            for doc in code_docs
        ]

    # -------------------------------
    # Q/A (문서 부족 시 의도 기반 재랭킹)
    # -------------------------------
    def answer_question(self, question: str) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"answer": "벡터 DB가 로드되지 않았습니다. 먼저 DB를 구축해주세요.", "sources": [], "metadata": {}}
        try:
            intents = self._intent_models(question)

            # 1차: 원 쿼리
            docs = self.search_documents(question, k=max(self.k, 16))

            # 2차: 확장 쿼리
            if len(docs) < max(4, self.k):
                exp_qs = self._expand_queries(question)
                more = self._multi_query_search(exp_qs, k_each=18, k_final=60)
                docs = self._dedup_docs(docs + more, limit=max(20, self.k * 4))

            # 3차: 의도 필터 & 스코어 재정렬 (설치 커맨드/무관 내용 하향)
            docs = self._filter_by_intent(docs, intents, min_keep=8)
            docs = sorted(docs, key=lambda d: self._custom_doc_score(question, d, intents=intents, code_mode=False), reverse=True)

            if not docs:
                return {"answer": "관련된 정보를 찾을 수 없습니다.", "sources": [], "metadata": {}}

            # 컨텍스트 빌드
            topn = min(len(docs), max(10, self.k))
            context_parts, sources = [], []
            for i, doc in enumerate(docs[:topn]):
                filename = doc.metadata.get("filename", "Unknown")
                content_type = doc.metadata.get("content_type", "Unknown")
                cell_index = doc.metadata.get("cell_index", "")
                context_parts.append(f"[문서 {i+1}] {filename} ({content_type})\n{doc.page_content}")
                sources.append({
                    "filename": filename,
                    "content_type": content_type,
                    "cell_index": cell_index,
                    "lecture_date": doc.metadata.get("lecture_date", "Unknown")
                })

            context = "\n\n".join(context_parts)
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            return {"answer": response.content, "sources": sources, "metadata": {"num_sources": len(docs)}}
        except Exception as e:
            return {"answer": f"답변 생성 중 오류: {e}", "sources": [], "metadata": {}}

    # -------------------------------
    # 라우팅
    # -------------------------------
    def _handle_summary(self, text: str) -> str:
        return self.get_lecture_summary_any(text)

    def _handle_code_search(self, text: str) -> str:
        snippets = self.get_code_snippets(text)
        if not snippets:
            return "관련 코드 스니펫을 찾지 못했습니다."
        blocks = []
        for s in snippets[: max(self.k, 5) ]:
            code = s["content"]
            if len(code) > 2500:
                code = code[:2500] + "\n# ... (생략)"
            blocks.append(
                f"{s['filename']} (날짜: {s['lecture_date']}, 셀: {s['cell_index']})\n\n"
                f"```python\n{code}\n```"
            )
        return "\n\n".join(blocks)

    def _handle_code_explain(self, text: str) -> str:
        return self.explain_code(text)

    def _handle_default(self, text: str) -> str:
        return self.answer_question(text).get("answer", "답변 생성 실패")

    def handle_user_input(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""

        # 명시적 접두사
        if s.startswith("코드 검색:"):
            return self._handle_code_search(s.split(":", 1)[1].strip())
        if s.startswith("강의 요약:"):
            return self._handle_summary(s.split(":", 1)[1].strip())
        if s.startswith("코드 설명:"):
            return self._handle_code_explain(s.split(":", 1)[1].strip())
        if s.startswith("회고록:"):
            return self.get_reflection_any(s.split(":", 1)[1].strip())

        # 회고록 자연어
        if ("회고록" in s):
            cleaned = (
                s.replace("회고록", " ")
                 .replace("만들어줘", " ")
                 .replace("작성", " ")
                 .replace("적어줘", " ")
                 .replace("써줘", " ")
                 .replace("해줘", " ")
            )
            cleaned = " ".join(cleaned.split())
            return self.get_reflection_any(cleaned)

        # 주차 패턴
        normalized = self._normalize_week_phrase(s)
        if re.search(r'\d{1,2}\s*월\s*\d{1,2}\s*주차', normalized):
            return self._handle_summary(normalized)

        # 문장 속 날짜 → 강의 요약
        if self._extract_date_from_text(s):
            return self._handle_summary(s)

        # 코드 의도
        if self._looks_like_code_intent(s):
            return self._handle_code_search(s)

        # 일반 Q/A
        return self._handle_default(s)

    # -------------------------------
    # 코드 설명
    # -------------------------------
    def explain_code(self, code_content: str, context: str = "") -> str:
        prompt = (
            "다음 Python 코드를 분석하고 설명해주세요:\n\n"
            f"{context}\n\n"
            "코드:\n"
            "```python\n"
            f"{code_content}\n"
            "```\n\n"
            "설명 시 포함할 내용:\n"
            "1. 코드 목적과 기능\n"
            "2. 주요 구성 요소\n"
            "3. 사용된 라이브러리\n"
            "4. 동작 순서\n"
            "5. 주의사항/개선점\n"
        )
        try:
            return self.llm.invoke(prompt).content
        except Exception as e:
            return f"코드 설명 생성 중 오류: {e}"


def main():
    rag = RAGSystem()
    print("== 주차 요약 ==")
    print(rag.handle_user_input("강의 요약: 6월 3주차"))
    print("\n== 단일 날짜 요약 ==")
    print(rag.handle_user_input("6월 11일 강의요약해줘"))
    print("\n== 코드 검색 (자연어) ==")
    print(rag.handle_user_input("CNN 모델 만드는 코드 찾아줘"))
    print("\n== 코드 검색 (키워드 단독) ==")
    print(rag.handle_user_input("KNN"))
    print("\n== 주제 검색 (LangChain) ==")
    print(rag.handle_user_input("langchain"))
    print("\n== 주제 검색 (RAG) ==")
    print(rag.handle_user_input("RAG"))
    print("\n== 주제 검색 (Retriever) ==")
    print(rag.handle_user_input("Retriever"))
    print("\n== 회고록 ==")
    print(rag.handle_user_input("회고록: 7월 3주차"))


if __name__ == "__main__":
    main()