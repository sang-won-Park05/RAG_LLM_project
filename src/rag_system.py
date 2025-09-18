"""
RAG 시스템 메인 클래스
벡터 검색과 LLM을 결합한 질의응답 시스템
"""

import os
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class RAGSystem:
    """RAG 시스템 메인 클래스"""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_model: str = "google/embeddinggemma-300m",
        llm_model: str = "gpt-4o-mini",
        k: int = 5
    ):
        """
        RAG 시스템 초기화

        Args:
            db_path: Chroma DB 경로
            embedding_model: 임베딩 모델명
            llm_model: LLM 모델명 (OpenAI)
            k: 검색할 문서 수
        """
        self.db_path = db_path
        self.k = k

        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # LLM 초기화 (OpenAI ChatGPT-4o-mini)
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0.1,
            max_tokens=2000
        )

        # 벡터 스토어 로드
        self.vectorstore = None
        self.ensemble_retriever = None
        self._load_vectorstore()

        # 프롬프트 템플릿
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
다음은 LLM 강의 자료에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 답해주세요:
질문: {question}

답변 시 다음 사항을 고려해주세요:
1. 제공된 강의 자료의 내용만을 기반으로 답변하세요
2. 소스 코드가 포함된 경우, 해당 파일명과 셀 번호를 명시하세요
3. 강의 날짜가 언급된 경우, 해당 날짜를 포함하여 답변하세요
4. 확실하지 않은 내용은 추측하지 말고 "제공된 자료에서 확인할 수 없습니다"라고 말하세요

답변:
            """
        )

    def _load_vectorstore(self):
        """벡터 스토어 로드"""
        if not os.path.exists(self.db_path):
            print(f"벡터 DB가 존재하지 않습니다: {self.db_path}")
            return

        try:
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

            # 앙상블 리트리버 설정
            self._setup_ensemble_retriever()
            print("벡터 스토어가 성공적으로 로드되었습니다.")

        except Exception as e:
            print(f"벡터 스토어 로드 중 오류: {e}")

    def _setup_ensemble_retriever(self):
        """앙상블 리트리버 설정 (벡터 검색 + BM25)"""
        if not self.vectorstore:
            return

        try:
            # 모든 문서 가져오기
            all_docs = self.vectorstore.get()
            documents = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
            ]

            # BM25 리트리버
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = self.k

            # 벡터 검색 리트리버
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

            # 앙상블 리트리버 (벡터: 0.7, BM25: 0.3)
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )

        except Exception as e:
            print(f"앙상블 리트리버 설정 중 오류: {e}")
            # 오류 시 벡터 검색만 사용
            self.ensemble_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def search_documents(self, query: str, filter_metadata: Optional[Dict] = None) -> List[Document]:
        """
        문서 검색

        Args:
            query: 검색 쿼리
            filter_metadata: 메타데이터 필터

        Returns:
            검색된 문서 리스트
        """
        if not self.vectorstore:
            return []

        try:
            if filter_metadata:
                # 메타데이터 필터가 있는 경우 벡터 검색만 사용
                results = self.vectorstore.similarity_search(
                    query,
                    k=self.k,
                    filter=filter_metadata
                )
            else:
                # 앙상블 리트리버 사용
                results = self.ensemble_retriever.get_relevant_documents(query)

            return results

        except Exception as e:
            print(f"문서 검색 중 오류: {e}")
            return []

    def get_code_snippets(self, query: str) -> List[Dict[str, Any]]:
        """
        코드 스니펫 검색

        Args:
            query: 검색 쿼리

        Returns:
            코드 스니펫 정보 리스트
        """
        # 코드 타입만 검색
        filter_metadata = {"content_type": "code"}
        code_docs = self.search_documents(query, filter_metadata)

        code_snippets = []
        for doc in code_docs:
            snippet_info = {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "cell_index": doc.metadata.get("cell_index", "Unknown"),
                "lecture_date": doc.metadata.get("lecture_date", "Unknown"),
                "libraries": doc.metadata.get("libraries", "").split(", ") if doc.metadata.get("libraries") else [],
                "model_types": doc.metadata.get("model_types", "").split(", ") if doc.metadata.get("model_types") else []
            }
            code_snippets.append(snippet_info)

        return code_snippets

    def get_lecture_summary(self, date: str) -> str:
        """
        특정 날짜 강의 요약

        Args:
            date: 강의 날짜 (YYYY-MM-DD 형식)

        Returns:
            강의 요약
        """
        filter_metadata = {"lecture_date": date}
        docs = self.search_documents("강의 내용 요약", filter_metadata)

        if not docs:
            return f"{date} 날짜의 강의 자료를 찾을 수 없습니다."

        # 해당 날짜의 모든 문서를 컨텍스트로 사용
        context = "\\n\\n".join([doc.page_content for doc in docs])

        summary_prompt = f"""
다음은 {date} 강의 자료입니다:

{context}

위 자료를 바탕으로 {date} 강의의 주요 내용을 요약해주세요.
- 다룬 주제들
- 실습한 코드나 모델
- 학습 목표
- 주요 개념

요약:
        """

        try:
            response = self.llm.invoke(summary_prompt)
            return response.content
        except Exception as e:
            return f"요약 생성 중 오류가 발생했습니다: {e}"

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            답변 정보 (answer, sources, metadata)
        """
        if not self.vectorstore:
            return {
                "answer": "벡터 DB가 로드되지 않았습니다. 먼저 DB를 구축해주세요.",
                "sources": [],
                "metadata": {}
            }

        try:
            # 관련 문서 검색
            relevant_docs = self.search_documents(question)

            if not relevant_docs:
                return {
                    "answer": "관련된 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "metadata": {}
                }

            # 컨텍스트 구성
            context_parts = []
            sources = []

            for i, doc in enumerate(relevant_docs):
                filename = doc.metadata.get("filename", "Unknown")
                content_type = doc.metadata.get("content_type", "Unknown")
                cell_index = doc.metadata.get("cell_index", "")

                context_parts.append(f"[문서 {i+1}] {filename} ({content_type})\\n{doc.page_content}")

                source_info = {
                    "filename": filename,
                    "content_type": content_type,
                    "cell_index": cell_index,
                    "lecture_date": doc.metadata.get("lecture_date", "Unknown")
                }
                sources.append(source_info)

            context = "\\n\\n".join(context_parts)

            # 프롬프트 생성 및 LLM 호출
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)

            return {
                "answer": response.content,
                "sources": sources,
                "metadata": {
                    "num_sources": len(relevant_docs),
                    "query": question
                }
            }

        except Exception as e:
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {e}",
                "sources": [],
                "metadata": {}
            }

    def explain_code(self, code_content: str, context: str = "") -> str:
        """
        코드 설명 생성

        Args:
            code_content: 설명할 코드
            context: 추가 컨텍스트

        Returns:
            코드 설명
        """
        explanation_prompt = f"""
다음 Python 코드를 분석하고 설명해주세요:

{context}

코드:
```python
{code_content}
```

설명 시 포함할 내용:
1. 코드의 전체적인 목적과 기능
2. 주요 구성 요소 (클래스, 함수, 변수 등)
3. 사용된 라이브러리와 그 역할
4. 코드의 동작 순서
5. 주의사항이나 개선점

설명:
        """

        try:
            response = self.llm.invoke(explanation_prompt)
            return response.content
        except Exception as e:
            return f"코드 설명 생성 중 오류가 발생했습니다: {e}"


def main():
    """테스트용 메인 함수"""
    rag_system = RAGSystem()

    # 테스트 질문들
    test_questions = [
        "CNN 모델 만드는 소스 코드 찾아줘",
        "RNN과 CNN 비교 설명해줘",
        "RAG 시스템이 뭐야?"
    ]

    for question in test_questions:
        print(f"\\n질문: {question}")
        result = rag_system.answer_question(question)
        print(f"답변: {result['answer']}")
        print(f"소스 수: {result['metadata'].get('num_sources', 0)}")


if __name__ == "__main__":
    main()