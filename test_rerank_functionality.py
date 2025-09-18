#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-ranking 기능 테스트 스크립트
RAG 시스템의 re-ranking 기능을 테스트하고 비교합니다.
"""

import os
import sys
from typing import List, Dict, Any
from src.rag_system import RAGSystem
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def test_reranking():
    """Re-ranking 기능을 테스트합니다."""

    print("=" * 60)
    print("Re-ranking 기능 테스트")
    print("=" * 60)

    # 1. 기본 RAG 시스템 (re-ranking 없음)
    print("\n1. 기본 RAG 시스템 초기화 (re-ranking 비활성화)")
    rag_basic = RAGSystem(
        use_reranking=False,
        k=5
    )

    # 2. Re-ranking RAG 시스템
    print("\n2. Re-ranking RAG 시스템 초기화 (re-ranking 활성화)")
    rag_rerank = RAGSystem(
        use_reranking=True,
        rerank_model="dragonkue/bge-reranker-v2-m3-ko",
        rerank_top_k=10,
        k=5
    )

    # 테스트 쿼리들
    test_queries = [
        "CNN 모델 구현",
        "RNN LSTM 예제",
        "LangChain RAG 시스템",
        "벡터 데이터베이스 Chroma",
        "딥러닝 모델 훈련"
    ]

    print("\n" + "=" * 60)
    print("검색 결과 비교")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 40}")
        print(f"테스트 쿼리 {i}: {query}")
        print(f"{'-' * 40}")

        # 기본 검색
        print("\n[기본 검색 결과]")
        try:
            basic_docs = rag_basic.search_documents(query, k=5)
            if basic_docs:
                for j, doc in enumerate(basic_docs[:3], 1):
                    filename = doc.metadata.get("filename", "Unknown")
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    print(f"  {j}. {filename}: {content_preview}...")
            else:
                print("  검색 결과 없음")
        except Exception as e:
            print(f"  오류: {e}")

        # Re-ranking 검색
        print("\n[Re-ranking 검색 결과]")
        try:
            rerank_docs = rag_rerank.search_documents(query, k=5)
            if rerank_docs:
                for j, doc in enumerate(rerank_docs[:3], 1):
                    filename = doc.metadata.get("filename", "Unknown")
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    print(f"  {j}. {filename}: {content_preview}...")
            else:
                print("  검색 결과 없음")
        except Exception as e:
            print(f"  오류: {e}")

        print()


def test_code_search_reranking():
    """코드 검색에서의 re-ranking 테스트"""

    print("\n" + "=" * 60)
    print("코드 검색 Re-ranking 테스트")
    print("=" * 60)

    # Re-ranking 활성화된 시스템
    rag_rerank = RAGSystem(
        use_reranking=True,
        rerank_model="dragonkue/bge-reranker-v2-m3-ko",
        rerank_top_k=10,
        k=3
    )

    code_queries = [
        "코드 검색: CNN 모델",
        "코드 검색: LSTM 구현",
        "코드 검색: LangChain retriever"
    ]

    for i, query in enumerate(code_queries, 1):
        print(f"\n{'-' * 40}")
        print(f"코드 검색 테스트 {i}: {query}")
        print(f"{'-' * 40}")

        try:
            # 코드 검색 (내부적으로 re-ranking 적용됨)
            result = rag_rerank.handle_user_input(query)
            print(result[:500] + "..." if len(result) > 500 else result)
        except Exception as e:
            print(f"오류: {e}")


def test_qa_reranking():
    """Q&A에서의 re-ranking 테스트"""

    print("\n" + "=" * 60)
    print("Q&A Re-ranking 테스트")
    print("=" * 60)

    # Re-ranking 활성화된 시스템
    rag_rerank = RAGSystem(
        use_reranking=True,
        rerank_model="dragonkue/bge-reranker-v2-m3-ko",
        rerank_top_k=10,
        k=5
    )

    qa_queries = [
        "CNN과 RNN의 차이점은 무엇인가요?",
        "LangChain에서 retriever는 어떻게 사용하나요?",
        "벡터 데이터베이스의 장점은 무엇인가요?"
    ]

    for i, query in enumerate(qa_queries, 1):
        print(f"\n{'-' * 40}")
        print(f"Q&A 테스트 {i}: {query}")
        print(f"{'-' * 40}")

        try:
            result = rag_rerank.answer_question(query)
            answer = result.get("answer", "답변 없음")
            sources = result.get("sources", [])

            print(f"답변: {answer[:300]}...")
            print(f"참조 소스 수: {len(sources)}")
            if sources:
                print("주요 소스:")
                for j, source in enumerate(sources[:2], 1):
                    print(f"  {j}. {source.get('filename', 'Unknown')}")
        except Exception as e:
            print(f"오류: {e}")


def main():
    """메인 테스트 함수"""

    # 벡터 DB 존재 확인
    if not os.path.exists("./chroma_db"):
        print("오류: 벡터 데이터베이스가 없습니다. 먼저 벡터 DB를 구축해주세요.")
        print("python src/vector_db_builder.py 실행 후 다시 시도하세요.")
        return

    try:
        # 기본 검색 vs Re-ranking 검색 비교
        test_reranking()

        # 코드 검색에서의 re-ranking
        test_code_search_reranking()

        # Q&A에서의 re-ranking
        test_qa_reranking()

        print("\n" + "=" * 60)
        print("Re-ranking 테스트 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()