#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-ranking 기능 간단 테스트
실제 데이터베이스 없이 re-ranking 모델 로딩 및 기본 기능을 테스트합니다.
"""

import sys
import os

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from sentence_transformers import CrossEncoder
    print("✅ sentence-transformers 패키지가 설치되어 있습니다.")
    RERANKER_AVAILABLE = True
except ImportError:
    print("❌ sentence-transformers 패키지가 설치되지 않았습니다.")
    RERANKER_AVAILABLE = False


def test_reranker_model():
    """Re-ranking 모델 로딩 테스트"""

    if not RERANKER_AVAILABLE:
        print("sentence-transformers가 없어서 테스트를 건너뜁니다.")
        return False

    print("\n" + "=" * 50)
    print("Re-ranking 모델 로딩 테스트")
    print("=" * 50)

    model_name = "dragonkue/bge-reranker-v2-m3-ko"
    print(f"모델 로딩 중: {model_name}")

    try:
        reranker = CrossEncoder(model_name)
        print("✅ Re-ranking 모델 로딩 성공!")

        # 간단한 re-ranking 테스트
        query = "딥러닝 CNN 모델"
        documents = [
            "CNN은 Convolutional Neural Network의 줄임말입니다.",
            "자연어 처리에서 RNN을 사용할 수 있습니다.",
            "CNN은 이미지 분류에 효과적인 딥러닝 모델입니다.",
            "파이썬에서 리스트를 정렬하는 방법입니다.",
            "Convolutional layer는 특징을 추출합니다."
        ]

        print(f"\n쿼리: '{query}'")
        print("문서들:")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc}")

        # Re-ranking 점수 계산
        query_doc_pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(query_doc_pairs)

        # 점수와 문서를 쌍으로 만들어 정렬
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        print("\nRe-ranking 결과 (점수 순):")
        for i, (score, doc) in enumerate(scored_docs, 1):
            print(f"  {i}. [점수: {score:.4f}] {doc}")

        return True

    except Exception as e:
        print(f"❌ Re-ranking 모델 로딩 실패: {e}")
        return False


def test_rag_system_initialization():
    """RAG 시스템 초기화 테스트"""

    print("\n" + "=" * 50)
    print("RAG 시스템 초기화 테스트")
    print("=" * 50)

    try:
        from src.rag_system import RAGSystem

        # Re-ranking 없는 시스템
        print("1. 기본 RAG 시스템 초기화 (re-ranking 비활성화)")
        rag_basic = RAGSystem(use_reranking=False)
        print("   ✅ 기본 시스템 초기화 성공")

        # Re-ranking 있는 시스템
        print("2. Re-ranking RAG 시스템 초기화 (re-ranking 활성화)")
        rag_rerank = RAGSystem(
            use_reranking=True,
            rerank_model="dragonkue/bge-reranker-v2-m3-ko",
            rerank_top_k=50
        )

        if rag_rerank.reranker is not None:
            print("   ✅ Re-ranking 시스템 초기화 성공")
            return True
        else:
            print("   ⚠️  Re-ranking 시스템 초기화되었지만 모델 로딩 실패")
            return False

    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""

    print("🔍 Re-ranking 기능 간단 테스트")
    print("=" * 60)

    # 1. Re-ranking 모델 단독 테스트
    model_test_success = test_reranker_model()

    # 2. RAG 시스템 통합 테스트
    system_test_success = test_rag_system_initialization()

    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"Re-ranking 모델 테스트: {'✅ 성공' if model_test_success else '❌ 실패'}")
    print(f"RAG 시스템 통합 테스트: {'✅ 성공' if system_test_success else '❌ 실패'}")

    if model_test_success and system_test_success:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("Re-ranking 기능을 사용할 준비가 완료되었습니다.")
        print("\n사용 방법:")
        print("  python main.py --rerank")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다.")
        print("오류를 확인하고 다시 시도해주세요.")


if __name__ == "__main__":
    main()