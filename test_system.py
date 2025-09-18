#!/usr/bin/env python3
"""
RAG 시스템 테스트 스크립트
벡터 DB 구축과 기본 기능을 테스트
"""

import sys
import os

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_db_builder import VectorDBBuilder
from src.rag_system import RAGSystem


def test_vector_db_building():
    """벡터 DB 구축 테스트"""
    print("=" * 60)
    print("🔧 벡터 DB 구축 테스트")
    print("=" * 60)

    try:
        # VectorDBBuilder 인스턴스 생성
        builder = VectorDBBuilder()

        # 데이터 디렉토리 확인
        data_dir = "./data/educational_materials"
        if not os.path.exists(data_dir):
            print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
            return False

        # 벡터 DB 구축
        print("📚 강의 자료 처리 중...")
        vectorstore = builder.build_vector_db(data_dir, "./test_chroma_db")

        if vectorstore:
            print("✅ 벡터 DB 구축 성공!")

            # 기본 통계 정보
            collection = vectorstore._collection
            count = collection.count()
            print(f"📊 총 문서 청크 수: {count}")

            return True
        else:
            print("❌ 벡터 DB 구축 실패")
            return False

    except Exception as e:
        print(f"❌ 벡터 DB 구축 중 오류: {e}")
        return False


def test_rag_system():
    """RAG 시스템 기능 테스트"""
    print("\\n" + "=" * 60)
    print("🤖 RAG 시스템 테스트")
    print("=" * 60)

    try:
        # RAG 시스템 초기화 (OpenAI API 키 필요)
        rag_system = RAGSystem(db_path="./test_chroma_db")

        if not rag_system.vectorstore:
            print("❌ 벡터 스토어 로드 실패")
            return False

        print("✅ RAG 시스템 초기화 성공!")

        # 테스트 질문들
        test_questions = [
            "CNN 모델에 대해 설명해줘",
            "딥러닝 학습 과정",
            "파이썬 코드 예시"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\\n🔍 테스트 {i}: {question}")
            print("-" * 40)

            # 질문 처리
            result = rag_system.answer_question(question)

            # 결과 출력
            print(f"📝 답변: {result['answer'][:200]}...")
            print(f"📚 참고 문서 수: {result['metadata'].get('num_sources', 0)}")

            if result['sources']:
                print("📎 소스 파일:")
                for j, source in enumerate(result['sources'][:3]):
                    print(f"  {j+1}. {source['filename']} ({source['content_type']})")

        return True

    except Exception as e:
        print(f"❌ RAG 시스템 테스트 중 오류: {e}")
        return False


def test_code_search():
    """코드 검색 기능 테스트"""
    print("\\n" + "=" * 60)
    print("💻 코드 검색 테스트")
    print("=" * 60)

    try:
        rag_system = RAGSystem(db_path="./test_chroma_db")

        # 코드 검색 테스트
        search_queries = [
            "CNN",
            "import torch",
            "def"
        ]

        for query in search_queries:
            print(f"\\n🔎 검색어: '{query}'")
            print("-" * 30)

            code_snippets = rag_system.get_code_snippets(query)
            print(f"📊 찾은 코드 스니펫 수: {len(code_snippets)}")

            for i, snippet in enumerate(code_snippets[:3]):
                print(f"  {i+1}. {snippet['filename']} (셀 {snippet['cell_index']})")
                if snippet['libraries']:
                    print(f"     라이브러리: {', '.join(snippet['libraries'])}")

        return True

    except Exception as e:
        print(f"❌ 코드 검색 테스트 중 오류: {e}")
        return False


def test_lecture_summary():
    """강의 요약 기능 테스트"""
    print("\\n" + "=" * 60)
    print("📅 강의 요약 테스트")
    print("=" * 60)

    try:
        rag_system = RAGSystem(db_path="./test_chroma_db")

        # 파일명에서 날짜 추출하여 테스트
        notebooks_dir = "./data/educational_materials/notebooks"
        if os.path.exists(notebooks_dir):
            files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

            if files:
                # 첫 번째 파일의 날짜 추출
                sample_file = files[0]
                print(f"📄 샘플 파일: {sample_file}")

                # 날짜 추출 시도
                import re
                date_pattern = r'(\\d{4}-\\d{2}-\\d{2})'
                match = re.search(date_pattern, sample_file)

                if match:
                    test_date = match.group(1)
                    print(f"📅 테스트 날짜: {test_date}")

                    summary = rag_system.get_lecture_summary(test_date)
                    print(f"📝 요약 (처음 300자): {summary[:300]}...")
                    return True

        print("⚠️ 날짜가 포함된 강의 파일을 찾을 수 없습니다.")
        return True

    except Exception as e:
        print(f"❌ 강의 요약 테스트 중 오류: {e}")
        return False


def cleanup():
    """테스트 파일 정리"""
    import shutil

    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        print(f"🧹 테스트 DB 삭제: {test_db_path}")


def main():
    """메인 테스트 함수"""
    print("🎓 LLM 강의 검색 RAG 시스템 테스트")
    print("=" * 60)

    results = []

    # 1. 벡터 DB 구축 테스트
    results.append(("벡터 DB 구축", test_vector_db_building()))

    # 2. RAG 시스템 테스트
    results.append(("RAG 시스템", test_rag_system()))

    # 3. 코드 검색 테스트
    results.append(("코드 검색", test_code_search()))

    # 4. 강의 요약 테스트
    results.append(("강의 요약", test_lecture_summary()))

    # 결과 요약
    print("\\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)

    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")

    # 정리
    cleanup()

    if passed == total:
        print("\\n🎉 모든 테스트가 성공했습니다!")
        print("이제 'python main.py'로 Gradio 앱을 실행할 수 있습니다.")
    else:
        print(f"\\n⚠️ {total-passed}개 테스트가 실패했습니다. 설정을 확인해주세요.")


if __name__ == "__main__":
    main()