"""
RAG 시스템 평가 실행 스크립트 (시각화 기능 포함)
"""

import os
import sys
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_system import RAGSystem
from src.evaluator import CustomEvaluatior
from src.rag_evaluation import RAGEvaluationPipeline

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def setup_environment():
    """환경 설정 및 검증"""

    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 설정하거나 환경변수를 export 해주세요.")
        return False

    # 벡터 DB 존재 확인
    if not os.path.exists("chroma_db"):
        print("❌ 벡터 데이터베이스가 존재하지 않습니다.")
        print("   먼저 'python src/vector_db_builder.py'를 실행하여 데이터베이스를 구축해주세요.")
        return False

    return True


def run_basic_evaluation(enable_visualization: bool = True):
    """기본 RAG 평가 실행"""

    print("🚀 RAG 시스템 기본 평가 시작")
    if enable_visualization:
        print("🎨 시각화 기능 활성화")
    print("="*50)

    try:
        # 1. RAG 시스템 초기화
        print("📚 RAG 시스템 로딩...")
        rag_system = RAGSystem()

        # 2. 평가용 LLM 초기화
        print("🤖 평가용 LLM 초기화...")
        eval_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )

        # 3. Embeddings 모델 초기화 (정량적 메트릭용)
        print("🔤 임베딩 모델 로딩...")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        # 4. CustomEvaluator 초기화
        evaluator = CustomEvaluatior(eval_llm, embeddings)

        # 5. 평가 파이프라인 생성 (시각화 옵션 포함)
        evaluation_pipeline = RAGEvaluationPipeline(rag_system, evaluator, enable_visualization=enable_visualization)

        # 6. 평가 실행
        print("📊 평가 실행 중...")
        results_df = evaluation_pipeline.run_comprehensive_evaluation()

        # 7. 대화형 시각화 표시 (옵션)
        if enable_visualization:
            show_interactive = input("\n대화형 시각화를 표시하시겠습니까? (y/n): ").strip().lower()
            if show_interactive in ['y', 'yes', '예']:
                evaluation_pipeline.show_interactive_visualizations(results_df)

        # 8. 결과 리포트 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"evaluation_results_{timestamp}"

        saved_files = evaluation_pipeline.generate_evaluation_report(
            results_df,
            save_path=report_path
        )

        print(f"\n✅ 평가 완료! 결과는 {report_path}_*.csv, *.json 파일에 저장되었습니다.")

        if enable_visualization and saved_files:
            print(f"\n📈 시각화 결과는 visualizations/ 폴더에 저장되었습니다.")
            print(f"   주요 파일: {saved_files.get('summary_report', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ 평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_configuration_comparison():
    """다양한 RAG 설정 비교 평가"""

    print("\n🔄 RAG 설정 비교 평가 시작")
    print("="*50)

    try:
        # 기본 설정으로 평가 파이프라인 초기화
        rag_system = RAGSystem()
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        evaluator = CustomEvaluatior(eval_llm, embeddings)
        evaluation_pipeline = RAGEvaluationPipeline(rag_system, evaluator)

        # 비교할 설정들
        configs = {
            "기본설정": {
                "use_reranking": False
            },
            "재순위화_적용": {
                "use_reranking": True
            }
        }

        # 설정 비교 실행
        comparison_df = evaluation_pipeline.compare_configurations(configs)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_df.to_csv(f"rag_config_comparison_{timestamp}.csv", index=False, encoding='utf-8')

        print(f"\n✅ 설정 비교 완료! 결과는 rag_config_comparison_{timestamp}.csv에 저장되었습니다.")

        return True

    except Exception as e:
        print(f"❌ 설정 비교 중 오류 발생: {e}")
        return False


def run_simple_test():
    """간단한 단일 쿼리 테스트"""

    print("\n🧪 단일 쿼리 테스트")
    print("="*30)

    try:
        # RAG 시스템 초기화
        rag_system = RAGSystem()
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        evaluator = CustomEvaluatior(eval_llm)

        # 테스트 쿼리
        test_query = "CNN 모델의 기본 구조를 설명해주세요"

        print(f"🔍 테스트 쿼리: {test_query}")

        # RAG 응답 생성
        response = rag_system.handle_user_input(test_query)
        print(f"\n🤖 RAG 응답:\n{response}")

        # 검색된 문서 확인
        retrieved_docs = rag_system.ensemble_retriever.get_relevant_documents(test_query)
        print(f"\n📚 검색된 문서 수: {len(retrieved_docs)}")

        # 평가 실행
        ground_truth = "CNN은 Convolutional Neural Network로, 합성곱층, 풀링층, 완전연결층으로 구성됩니다."

        evaluation_result = evaluator.evaluate_response(
            query=test_query,
            response=response,
            ground_truth=ground_truth
        )

        print(f"\n📊 평가 결과:\n{evaluation_result}")

        return True

    except Exception as e:
        print(f"❌ 단일 쿼리 테스트 중 오류 발생: {e}")
        return False


def main():
    """메인 실행 함수"""

    print("📊 RAG 시스템 평가 도구")
    print("="*50)

    # 환경 설정 확인
    if not setup_environment():
        return

    # 사용자 선택
    print("\n평가 유형을 선택하세요:")
    print("1. 간단한 단일 쿼리 테스트")
    print("2. 기본 종합 평가 (시각화 포함)")
    print("3. 기본 종합 평가 (시각화 없음)")
    print("4. 설정 비교 평가")
    print("5. 모든 평가 실행")

    choice = input("\n선택 (1-5): ").strip()

    if choice == "1":
        run_simple_test()
    elif choice == "2":
        run_basic_evaluation(enable_visualization=True)
    elif choice == "3":
        run_basic_evaluation(enable_visualization=False)
    elif choice == "4":
        run_configuration_comparison()
    elif choice == "5":
        print("\n🚀 모든 평가 실행...")
        run_simple_test()
        print("\n" + "="*50)
        run_basic_evaluation(enable_visualization=True)
        print("\n" + "="*50)
        run_configuration_comparison()
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()