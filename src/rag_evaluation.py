"""
RAG 시스템 평가를 위한 종합적인 평가 스크립트
CustomEvaluator를 활용하여 다양한 메트릭으로 RAG 성능을 평가
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.rag_system import RAGSystem
from src.evaluator import CustomEvaluatior
from src.rag_visualizer import RAGVisualizationDashboard, create_summary_report
from langchain_openai import ChatOpenAI

class RAGEvaluationPipeline:
    def __init__(self, rag_system: RAGSystem, evaluator: CustomEvaluatior, enable_visualization: bool = True):
        """
        RAG 평가 파이프라인 초기화

        Args:
            rag_system: 평가할 RAG 시스템
            evaluator: CustomEvaluator 인스턴스
            enable_visualization: 시각화 기능 활성화 여부
        """
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.evaluation_results = []
        self.enable_visualization = enable_visualization

        # 시각화 대시보드 초기화
        if self.enable_visualization:
            self.visualizer = RAGVisualizationDashboard()

    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """
        테스트 데이터셋 생성 - 실제 교육 자료 기반으로 생성된 현실적인 질문들
        data/educational_materials의 PDF와 notebook 파일들을 분석하여 생성
        """
        test_queries = [
            # LangChain & RAG 관련 (2025-08-25 langchain_2.ipynb 기반)
            {
                "question": "LangChain에서 한국 전통 음식 정보를 검색하는 RAG 시스템을 구현하는 방법을 설명해주세요",
                "ground_truth": "LangChain에서 RAG 시스템을 구현하려면 벡터 데이터베이스(Chroma), 임베딩 모델, LLM을 조합하여 문서 검색 후 답변을 생성하는 파이프라인을 구축해야 합니다. 한국 전통 음식 데이터를 벡터화하고 유사도 검색을 통해 관련 정보를 찾아 답변을 생성합니다.",
                "query_type": "theory",
                "difficulty": "medium"
            },
            {
                "question": "코드 검색: LangChain Document 클래스 사용법",
                "ground_truth": "LangChain의 Document 클래스는 page_content와 metadata 속성을 가지며, 텍스트 데이터와 메타데이터를 함께 저장합니다. from langchain.schema import Document로 import하여 사용할 수 있습니다.",
                "query_type": "code",
                "difficulty": "easy"
            },

            # 토큰화 관련 (2025-08-11 BPE_Unigram.ipynb 기반)
            {
                "question": "BPE(Byte Pair Encoding)와 Unigram 토큰화 방법의 차이점은 무엇인가요?",
                "ground_truth": "BPE는 빈도가 높은 문자 쌍을 반복적으로 병합하여 서브워드를 생성하는 방식이고, Unigram은 확률 모델을 사용하여 가장 가능성이 높은 서브워드 분할을 찾는 방식입니다. BPE는 deterministic하지만 Unigram은 probabilistic 접근법을 사용합니다.",
                "query_type": "theory",
                "difficulty": "hard"
            },
            {
                "question": "SentencePiece 라이브러리에서 BPE 모델을 학습시키는 코드를 설명해주세요",
                "ground_truth": "SentencePiece에서 BPE 모델 학습은 spm.SentencePieceTrainer.train()을 사용하며, input 파일, model_prefix, vocab_size, model_type='bpe' 등의 파라미터를 설정합니다.",
                "query_type": "code",
                "difficulty": "medium"
            },

            # GAN 관련 (2025-09-12 멀티모달_GAN_ex1.ipynb 기반)
            {
                "question": "조건부 GAN(Conditional GAN)에서 텍스트 조건을 어떻게 구현하나요?",
                "ground_truth": "조건부 GAN에서는 임베딩 레이어를 사용하여 텍스트 레이블을 벡터로 변환하고, 이를 생성자와 판별자의 입력에 concatenate합니다. nn.Embedding을 사용하여 클래스 레이블을 임베딩으로 변환한 후 노이즈 벡터와 결합합니다.",
                "query_type": "theory",
                "difficulty": "hard"
            },
            {
                "question": "PyTorch에서 MNIST 데이터로 GAN을 학습시킬 때 사용하는 손실 함수는 무엇인가요?",
                "ground_truth": "MNIST GAN 학습에는 주로 Binary Cross Entropy Loss(nn.BCELoss)를 사용합니다. 생성자는 판별자가 가짜 이미지를 진짜로 판별하도록 학습하고, 판별자는 진짜와 가짜를 구분하도록 학습합니다.",
                "query_type": "code",
                "difficulty": "medium"
            },

            # 머신러닝 기초 (KNN, 영화추천 등)
            {
                "question": "KNN 알고리즘에서 K값을 선택하는 기준은 무엇인가요?",
                "ground_truth": "KNN에서 K값은 홀수로 선택하여 동점을 방지하고, 교차 검증을 통해 최적값을 찾습니다. K가 너무 작으면 노이즈에 민감하고, 너무 크면 결정 경계가 smooth해져 underfitting이 발생할 수 있습니다.",
                "query_type": "theory",
                "difficulty": "medium"
            },
            {
                "question": "영화 추천 시스템에서 TF-IDF와 코사인 유사도를 사용하는 이유를 설명해주세요",
                "ground_truth": "TF-IDF는 문서의 중요한 키워드를 가중치로 표현하고, 코사인 유사도는 벡터 간의 방향성을 측정하여 문서 길이에 영향받지 않는 순수한 유사성을 계산합니다. 이를 통해 장르, 줄거리 등의 텍스트 정보로 유사한 영화를 찾을 수 있습니다.",
                "query_type": "theory",
                "difficulty": "medium"
            },
            # 평가 지표 관련
            {
                "question": "BLEU와 ROUGE 평가 지표의 차이점과 사용 용도를 설명해주세요",
                "ground_truth": "BLEU는 기계번역 품질 평가에 주로 사용되며 n-gram 정밀도를 측정합니다. ROUGE는 자동요약 평가에 사용되며 recall 기반으로 참조 요약과 생성 요약 간의 겹치는 정도를 측정합니다. BLEU는 정밀도, ROUGE는 재현율에 중점을 둡니다.",
                "query_type": "theory",
                "difficulty": "medium"
            },
            # 데이터 전처리
            {
                "question": "데이터 전처리에서 정규화(Normalization)와 표준화(Standardization)의 차이점은 무엇인가요?",
                "ground_truth": "정규화는 데이터를 0-1 범위로 스케일링하는 Min-Max scaling이고, 표준화는 평균을 0, 표준편차를 1로 만드는 Z-score scaling입니다. 정규화는 이상치에 민감하지만 표준화는 상대적으로 강건합니다.",
                "query_type": "theory",
                "difficulty": "medium"
            }
        ]
        return test_queries

    def evaluate_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 쿼리에 대한 종합 평가

        Args:
            query_data: 쿼리 정보 (question, ground_truth, query_type, difficulty)

        Returns:
            평가 결과 딕셔너리
        """
        question = query_data["question"]
        ground_truth = query_data["ground_truth"]

        # 1. RAG 시스템으로 답변 생성
        response = self.rag_system.handle_user_input(question)

        # 2. 관련 문서 검색
        retrieved_docs = self.rag_system.ensemble_retriever.get_relevant_documents(question)
        doc_contents = [doc.page_content for doc in retrieved_docs[:5]]  # 상위 5개 문서

        # 3. CustomEvaluator로 응답 품질 평가
        response_evaluation = self.evaluator.evaluate_response(
            query=question,
            response=response,
            ground_truth=ground_truth
        )

        # 4. 검색된 문서 관련성 평가
        retrieval_evaluation = self.evaluator.evaluate_retrieval(
            query=question,
            retrieved_documents=doc_contents
        )

        # 5. 정량적 메트릭 계산 (embeddings가 있는 경우)
        quantitative_metrics = {}
        try:
            # 임시로 평가용 DataFrame 생성
            eval_df = pd.DataFrame([{
                'question': question,
                'answer': response,
                'ground_truth': ground_truth,
                'contexts': doc_contents
            }])

            # embeddings 속성이 있는지 확인
            if hasattr(self.evaluator, 'embeddings') and self.evaluator.embeddings is not None:
                quantitative_metrics = self.evaluator.evaluate_response_all(eval_df)
            else:
                # embeddings가 없으면 간단한 메트릭만 계산
                quantitative_metrics = {
                    'answer_similarity': self.evaluator.calculate_answer_similarity(response, ground_truth),
                    'faithfulness': self.evaluator.calculate_faithfulness(response, doc_contents)
                }
        except Exception as e:
            print(f"정량적 메트릭 계산 중 오류: {e}")
            quantitative_metrics = {}

        # 6. 결과 종합
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'query_data': query_data,
            'response': response,
            'retrieved_docs_count': len(retrieved_docs),
            'response_evaluation': response_evaluation,
            'retrieval_evaluation': retrieval_evaluation,
            'quantitative_metrics': quantitative_metrics,
            'execution_time': None  # 실행 시간 측정 추가 가능
        }

        return evaluation_result

    def run_comprehensive_evaluation(self, test_dataset: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        전체 테스트 데이터셋에 대한 종합 평가 실행

        Args:
            test_dataset: 테스트 데이터셋 (None이면 기본 데이터셋 사용)

        Returns:
            평가 결과 DataFrame
        """
        if test_dataset is None:
            test_dataset = self.create_test_dataset()

        print(f"📊 RAG 시스템 종합 평가 시작 (총 {len(test_dataset)}개 쿼리)")
        print("=" * 60)

        for i, query_data in enumerate(tqdm(test_dataset, desc="평가 진행중")):
            print(f"\n🔍 쿼리 {i+1}: {query_data['question'][:50]}...")

            try:
                result = self.evaluate_single_query(query_data)
                self.evaluation_results.append(result)

                # 간단한 결과 출력
                if 'answer_similarity' in result['quantitative_metrics']:
                    similarity = result['quantitative_metrics']['answer_similarity']
                    print(f"   📈 답변 유사도: {similarity:.3f}")

            except Exception as e:
                print(f"   ❌ 평가 실패: {e}")
                continue

        print(f"\n✅ 평가 완료! 총 {len(self.evaluation_results)}개 결과")

        # 결과를 DataFrame으로 변환
        return self.create_results_dataframe()

    def create_results_dataframe(self) -> pd.DataFrame:
        """평가 결과를 DataFrame으로 변환"""
        rows = []

        for result in self.evaluation_results:
            query_data = result['query_data']
            metrics = result['quantitative_metrics']

            row = {
                'question': query_data['question'],
                'query_type': query_data['query_type'],
                'difficulty': query_data['difficulty'],
                'response_length': len(result['response']),
                'retrieved_docs_count': result['retrieved_docs_count'],
                **metrics  # 정량적 메트릭 추가
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_evaluation_report(self, results_df: pd.DataFrame, save_path: str = None):
        """
        평가 결과 리포트 생성 (시각화 포함)

        Args:
            results_df: 평가 결과 DataFrame
            save_path: 리포트 저장 경로
        """
        print("\n" + "="*60)
        print("📊 RAG 시스템 평가 리포트")
        print("="*60)

        # 1. 전체 통계
        print(f"\n📈 전체 통계:")
        print(f"   • 총 평가 쿼리 수: {len(results_df)}")
        print(f"   • 평균 응답 길이: {results_df['response_length'].mean():.1f} 문자")
        print(f"   • 평균 검색 문서 수: {results_df['retrieved_docs_count'].mean():.1f}")

        # 2. 정량적 메트릭 통계
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        metric_columns = [col for col in numeric_columns if col not in ['response_length', 'retrieved_docs_count']]

        if metric_columns:
            print(f"\n📊 성능 메트릭 (평균):")
            for metric in metric_columns:
                avg_score = results_df[metric].mean()
                print(f"   • {metric}: {avg_score:.3f}")

        # 3. 쿼리 타입별 분석
        if 'query_type' in results_df.columns:
            print(f"\n🎯 쿼리 타입별 성능:")
            type_stats = results_df.groupby('query_type').agg({
                col: 'mean' for col in metric_columns
            }).round(3)
            print(type_stats.to_string())

        # 4. 난이도별 분석
        if 'difficulty' in results_df.columns:
            print(f"\n⚡ 난이도별 성능:")
            difficulty_stats = results_df.groupby('difficulty').agg({
                col: 'mean' for col in metric_columns
            }).round(3)
            print(difficulty_stats.to_string())

        # 5. 상세 결과 출력
        print(f"\n📋 상세 평가 결과:")
        for i, result in enumerate(self.evaluation_results[:3]):  # 처음 3개만 출력
            print(f"\n--- 쿼리 {i+1} ---")
            print(f"질문: {result['query_data']['question']}")
            print(f"응답: {result['response'][:200]}...")
            print(f"응답 평가:\n{result['response_evaluation'][:300]}...")

        # 6. 시각화 생성 및 저장
        saved_files = {}
        if self.enable_visualization and save_path:
            try:
                print(f"\n🎨 시각화 생성 중...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 통합 대시보드 저장
                dashboard_path = self.visualizer.save_unified_dashboard(
                    results_df,
                    detailed_results=self.evaluation_results,
                    save_dir="visualizations",
                    timestamp=timestamp
                )

                saved_files = {'unified_dashboard': dashboard_path}
                print(f"📈 통합 대시보드 저장 완료: {dashboard_path}")

            except Exception as e:
                print(f"❌ 시각화 생성 중 오류: {e}")
        return saved_files

    def show_interactive_visualizations(self, results_df: pd.DataFrame):
        """
        통합 대화형 시각화를 실시간으로 표시

        Args:
            results_df: 평가 결과 DataFrame
        """
        if not self.enable_visualization:
            print("시각화가 비활성화되어 있습니다.")
            return

        print("\n🎨 통합 대시보드 생성 중...")

        try:
            # 통합 HTML 대시보드 생성
            unified_html = self.visualizer.create_unified_dashboard(results_df, self.evaluation_results)

            # 임시 파일로 저장하고 브라우저에서 열기
            import tempfile
            import webbrowser

            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(unified_html)
                temp_path = f.name

            print(f"📈 통합 대시보드를 브라우저에서 여는 중: {temp_path}")
            webbrowser.open(f'file://{temp_path}')
            print("✅ 통합 대시보드가 브라우저에서 열렸습니다!")

        except Exception as e:
            print(f"❌ 시각화 표시 중 오류: {e}")

    def compare_configurations(self, configs: Dict[str, Dict]) -> pd.DataFrame:
        """
        다양한 RAG 설정 비교 평가

        Args:
            configs: 설정 이름과 RAG 시스템 설정 딕셔너리

        Returns:
            비교 결과 DataFrame
        """
        print("\n🔄 RAG 설정 비교 평가 시작")
        print("="*40)

        comparison_results = []
        test_dataset = self.create_test_dataset()

        for config_name, config_params in configs.items():
            print(f"\n📝 {config_name} 설정 평가 중...")

            # 새로운 RAG 시스템 생성
            try:
                rag_system = RAGSystem(**config_params)
                evaluator = RAGEvaluationPipeline(rag_system, self.evaluator)

                # 평가 실행
                results_df = evaluator.run_comprehensive_evaluation(test_dataset)

                # 결과 요약
                numeric_columns = results_df.select_dtypes(include=[np.number]).columns
                metric_columns = [col for col in numeric_columns if col not in ['response_length', 'retrieved_docs_count']]

                config_result = {'configuration': config_name}
                for metric in metric_columns:
                    config_result[metric] = results_df[metric].mean()

                comparison_results.append(config_result)

            except Exception as e:
                print(f"❌ {config_name} 설정 평가 실패: {e}")
                continue

        comparison_df = pd.DataFrame(comparison_results)

        print("\n📊 설정 비교 결과:")
        print(comparison_df.to_string(index=False))

        return comparison_df


def main():
    """RAG 평가 파이프라인 실행 예시"""

    # 1. RAG 시스템과 평가자 초기화
    print("🚀 RAG 평가 시스템 초기화...")

    try:
        # RAG 시스템 로드
        rag_system = RAGSystem()

        # LLM 초기화 (평가용)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )

        # CustomEvaluator 초기화
        evaluator = CustomEvaluatior(llm)

        # 평가 파이프라인 생성
        evaluation_pipeline = RAGEvaluationPipeline(rag_system, evaluator)

        # 2. 종합 평가 실행
        print("\n📊 종합 평가 실행...")
        results_df = evaluation_pipeline.run_comprehensive_evaluation()

        # 3. 평가 리포트 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"evaluation_results_{timestamp}"

        evaluation_pipeline.generate_evaluation_report(
            results_df,
            save_path=report_path
        )

        # 4. 설정 비교 (선택사항)
        print("\n🔄 다양한 설정 비교...")
        configs = {
            "basic": {"use_reranking": False},
            "with_reranking": {"use_reranking": True},
            # "custom_weights": {"ensemble_weights": [0.8, 0.2]}  # 필요시 추가
        }

        comparison_df = evaluation_pipeline.compare_configurations(configs)
        comparison_df.to_csv(f"rag_config_comparison_{timestamp}.csv", index=False)

        print(f"\n✅ 모든 평가 완료! 결과 파일들이 저장되었습니다.")

    except Exception as e:
        print(f"❌ 평가 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()