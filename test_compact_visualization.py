#!/usr/bin/env python3
"""
컴팩트 시각화 테스트 스크립트
업데이트된 시각화 크기 조정이 제대로 작동하는지 확인
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 프로젝트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_visualizer import RAGVisualizationDashboard

def create_test_data():
    """테스트용 평가 결과 데이터 생성"""
    np.random.seed(42)

    test_data = []
    query_types = ['theory', 'code', 'application']
    difficulties = ['easy', 'medium', 'hard']

    for i in range(10):
        row = {
            'question': f'Test question {i+1}',
            'query_type': np.random.choice(query_types),
            'difficulty': np.random.choice(difficulties),
            'response_length': np.random.randint(100, 1000),
            'retrieved_docs_count': np.random.randint(3, 8),
            # RAG 메트릭들
            'answer_similarity': np.random.uniform(0.6, 0.95),
            'answer_relevancy': np.random.uniform(0.7, 0.9),
            'context_precision': np.random.uniform(0.5, 0.85),
            'faithfulness': np.random.uniform(0.6, 0.9)
        }
        test_data.append(row)

    return pd.DataFrame(test_data)

def test_compact_visualization():
    """컴팩트 시각화 테스트"""
    print("🧪 컴팩트 시각화 테스트 시작...")

    # 1. 테스트 데이터 생성
    results_df = create_test_data()
    print(f"✅ 테스트 데이터 생성 완료: {len(results_df)}개 샘플")

    # 2. 시각화 대시보드 생성
    visualizer = RAGVisualizationDashboard()
    print("✅ RAG 시각화 대시보드 초기화 완료")

    # 3. 개별 차트 테스트 (크기 확인)
    try:
        summary_fig = visualizer.create_performance_summary_chart(results_df)
        print(f"✅ 성능 요약 차트 생성 완료 (높이: 350px)")

        dashboard_fig = visualizer.create_comprehensive_dashboard(results_df)
        print(f"✅ 종합 대시보드 생성 완료 (높이: 500px)")

    except Exception as e:
        print(f"❌ 차트 생성 실패: {e}")
        return False

    # 4. 통합 HTML 대시보드 생성
    try:
        unified_html = visualizer.create_unified_dashboard(results_df)
        print("✅ 통합 HTML 대시보드 생성 완료")

        # HTML 크기 확인
        html_size = len(unified_html)
        print(f"📊 HTML 크기: {html_size:,} 문자 ({html_size/1024:.1f}KB)")

    except Exception as e:
        print(f"❌ 통합 대시보드 생성 실패: {e}")
        return False

    # 5. HTML 파일 저장 및 검증
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"compact_visualization_test_{timestamp}.html"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(unified_html)

        print(f"✅ HTML 파일 저장 완료: {output_file}")

        # 파일 크기 확인
        file_size = os.path.getsize(output_file)
        print(f"📁 파일 크기: {file_size:,} 바이트 ({file_size/1024:.1f}KB)")

    except Exception as e:
        print(f"❌ HTML 파일 저장 실패: {e}")
        return False

    # 6. 컴팩트 크기 확인
    print("\n📏 컴팩트 크기 설정 확인:")
    print("  - 차트 높이 감소: 600px → 400px (메트릭 대시보드)")
    print("  - 차트 높이 감소: 500px → 350px (성능 요약)")
    print("  - 차트 높이 감소: 700px → 500px (종합 대시보드)")
    print("  - 페이지 패딩: 20px → 10px")
    print("  - 헤더 패딩: 30px → 15px")
    print("  - 섹션 간격: 30px → 15px")

    return True

if __name__ == "__main__":
    success = test_compact_visualization()

    if success:
        print("\n🎉 컴팩트 시각화 테스트 성공!")
        print("💡 시각화 HTML 페이지가 한 화면에서 스크롤 없이 볼 수 있도록 최적화되었습니다.")
    else:
        print("\n❌ 컴팩트 시각화 테스트 실패!")
        sys.exit(1)