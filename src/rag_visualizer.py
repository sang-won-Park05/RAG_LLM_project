"""
RAG 시스템 평가 결과 시각화 모듈
RAG 전용 메트릭의 시각적 분석 및 리포팅 기능 제공
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

class RAGVisualizationDashboard:
    """RAG 시스템 평가 결과 종합 시각화 클래스"""

    def __init__(self, style_theme: str = "plotly_white"):
        """
        RAG 시각화 대시보드 초기화

        Args:
            style_theme: Plotly 테마 (plotly_white, plotly_dark, ggplot2 등)
        """
        self.style_theme = style_theme
        self.rag_colors = {
            'primary': '#2E86AB',       # 파란색 (검색/Retrieval)
            'secondary': '#A23B72',     # 자주색 (생성/Generation)
            'accent': '#F18F01',        # 주황색 (품질/Quality)
            'success': '#C73E1D',       # 빨간색 (성능/Performance)
            'neutral': '#6C757D',       # 회색 (기타)
            'gradient': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        }

    def create_rag_metrics_overview(self, results_df: pd.DataFrame) -> go.Figure:
        """
        RAG 핵심 메트릭 개요 대시보드

        Args:
            results_df: 평가 결과 DataFrame

        Returns:
            Plotly Figure 객체
        """
        # RAG 전용 메트릭 추출
        rag_metrics = [
            'answer_similarity',     # 답변 유사도
            'answer_relevancy',     # 답변 관련성
            'context_precision',    # 컨텍스트 정확도
            'faithfulness'          # 충실도
        ]

        available_metrics = [m for m in rag_metrics if m in results_df.columns]

        if not available_metrics:
            raise ValueError("RAG 메트릭이 데이터에 없습니다.")

        # 서브플롯 생성 (2x2 그리드)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '답변 유사도 (Answer Similarity)',
                '답변 관련성 (Answer Relevancy)',
                '컨텍스트 정확도 (Context Precision)',
                '충실도 (Faithfulness)'
            ],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )

        # 각 메트릭에 대한 게이지 차트
        positions = [(1,1), (1,2), (2,1), (2,2)]
        metric_names = ['답변 유사도', '답변 관련성', '컨텍스트 정확도', '충실도']

        for i, (metric, name) in enumerate(zip(available_metrics, metric_names)):
            if metric in results_df.columns:
                score = results_df[metric].mean()

                # 성능 등급 결정
                if score >= 0.8:
                    color = self.rag_colors['success']
                    grade = "Excellent"
                elif score >= 0.6:
                    color = self.rag_colors['accent']
                    grade = "Good"
                elif score >= 0.4:
                    color = self.rag_colors['secondary']
                    grade = "Fair"
                else:
                    color = self.rag_colors['neutral']
                    grade = "Poor"

                row, col = positions[i]
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{name}<br><span style='font-size:12px'>{grade}</span>"},
                        delta={'reference': 0.7, 'position': "top"},
                        gauge={
                            'axis': {'range': [None, 1]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 0.4], 'color': "lightgray"},
                                {'range': [0.4, 0.7], 'color': "gray"},
                                {'range': [0.7, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.8
                            }
                        }
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title="RAG 시스템 핵심 성능 메트릭 대시보드",
            template=self.style_theme,
            height=400,
            showlegend=False
        )

        return fig

    def create_performance_summary_chart(self, results_df: pd.DataFrame) -> go.Figure:
        """
        성능 요약 바 차트 (핵심 메트릭만)

        Args:
            results_df: 평가 결과 DataFrame

        Returns:
            Plotly Figure 객체
        """
        # RAG 메트릭
        rag_metrics = ['answer_similarity', 'answer_relevancy', 'context_precision', 'faithfulness']
        available_metrics = [m for m in rag_metrics if m in results_df.columns]

        if not available_metrics:
            raise ValueError("RAG 메트릭이 데이터에 없습니다.")

        # 평균값 계산
        avg_scores = [results_df[metric].mean() for metric in available_metrics]

        # 영어 라벨
        metric_labels = {
            'answer_similarity': 'Answer Similarity',
            'answer_relevancy': 'Answer Relevancy',
            'context_precision': 'Context Precision',
            'faithfulness': 'Faithfulness'
        }

        english_labels = [metric_labels.get(m, m) for m in available_metrics]

        # 바 차트 생성
        fig = go.Figure(data=[
            go.Bar(
                x=english_labels,
                y=avg_scores,
                marker_color=self.rag_colors['gradient'][:len(available_metrics)],
                text=[f'{score:.3f}' for score in avg_scores],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="RAG 시스템 핵심 성능 메트릭",
            xaxis_title="평가 메트릭",
            yaxis_title="평균 점수",
            yaxis=dict(range=[0, 1]),
            template=self.style_theme,
            height=350
        )

        return fig

    def create_comprehensive_dashboard(self, results_df: pd.DataFrame,
                                    detailed_results: List[Dict] = None) -> go.Figure:
        """
        단순화된 RAG 평가 대시보드 (핵심 지표만)

        Args:
            results_df: 평가 결과 DataFrame
            detailed_results: 상세 평가 결과 리스트

        Returns:
            Plotly Figure 객체
        """
        # 2x2 레이아웃으로 단순화
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '전체 성능 요약', '메트릭별 상세 점수',
                '평가 통계', '성능 분포'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # RAG 메트릭
        rag_metrics = ['answer_similarity', 'answer_relevancy', 'context_precision', 'faithfulness']
        available_metrics = [m for m in rag_metrics if m in results_df.columns]

        # 1. 전체 성능 요약 (게이지)
        if available_metrics:
            overall_score = results_df[available_metrics].mean().mean()

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "전체 RAG 성능"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': self.rag_colors['primary']},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "lightgreen"}
                        ]
                    }
                ),
                row=1, col=1
            )

            # 2. 메트릭별 바 차트
            metric_labels = {
                'answer_similarity': 'Answer Similarity',
                'answer_relevancy': 'Answer Relevancy',
                'context_precision': 'Context Precision',
                'faithfulness': 'Faithfulness'
            }

            english_labels = [metric_labels.get(m, m) for m in available_metrics]
            avg_scores = [results_df[metric].mean() for metric in available_metrics]

            fig.add_trace(
                go.Bar(
                    x=english_labels,
                    y=avg_scores,
                    marker_color=self.rag_colors['gradient'][:len(available_metrics)],
                    text=[f'{score:.3f}' for score in avg_scores],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )

            # 3. 통계 테이블
            stats_data = []
            for metric in available_metrics:
                stats_data.append([
                    metric_labels.get(metric, metric),
                    f"{results_df[metric].mean():.3f}",
                    f"{results_df[metric].std():.3f}",
                    f"{results_df[metric].min():.3f}",
                    f"{results_df[metric].max():.3f}"
                ])

            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Mean', 'Std Dev', 'Min', 'Max']),
                    cells=dict(values=list(zip(*stats_data)))
                ),
                row=2, col=1
            )

            # 4. 전체 점수 분포 히스토그램
            if len(available_metrics) > 0:
                overall_scores = results_df[available_metrics].mean(axis=1)

                fig.add_trace(
                    go.Histogram(
                        x=overall_scores,
                        nbinsx=10,
                        marker_color=self.rag_colors['accent'],
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            title="RAG 시스템 핵심 성능 대시보드",
            template=self.style_theme,
            height=500,
            showlegend=False
        )

        return fig

    def create_unified_dashboard(self, results_df: pd.DataFrame,
                               detailed_results: List[Dict] = None) -> str:
        """
        모든 시각화를 포함한 통합 HTML 대시보드 생성

        Args:
            results_df: 평가 결과 DataFrame
            detailed_results: 상세 평가 결과

        Returns:
            HTML 문자열
        """
        # 각 시각화 생성 (게이지 제거)
        summary_fig = self.create_performance_summary_chart(results_df)
        dashboard_fig = self.create_comprehensive_dashboard(results_df, detailed_results)

        # HTML로 변환 (div만 추출)
        summary_div = summary_fig.to_html(include_plotlyjs=False, div_id="performance_summary")
        dashboard_div = dashboard_fig.to_html(include_plotlyjs=False, div_id="comprehensive_dashboard")

        # RAG 메트릭 통계 계산
        rag_metrics = ['answer_similarity', 'answer_relevancy', 'context_precision', 'faithfulness']
        available_metrics = [m for m in rag_metrics if m in results_df.columns]

        stats_html = ""
        if available_metrics:
            overall_score = results_df[available_metrics].mean().mean()
            metric_labels = {
                'answer_similarity': 'Answer Similarity',
                'answer_relevancy': 'Answer Relevancy',
                'context_precision': 'Context Precision',
                'faithfulness': 'Faithfulness'
            }

            stats_html = f"""
            <div class="stats-summary">
                <h3>📊 핵심 성능 지표</h3>
                <div class="overall-score">
                    <h4>전체 평균 성능: <span class="score">{overall_score:.3f}</span></h4>
                </div>
                <div class="metrics-grid">
            """

            for metric in available_metrics:
                score = results_df[metric].mean()
                label = metric_labels.get(metric, metric)
                stats_html += f"""
                    <div class="metric-item">
                        <span class="metric-label">{label}</span>
                        <span class="metric-score">{score:.3f}</span>
                    </div>
                """

            stats_html += """
                </div>
            </div>
            """

        # 통합 HTML 생성
        unified_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG 시스템 통합 평가 대시보드</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 10px;
                    background-color: #f8f9fa;
                    color: #333;
                    font-size: 14px;
                }}
                .header {{
                    background: linear-gradient(135deg, #2E86AB, #A23B72);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 1.8em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 5px 0 0 0;
                    opacity: 0.9;
                    font-size: 0.9em;
                }}
                .stats-summary {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .overall-score {{
                    text-align: center;
                    margin-bottom: 15px;
                    padding: 12px;
                    background: #f8f9fa;
                    border-radius: 6px;
                }}
                .score {{
                    color: #2E86AB;
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 10px;
                }}
                .metric-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 6px;
                    border-left: 3px solid #2E86AB;
                }}
                .metric-label {{
                    font-weight: 500;
                    color: #555;
                }}
                .metric-score {{
                    font-weight: bold;
                    color: #2E86AB;
                    font-size: 1.1em;
                }}
                .visualization-section {{
                    background: white;
                    margin-bottom: 15px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .section-header {{
                    background: #2E86AB;
                    color: white;
                    padding: 12px;
                    margin: 0;
                    font-size: 1.1em;
                    font-weight: 500;
                }}
                .section-content {{
                    padding: 10px;
                }}
                .nav-tabs {{
                    display: flex;
                    background: #e9ecef;
                    margin: 0;
                    padding: 0;
                    list-style: none;
                    border-radius: 8px 8px 0 0;
                }}
                .nav-tab {{
                    flex: 1;
                    text-align: center;
                    padding: 10px;
                    cursor: pointer;
                    background: #e9ecef;
                    border: none;
                    font-size: 0.9em;
                    transition: all 0.3s ease;
                }}
                .nav-tab.active {{
                    background: #2E86AB;
                    color: white;
                }}
                .nav-tab:hover {{
                    background: #A23B72;
                    color: white;
                }}
                .tab-content {{
                    display: none;
                    padding: 10px;
                    background: white;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .footer {{
                    text-align: center;
                    padding: 10px;
                    color: #666;
                    background: white;
                    border-radius: 8px;
                    margin-top: 15px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 RAG 시스템 평가 대시보드</h1>
                <p>생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>총 평가 쿼리: {len(results_df)}개</p>
            </div>

            {stats_html}

            <div class="visualization-section">
                <ul class="nav-tabs">
                    <li class="nav-tab active" onclick="showTab('tab1', this)">📊 Performance Summary</li>
                    <li class="nav-tab" onclick="showTab('tab2', this)">📈 Comprehensive Dashboard</li>
                </ul>

                <div id="tab1" class="tab-content active">
                    {summary_div}
                </div>

                <div id="tab2" class="tab-content">
                    {dashboard_div}
                </div>
            </div>

            <div class="footer">
                <p>🤖 RAG 시스템 평가 결과 | 자동 생성됨</p>
            </div>

            <script>
                function showTab(tabId, element) {{
                    // 모든 탭 내용 숨기기
                    var contents = document.querySelectorAll('.tab-content');
                    contents.forEach(function(content) {{
                        content.classList.remove('active');
                    }});

                    // 모든 탭 버튼 비활성화
                    var tabs = document.querySelectorAll('.nav-tab');
                    tabs.forEach(function(tab) {{
                        tab.classList.remove('active');
                    }});

                    // 선택된 탭 활성화
                    document.getElementById(tabId).classList.add('active');
                    element.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """

        return unified_html

    def save_unified_dashboard(self, results_df: pd.DataFrame,
                             detailed_results: List[Dict] = None,
                             save_dir: str = "visualizations",
                             timestamp: str = None) -> str:
        """
        통합 대시보드를 파일로 저장

        Args:
            results_df: 평가 결과 DataFrame
            detailed_results: 상세 평가 결과
            save_dir: 저장 디렉토리
            timestamp: 타임스탬프

        Returns:
            저장된 파일 경로
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 통합 HTML 생성
        unified_html = self.create_unified_dashboard(results_df, detailed_results)

        # 파일 저장
        dashboard_path = os.path.join(save_dir, f"rag_unified_dashboard_{timestamp}.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(unified_html)

        return dashboard_path

def create_summary_report(results_df: pd.DataFrame, saved_files: Dict[str, str]) -> str:
    """
    시각화 결과 요약 리포트 생성

    Args:
        results_df: 평가 결과 DataFrame
        saved_files: 저장된 파일 경로 딕셔너리

    Returns:
        HTML 형식의 요약 리포트
    """
    # RAG 메트릭 통계
    rag_metrics = ['answer_similarity', 'answer_relevancy', 'context_precision', 'faithfulness']
    available_metrics = [m for m in rag_metrics if m in results_df.columns]

    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG 시스템 평가 리포트</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #2E86AB; color: white; padding: 20px; border-radius: 10px; }}
            .metric {{ display: inline-block; margin: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
            .links {{ margin: 30px 0; }}
            .links a {{ display: block; margin: 10px 0; color: #2E86AB; text-decoration: none; }}
            .links a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 RAG 시스템 평가 리포트</h1>
            <p>생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <h2>📊 주요 성능 지표</h2>
    """

    # 메트릭별 통계 추가
    for metric in available_metrics:
        if metric in results_df.columns:
            avg_score = results_df[metric].mean()
            metric_name = {
                'answer_similarity': '답변 유사도',
                'answer_relevancy': '답변 관련성',
                'context_precision': '컨텍스트 정확도',
                'faithfulness': '충실도'
            }.get(metric, metric)

            report_html += f"""
            <div class="metric">
                <h3>{metric_name}</h3>
                <p style="font-size: 24px; font-weight: bold; color: #2E86AB;">{avg_score:.3f}</p>
            </div>
            """

    # 링크 섹션 추가
    report_html += """
        <h2>📈 상세 시각화 결과</h2>
        <div class="links">
    """

    link_names = {
        'metrics_overview': '🎯 핵심 메트릭 게이지 대시보드',
        'performance_summary': '📊 성능 요약 차트',
        'comprehensive_dashboard': '📈 종합 대시보드'
    }

    for key, file_path in saved_files.items():
        link_name = link_names.get(key, key)
        report_html += f'<a href="{file_path}" target="_blank">{link_name}</a>\n'

    report_html += """
        </div>

        <h2>📋 평가 요약</h2>
        <ul>
    """

    report_html += f"<li>총 평가 쿼리 수: {len(results_df)}</li>"
    report_html += f"<li>평균 응답 길이: {results_df['response_length'].mean():.1f} 문자</li>" if 'response_length' in results_df.columns else ""
    report_html += f"<li>평균 검색 문서 수: {results_df['retrieved_docs_count'].mean():.1f}</li>" if 'retrieved_docs_count' in results_df.columns else ""

    if available_metrics:
        overall_score = results_df[available_metrics].mean().mean()
        report_html += f"<li>전체 평균 성능: {overall_score:.3f}</li>"

    report_html += """
        </ul>
    </body>
    </html>
    """

    return report_html