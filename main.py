#!/usr/bin/env python3
"""
LLM 강의 검색 및 Help RAG Agent
메인 실행 스크립트
"""

import sys
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gradio_app import GradioRAGApp


def main():
    """메인 실행 함수"""
    print("🎓 LLM 강의 검색 & Help RAG Agent 시작")
    print("=" * 50)

    # Gradio 애플리케이션 생성 및 실행
    app = GradioRAGApp()

    print("Gradio 웹 인터페이스를 시작합니다...")
    print("브라우저에서 http://localhost:7860 으로 접속하세요.")

    # 애플리케이션 실행
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()