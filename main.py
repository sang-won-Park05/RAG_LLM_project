#!/usr/bin/env python3
"""
LLM 강의 검색 및 Help RAG Agent
메인 실행 스크립트
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gradio_app import GradioRAGApp


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="LLM 강의 검색 & Help RAG Agent")
    parser.add_argument(
        "--norerank",
        action="store_true",
        help="Re-ranking 기능 비활성화 (dragonkue/bge-reranker-v2-m3-ko)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="서버 포트 번호 (기본값: 7860)"
    )

    args = parser.parse_args()

    print("🎓 LLM 강의 검색 & Help RAG Agent 시작")
    print("=" * 50)

    if args.norerank:
        print("✨ Re-ranking 기능이 비활성화됩니다.")
        print("   Re-ranking을 사용하려면 --rerank 옵션을 추가하세요.")
    else:
        print("📋 기본 검색 모드 (Re-ranking 활성화)")
        print("   모델: dragonkue/bge-reranker-v2-m3-ko")
        

    # Gradio 애플리케이션 생성 및 실행
    app = GradioRAGApp(use_reranking= not args.norerank)

    print(f"Gradio 웹 인터페이스를 시작합니다... (포트: {args.port})")
    print(f"브라우저에서 http://localhost:{args.port} 으로 접속하세요.")

    # 애플리케이션 실행
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()