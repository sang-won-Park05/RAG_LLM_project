# src/gradio_app.py
# -*- coding: utf-8 -*-
"""
Gradio 웹 인터페이스
RAG 시스템을 위한 채팅 UI
- 모드 전환: 채팅 / 회고록작성
- 회고록 모드에서 자연어(예: '7월 2번째 주 회고록 적어줘')도 인식하도록 전처리
- Chatbot(type='messages') 사용
"""

import os
from typing import List, Tuple

import gradio as gr

from rag_system import RAGSystem


class GradioRAGApp:
    """Gradio RAG 애플리케이션 클래스"""

    def __init__(self, use_reranking: bool = True):
        """초기화"""
        self.rag_system = None
        self.use_reranking = use_reranking
        self._initialize_system()

    def _initialize_system(self):
        """RAG 시스템 초기화"""
        try:
            self.rag_system = RAGSystem(
                use_reranking=self.use_reranking,
                rerank_model="dragonkue/bge-reranker-v2-m3-ko",
                rerank_top_k=20
            )
            rerank_status = "활성화" if self.use_reranking else "비활성화"
            print(f"RAG 시스템이 초기화되었습니다. (Re-ranking: {rerank_status})")
        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류: {e}")

    # -------------------------------
    # 회고록 입력 전처리 유틸
    # -------------------------------
    def _normalize_week_phrase(self, s: str) -> str:
        """
        '7월 2번째 주' / '7월 둘째 주' / '7월 2주' -> '7월 2주차' 로 표준화
        """
        import re

        # 한글 서수 → 숫자
        ord_map = {
            "첫째": 1, "첫번째": 1,
            "둘째": 2, "두번째": 2,
            "셋째": 3, "세번째": 3,
            "넷째": 4, "네번째": 4,
            "다섯째": 5, "다섯번째": 5,
        }
        for k, v in ord_map.items():
            s = s.replace(f"{k} 주", f"{v}주")
            s = s.replace(f"{k}주", f"{v}주")

        # 'N번째 주' → 'N주', 'N 주' → 'N주'
        s = re.sub(r"(\d+)\s*번째\s*주", r"\1주", s)
        s = re.sub(r"(\d+)\s*주(?!차)", r"\1주차", s)  # '주'로 끝나면 '주차'로

        # '월 0X 주차' 사이 공백 정리
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _clean_reflection_text(self, s: str) -> str:
        """
        회고록 자연어에서 불필요한 말 제거:
        예: '회고록', '만들어줘', '적어줘', '작성', '써줘', '해줘', '부탁' 등
        """
        import re
        noise = [
            "회고록", "만들어줘", "작성", "적어줘", "써줘", "써", "해줘", "부탁", "좀", "요", "주세요"
        ]
        for w in noise:
            s = s.replace(w, " ")
        # 괄호/기호류 정리
        s = s.replace(":", " ").replace("–", " ").replace("-", " ").replace("~", " ")
        s = " ".join(s.split())
        s = self._normalize_week_phrase(s)
        return s.strip()

    # -------------------------------
    # 대화 콜백 (Chatbot type='messages')
    # -------------------------------
    def chat_response(self, message: str, history: list, mode: str) -> Tuple[str, list]:
        """
        채팅 응답 생성
        - Chatbot(type='messages') 사용: history는 [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
        mode: '채팅' | '회고록작성'
        """
        if not self.rag_system:
            response = "RAG 시스템이 초기화되지 않았습니다."
            history = (history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return "", history

        text = (message or "").strip()
        if not text:
            return "", history

        try:
            if mode == "회고록작성":
                # 접두사 없어도 OK: 자연어 정리 → get_reflection_any
                target = self._clean_reflection_text(text)
                # 사용자가 '강의 요약:'을 실수로 쓴 경우도 회고록 모드에서는 회고록으로 해석
                if target.startswith("강의 요약"):
                    target = target.replace("강의 요약", "").strip()
                response = self.rag_system.get_reflection_any(target if target else text)
            else:
                # 채팅 모드: 모든 라우팅을 RAG로 위임
                response = self.rag_system.handle_user_input(text)

            history = (history or [])
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        except Exception as e:
            error_response = f"오류가 발생했습니다: {e}"
            history = (history or [])
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return "", history

    # -------------------------------
    # Gradio UI
    # -------------------------------
    def clear_chat(self) -> list:
        """채팅 기록 초기화"""
        return []

    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(
            title="LLM 강의 검색 RAG 시스템",
            theme=gr.themes.Soft(),
            css="""
            .chatbot { height: 520px; }
            .chat-message { font-size: 14px; }
            """
        ) as interface:

            gr.Markdown("""
# 🎓 LLM 강의 검색 & Help RAG Agent

강의 자료(Jupyter 노트북, PDF)를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

## 📝 사용법
- **일반 질문**: `RNN과 CNN의 차이점은?`
- **코드 검색**: `코드 검색: CNN 모델` _또는_ `CNN 모델 만드는 코드 찾아줘`
- **강의 요약(날짜)**: `강의 요약: 2025-09-10`
- **강의 요약(주차)**: `강의 요약: 6월 3주차` _또는_ 그냥 `6월 3주차`
- **회고록**: `회고록: 7월 3주차` _또는_ 상단에서 **회고록작성** 모드 선택 후 `7월 3주차` / `7월 2번째 주`
- **코드 설명**: `코드 설명: [코드 내용 붙여넣기]`
""")

            # 🔘 모드 상태
            mode_state = gr.State("채팅")

            with gr.Row():
                chat_mode_btn = gr.Button("채팅", variant="secondary")
                retro_mode_btn = gr.Button("회고록작성", variant="secondary")
                mode_indicator = gr.Markdown("**모드:** 채팅")

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="채팅",
                        height=520,
                        show_label=True,
                        container=True,
                        show_copy_button=True,
                        type="messages",  # 권장 포맷
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="메시지",
                            placeholder="예) 강의 요약: 6월 3주차 / 회고록: 7월 3주차 / CNN 모델 만드는 코드 찾아줘",
                            container=False,
                            scale=4
                        )
                        send_btn = gr.Button("전송", variant="primary", scale=1)
                        clear_btn = gr.Button("대화 초기화", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### 📋 명령어 예시")
                    ex1 = gr.Button("CNN 모델 만드는 코드 찾아줘", variant="secondary")
                    ex2 = gr.Button("코드 검색: 딥러닝", variant="secondary")
                    ex3 = gr.Button("강의 요약: 2025-09-16", variant="secondary")
                    ex4 = gr.Button("강의 요약: 6월 3주차", variant="secondary")
                    ex5 = gr.Button("회고록: 7월 3주차", variant="secondary")
                    ex6 = gr.Button("RNN과 CNN 비교해줘", variant="secondary")
                    ex7 = gr.Button("코드 설명: 이 코드 뭐하는 거야?", variant="secondary")

            # 모드 버튼 → 상태/표시 갱신
            chat_mode_btn.click(lambda: ("채팅", "**모드:** 채팅"), inputs=None, outputs=[mode_state, mode_indicator])
            retro_mode_btn.click(lambda: ("회고록작성", "**모드:** 회고록작성"), inputs=None, outputs=[mode_state, mode_indicator])

            # 전송 이벤트 (mode_state 함께 전달)
            msg.submit(self.chat_response, inputs=[msg, chatbot, mode_state], outputs=[msg, chatbot])
            send_btn.click(self.chat_response, inputs=[msg, chatbot, mode_state], outputs=[msg, chatbot])

            # 초기화
            clear_btn.click(self.clear_chat, outputs=[chatbot])

            # 예시 버튼 → 입력창 채우기 (lambda 로 고정값 반환)
            ex1.click(lambda: "CNN 모델 만드는 코드 찾아줘", inputs=None, outputs=msg)
            ex2.click(lambda: "코드 검색: 딥러닝", inputs=None, outputs=msg)
            ex3.click(lambda: "강의 요약: 2025-09-16", inputs=None, outputs=msg)
            ex4.click(lambda: "강의 요약: 6월 3주차", inputs=None, outputs=msg)
            ex5.click(lambda: "회고록: 7월 3주차", inputs=None, outputs=msg)
            ex6.click(lambda: "RNN과 CNN 비교해줘", inputs=None, outputs=msg)
            ex7.click(lambda: "코드 설명: 이 코드 뭐하는 거야?", inputs=None, outputs=msg)

        return interface

    def launch(self, **kwargs):
        """애플리케이션 실행"""
        print("Gradio 웹 인터페이스를 시작합니다...")
        print("브라우저에서 http://localhost:7860 으로 접속하세요.")
        app = self.create_interface()
        kwargs.setdefault("server_name", "0.0.0.0")
        kwargs.setdefault("server_port", 7860)
        kwargs.setdefault("share", False)
        kwargs.setdefault("debug", True)
        return app.launch(**kwargs)


def main():
    """메인 실행 함수"""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    app = GradioRAGApp()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()