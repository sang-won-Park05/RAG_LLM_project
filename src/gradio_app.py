"""
Gradio 웹 인터페이스
RAG 시스템을 위한 채팅 UI
"""

import os
import gradio as gr
from typing import List, Tuple, Dict, Any

from rag_system import RAGSystem
from vector_db_builder import VectorDBBuilder


class GradioRAGApp:
    """Gradio RAG 애플리케이션 클래스"""

    def __init__(self):
        """초기화"""
        self.rag_system = None
        self.chat_history = []
        self._initialize_system()

    def _initialize_system(self):
        """RAG 시스템 초기화"""
        try:
            self.rag_system = RAGSystem()
            print("RAG 시스템이 초기화되었습니다.")
        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류: {e}")

    def chat_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        채팅 응답 생성

        Args:
            message: 사용자 메시지
            history: 채팅 히스토리

        Returns:
            (빈 문자열, 업데이트된 히스토리)
        """
        if not self.rag_system:
            response = "RAG 시스템이 초기화되지 않았습니다."
            history.append([message, response])
            return "", history

        try:
            # 특별한 명령어 처리
            if message.lower().startswith("코드 검색:"):
                query = message[5:].strip()
                response = self._handle_code_search(query)
            elif message.lower().startswith("강의 요약:"):
                date_text = message[5:].strip()
                response = self._handle_lecture_summary(date_text)
            elif message.lower().startswith("코드 설명:"):
                # 이전 대화에서 선택된 코드에 대한 설명
                response = self._handle_code_explanation(message[5:].strip())
            else:
                # 일반 질문 처리
                result = self.rag_system.answer_question(message)
                response = self._format_response(result)

            history.append([message, response])
            return "", history

        except Exception as e:
            error_response = f"오류가 발생했습니다: {e}"
            history.append([message, error_response])
            return "", history

    def _handle_code_search(self, query: str) -> str:
        """코드 검색 처리"""
        code_snippets = self.rag_system.get_code_snippets(query)

        if not code_snippets:
            return f"'{query}'와 관련된 코드를 찾을 수 없습니다."

        response = f"'{query}'와 관련된 {len(code_snippets)}개의 코드를 찾았습니다:\\n\\n"

        for i, snippet in enumerate(code_snippets):
            response += f"**{i+1}. {snippet['filename']} (셀 {snippet['cell_index']})**\\n"
            response += f"- 강의 날짜: {snippet['lecture_date']}\\n"

            if snippet['libraries']:
                response += f"- 사용 라이브러리: {', '.join(snippet['libraries'])}\\n"
            if snippet['model_types']:
                response += f"- 모델 타입: {', '.join(snippet['model_types'])}\\n"

            # 코드 미리보기 (처음 200자)
            preview = snippet['content'][:200]
            if len(snippet['content']) > 200:
                preview += "..."

            response += f"```python\\n{preview}\\n```\\n"
            response += f"[전체 코드 보기 #{i+1}]\\n\\n"

        response += "\\n전체 코드를 보려면 '[전체 코드 보기 #번호]'를 클릭하거나 입력하세요."
        return response

    def _handle_lecture_summary(self, date_text: str) -> str:
        """강의 요약 처리"""
        import re

        # 텍스트에서 날짜 패턴 추출
        # YYYY-MM-DD 패턴 찾기
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        date_match = re.search(date_pattern, date_text)

        if date_match:
            formatted_date = date_match.group(1)
        else:
            # YYYYMMDD 패턴 찾기
            date_pattern = r'(\d{8})'
            date_match = re.search(date_pattern, date_text)
            if date_match:
                date_str = date_match.group(1)
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                return f"올바른 날짜 형식을 입력해주세요. (예: 2025-09-10 또는 20250910)"

        summary = self.rag_system.get_lecture_summary(formatted_date)
        return summary

    def _handle_code_explanation(self, query: str) -> str:
        """코드 설명 처리"""
        # 단순히 일반 질문으로 처리
        result = self.rag_system.answer_question(f"다음 코드에 대해 설명해줘: {query}")
        return self._format_response(result)

    def _format_response(self, result: Dict[str, Any]) -> str:
        """응답 포맷팅"""
        response = result['answer']

        if result['sources']:
            response += "\\n\\n**참고 자료:**\\n"
            for i, source in enumerate(result['sources']):
                response += f"{i+1}. {source['filename']}"
                if source['content_type'] == 'code':
                    response += f" (코드 셀 {source['cell_index']})"
                response += f" - {source['lecture_date']}\\n"

        return response

    def clear_chat(self) -> List[List[str]]:
        """채팅 기록 초기화"""
        return []

    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(
            title="LLM 강의 검색 RAG 시스템",
            theme=gr.themes.Soft(),
            css="""
            .chatbot { height: 500px; }
            .chat-message { font-size: 14px; }
            """
        ) as interface:

            gr.Markdown("""
            # 🎓 LLM 강의 검색 & Help RAG Agent

            강의 자료(Jupyter 노트북, PDF)를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

            ## 📝 사용법:
            - **일반 질문**: "RNN과 CNN의 차이점은?"
            - **코드 검색**: "코드 검색: CNN 모델"
            - **강의 요약**: "강의 요약: 2025-09-10"
            - **코드 설명**: 코드를 선택한 후 "코드 설명: [코드 내용]"
            """)

            with gr.Row():
                with gr.Column(scale=3):
                    # 채팅 인터페이스
                    chatbot = gr.Chatbot(
                        label="채팅",
                        height=500,
                        show_label=True,
                        container=True,
                        show_copy_button=True
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="메시지",
                            placeholder="질문을 입력하세요... (예: CNN 모델 만드는 코드 찾아줘)",
                            container=False,
                            scale=4
                        )
                        submit_btn = gr.Button("전송", variant="primary", scale=1)
                        clear_btn = gr.Button("대화 초기화", scale=1)

                with gr.Column(scale=1):

                    gr.Markdown("""
                    ### 📋 명령어 예시:
                    - `CNN 모델 만드는 코드 찾아줘`
                    - `코드 검색: 딥러닝`
                    - `강의 요약: 2024-09-16`
                    - `RNN과 CNN 비교해줘`
                    - `코드 설명: 이 코드 뭐하는 거야?`
                    """)

            # 이벤트 핸들러
            msg.submit(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )

            submit_btn.click(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )

            clear_btn.click(
                self.clear_chat,
                outputs=[chatbot]
            )

        return interface

    def launch(self, **kwargs):
        """애플리케이션 실행"""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """메인 실행 함수"""
    import os

    # 필요한 모듈 import
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # 애플리케이션 생성 및 실행
    app = GradioRAGApp()

    # Gradio 애플리케이션 실행
    app.launch(
        server_name="0.0.0.0",  # 모든 IP에서 접근 가능
        server_port=7860,       # 포트 번호
        share=False,            # 퍼블릭 링크 생성 여부
        debug=True             # 디버그 모드
    )


if __name__ == "__main__":
    main()