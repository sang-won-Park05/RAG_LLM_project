"""
Vector DB 구축 모듈
Jupyter 노트북과 PDF 파일을 처리하여 Chroma DB에 임베딩하는 기능
"""

import os
import re
import json
import pickle
from typing import List, Dict, Any

import nbformat
import pymupdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata


class VectorDBBuilder:
    """벡터 DB 구축을 위한 클래스"""

    def __init__(self, embedding_model: str = "google/embeddinggemma-300m"):
        """
        초기화
        Args:
            embedding_model: 사용할 임베딩 모델
        """
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def extract_date_from_filename(self, filename: str) -> str:
        """
        파일명에서 날짜 추출
        예: "2024-09-16_LangChain_RAG.ipynb" -> "2024-09-16"
        """
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, filename)
        return match.group(1) if match else "unknown"

    def process_notebook(self, notebook_path: str) -> List[Document]:
        """
        Jupyter 노트북 파일을 처리하여 Document 객체 리스트 반환
        코드와 마크다운을 분리하여 처리
        """
        documents = []
        filename = os.path.basename(notebook_path)
        lecture_date = self.extract_date_from_filename(filename)

        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)

            for cell_idx, cell in enumerate(notebook.cells):
                content = cell.source.strip()
                if not content:
                    continue

                # 기본 메타데이터
                metadata = {
                    "source": notebook_path,
                    "filename": filename,
                    "lecture_date": lecture_date,
                    "cell_index": cell_idx,
                    "cell_type": cell.cell_type,
                    "file_type": "notebook"
                }

                if cell.cell_type == "code":
                    # 코드 셀 처리
                    metadata.update({
                        "content_type": "code",
                        "language": "python"
                    })

                    # 코드 내용 분석하여 추가 메타데이터 생성
                    code_metadata = self._analyze_code_content(content)
                    metadata.update(code_metadata)

                elif cell.cell_type == "markdown":
                    # 마크다운 셀 처리
                    metadata.update({
                        "content_type": "markdown"
                    })

                    # 마크다운 헤더 레벨 분석
                    header_level = self._get_header_level(content)
                    if header_level:
                        metadata["header_level"] = header_level

                # Document 생성
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)

        except Exception as e:
            print(f"노트북 처리 중 오류 발생 {notebook_path}: {e}")

        return documents

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일을 처리하여 Document 객체 리스트 반환 (PyMuPDF 사용)
        """
        documents = []
        filename = os.path.basename(pdf_path)
        lecture_date = self.extract_date_from_filename(filename)

        try:
            # PyMuPDF로 PDF 열기
            pdf_document = pymupdf.open(pdf_path)
            full_text = ""

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()

                if text.strip():
                    full_text += f"\\n--- 페이지 {page_num + 1} ---\\n{text}\\n"

            pdf_document.close()

            if full_text.strip():
                # 텍스트를 청크로 분할
                chunks = self.text_splitter.split_text(full_text)

                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": pdf_path,
                        "filename": filename,
                        "lecture_date": lecture_date,
                        "chunk_index": i,
                        "file_type": "pdf",
                        "content_type": "text"
                    }

                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    documents.append(doc)

        except Exception as e:
            print(f"PDF 처리 중 오류 발생 {pdf_path}: {e}")

        return documents

    def _analyze_code_content(self, code: str) -> Dict[str, Any]:
        """
        코드 내용을 분석하여 메타데이터 생성
        """
        metadata = {}

        # 주요 라이브러리 import 검사
        libraries = []
        if "import torch" in code or "from torch" in code:
            libraries.append("pytorch")
        if "import tensorflow" in code or "from tensorflow" in code:
            libraries.append("tensorflow")
        if "import pandas" in code or "from pandas" in code:
            libraries.append("pandas")
        if "import numpy" in code or "from numpy" in code:
            libraries.append("numpy")
        if "import matplotlib" in code or "from matplotlib" in code:
            libraries.append("matplotlib")
        if "import sklearn" in code or "from sklearn" in code:
            libraries.append("sklearn")
        if "import langchain" in code or "from langchain" in code:
            libraries.append("langchain")

        if libraries:
            metadata["libraries"] = ", ".join(libraries)  # 리스트를 문자열로 변환

        # 모델 타입 검사
        model_types = []
        if any(keyword in code.lower() for keyword in ["cnn", "conv", "convolution"]):
            model_types.append("CNN")
        if any(keyword in code.lower() for keyword in ["rnn", "lstm", "gru"]):
            model_types.append("RNN")
        if any(keyword in code.lower() for keyword in ["transformer", "attention"]):
            model_types.append("Transformer")
        if any(keyword in code.lower() for keyword in ["rag", "retrieval"]):
            model_types.append("RAG")

        if model_types:
            metadata["model_types"] = ", ".join(model_types)  # 리스트를 문자열로 변환

        # 함수 정의 검사
        if "def " in code:
            metadata["has_function_definition"] = True

        # 클래스 정의 검사
        if "class " in code:
            metadata["has_class_definition"] = True

        return metadata

    def _get_header_level(self, markdown_text: str) -> int:
        """
        마크다운 헤더 레벨 추출
        """
        lines = markdown_text.split('\\n')
        for line in lines:
            if line.strip().startswith('#'):
                return len(line.split()[0])
        return 0

    def save_documents_for_bm25(self, documents: List[Document], db_path: str):
        """
        BM25용 원본 Documents를 별도 파일로 저장
        """
        os.makedirs(db_path, exist_ok=True)
        documents_path = os.path.join(db_path, "documents_for_bm25.pkl")

        try:
            with open(documents_path, 'wb') as f:
                pickle.dump(documents, f)
            print(f"BM25용 Documents 저장 완료: {documents_path}")
        except Exception as e:
            print(f"Documents 저장 중 오류: {e}")

    def build_vector_db(self, data_dir: str, db_path: str = "./chroma_db") -> Chroma:
        """
        데이터 디렉토리의 모든 파일을 처리하여 벡터 DB 구축
        """
        all_documents = []

        # notebooks 디렉토리 처리
        notebooks_dir = os.path.join(data_dir, "notebooks")
        if os.path.exists(notebooks_dir):
            print("Jupyter 노트북 파일 처리 중...")
            for filename in os.listdir(notebooks_dir):
                if filename.endswith('.ipynb'):
                    notebook_path = os.path.join(notebooks_dir, filename)
                    print(f"처리 중: {filename}")
                    documents = self.process_notebook(notebook_path)
                    all_documents.extend(documents)


        # pdfs 디렉토리 처리
        pdfs_dir = os.path.join(data_dir, "pdfs")
        if os.path.exists(pdfs_dir):
            print("PDF 파일 처리 중...")
            for filename in os.listdir(pdfs_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(pdfs_dir, filename)
                    print(f"처리 중: {filename}")
                    documents = self.process_pdf(pdf_path)
                    all_documents.extend(documents)


        if not all_documents:
            print("처리할 문서가 없습니다.")
            return None

        print(f"총 {len(all_documents)}개의 문서 청크가 생성되었습니다.")

        # 복잡한 메타데이터 필터링
        print("메타데이터 필터링 중...")
        filtered_documents = []
        for doc in all_documents:
            # Document 객체인지 확인
            if hasattr(doc, 'metadata'):
                try:
                    filtered_doc = filter_complex_metadata(doc)
                    filtered_documents.append(filtered_doc)
                except Exception as e:
                    print(f"메타데이터 필터링 오류, 원본 문서 사용: {e}")
                    filtered_documents.append(doc)
            else:
                print(f"잘못된 문서 타입: {type(doc)}, 건너뛰기")

        # BM25용 원본 Documents 저장
        self.save_documents_for_bm25(all_documents, db_path)

        # Chroma DB에 저장
        print("벡터 DB 구축 중...")
        vectorstore = Chroma.from_documents(
            documents=filtered_documents,
            embedding=self.embeddings,
            persist_directory=db_path
        )

        print(f"벡터 DB가 {db_path}에 저장되었습니다.")
        return vectorstore


def main():
    """메인 실행 함수"""
    # VectorDBBuilder 인스턴스 생성
    builder = VectorDBBuilder()

    # 벡터 DB 구축
    data_directory = "./data/educational_materials"
    db_directory = "./chroma_db"

    vectorstore = builder.build_vector_db(data_directory, db_directory)

    if vectorstore:
        print("벡터 DB 구축이 완료되었습니다!")

        # 간단한 테스트 쿼리
        test_query = "BPE_Unigram 사용 코드"
        results = vectorstore.similarity_search(test_query, k=10)
        print(f"\\n테스트 쿼리 '{test_query}'에 대한 결과:")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.metadata.get('filename', 'Unknown')} - {doc.metadata.get('content_type', 'Unknown')}")
            print(f"content: {doc.page_content}")


if __name__ == "__main__":

    from dotenv import load_dotenv
    # .env 파일 로드
    load_dotenv()    
    main()