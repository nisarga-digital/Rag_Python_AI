import io
import os
import tempfile

import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_pdf(file_bytes: bytes, chunk_size: int, chunk_overlap: int) -> tuple:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    splitter = _build_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs), len(docs)


def load_csv(file_bytes: bytes, chunk_size: int, chunk_overlap: int) -> tuple:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = CSVLoader(file_path=tmp_path, encoding="utf-8").load()
    finally:
        os.unlink(tmp_path)

    splitter = _build_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs), len(docs)


def csv_to_dataframe(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes))
