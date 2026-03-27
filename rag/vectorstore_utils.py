import json
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings

VECTORSTORE_DIR = Path(__file__).resolve().parent / "storage" / "faiss_index"
VECTORSTORE_META_PATH = VECTORSTORE_DIR / "metadata.json"


def build_or_merge_vectorstore(existing_vs, new_docs: list, api_key: str):
    embeddings = get_embeddings(api_key)
    new_vs = FAISS.from_documents(new_docs, embeddings)
    if existing_vs is None:
        return new_vs
    existing_vs.merge_from(new_vs)
    return existing_vs


def save_vectorstore(vectorstore, doc_sources: list | None = None) -> None:
    """Persist the FAISS index and lightweight source metadata to disk."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    metadata = {"doc_sources": doc_sources or []}
    VECTORSTORE_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_vectorstore(api_key: str):
    """Load a previously persisted FAISS index from disk."""
    index_file = VECTORSTORE_DIR / "index.faiss"
    store_file = VECTORSTORE_DIR / "index.pkl"
    if not index_file.exists() or not store_file.exists():
        return None, []

    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    doc_sources = []
    if VECTORSTORE_META_PATH.exists():
        metadata = json.loads(VECTORSTORE_META_PATH.read_text(encoding="utf-8"))
        doc_sources = metadata.get("doc_sources", [])

    return vectorstore, doc_sources


def reset_vectorstore() -> None:
    """Remove the persisted FAISS index and its metadata from disk."""
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
