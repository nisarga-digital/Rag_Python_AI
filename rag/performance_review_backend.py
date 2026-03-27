from dotenv import load_dotenv

from document_loaders import csv_to_dataframe, load_csv, load_pdf
from embeddings import get_embeddings, get_llm
from export_utils import build_export_row, export_filename, export_to_csv_bytes
from memory import (
    DB_PATH,
    clear_memory as clear_sqlite_memory,
    create_session,
    get_all_sessions,
    get_employees_mentioned,
    get_summaries,
    init_db,
    load_full_history,
    load_memory_for_llm,
    update_session_files,
    update_memory,
)
from prompts import SYSTEM_PROMPT, get_prompt
from query_engine import run_query as execute_query
from vectorstore_utils import (
    VECTORSTORE_DIR,
    build_or_merge_vectorstore,
    load_vectorstore,
    reset_vectorstore,
    save_vectorstore,
)

load_dotenv()

# One shared session for the current app shape.
GLOBAL_SESSION = "global"
init_db()
create_session(GLOBAL_SESSION)


def clear_memory(session_id: str = GLOBAL_SESSION):
    """Keep the old no-arg UI call working while supporting explicit sessions."""
    return clear_sqlite_memory(session_id=session_id)


def run_query(
    vectorstore,
    memory,
    query: str,
    api_key: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
    session_id: str = GLOBAL_SESSION,
) -> dict:
    return execute_query(
        vectorstore=vectorstore,
        memory=memory,
        query=query,
        api_key=api_key,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        session_id=session_id,
    )


__all__ = [
    "DB_PATH",
    "GLOBAL_SESSION",
    "SYSTEM_PROMPT",
    "build_export_row",
    "build_or_merge_vectorstore",
    "clear_memory",
    "csv_to_dataframe",
    "export_filename",
    "export_to_csv_bytes",
    "get_all_sessions",
    "get_employees_mentioned",
    "get_embeddings",
    "get_llm",
    "get_prompt",
    "get_summaries",
    "load_csv",
    "load_full_history",
    "load_memory_for_llm",
    "load_pdf",
    "load_vectorstore",
    "reset_vectorstore",
    "run_query",
    "save_vectorstore",
    "update_memory",
    "update_session_files",
    "VECTORSTORE_DIR",
]
