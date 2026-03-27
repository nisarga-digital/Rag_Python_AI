from datetime import datetime

import pandas as pd


def build_export_row(
    query: str,
    answer: str,
    elapsed: float,
    sources_count: int,
    memory_turns: int,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "question": query,
        "answer": answer,
        "response_time_s": elapsed,
        "sources_used": sources_count,
        "memory_turns": memory_turns,
    }


def export_to_csv_bytes(qa_log: list) -> bytes:
    if not qa_log:
        return b""
    return pd.DataFrame(qa_log).to_csv(index=False).encode("utf-8")


def export_filename() -> str:
    return f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
