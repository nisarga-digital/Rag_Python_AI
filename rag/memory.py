"""
memory.py — Long-Term SQLite Memory for HR Performance Analysis Assistant

Replaces the in-session list-based memory with a persistent SQLite store.
Stores: full Q&A history, employee names, analysis summaries, session metadata.
Scope: Global (one shared history across all sessions).
"""

import sqlite3
import json
import re
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = "hr_memory.db"          # SQLite file created in your project folder
MAX_MEMORY_MESSAGES = 20          # how many messages to load back into LLM context


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str = DB_PATH) -> None:
    """
    Create all tables if they don't exist yet.
    Safe to call on every app startup.
    """
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            -- Full Q&A conversation log
            CREATE TABLE IF NOT EXISTS conversations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL,
                timestamp     TEXT    NOT NULL,
                role          TEXT    NOT NULL CHECK(role IN ('human', 'ai')),
                content       TEXT    NOT NULL
            );

            -- Employee names extracted from queries/answers
            CREATE TABLE IF NOT EXISTS employees (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL,
                timestamp     TEXT    NOT NULL,
                employee_name TEXT    NOT NULL
            );

            -- Short analysis summaries (one per Q&A turn)
            CREATE TABLE IF NOT EXISTS summaries (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL,
                timestamp     TEXT    NOT NULL,
                question      TEXT    NOT NULL,
                summary       TEXT    NOT NULL,
                sources_used  INTEGER DEFAULT 0,
                elapsed_s     REAL    DEFAULT 0.0
            );

            -- Session-level metadata
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT    PRIMARY KEY,
                started_at    TEXT    NOT NULL,
                last_active   TEXT    NOT NULL,
                turn_count    INTEGER DEFAULT 0,
                files_loaded  TEXT    DEFAULT '[]'  -- JSON array of filenames
            );

            CREATE INDEX IF NOT EXISTS idx_conv_session  ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_emp_session   ON employees(session_id);
            CREATE INDEX IF NOT EXISTS idx_emp_name      ON employees(employee_name);
            CREATE INDEX IF NOT EXISTS idx_summ_session  ON summaries(session_id);
        """)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_session(session_id: str, db_path: str = DB_PATH) -> None:
    """Register a new session (or update last_active if it already exists)."""
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, started_at, last_active, turn_count, files_loaded)
            VALUES (?, ?, ?, 0, '[]')
            ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
        """, (session_id, now, now))


def update_session_files(session_id: str, filenames: List[str], db_path: str = DB_PATH) -> None:
    """Persist the list of files loaded in this session."""
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            UPDATE sessions SET files_loaded = ?, last_active = ? WHERE session_id = ?
        """, (json.dumps(filenames), now, session_id))


def get_all_sessions(db_path: str = DB_PATH) -> List[dict]:
    """Return all sessions ordered by most recent activity."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT session_id, started_at, last_active, turn_count, files_loaded
            FROM sessions ORDER BY last_active DESC
        """).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_turn(
    session_id: str,
    query: str,
    answer: str,
    db_path: str = DB_PATH,
) -> None:
    """Save a single Human + AI turn to the conversations table."""
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO conversations (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
            [
                (session_id, now, "human", query),
                (session_id, now, "ai",    answer),
            ],
        )
        conn.execute("""
            UPDATE sessions
            SET turn_count = turn_count + 1, last_active = ?
            WHERE session_id = ?
        """, (now, session_id))


def load_memory_for_llm(
    session_id: str,
    max_messages: int = MAX_MEMORY_MESSAGES,
    db_path: str = DB_PATH,
) -> List:
    """
    Load the most recent N messages from SQLite and return them as
    LangChain HumanMessage / AIMessage objects for injection into the LLM prompt.
    Global scope: pass session_id=GLOBAL_SESSION to load all history.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT role, content FROM conversations
            WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """, (session_id, max_messages)).fetchall()

    # Rows come back newest-first; reverse so oldest is first for the LLM
    rows = list(reversed(rows))

    messages = []
    for role, content in rows:
        if role == "human":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def load_full_history(session_id: str, db_path: str = DB_PATH) -> List[dict]:
    """
    Return the FULL conversation history for a session as plain dicts.
    Useful for UI display (chat history panel) or export.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, timestamp, role, content
            FROM conversations WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# EMPLOYEE NAME TRACKING
# ══════════════════════════════════════════════════════════════════════════════

# Very lightweight name extractor — no NLP dependency.
# Catches patterns like "analyse John Smith", "about Sarah Connor", etc.
_NAME_PATTERNS = [
    r"\b(?:about|for|analyse|analyze|review|of|employee)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?:'s|\s+performance|\s+review|\s+data|\s+profile)",
]


def extract_and_save_employees(
    session_id: str,
    text: str,
    db_path: str = DB_PATH,
) -> List[str]:
    """
    Extract employee names from a query/answer string and persist them.
    Returns the list of names found (deduplicated).
    """
    found = set()
    for pattern in _NAME_PATTERNS:
        for match in re.finditer(pattern, text):
            found.add(match.group(1).strip())

    if not found:
        return []

    now = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        # Only insert names not already recorded for this session
        existing = {
            row[0] for row in conn.execute(
                "SELECT employee_name FROM employees WHERE session_id = ?", (session_id,)
            ).fetchall()
        }
        new_names = found - existing
        if new_names:
            conn.executemany(
                "INSERT INTO employees (session_id, timestamp, employee_name) VALUES (?, ?, ?)",
                [(session_id, now, name) for name in new_names],
            )
    return list(found)


def get_employees_mentioned(session_id: str, db_path: str = DB_PATH) -> List[str]:
    """Return all employee names ever mentioned in a session (or globally)."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT DISTINCT employee_name FROM employees
            WHERE session_id = ?
            ORDER BY employee_name
        """, (session_id,)).fetchall()
    return [r[0] for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(
    session_id: str,
    question: str,
    answer: str,
    sources_used: int = 0,
    elapsed_s: float = 0.0,
    db_path: str = DB_PATH,
) -> None:
    """
    Persist a short summary (first 400 chars of the answer) for quick review
    without loading the full conversation.
    """
    summary = answer[:400].rsplit(" ", 1)[0] + "…" if len(answer) > 400 else answer
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO summaries (session_id, timestamp, question, summary, sources_used, elapsed_s)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, now, question, summary, sources_used, elapsed_s))


def get_summaries(
    session_id: Optional[str] = None,
    limit: int = 50,
    db_path: str = DB_PATH,
) -> List[dict]:
    """
    Retrieve analysis summaries.
    Pass session_id=None to get summaries across ALL sessions (global view).
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if session_id:
            rows = conn.execute("""
                SELECT * FROM summaries WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
            """, (session_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM summaries ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED update_memory  (drop-in replacement for the original function)
# ══════════════════════════════════════════════════════════════════════════════

def update_memory(
    session_id: str,
    query: str,
    answer: str,
    sources_count: int = 0,
    elapsed: float = 0.0,
    db_path: str = DB_PATH,
) -> List:
    """
    Persist the turn to SQLite and return an up-to-date LangChain message list
    for the LLM context window — same shape as the old in-memory list.

    Drop-in replacement:
        OLD: memory = update_memory(memory, query, answer)
        NEW: memory = update_memory(session_id, query, answer, sources_count, elapsed)
    """
    save_turn(session_id, query, answer, db_path)
    save_summary(session_id, query, answer, sources_count, elapsed, db_path)
    extract_and_save_employees(session_id, query + " " + answer, db_path)
    return load_memory_for_llm(session_id, MAX_MEMORY_MESSAGES, db_path)


def clear_memory(session_id: str, db_path: str = DB_PATH) -> List:
    """
    Delete all conversation rows for a session.
    Returns an empty list (compatible with old clear_memory() return).
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM summaries    WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM employees    WHERE session_id = ?", (session_id,))
        conn.execute("UPDATE sessions SET turn_count = 0 WHERE session_id = ?", (session_id,))
    return []