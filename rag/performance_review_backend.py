
import os
import io
import time
import tempfile
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM EMBEDDINGS — calls REST v1 directly, no SDK version issues
# ══════════════════════════════════════════════════════════════════════════════

def _validate_api_key(api_key: str) -> str:
    """Fail fast on empty or obviously malformed Gemini API keys."""
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file or enter it in the app before uploading files."
        )

    api_key = api_key.strip()
    if len(api_key) < 20:
        raise ValueError(
            "GEMINI_API_KEY looks incomplete. Double-check that the full Gemini API key was copied into .env."
        )
    return api_key


def get_embeddings(api_key: str) -> Embeddings:
    api_key = _validate_api_key(api_key)
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert HR Performance Analysis Assistant with deep experience in talent management,
employee development, and organizational psychology.

Your role is to provide DETAILED, STRUCTURED, and ACTIONABLE performance analyses based strictly on the
provided context (performance reviews, employee records, CSV data). Never fabricate data.

ANALYSIS FRAMEWORK — always follow this when relevant:

1. PERFORMANCE SUMMARY
- Overall rating / score with context
- Trend (improving / declining / stable) if multi-period data exists

2. STRENGTHS ANALYSIS
- Core technical competencies
- Soft skills and leadership qualities
- Demonstrated achievements with specifics

3. AREAS FOR IMPROVEMENT
- Skill gaps with severity (critical / moderate / minor)
- Behavioral patterns needing attention
- Specific examples from reviews

4. GOAL ASSESSMENT
- Progress on previously set goals
- Recommended SMART goals for next period

5. DEVELOPMENT RECOMMENDATIONS
- Training / certifications to pursue
- Mentoring or coaching suggestions
- Stretch assignments or projects

6. PROMOTION / GROWTH READINESS
- Readiness level (Ready Now / 6-12 months / 12+ months)
- Justification based on data

7. KEY RISKS
- Retention risk if visible
- Performance risks going forward

CONVERSATION MEMORY:
You have access to the full conversation history. Use it to reference previous questions,
build on earlier analysis, and track employee names mentioned in this session.

FORMATTING RULES:
- Always use headers and bullet points for structured analysis
- Use emoji section markers for readability
- Quantify wherever possible (ratings, years, percentages)
- End with a QUICK INSIGHT summary (2-3 sentences max)

Context from documents:
{context}"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(api_key: str, temperature: float, max_tokens: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(file_bytes: bytes, chunk_size: int, chunk_overlap: int) -> tuple:
    """Load PDF from raw bytes. Returns (split_docs, page_count)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    docs = PyPDFLoader(tmp_path).load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs), len(docs)


def load_csv(file_bytes: bytes, chunk_size: int, chunk_overlap: int) -> tuple:
    """Load CSV from raw bytes. Returns (split_docs, row_count)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    docs = CSVLoader(file_path=tmp_path, encoding="utf-8").load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs), len(docs)


def csv_to_dataframe(file_bytes: bytes):
    """Parse raw CSV bytes into a DataFrame for preview."""
    return pd.read_csv(io.BytesIO(file_bytes))


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════

def build_or_merge_vectorstore(existing_vs, new_docs: list, api_key: str):
    """Create a new FAISS store or merge new docs into an existing one."""
    embeddings = get_embeddings(api_key)
    new_vs = FAISS.from_documents(new_docs, embeddings)
    if existing_vs is None:
        return new_vs
    existing_vs.merge_from(new_vs)
    return existing_vs


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════════════════════

MAX_MEMORY = 20  # 10 turns x 2 messages


def update_memory(memory: list, query: str, answer: str) -> list:
    """Append a Human/AI pair and trim to MAX_MEMORY."""
    memory.append(HumanMessage(content=query))
    memory.append(AIMessage(content=answer))
    if len(memory) > MAX_MEMORY:
        memory = memory[-MAX_MEMORY:]
    return memory


def clear_memory() -> list:
    return []


# ══════════════════════════════════════════════════════════════════════════════
# RAG QUERY
# ══════════════════════════════════════════════════════════════════════════════

def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def run_query(
    vectorstore,
    memory: list,
    query: str,
    api_key: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    """
    Run a RAG query with conversation memory.
    Returns {"answer", "sources", "elapsed", "memory"}.
    """
    llm       = get_llm(api_key, temperature, max_tokens)
    prompt    = get_prompt()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    t0          = time.time()
    source_docs = retriever.invoke(query)
    context_str = _format_docs(source_docs)

    prompt_input = {
        "context":      context_str,
        "chat_history": memory,
        "input":        query,
    }

    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke(prompt_input)

    elapsed        = round(time.time() - t0, 2)
    updated_memory = update_memory(memory, query, answer)

    return {
        "answer":  answer,
        "sources": source_docs,
        "elapsed": elapsed,
        "memory":  updated_memory,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_export_row(
    query: str,
    answer: str,
    elapsed: float,
    sources_count: int,
    memory_turns: int,
) -> dict:
    return {
        "timestamp":       datetime.now().isoformat(),
        "question":        query,
        "answer":          answer,
        "response_time_s": elapsed,
        "sources_used":    sources_count,
        "memory_turns":    memory_turns,
    }


def export_to_csv_bytes(qa_log: list) -> bytes:
    if not qa_log:
        return b""
    return pd.DataFrame(qa_log).to_csv(index=False).encode("utf-8")


def export_filename() -> str:
    return f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
