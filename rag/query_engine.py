import time

from langchain_core.output_parsers import StrOutputParser

from embeddings import get_llm
from prompts import get_prompt
from memory import load_memory_for_llm, update_memory


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def run_query(
    vectorstore,
    memory,
    query: str,
    api_key: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
    session_id: str,
) -> dict:
    """Run the full RAG query flow and persist memory updates."""
    llm = get_llm(api_key, temperature, max_tokens)
    prompt = get_prompt()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    t0 = time.time()
    source_docs = retriever.invoke(query)
    context_str = format_docs(source_docs)
    chat_history = load_memory_for_llm(session_id)

    prompt_input = {
        "context": context_str,
        "chat_history": chat_history,
        "input": query,
    }

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(prompt_input)
    elapsed = round(time.time() - t0, 2)

    updated_memory = update_memory(
        session_id=session_id,
        query=query,
        answer=answer,
        sources_count=len(source_docs),
        elapsed=elapsed,
    )

    return {
        "answer": answer,
        "sources": source_docs,
        "elapsed": elapsed,
        "memory": updated_memory,
    }
