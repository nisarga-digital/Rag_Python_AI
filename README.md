# Rag Python AI

Streamlit-based RAG app for performance review analysis using Gemini, FAISS, and PDF/CSV document ingestion.

## Project Structure

- `rag/app.py`: Streamlit UI
- `rag/performance_review_backend.py`: document loading, embeddings, vector store, and RAG logic
- `rag/rag.py`: smaller standalone RAG example
- `assets/sample_docs/`: sample documents
- `examples/`: scratch and learning scripts kept out of the main app flow

## Run Locally

1. Create a `.env` file with `GEMINI_API_KEY=your_key_here`
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app:

```powershell
streamlit run rag/app.py
```

## Notes

- The repo ignores local virtual environments and `.env` secrets.
- The main production app lives under `rag/`.
