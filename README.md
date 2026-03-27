# Rag Python AI

Streamlit-based RAG app for performance review analysis using Gemini, FAISS, and PDF/CSV document ingestion.

## Project Structure

- `rag/app.py`: Streamlit UI
- `rag/performance_review_backend.py`: document loading, embeddings, vector store, and RAG logic
- `rag/rag.py`: smaller standalone RAG example
- `assets/sample_docs/`: sample documents
- `examples/`: scratch and learning scripts kept out of the main app flow

## Run Locally

### Git Bash

```bash
git clone https://github.com/nisarga-digital/Rag_Python_AI.git
cd Rag_Python_AI

python -m venv .venv
source .venv/Scripts/activate

pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env

streamlit run rag/app.py
```

If `python` does not work in Git Bash, try `py`:

```bash
py -m venv .venv
source .venv/Scripts/activate
```

### PowerShell

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
