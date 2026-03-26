import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA          # ✅ THIS LINE IS FIXED

# ── Load API Key ──────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Make sure your .env file exists and has the key.")

# ── Main RAG Function ─────────────────────────────────────────────────────────
def process_pdf_rag(pdf_file_path: str, user_query: str) -> str:

    print(f"[1/6] Loading PDF: {pdf_file_path}")
    loader = PyPDFLoader(pdf_file_path)
    data = loader.load()
    print(f"Loaded {len(data)} page(s)")   

    print("[2/6] Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(data)
    print(f"      Created {len(docs)} chunk(s)")

    print("[3/6] Loading Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    print("[4/6] Building FAISS Vector Store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("[5/6] Loading Gemini 2.5 Flash LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=2048
    )

    print("[6/6] Running RAG chain...")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )

    result = rag_chain.invoke({"query": user_query})
    return result["result"]


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PDF_PATH = os.path.join("assets", "sample_docs", "Nisarga_N_Resumepdf.pdf")
    QUERY    = "what are my skills and experience?"

    answer = process_pdf_rag(PDF_PATH, QUERY)

    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)

