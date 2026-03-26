
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from performance_review_backend import (
    load_pdf,
    load_csv,
    csv_to_dataframe,
    build_or_merge_vectorstore,
    run_query,
    clear_memory,
    build_export_row,
    export_to_csv_bytes,
    export_filename,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Performance Review Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16192a 100%);
        border-right: 1px solid #2d3561;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2140 0%, #252a4a 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .user-bubble {
        background: linear-gradient(135deg, #1a2a4a 0%, #1e3255 100%);
        border: 1px solid #2d4a7a;
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin: 8px 0 4px auto;
        color: #c8deff;
        font-size: 14px;
        max-width: 85%;
        text-align: right;
    }
    .ai-bubble {
        background: linear-gradient(135deg, #0d2137 0%, #112940 100%);
        border: 1px solid #1a5080;
        border-left: 3px solid #00b4d8;
        border-radius: 4px 12px 12px 12px;
        padding: 16px 20px;
        margin: 4px 0 8px 0;
        color: #e0f0ff;
        font-size: 14px;
        line-height: 1.7;
        max-width: 95%;
    }
    .memory-badge {
        display: inline-block;
        background: #0d3320;
        color: #4cde8e;
        border: 1px solid #1a7a45;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 600;
    }
    .step-badge {
        display: inline-block;
        background: #1a3a5c;
        color: #00b4d8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .csv-badge {
        display: inline-block;
        background: #1a3020;
        color: #6ee7a0;
        border: 1px solid #2a7040;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
    }
    .pdf-badge {
        display: inline-block;
        background: #2a1a30;
        color: #c084fc;
        border: 1px solid #6030a0;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0077b6 0%, #0096c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }
    h1, h2, h3 { color: #e0f0ff !important; }
    .stTextArea textarea {
        background: #1a1d2e !important;
        color: #e0f0ff !important;
        border: 1px solid #2d3561 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

SESSION_DEFAULTS = {
    "vectorstore":  None,
    "doc_sources":  [],
    "chat_history": [],
    "lc_memory":    [],
    "qa_export":    [],
    "csv_df":       None,
}

for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

SAMPLES = [
    "Give me a full detailed performance analysis of Arjun Reddy",
    "Compare Arjun and Rahul — strengths, gaps, and growth readiness",
    "Who are the top performers and why? Detailed justification",
    "What development recommendations do you have for Sneha Sharma?",
    "Which employees show retention risk based on reviews?",
    "Generate a detailed review template for a Machine Learning Engineer",
    "Summarize all employees below rating 4 with improvement plans",
    "What SMART goals should be set for Rahul Verma next quarter?",
]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AIza...",
    )

    st.markdown("### 📂 Upload Documents")
    st.caption("PDF and CSV supported — upload multiple files")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("### 🛠 RAG Settings")
    chunk_size    = st.slider("Chunk Size",      200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap",   0,   500,  150,  50)
    top_k         = st.slider("Top-K Retrieval", 1,   10,   5)

    st.markdown("### 🤖 LLM Settings")
    temperature = st.slider("Temperature",       0.0, 1.0,  0.2, 0.05)
    max_tokens  = st.slider("Max Output Tokens", 512, 4096, 2048, 256)

    st.markdown("---")
    build_btn = st.button(
        "🔨 Build / Update Index",
        disabled=(not uploaded_files or not api_key),
    )

    if build_btn and uploaded_files and api_key:
        already_indexed = [s["name"] for s in st.session_state.doc_sources]
        for uf in uploaded_files:
            if uf.name in already_indexed:
                st.sidebar.warning(f"⚠️ {uf.name} already indexed — skipped")
                continue
            with st.spinner(f"Indexing {uf.name}..."):
                try:
                    ext = uf.name.rsplit(".", 1)[-1].lower()
                    if ext == "pdf":
                        docs, n = load_pdf(uf.getvalue(), chunk_size, chunk_overlap)
                        dtype = "pdf"
                    else:
                        docs, n = load_csv(uf.getvalue(), chunk_size, chunk_overlap)
                        dtype = "csv"
                        st.session_state.csv_df = csv_to_dataframe(uf.getvalue())

                    st.session_state.vectorstore = build_or_merge_vectorstore(
                        st.session_state.vectorstore, docs, api_key
                    )
                    st.session_state.doc_sources.append(
                        {"name": uf.name, "type": dtype, "chunks": len(docs), "pages": n}
                    )
                    st.sidebar.success(f"✅ {uf.name} → {len(docs)} chunks")
                except Exception as e:
                    st.sidebar.error(f"❌ {uf.name}: {e}")

    # Indexed sources list
    if st.session_state.doc_sources:
        st.markdown("### 📁 Indexed Sources")
        for src in st.session_state.doc_sources:
            badge_cls = "csv-badge" if src["type"] == "csv" else "pdf-badge"
            st.markdown(f"""
            <div class="metric-card" style="padding:10px 14px;">
                <span class="{badge_cls}">{src['type'].upper()}</span>
                <small><b> {src['name']}</b></small><br/>
                <small style="color:#7090b0">{src['chunks']} chunks &middot; {src['pages']} rows/pages</small>
            </div>""", unsafe_allow_html=True)

    # Memory badge
    turns = len(st.session_state.lc_memory) // 2
    if turns:
        st.markdown(
            f'<br/><span class="memory-badge">🧠 Memory: {turns} turn(s) active</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.lc_memory = clear_memory()
            st.rerun()
    with col_b:
        if st.button("🔄 Reset All"):
            for k, v in SESSION_DEFAULTS.items():
                st.session_state[k] = v
            st.cache_resource.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER + METRICS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style="text-align:center; font-size:2.3rem; margin-bottom:4px;">
    📋 Performance Review Assistant
</h1>
<p style="text-align:center; color:#7090b0; margin-bottom:24px;">
    RAG · Conversation Memory · CSV + PDF Support · Gemini 2.5 Flash
</p>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Index",          "✅ Ready" if st.session_state.vectorstore else "⏳ Empty")
c2.metric("Sources Loaded", len(st.session_state.doc_sources))
c3.metric("Memory Turns",   len(st.session_state.lc_memory) // 2)
c4.metric("Q&A Count",      len(st.session_state.qa_export))

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_chat, tab_preview, tab_export = st.tabs([
    "💬 Chat & Analysis",
    "📊 CSV Preview",
    "💾 Export Results",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — CHAT
# ──────────────────────────────────────────────────────────────────────────────

with tab_chat:

    with st.expander("💡 Sample questions — click to prefill"):
        cols = st.columns(2)
        for i, q in enumerate(SAMPLES):
            if cols[i % 2].button(q, key=f"sq_{i}"):
                st.session_state["prefill"] = q
                st.rerun()

    if st.session_state.chat_history:
        st.markdown("### 🗨️ Conversation")
        for msg in st.session_state.chat_history:
            ts = msg.get("ts", "")
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">'
                    f'🧑 You &nbsp;·&nbsp; {ts}<br/><br/>{msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                answer_html = msg["content"].replace("\n", "<br/>")
                elapsed = msg.get("elapsed", "?")
                st.markdown(
                    f'<div class="ai-bubble">'
                    f'<b>🤖 Assistant</b> &nbsp;·&nbsp; {ts} &nbsp;·&nbsp; ⏱ {elapsed}s'
                    f'<br/><br/>{answer_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Source chunks used: {len(sources)}"):
                        for j, doc in enumerate(sources):
                            pg  = doc.metadata.get("page", doc.metadata.get("row", "?"))
                            src = os.path.basename(doc.metadata.get("source", "unknown"))
                            st.markdown(
                                f'<div class="metric-card" style="padding:10px;">'
                                f'<span class="step-badge">#{j+1}</span>'
                                f'<small style="color:#7090b0">{src} · row/page {pg}</small>'
                                f'<p style="color:#a0b8cc;font-size:12px;margin-top:6px;">'
                                f'{doc.page_content[:380]}...</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    st.markdown("---")
    prefill = st.session_state.pop("prefill", "")
    query = st.text_area(
        "Your question",
        value=prefill,
        height=110,
        placeholder="e.g. Give me a full performance analysis of Sneha Sharma including SMART goals and development plan",
        label_visibility="collapsed",
    )

    ask_col, clr_col = st.columns([4, 1])
    with ask_col:
        ask_btn = st.button(
            "🚀 Analyse",
            disabled=(not st.session_state.vectorstore or not query.strip()),
        )
    with clr_col:
        if st.button("✂️ Clear"):
            st.session_state["prefill"] = ""
            st.rerun()

    if ask_btn and query.strip():
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
        elif not st.session_state.vectorstore:
            st.error("Upload documents and click 'Build / Update Index' first.")
        else:
            with st.spinner("🔍 Analysing with memory context..."):
                try:
                    result = run_query(
                        vectorstore=st.session_state.vectorstore,
                        memory=st.session_state.lc_memory,
                        query=query,
                        api_key=api_key,
                        top_k=top_k,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.session_state.lc_memory = result["memory"]
                    ts = datetime.now().strftime("%H:%M:%S")

                    st.session_state.chat_history.append(
                        {"role": "user", "content": query, "ts": ts}
                    )
                    st.session_state.chat_history.append({
                        "role":    "ai",
                        "content": result["answer"],
                        "elapsed": result["elapsed"],
                        "sources": result["sources"],
                        "ts":      ts,
                    })
                    st.session_state.qa_export.append(
                        build_export_row(
                            query=query,
                            answer=result["answer"],
                            elapsed=result["elapsed"],
                            sources_count=len(result["sources"]),
                            memory_turns=len(st.session_state.lc_memory) // 2,
                        )
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#506070;">
            <div style="font-size:3.5rem;">📋</div>
            <h3 style="color:#7090b0!important;">Ready for Deep HR Analysis</h3>
            <p>1. Paste Gemini API key &nbsp;·&nbsp; 2. Upload PDF or CSV &nbsp;·&nbsp;
               3. Build Index &nbsp;·&nbsp; 4. Ask anything</p>
            <p style="font-size:12px;margin-top:8px;color:#405060;">
                Memory tracks your conversation — ask follow-ups like
                "Now compare them" or "What about her goals?"
            </p>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CSV PREVIEW
# ──────────────────────────────────────────────────────────────────────────────

with tab_preview:
    if st.session_state.csv_df is not None:
        df = st.session_state.csv_df
        st.markdown(f"### 📊 CSV Data — {df.shape[0]} rows × {df.shape[1]} columns")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            st.markdown("#### 📈 Numeric Summary")
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
        st.markdown("#### 🗂 Full Table")
        st.dataframe(df, use_container_width=True, height=420)
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="preview_data.csv",
            mime="text/csv",
        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#506070;">
            <div style="font-size:3rem;">📂</div>
            <p>Upload a CSV file and build the index to preview it here.</p>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — EXPORT
# ──────────────────────────────────────────────────────────────────────────────

with tab_export:
    st.markdown("### 💾 Export Q&A Session Log")
    if st.session_state.qa_export:
        export_df = pd.DataFrame(st.session_state.qa_export)
        st.dataframe(export_df, use_container_width=True, height=350)

        dl_col, clr_col = st.columns([2, 1])
        with dl_col:
            st.download_button(
                "⬇️ Download Q&A as CSV",
                data=export_to_csv_bytes(st.session_state.qa_export),
                file_name=export_filename(),
                mime="text/csv",
                use_container_width=True,
            )
        with clr_col:
            if st.button("🗑 Clear Log", use_container_width=True):
                st.session_state.qa_export = []
                st.rerun()

        avg_t   = export_df["response_time_s"].mean()
        max_mem = export_df["memory_turns"].max()
        st.markdown(f"""
        <div class="metric-card">
            <b>Total queries:</b> {len(export_df)} &nbsp;|&nbsp;
            <b>Avg response time:</b> {avg_t:.1f}s &nbsp;|&nbsp;
            <b>Max memory turns:</b> {max_mem}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#506070;">
            <div style="font-size:3rem;">💾</div>
            <p>Your Q&A session log appears here after your first analysis.<br/>
            Export the full session as a CSV for reporting or HR audit trails.</p>
        </div>""", unsafe_allow_html=True)