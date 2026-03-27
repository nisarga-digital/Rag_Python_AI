import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from performance_review_backend import (
    GLOBAL_SESSION,
    load_pdf,
    load_csv,
    csv_to_dataframe,
    build_or_merge_vectorstore,
    load_full_history,
    load_memory_for_llm,
    load_vectorstore,
    run_query,
    save_vectorstore,
    reset_vectorstore,
    clear_memory,
    build_export_row,
    export_to_csv_bytes,
    export_filename,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PerfIQ — Performance Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _looks_like_markdown_table(block: str) -> bool:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if "|" not in lines[0]:
        return False
    return bool(re.match(r"^\s*\|?[\s:-]+\|[\s|:-]*\|?\s*$", lines[1]))


def _markdown_table_to_dataframe(block: str) -> pd.DataFrame | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    def parse_row(line: str) -> list[str]:
        line = line.strip().strip("|")
        return [cell.strip() for cell in line.split("|")]

    headers = parse_row(lines[0])
    rows = [parse_row(line) for line in lines[2:]]
    rows = [row for row in rows if len(row) == len(headers)]
    if not headers:
        return None
    return pd.DataFrame(rows, columns=headers)


def _render_assistant_content(content: str) -> None:
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        if not block.strip():
            continue
        if _looks_like_markdown_table(block):
            df = _markdown_table_to_dataframe(block)
            if df is not None:
                st.table(df)
                continue
        st.markdown(block)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════
   ROOT DESIGN TOKENS
══════════════════════════════════════════════ */
:root {
    /* Background layers */
    --bg-base:      #05080f;
    --bg-orb-1:     rgba(99, 102, 241, 0.18);
    --bg-orb-2:     rgba(139, 92, 246, 0.13);
    --bg-orb-3:     rgba(6, 182, 212, 0.10);

    /* Glass surfaces */
    --glass-white:  rgba(255, 255, 255, 0.05);
    --glass-white2: rgba(255, 255, 255, 0.09);
    --glass-dark:   rgba(5, 8, 15, 0.55);
    --glass-blur:   blur(20px);
    --glass-blur-heavy: blur(32px);

    /* Borders */
    --border:       rgba(255, 255, 255, 0.08);
    --border-hi:    rgba(255, 255, 255, 0.16);
    --border-user:  rgba(139, 92, 246, 0.45);
    --border-ai:    rgba(6, 182, 212, 0.40);
    --border-glow-u: rgba(139, 92, 246, 0.6);
    --border-glow-a: rgba(6, 182, 212, 0.55);

    /* Accent palette */
    --violet:       #8b5cf6;
    --violet-light: #a78bfa;
    --cyan:         #06b6d4;
    --cyan-light:   #67e8f9;
    --emerald:      #10b981;
    --rose:         #f43f5e;
    --amber:        #f59e0b;

    /* Text */
    --text-hi:      #f1f5ff;
    --text-mid:     #8898b0;
    --text-lo:      #3a4a62;

    /* Misc */
    --font:         'Outfit', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --r:            16px;
    --r-sm:         10px;
    --r-pill:       9999px;
    --transition:   all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ══════════════════════════════════════════════
   ANIMATED BACKGROUND — FLOATING ORBS
══════════════════════════════════════════════ */
.stApp {
    background-color: var(--bg-base) !important;
    font-family: var(--font);
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 15% 10%,  var(--bg-orb-1) 0%, transparent 65%),
        radial-gradient(ellipse 55% 45% at 85% 20%,  var(--bg-orb-2) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 50% 90%,  var(--bg-orb-3) 0%, transparent 65%),
        radial-gradient(ellipse 40% 35% at 90% 70%,  rgba(139,92,246,0.08) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
    animation: orb-drift 18s ease-in-out infinite alternate;
}
@keyframes orb-drift {
    0%   { opacity: 1;    transform: scale(1)    translateY(0px); }
    50%  { opacity: 0.85; transform: scale(1.04) translateY(-12px); }
    100% { opacity: 1;    transform: scale(0.97) translateY(6px); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5, 8, 15, 0.75) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: var(--glass-blur-heavy) !important;
    -webkit-backdrop-filter: var(--glass-blur-heavy) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font) !important; }

/* ══════════════════════════════════════════════
   GLOBAL TYPOGRAPHY + CHROME
══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }
* { font-family: var(--font); }
h1, h2, h3, h4 { color: var(--text-hi) !important; letter-spacing: -0.02em; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: var(--r-pill); }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ══════════════════════════════════════════════
   SIDEBAR LABELS + INPUTS
══════════════════════════════════════════════ */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    color: var(--text-mid) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-hi) !important;
    font-size: 14px !important;
    transition: var(--transition);
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.12), 0 0 20px rgba(139,92,246,0.08) !important;
    background: rgba(255,255,255,0.06) !important;
}

/* ══════════════════════════════════════════════
   SLIDERS
══════════════════════════════════════════════ */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--violet) !important;
    border-color: var(--violet-light) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.2) !important;
}
.stSlider [data-baseweb="track"] > div:first-child { background: var(--violet) !important; }
.stSlider p { color: var(--text-mid) !important; font-size: 11px !important; }

/* ══════════════════════════════════════════════
   BUTTONS — GLASS STYLE
══════════════════════════════════════════════ */
.stButton > button {
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: var(--r-sm) !important;
    border: 1px solid var(--border-hi) !important;
    background: var(--glass-white) !important;
    backdrop-filter: blur(10px) !important;
    color: var(--text-hi) !important;
    transition: var(--transition) !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: var(--glass-white2) !important;
    border-color: rgba(255,255,255,0.25) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3), 0 0 12px rgba(139,92,246,0.1) !important;
}

/* Primary CTA */
.primary-btn > button {
    background: linear-gradient(135deg,
        rgba(139,92,246,0.25) 0%,
        rgba(6,182,212,0.18) 100%) !important;
    border: 1px solid rgba(139,92,246,0.5) !important;
    color: #c4b5fd !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 0 0 20px rgba(139,92,246,0.15) !important;
}
.primary-btn > button:hover {
    background: linear-gradient(135deg,
        rgba(139,92,246,0.38) 0%,
        rgba(6,182,212,0.28) 100%) !important;
    border-color: rgba(139,92,246,0.7) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.12),
                0 0 32px rgba(139,92,246,0.3),
                0 8px 24px rgba(0,0,0,0.3) !important;
    transform: translateY(-2px);
}

/* ══════════════════════════════════════════════
   TABS
══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-lo) !important;
    letter-spacing: 0.04em;
    padding: 0.65rem 1.4rem !important;
    border-radius: 0 !important;
    background: transparent !important;
    border: none !important;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan-light) !important;
    border-bottom: 2px solid var(--cyan) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ══════════════════════════════════════════════
   METRICS — GLASS CARDS
══════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--glass-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 1rem 1.3rem !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 4px 16px rgba(0,0,0,0.25) !important;
    transition: var(--transition) !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hi) !important;
    transform: translateY(-3px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08),
                0 12px 32px rgba(0,0,0,0.35),
                0 0 16px rgba(139,92,246,0.08) !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-mid) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
}

/* ══════════════════════════════════════════════
   EXPANDERS
══════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: var(--glass-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-mid) !important;
    font-size: 13px !important;
    backdrop-filter: blur(10px) !important;
}
.streamlit-expanderContent {
    background: rgba(5,8,15,0.7) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--r-sm) var(--r-sm) !important;
}

/* ══════════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--glass-white) !important;
    border: 1px dashed rgba(139,92,246,0.3) !important;
    border-radius: var(--r) !important;
    padding: 1rem !important;
    backdrop-filter: blur(8px) !important;
    transition: var(--transition);
}
[data-testid="stFileUploader"]:hover {
    background: var(--glass-white2) !important;
    border-color: rgba(139,92,246,0.55) !important;
    box-shadow: 0 0 20px rgba(139,92,246,0.1) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-mid) !important; font-size: 12px !important; }

/* ══════════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════════ */
.stDataFrame { border-radius: var(--r) !important; overflow: hidden; }
[data-testid="stDataFrame"] > div {
    background: var(--glass-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    backdrop-filter: var(--glass-blur) !important;
}

/* ══════════════════════════════════════════════
   HERO SECTION
══════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 1.2rem 0 0.8rem;
    position: relative;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f1f5ff 0%, #c4b5fd 40%, #67e8f9 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-lo);
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.hero-pills {
    display: flex;
    gap: 6px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.hero-pill {
    background: var(--glass-white);
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    border-radius: var(--r-pill);
    padding: 3px 12px;
    font-size: 11px;
    color: var(--text-mid);
    font-weight: 500;
}

/* ══════════════════════════════════════════════
   SIDEBAR COMPONENTS
══════════════════════════════════════════════ */
.sb-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-lo);
    padding: 1.1rem 0 0.35rem;
    border-top: 1px solid var(--border);
    margin-top: 0.4rem;
}
.sb-section-first { border-top: none; padding-top: 0.2rem; }

.src-card {
    background: var(--glass-white);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 9px 12px;
    margin-bottom: 5px;
    backdrop-filter: blur(8px);
    transition: var(--transition);
}
.src-card:hover { border-color: var(--border-hi); }
.src-type-pdf {
    color: #c084fc; background: rgba(192,132,252,0.12);
    border-radius: 4px; padding: 1px 6px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
}
.src-type-csv {
    color: var(--emerald); background: rgba(16,185,129,0.1);
    border-radius: 4px; padding: 1px 6px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
}
.src-name { font-size: 12px; color: var(--text-hi); font-weight: 500; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.src-meta { font-size: 11px; color: var(--text-lo); margin-top: 2px; }

.mem-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.3);
    color: var(--violet-light); border-radius: var(--r-pill);
    padding: 3px 10px; font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
}

/* ══════════════════════════════════════════════
   ██████████  CHAT BUBBLES  ██████████
   The crown jewel — full glassmorphism treatment
══════════════════════════════════════════════ */

/* ── Chat wrapper ── */
.chat-area {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 0;
}

/* ── User message ── */
.bubble-user {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0 2px;
    animation: slide-in-right 0.28s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes slide-in-right {
    from { opacity: 0; transform: translateX(18px) scale(0.96); }
    to   { opacity: 1; transform: translateX(0)   scale(1); }
}
.bubble-user-inner {
    position: relative;
    max-width: 72%;
    padding: 13px 17px;
    /* Glassmorphism */
    background: linear-gradient(135deg,
        rgba(139,92,246,0.18) 0%,
        rgba(139,92,246,0.10) 60%,
        rgba(99,102,241,0.08) 100%);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid rgba(139,92,246,0.38);
    border-bottom: 1px solid rgba(139,92,246,0.22);
    border-radius: 18px 18px 4px 18px;
    /* Inner glow */
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.14),
        inset 0 -1px 0 rgba(0,0,0,0.15),
        0 4px 24px rgba(139,92,246,0.18),
        0 12px 40px rgba(0,0,0,0.3);
    color: #e8d9ff;
    font-size: 14px;
    line-height: 1.65;
    word-break: break-word;
}
/* Shimmer highlight on top edge */
.bubble-user-inner::before {
    content: '';
    position: absolute;
    top: 0; left: 12px; right: 12px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    border-radius: var(--r-pill);
}
.bubble-user-meta {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(167,139,250,0.6);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 5px;
    justify-content: flex-end;
}

/* ── AI message ── */
.bubble-ai {
    display: flex;
    justify-content: flex-start;
    margin: 2px 0 6px;
    animation: slide-in-left 0.28s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes slide-in-left {
    from { opacity: 0; transform: translateX(-18px) scale(0.96); }
    to   { opacity: 1; transform: translateX(0)    scale(1); }
}
.bubble-ai-inner {
    position: relative;
    max-width: 88%;
    padding: 16px 20px;
    /* Glassmorphism — darker, cooler tint */
    background: linear-gradient(135deg,
        rgba(6,182,212,0.10) 0%,
        rgba(5,8,15,0.65) 40%,
        rgba(5,8,15,0.55) 100%);
    backdrop-filter: var(--glass-blur-heavy);
    -webkit-backdrop-filter: var(--glass-blur-heavy);
    border: 1px solid rgba(6,182,212,0.28);
    border-left: 2px solid var(--cyan);
    border-radius: 4px 18px 18px 18px;
    /* Depth shadows */
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.07),
        inset 0 -1px 0 rgba(0,0,0,0.2),
        inset 1px 0 0 rgba(6,182,212,0.15),
        0 6px 32px rgba(0,0,0,0.4),
        0 0 40px rgba(6,182,212,0.06);
    color: var(--text-hi);
    font-size: 14px;
    line-height: 1.78;
    word-break: break-word;
}
/* Top shimmer */
.bubble-ai-inner::before {
    content: '';
    position: absolute;
    top: 0; left: 10px; right: 10px;
    height: 1px;
    background: linear-gradient(90deg, rgba(6,182,212,0.3), transparent 40%);
    border-radius: var(--r-pill);
}
/* Subtle AI avatar dot */
.bubble-ai-inner::after {
    content: '✦';
    position: absolute;
    top: -10px; left: -10px;
    width: 20px; height: 20px;
    background: linear-gradient(135deg, rgba(6,182,212,0.9), rgba(139,92,246,0.7));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 8px;
    color: white;
    box-shadow: 0 0 10px rgba(6,182,212,0.5);
    line-height: 20px;
    text-align: center;
}
.ai-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.ai-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--cyan-light);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ai-time { font-size: 10px; color: var(--text-lo); }
.ai-speed {
    display: inline-flex; align-items: center; gap: 3px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25);
    border-radius: var(--r-pill); padding: 1px 8px;
    font-size: 10px; color: var(--emerald); font-weight: 600;
}
.ai-body { color: rgba(241,245,255,0.9); }
.ai-body p { margin: 0.4em 0; }

/* ── Typing indicator (for future use) ── */
.typing-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    margin-right: 3px;
    animation: blink 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.9); }
    40%            { opacity: 1;   transform: scale(1.1); }
}

/* ── Chat divider with gradient ── */
.chat-day-divider {
    display: flex; align-items: center; gap: 10px;
    margin: 12px 0; color: var(--text-lo); font-size: 11px;
    font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.chat-day-divider::before, .chat-day-divider::after {
    content: ''; flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ══════════════════════════════════════════════
   SOURCE CHUNK CARDS
══════════════════════════════════════════════ */
.chunk-card {
    background: rgba(5,8,15,0.7);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 10px 13px;
    margin-bottom: 6px;
    backdrop-filter: blur(8px);
    transition: var(--transition);
}
.chunk-card:hover { border-color: var(--border-hi); }
.chunk-idx {
    display: inline-block;
    background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.3);
    border-radius: var(--r-pill); padding: 1px 8px;
    font-size: 10px; color: var(--cyan); font-weight: 700; margin-right: 6px;
}
.chunk-src { font-size: 11px; color: var(--text-lo); }
.chunk-text { font-size: 12px; color: rgba(136,152,176,0.8); margin-top: 6px; line-height: 1.55; font-family: var(--font-mono); }

/* ══════════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════════ */
.empty-state { text-align: center; padding: 60px 20px; }
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-title { font-size: 1.15rem; font-weight: 700; color: var(--text-mid); margin-bottom: 0.5rem; }
.empty-body { font-size: 13px; color: var(--text-lo); line-height: 1.7; }
.steps-row {
    display: flex; gap: 6px; justify-content: center; flex-wrap: wrap; margin-top: 1.3rem;
}
.step-item {
    display: flex; align-items: center; gap: 6px;
    background: var(--glass-white); border: 1px solid var(--border);
    backdrop-filter: blur(8px); border-radius: var(--r-pill);
    padding: 5px 14px; font-size: 12px; color: var(--text-mid);
}
.step-num {
    width: 18px; height: 18px;
    background: rgba(139,92,246,0.2); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: var(--violet-light); font-weight: 700;
}

/* ══════════════════════════════════════════════
   MISC
══════════════════════════════════════════════ */
.div { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

.stat-bar {
    background: var(--glass-white);
    border: 1px solid var(--border);
    backdrop-filter: var(--glass-blur);
    border-radius: var(--r);
    padding: 16px 24px;
    display: flex; gap: 24px; flex-wrap: wrap; align-items: center;
}
.stat-item { text-align: center; }
.stat-val { font-size: 1.4rem; font-weight: 700; color: var(--text-hi); }
.stat-lbl { font-size: 10px; color: var(--text-lo); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
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
    "vectorstore_bootstrap_key": None,
    "chat_bootstrapped": False,
}
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.chat_bootstrapped:
    stored_history = load_full_history(GLOBAL_SESSION)
    if stored_history:
        st.session_state.chat_history = [
            {
                "role": "user" if row["role"] == "human" else "ai",
                "content": row["content"],
                "ts": datetime.fromisoformat(row["timestamp"]).strftime("%H:%M:%S"),
            }
            for row in stored_history
        ]
        st.session_state.lc_memory = load_memory_for_llm(GLOBAL_SESSION)
    st.session_state.chat_bootstrapped = True

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
    # Brand
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div style="font-size:1.4rem; font-weight:800; letter-spacing:-0.03em; color:#f1f5ff;">
            ✦ PerfIQ
        </div>
        <div style="font-size:10px; color:#3a4a62; text-transform:uppercase;
                    letter-spacing:0.14em; margin-top:3px;">
            Performance Intelligence
        </div>
        <div style="width:32px; height:2px; margin-top:8px;
                    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
                    border-radius:9999px;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section sb-section-first">Authentication</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AIza...",
        label_visibility="visible",
    )

    if api_key and st.session_state.vectorstore_bootstrap_key != api_key:
        try:
            saved_vectorstore, saved_sources = load_vectorstore(api_key)
            if saved_vectorstore is not None:
                st.session_state.vectorstore = saved_vectorstore
                st.session_state.doc_sources = saved_sources
        except Exception:
            pass
        finally:
            st.session_state.vectorstore_bootstrap_key = api_key

    st.markdown('<div class="sb-section">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or CSV",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-section">RAG Parameters</div>', unsafe_allow_html=True)
    chunk_size    = st.slider("Chunk Size",      200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap",   0,   500,  150,  50)
    top_k         = st.slider("Top-K Retrieval", 1,   10,   5)

    st.markdown('<div class="sb-section">Model Settings</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature",       0.0, 1.0,  0.2, 0.05)
    max_tokens  = st.slider("Max Output Tokens", 512, 4096, 2048, 256)

    st.markdown("")
    build_btn = st.button(
        "⚡ Build / Update Index",
        disabled=(not uploaded_files or not api_key),
        use_container_width=True,
    )

    if build_btn and uploaded_files and api_key:
        already_indexed = [s["name"] for s in st.session_state.doc_sources]
        for uf in uploaded_files:
            if uf.name in already_indexed:
                st.sidebar.warning(f"Already indexed: {uf.name}")
                continue
            with st.spinner(f"Indexing {uf.name}…"):
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
                    save_vectorstore(
                        st.session_state.vectorstore,
                        st.session_state.doc_sources,
                    )
                    st.sidebar.success(f"✅ {uf.name} — {len(docs)} chunks")
                except Exception as e:
                    st.sidebar.error(f"❌ {uf.name}: {e}")

    # Indexed sources
    if st.session_state.doc_sources:
        st.markdown('<div class="sb-section">Indexed Sources</div>', unsafe_allow_html=True)
        for src in st.session_state.doc_sources:
            badge_cls = "src-type-pdf" if src["type"] == "pdf" else "src-type-csv"
            st.markdown(f"""
            <div class="src-card">
                <span class="{badge_cls}">{src['type'].upper()}</span>
                <div class="src-name">{src['name']}</div>
                <div class="src-meta">{src['chunks']} chunks &middot; {src['pages']} rows/pages</div>
            </div>""", unsafe_allow_html=True)

    # Memory
    turns = len(st.session_state.lc_memory) // 2
    if turns:
        st.markdown(f'<br/><span class="mem-tag">🧠 {turns} memory turn{"s" if turns != 1 else ""} active</span>',
                    unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.lc_memory = clear_memory()
            st.rerun()
    with col_b:
        if st.button("↺ Reset All", use_container_width=True):
            for k, v in SESSION_DEFAULTS.items():
                st.session_state[k] = v
            reset_vectorstore()
            st.cache_resource.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-title">Performance Review Intelligence</div>
    <div class="hero-sub">AI-powered · Conversational Memory · PDF & CSV</div>
    <div class="hero-pills">
        <span class="hero-pill">✦ RAG</span>
        <span class="hero-pill">🧠 Memory</span>
        <span class="hero-pill">📄 PDF + CSV</span>
        <span class="hero-pill">⚡ Gemini 2.5 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metric row ──
is_ready = st.session_state.vectorstore is not None
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Index Status",    "Ready ✓" if is_ready else "Empty")
with c2:
    st.metric("Sources Loaded",  len(st.session_state.doc_sources))
with c3:
    st.metric("Memory Turns",    len(st.session_state.lc_memory) // 2)
with c4:
    st.metric("Queries Run",     len(st.session_state.qa_export))

st.markdown("<br/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_chat, tab_preview, tab_export = st.tabs([
    "💬  Chat & Analysis",
    "📊  CSV Preview",
    "💾  Export Log",
])

# ─────────────────────────────── TAB 1 — CHAT ────────────────────────────────

with tab_chat:

    # Sample questions
    with st.expander("✦ Sample Questions — click to use"):
        col1, col2 = st.columns(2)
        for i, q in enumerate(SAMPLES):
            if (col1 if i % 2 == 0 else col2).button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["prefill"] = q
                st.rerun()

    # Conversation history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            ts = msg.get("ts", "")
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="bubble-user">
                    <div class="bubble-user-inner">
                        <div class="bubble-user-meta">
                            <span>You</span>
                            <span style="opacity:0.5">·</span>
                            <span>{ts}</span>
                        </div>
                        {msg["content"]}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                elapsed = msg.get("elapsed", "?")
                st.markdown(f"""
                <div class="bubble-ai">
                    <div class="bubble-ai-inner">
                        <div class="ai-header">
                            <span class="ai-label">Assistant</span>
                            <span class="ai-time">{ts}</span>
                            <span class="ai-speed">⚡ {elapsed}s</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                _render_assistant_content(msg["content"])

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 {len(sources)} source chunks retrieved"):
                        for j, doc in enumerate(sources):
                            pg  = doc.metadata.get("page", doc.metadata.get("row", "?"))
                            src = os.path.basename(doc.metadata.get("source", "unknown"))
                            st.markdown(f"""
                            <div class="chunk-card">
                                <span class="chunk-idx">#{j+1}</span>
                                <span class="chunk-src">{src} · row/page {pg}</span>
                                <div class="chunk-text">{doc.page_content[:380]}…</div>
                            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="div"/>', unsafe_allow_html=True)

    # Input area
    prefill = st.session_state.pop("prefill", "")
    query = st.text_area(
        "Ask anything about your documents",
        value=prefill,
        height=100,
        placeholder="e.g. Give me a full performance analysis of Sneha Sharma including SMART goals and a 90-day development plan…",
        label_visibility="collapsed",
    )

    ask_col, clr_col = st.columns([5, 1])
    with ask_col:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        ask_btn = st.button(
            "🚀  Analyse",
            disabled=(not st.session_state.vectorstore or not query.strip()),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with clr_col:
        if st.button("✂️ Clear", use_container_width=True):
            st.session_state["prefill"] = ""
            st.rerun()

    if ask_btn and query.strip():
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
        elif not st.session_state.vectorstore:
            st.error("Upload documents and build the index first.")
        else:
            with st.spinner("Analysing with memory context…"):
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
                    st.error(f"Error: {e}")

    # Empty state
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <div class="empty-title">Ready for Deep HR Analysis</div>
            <div class="empty-body">
                Ask anything about your employees, performance trends, or generate review templates.<br/>
                Memory tracks your full conversation — ask follow-ups naturally.
            </div>
            <div class="steps-row">
                <div class="step-item"><div class="step-num">1</div> Paste API key</div>
                <div class="step-item"><div class="step-num">2</div> Upload PDF / CSV</div>
                <div class="step-item"><div class="step-num">3</div> Build Index</div>
                <div class="step-item"><div class="step-num">4</div> Ask anything</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────── TAB 2 — CSV PREVIEW ─────────────────────────

with tab_preview:
    if st.session_state.csv_df is not None:
        df = st.session_state.csv_df
        st.markdown(f"### 📊 Data Preview — {df.shape[0]} rows × {df.shape[1]} columns")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            st.markdown("#### Numeric Summary")
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
        st.markdown("#### Full Table")
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="preview_data.csv",
            mime="text/csv",
        )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <div class="empty-title">No CSV Loaded</div>
            <div class="empty-body">Upload a CSV file and build the index to preview your data here.</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────── TAB 3 — EXPORT ──────────────────────────────

with tab_export:
    st.markdown("### 💾 Q&A Session Log")
    if st.session_state.qa_export:
        export_df = pd.DataFrame(st.session_state.qa_export)
        st.dataframe(export_df, use_container_width=True, height=340)

        avg_t   = export_df["response_time_s"].mean()
        max_mem = export_df["memory_turns"].max()
        st.markdown(f"""
        <div class="stat-bar">
            <div class="stat-item">
                <div class="stat-val">{len(export_df)}</div>
                <div class="stat-lbl">Total Queries</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{avg_t:.1f}s</div>
                <div class="stat-lbl">Avg Response</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{max_mem}</div>
                <div class="stat-lbl">Max Memory Turns</div>
            </div>
        </div>
        <br/>""", unsafe_allow_html=True)

        dl_col, clr_col = st.columns([3, 1])
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
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💾</div>
            <div class="empty-title">No Session Data Yet</div>
            <div class="empty-body">
                Your Q&A log appears here after your first query.<br/>
                Export the full session as CSV for reporting or HR audit trails.
            </div>
        </div>""", unsafe_allow_html=True)
