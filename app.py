"""AI Customer Support Automation Desk — Production Streamlit UI.

Enterprise-grade multi-agent support console with live agent execution
visualizer, real-time analytics dashboard, and polished ticket history viewer.
Built with AutoGen v0.4+, ChromaDB RAG, and Streamlit.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

from core.config import get_available_provider, get_model_client, LOG_FILE, KB_DIR
from core.team import process_complaint_stream
from tools.ticket_logger import CSV_COLUMNS

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Support Desk · AutoGen Multi-Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Deep Indigo / Slate Professional Theme
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global polish ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label { color: #94a3b8 !important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }
section[data-testid="stSidebar"] hr { border-color: #334155 !important; }
section[data-testid="stSidebar"] .stButton > button { background: #334155; border: 1px solid #475569; color: #f1f5f9 !important; border-radius: 8px; transition: all 0.2s; }
section[data-testid="stSidebar"] .stButton > button:hover { background: #475569; border-color: #64748b; }

/* ── Header banner ─────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1e293b 0%, #312e81 50%, #4338ca 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -30%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 { margin: 0 0 0.3rem 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; }
.hero-banner p  { margin: 0; font-size: 0.95rem; color: #c7d2fe; font-weight: 400; }

/* ── Metric cards ──────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: box-shadow 0.2s, transform 0.2s;
}
.kpi-card:hover { box-shadow: 0 8px 25px rgba(0,0,0,0.08); transform: translateY(-2px); }
.kpi-value { font-size: 2rem; font-weight: 800; color: #1e293b; line-height: 1.1; }
.kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; }
.kpi-accent-blue   .kpi-value { color: #4338ca; }
.kpi-accent-green  .kpi-value { color: #16a34a; }
.kpi-accent-red    .kpi-value { color: #dc2626; }
.kpi-accent-amber  .kpi-value { color: #d97706; }

/* ── Status badges / pills ─────────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
}
.pill-resolved {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
    border: 1px solid #86efac;
}
.pill-escalated {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
    border: 1px solid #fca5a5;
}
.pill-icon { font-size: 1.05rem; }

/* ── Agent timeline step card ──────────────────────────── */
.agent-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.agent-step:hover { border-left-color: #4338ca; }
.agent-step-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.5rem;
}
.agent-step-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.agent-step-icon.query    { background: #ede9fe; }
.agent-step-icon.kb       { background: #e0f2fe; }
.agent-step-icon.resolve  { background: #fef3c7; }
.agent-step-icon.log      { background: #d1fae5; }
.agent-step-name  { font-weight: 700; font-size: 0.88rem; color: #1e293b; }
.agent-step-role  { font-size: 0.78rem; color: #64748b; }

/* ── Response card ─────────────────────────────────────── */
.response-card {
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.65;
    position: relative;
}
.response-card.resolved {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1px solid #86efac;
    color: #14532d;
}
.response-card.escalated {
    background: linear-gradient(135deg, #fef2f2, #fff1f2);
    border: 1px solid #fca5a5;
    color: #7f1d1d;
}

/* ── Detail grid ───────────────────────────────────────── */
.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}
.detail-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.detail-item .label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.detail-item .value { font-size: 1rem; font-weight: 700; color: #1e293b; margin-top: 2px; }

/* ── Complaint quote ───────────────────────────────────── */
.complaint-quote {
    background: #f1f5f9;
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    font-style: italic;
    color: #334155;
    margin: 0.75rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Section titles ────────────────────────────────────── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Sidebar knowledge-base chips ──────────────────────── */
.kb-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #334155;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 4px 10px;
    margin: 3px 2px;
    font-size: 0.78rem;
    color: #cbd5e1 !important;
}

/* ── Polish native Streamlit elements ──────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 24px;
    font-weight: 600;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}

/* ═══════════════════════════════════════════════════════════
   VERTICAL STEP-BY-STEP CONNECTED WORKFLOW (DOTS & LINE)
   ═══════════════════════════════════════════════════════════ */
.workflow-tree {
    position: relative;
    padding-left: 8px;
    margin: 1rem 0;
}

.tree-node {
    position: relative;
    padding-bottom: 1.5rem;
    padding-left: 2.4rem;
}

.tree-node:last-child {
    padding-bottom: 0.25rem;
}

/* Continuous connecting line between dots */
.tree-node::before {
    content: '';
    position: absolute;
    left: 13px;
    top: 30px;
    bottom: -6px;
    width: 3px;
    background: #e2e8f0;
    transition: background 0.4s ease, box-shadow 0.4s ease;
}

/* Line turns bright green when step is completed */
.tree-node.completed::before {
    background: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.tree-node:last-child::before {
    display: none;
}

/* Node indicator circle */
.tree-dot {
    position: absolute;
    left: 0;
    top: 2px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 700;
    z-index: 2;
    transition: all 0.3s ease;
}

/* Pending State */
.tree-node.pending .tree-dot {
    background: #f1f5f9;
    border: 2px solid #cbd5e1;
    color: #94a3b8;
}

/* Active / Running State (Pulsing blue/indigo) */
.tree-node.active .tree-dot {
    background: #6366f1;
    border: 3px solid #c7d2fe;
    color: #ffffff;
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7);
    animation: pulse-ring 1.5s infinite;
}

@keyframes pulse-ring {
    0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
    100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
}

/* Completed State (Vibrant emerald green with checkmark) */
.tree-node.completed .tree-dot {
    background: #10b981;
    border: 2px solid #059669;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
}

/* Step Content Card */
.tree-content {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem 1.15rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    transition: all 0.2s ease;
}

.tree-node.active .tree-content {
    border-color: #a5b4fc;
    background: #f8faff;
}

.tree-node.completed .tree-content {
    border-color: #bbf7d0;
}

.tree-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.tree-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 6px;
}

.tree-desc {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 2px;
}

.tree-badge {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 999px;
}

.tree-node.pending .tree-badge { background: #f1f5f9; color: #94a3b8; }
.tree-node.active .tree-badge  { background: #ede9fe; color: #4f46e5; }
.tree-node.completed .tree-badge { background: #dcfce7; color: #15803d; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS — Sample Customer Complaints
# ═══════════════════════════════════════════════════════════════════
SAMPLE_COMPLAINTS: Dict[str, Any] = {
    "— Select a recent complaint —": None,
    "Priya Sharma · ORD-4512 — Delayed delivery inquiry": {
        "name": "Priya Sharma", "order_id": "ORD-4512",
        "complaint": "Where is my order? It was supposed to arrive yesterday.",
    },
    "Alex Johnson · ORD-8821 — Shoe return & refund request": {
        "name": "Alex Johnson", "order_id": "ORD-8821",
        "complaint": "I want to return the shoes I bought last week. How do I get a refund?",
    },
    "Carlos Gomez · ORD-7740 — Unable to access account": {
        "name": "Carlos Gomez", "order_id": "ORD-7740",
        "complaint": "I forgot my password and can't log in.",
    },
    "Sarah Miller · ORD-3309 — Damaged product received": {
        "name": "Sarah Miller", "order_id": "ORD-3309",
        "complaint": "The blender I received is broken. Can I get a replacement?",
    },
    "Robert Vance · ORD-9901 — Multiple card charges": {
        "name": "Robert Vance", "order_id": "ORD-9901",
        "complaint": "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    },
    "Evelyn Wright · ORD-6150 — Corporate gifting inquiry": {
        "name": "Evelyn Wright", "order_id": "ORD-6150",
        "complaint": "Do you sell gift cards for a company of 500 employees with custom branding?",
    },
    "Daniel Craig · ORD-7712 — Shipping address update": {
        "name": "Daniel Craig", "order_id": "ORD-7712",
        "complaint": "I just placed an order 20 minutes ago. How do I change my shipping address?",
    },
    "Maya Patel · ORD-2284 — Suspected account breach": {
        "name": "Maya Patel", "order_id": "ORD-2284",
        "complaint": "I believe my account has been hacked and someone placed unauthorized orders with my saved card. Block my account immediately!",
    },
}

AGENT_META = {
    "Query_Agent": {
        "icon": "🔍", "css": "query",
        "label": "Query Analyst",
        "desc": "Extracted intent, urgency & sentiment",
    },
    "Knowledge_Agent": {
        "icon": "📚", "css": "kb",
        "label": "Knowledge Retriever",
        "desc": "Searched ChromaDB vector store",
    },
    "Resolution_Agent": {
        "icon": "✍️", "css": "resolve",
        "label": "Resolution Drafter",
        "desc": "Composed grounded customer reply",
    },
    "Log_Agent": {
        "icon": "📋", "css": "log",
        "label": "QA & Logger",
        "desc": "Evaluated escalation rules & persisted ticket",
    },
}


# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "latest_result": None,
    "latest_updates": [],
    "is_processing": False,
    "run_count": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
WORKFLOW_STEPS = [
    {
        "key": "Query_Agent",
        "icon": "🔍",
        "title": "Query Analyst",
        "desc": "Extracts customer intent, urgency level & emotional sentiment",
    },
    {
        "key": "Knowledge_Agent",
        "icon": "📚",
        "title": "Knowledge Retriever",
        "desc": "Executes semantic RAG search across ChromaDB vector store",
    },
    {
        "key": "Resolution_Agent",
        "icon": "✍️",
        "title": "Resolution Drafter",
        "desc": "Drafts grounded reply strictly matching company policy facts",
    },
    {
        "key": "Log_Agent",
        "icon": "📋",
        "title": "QA & Auditor",
        "desc": "Evaluates 5 escalation criteria & records ticket to CSV",
    },
]


def render_connected_workflow(
    active_agent_key: str = None,
    completed_keys: list = None,
    step_summaries: dict = None,
) -> str:
    """Generates connected vertical tree HTML with animated dots and green connecting lines.

    Constructed strictly without leading whitespace or indentation to prevent
    Markdown parsers from interpreting any tag as an indented code block.
    """
    completed_keys = completed_keys or []
    step_summaries = step_summaries or {}
    lines = ['<div class="workflow-tree">']

    for idx, step in enumerate(WORKFLOW_STEPS, 1):
        key = step["key"]
        if key in completed_keys:
            state = "completed"
            dot_content = "✓"
            badge = "Completed"
        elif key == active_agent_key:
            state = "active"
            dot_content = str(idx)
            badge = "In Progress"
        else:
            state = "pending"
            dot_content = str(idx)
            badge = "Pending"

        summary_text = step_summaries.get(key)
        if summary_text:
            desc_html = f'<div class="tree-desc" style="color:#0f172a; font-weight:500;">{summary_text}</div>'
        else:
            desc_html = f'<div class="tree-desc">{step["desc"]}</div>'

        lines.append(
            f'<div class="tree-node {state}">'
            f'<div class="tree-dot">{dot_content}</div>'
            f'<div class="tree-content">'
            f'<div class="tree-header">'
            f'<div class="tree-title">{step["icon"]} {step["title"]}</div>'
            f'<span class="tree-badge">{badge}</span>'
            f'</div>'
            f'{desc_html}'
            f'</div>'
            f'</div>'
        )

    lines.append('</div>')
    return "".join(lines)


def get_step_summaries(res: dict) -> dict:
    """Extract human-readable one-line summaries for completed steps."""
    if not res:
        return {}
    return {
        "Query_Agent": f"Category: {res.get('category', 'General')} · Urgency: {res.get('urgency', 'Low')} · Sentiment: {res.get('sentiment', 'Neutral')}",
        "Knowledge_Agent": f"Sources consulted: {res.get('kb_sources', 'None')}",
        "Resolution_Agent": f"Grounded reply ready · Confidence: {res.get('confidence', 'High')}",
        "Log_Agent": f"Decision: {res.get('status', 'Resolved')} · Ticket: {res.get('ticket_id', 'N/A')}",
    }


def load_logs() -> pd.DataFrame:
    """Load ticket CSV with graceful fallback."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)
        return df
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


def render_kpi(value: str, label: str, accent: str = "blue") -> str:
    """Return HTML for a single KPI metric card."""
    return f"""
    <div class="kpi-card kpi-accent-{accent}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def render_agent_step(source: str, content: str) -> None:
    """Render a single agent timeline step as a styled card."""
    meta = AGENT_META.get(source, {"icon": "🤖", "css": "query", "label": source, "desc": ""})
    st.markdown(f"""
    <div class="agent-step">
        <div class="agent-step-header">
            <div class="agent-step-icon {meta['css']}">{meta['icon']}</div>
            <div>
                <div class="agent-step-name">{meta['label']}</div>
                <div class="agent-step-role">{meta['desc']}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.code(content, language="markdown")


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size:2rem;">🛡️</span>
        <div style="font-size:1.1rem; font-weight:800; margin-top:0.25rem; color:#f1f5f9;">
            AI Support Desk
        </div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">
            AutoGen v0.4+ · ChromaDB · Streamlit
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Provider selection ──
    st.markdown('<p style="font-size:0.72rem; color:#94a3b8 !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">LLM Provider</p>', unsafe_allow_html=True)
    env_info = get_available_provider()
    detected_provider = env_info.get("provider") or "openai"

    provider_map = {"OpenAI": "openai", "Google Gemini": "gemini", "Groq": "groq", "Custom / Ollama": "custom"}
    reverse_map = {v: k for k, v in provider_map.items()}
    provider_choice = st.selectbox(
        "LLM Provider",
        options=list(provider_map.keys()),
        index=list(provider_map.values()).index(detected_provider) if detected_provider in provider_map.values() else 0,
        label_visibility="collapsed",
    )
    selected_provider_name = provider_map[provider_choice]

    # ── API key ──
    default_key = env_info.get("api_key") or ""
    api_key_input = st.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Auto-loaded from .env · Paste a key here to override",
    )

    # ── Model & base URL ──
    model_defaults = {
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "gemini": "gemini-2.5-flash",
        "groq": "llama-3.3-70b-versatile",
        "custom": "llama3.1",
    }
    base_url_defaults = {
        "openai": "",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "groq": "https://api.groq.com/openai/v1",
        "custom": "http://localhost:11434/v1",
    }
    custom_model = st.text_input("Model", value=model_defaults.get(selected_provider_name, "gpt-4o-mini"))

    if selected_provider_name == "custom":
        selected_base_url = st.text_input("Base URL", value=base_url_defaults["custom"])
    else:
        selected_base_url = base_url_defaults.get(selected_provider_name) or None

    # ── Connection status ──
    if api_key_input and not api_key_input.startswith("your_"):
        st.success(f"✅ Connected — {provider_choice}", icon="🔑")
    else:
        st.warning("API key required", icon="⚠️")

    st.markdown("---")

    # ── Knowledge base preview ──
    st.markdown('<p style="font-size:0.72rem; color:#94a3b8 !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Knowledge Base Documents</p>', unsafe_allow_html=True)
    kb_files = sorted(KB_DIR.glob("*.md"))
    if kb_files:
        chips_html = "".join(f'<span class="kb-chip">📄 {f.name}</span>' for f in kb_files)
        st.markdown(chips_html, unsafe_allow_html=True)
        with st.expander("Inspect KB Contents"):
            for kf in kb_files:
                st.caption(f"**{kf.name}**")
                st.code(kf.read_text(encoding="utf-8")[:350] + "\n…", language="markdown")

    st.markdown("---")

    # ── Quick stats from CSV ──
    _df = load_logs()
    _total = len(_df)
    st.markdown('<p style="font-size:0.72rem; color:#94a3b8 !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">Session Stats</p>', unsafe_allow_html=True)
    _resolved = len(_df[_df["status"] == "Resolved"]) if _total else 0
    _escalated = len(_df[_df["status"] == "Escalated"]) if _total else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", _total)
    c2.metric("✅", _resolved)
    c3.metric("🚨", _escalated)

    if st.button("🗑️  Clear Logs", use_container_width=True):
        if LOG_FILE.exists():
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(LOG_FILE, index=False)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
#  HERO BANNER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🛡️ AI Customer Support Desk</h1>
    <p>Multi-agent orchestration with AutoGen v0.4+ · ChromaDB vector RAG · Real-time analytics</p>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab_console, tab_analytics, tab_logs = st.tabs([
    "💬  Support Console",
    "📊  Analytics Dashboard",
    "📋  Ticket History",
])


# ═══════════════════════════════════════════════════════════════════
#  TAB 1 — SUPPORT CONSOLE
# ═══════════════════════════════════════════════════════════════════
with tab_console:
    col_form, col_result = st.columns([5, 7], gap="large")

    # ── LEFT: Complaint Form ──────────────────────────────
    with col_form:
        st.markdown('<div class="section-title">📝 New Customer Complaint</div>', unsafe_allow_html=True)

        with st.container(border=True):
            selected_case = st.selectbox(
                "Recent Complaints",
                options=list(SAMPLE_COMPLAINTS.keys()),
                key="test_case_select",
            )
            preset = SAMPLE_COMPLAINTS.get(selected_case)
            _dn = preset["name"] if preset else "Priya Sharma"
            _do = preset["order_id"] if preset else "ORD-4512"
            _dc = preset["complaint"] if preset else ""

            c_name, c_order = st.columns(2)
            with c_name:
                customer_name = st.text_input("Customer Name", value=_dn, key="inp_name")
            with c_order:
                order_id = st.text_input("Order ID", value=_do, key="inp_order")

            complaint_text = st.text_area(
                "Complaint / Question",
                value=_dc,
                height=125,
                placeholder="Describe the issue — late shipment, broken product, billing dispute …",
                key="inp_complaint",
            )

            submit = st.button(
                "🚀  Process with AI Agents",
                type="primary",
                use_container_width=True,
            )

    # ── RIGHT: Agent Execution & Results ──────────────────
    with col_result:
        st.markdown('<div class="section-title">🤖 Multi-Agent Workflow</div>', unsafe_allow_html=True)
        tree_placeholder = st.empty()

        # ── Handle new submission ──
        if submit:
            if not complaint_text.strip():
                st.error("Please enter a customer complaint before submitting.", icon="⚠️")
            elif not api_key_input or api_key_input.startswith("your_"):
                st.error("Configure a valid API key in the sidebar (or .env file) to run AI agents.", icon="🔑")
            else:
                st.session_state["is_processing"] = True
                st.session_state["latest_result"] = None
                st.session_state["latest_updates"] = []

                completed_agents = []
                live_summaries = {}

                # Stage 0: Step 1 (Query Agent) starts as active
                tree_placeholder.markdown(
                    render_connected_workflow(active_agent_key="Query_Agent", completed_keys=[]),
                    unsafe_allow_html=True,
                )

                try:
                    client = get_model_client(
                        api_key=api_key_input,
                        model=custom_model,
                        provider=selected_provider_name,
                        base_url=selected_base_url,
                    )

                    workflow_state: Dict[str, Any] = {"final_data": None, "updates": []}

                    next_agent_lookup = {
                        "Query_Agent": "Knowledge_Agent",
                        "Knowledge_Agent": "Resolution_Agent",
                        "Resolution_Agent": "Log_Agent",
                        "Log_Agent": None,
                    }

                    async def _run():
                        async for event in process_complaint_stream(
                            complaint=complaint_text,
                            customer_name=customer_name,
                            order_id=order_id,
                            model_client=client,
                        ):
                            if event["type"] == "agent_message":
                                src = event["source"]
                                cnt = event["content"]
                                workflow_state["updates"].append((src, cnt))
                                if src not in completed_agents:
                                    completed_agents.append(src)

                                # Live inline summaries per agent
                                if src == "Query_Agent":
                                    cat_m = re.search(r"Category:\s*([A-Za-z]+)", cnt, re.I)
                                    urg_m = re.search(r"Urgency:\s*(Low|Medium|High)", cnt, re.I)
                                    sent_m = re.search(r"Sentiment:\s*([A-Za-z ]+)", cnt, re.I)
                                    cat_val = cat_m.group(1) if cat_m else "Classified"
                                    urg_val = urg_m.group(1) if urg_m else "Normal"
                                    sent_val = sent_m.group(1).strip() if sent_m else "Neutral"
                                    live_summaries["Query_Agent"] = f"Category: {cat_val} · Urgency: {urg_val} · Sentiment: {sent_val}"
                                elif src == "Knowledge_Agent":
                                    sources = re.findall(r"\[Source:\s*([^\]]+)\]", cnt)
                                    if sources:
                                        live_summaries["Knowledge_Agent"] = f"Sources: {', '.join(dict.fromkeys(sources))}"
                                    else:
                                        live_summaries["Knowledge_Agent"] = "No matching documents found in KB"
                                elif src == "Resolution_Agent":
                                    conf_m = re.search(r"Confidence:\s*(High|Low)", cnt, re.I)
                                    conf_val = conf_m.group(1) if conf_m else "High"
                                    live_summaries["Resolution_Agent"] = f"Resolution drafted · Confidence: {conf_val}"
                                elif src == "Log_Agent":
                                    stat_m = re.search(r"Status:\s*(Resolved|Escalated)", cnt, re.I)
                                    stat_val = stat_m.group(1) if stat_m else "Completed"
                                    live_summaries["Log_Agent"] = f"Outcome: {stat_val} · Ticket logged to CSV"

                                # Next node becomes active, prior nodes & line turn green
                                next_step = next_agent_lookup.get(src)
                                tree_placeholder.markdown(
                                    render_connected_workflow(
                                        active_agent_key=next_step,
                                        completed_keys=completed_agents,
                                        step_summaries=live_summaries,
                                    ),
                                    unsafe_allow_html=True,
                                )
                            elif event["type"] == "final_result":
                                workflow_state["final_data"] = event["result"]

                    asyncio.run(_run())

                    # Final state: all 4 nodes green with full connecting line and consolidated summaries
                    final_res = workflow_state["final_data"]
                    final_summaries = get_step_summaries(final_res) if final_res else live_summaries
                    tree_placeholder.markdown(
                        render_connected_workflow(
                            active_agent_key=None,
                            completed_keys=[s["key"] for s in WORKFLOW_STEPS],
                            step_summaries=final_summaries,
                        ),
                        unsafe_allow_html=True,
                    )

                    st.session_state["latest_result"] = workflow_state["final_data"]
                    st.session_state["latest_updates"] = workflow_state["updates"]
                    st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1

                except Exception as exc:
                    error_msg = str(exc)
                    if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                        st.error("**Rate-limit hit.** Wait a moment and try again, or switch to a different LLM provider in the sidebar.", icon="⏳")
                    elif "auth" in error_msg.lower() or "key" in error_msg.lower() or "401" in error_msg:
                        st.error("**Authentication error.** Double-check your API key in the sidebar.", icon="🔑")
                    else:
                        st.error(f"**Agent error:** {error_msg}", icon="🚨")
                finally:
                    st.session_state["is_processing"] = False
        else:
            # Steady-state: Render workflow tree if result exists, or initial pending state
            existing_res = st.session_state.get("latest_result")
            if existing_res:
                tree_placeholder.markdown(
                    render_connected_workflow(
                        active_agent_key=None,
                        completed_keys=[s["key"] for s in WORKFLOW_STEPS],
                        step_summaries=get_step_summaries(existing_res),
                    ),
                    unsafe_allow_html=True,
                )
            else:
                tree_placeholder.markdown(
                    render_connected_workflow(active_agent_key=None, completed_keys=[]),
                    unsafe_allow_html=True,
                )

        # ── Display persisted result (survives re-runs) ──
        res = st.session_state.get("latest_result")
        if res:
            is_esc = res["status"].lower() == "escalated"

            # Status pill
            if is_esc:
                st.markdown(
                    '<div class="pill pill-escalated"><span class="pill-icon">🚨</span> ESCALATED — Forwarded to Human Agent</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="pill pill-resolved"><span class="pill-icon">✅</span> RESOLVED — Handled by AI Support</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")  # spacer

            # Customer complaint echo
            st.markdown(f'<div class="complaint-quote">"{res["complaint"]}"</div>', unsafe_allow_html=True)

            # Proposed response
            st.markdown('<div class="section-title">💬 Proposed Customer Response</div>', unsafe_allow_html=True)
            css_cls = "escalated" if is_esc else "resolved"
            st.markdown(f'<div class="response-card {css_cls}">{res["response"]}</div>', unsafe_allow_html=True)

            # Detail grid
            urg_color = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}.get(res["urgency"].lower(), "#64748b")
            sent_color = {"very angry": "#dc2626", "angry": "#ef4444", "frustrated": "#f59e0b", "neutral": "#64748b", "calm": "#16a34a"}.get(res["sentiment"].lower(), "#64748b")
            conf_color = "#16a34a" if res["confidence"].lower() == "high" else "#dc2626"

            st.markdown(f"""
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="label">Ticket ID</div>
                    <div class="value">{res.get("ticket_id", "N/A")}</div>
                </div>
                <div class="detail-item">
                    <div class="label">Category</div>
                    <div class="value">{res.get("category", "General")}</div>
                </div>
                <div class="detail-item">
                    <div class="label">Urgency</div>
                    <div class="value" style="color:{urg_color}">{res.get("urgency", "Low")}</div>
                </div>
                <div class="detail-item">
                    <div class="label">Sentiment</div>
                    <div class="value" style="color:{sent_color}">{res.get("sentiment", "Neutral")}</div>
                </div>
                <div class="detail-item">
                    <div class="label">Confidence</div>
                    <div class="value" style="color:{conf_color}">{res.get("confidence", "High")}</div>
                </div>
                <div class="detail-item">
                    <div class="label">KB Sources</div>
                    <div class="value" style="font-size:0.82rem">{res.get("kb_sources", "None")}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Escalation reason callout
            if is_esc and res.get("escalation_reason", "None") != "None":
                st.error(f"**Escalation Reason:** {res['escalation_reason']}", icon="🚨")

            # Agent thought-process timeline
            updates: List[Tuple[str, str]] = st.session_state.get("latest_updates", [])
            if updates:
                with st.expander("🕵️  Inspect Agent Step-by-Step Outputs", expanded=False):
                    for src, cnt in updates:
                        render_agent_step(src, cnt)
        else:
            # Empty state
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🤖</div>
                <div style="font-size:1rem; font-weight:600; color:#64748b;">No active workflow</div>
                <div style="font-size:0.85rem; margin-top:0.3rem;">
                    Fill out a complaint on the left and click <b>Process with AI Agents</b> to begin.
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab_analytics:
    df_all = load_logs()
    total = len(df_all)

    if total == 0:
        st.info("No tickets logged yet. Process a complaint in the **Support Console** tab to see analytics here.", icon="📊")
    else:
        resolved = len(df_all[df_all["status"] == "Resolved"])
        escalated = len(df_all[df_all["status"] == "Escalated"])
        high_urg = len(df_all[df_all["urgency"] == "High"]) if "urgency" in df_all.columns else 0
        res_rate = f"{resolved / total * 100:.0f}%" if total else "—"

        # ── KPI row ──
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(render_kpi(str(total), "Total Tickets", "blue"), unsafe_allow_html=True)
        with k2:
            st.markdown(render_kpi(f"{resolved}  ({res_rate})", "Resolved", "green"), unsafe_allow_html=True)
        with k3:
            st.markdown(render_kpi(str(escalated), "Escalated", "red"), unsafe_allow_html=True)
        with k4:
            st.markdown(render_kpi(str(high_urg), "High Urgency", "amber"), unsafe_allow_html=True)

        st.markdown("")  # spacer

        # ── Charts ──
        ch1, ch2 = st.columns(2)
        with ch1:
            with st.container(border=True):
                st.markdown("##### Tickets by Category")
                if "category" in df_all.columns:
                    cat_counts = df_all["category"].value_counts()
                    st.bar_chart(cat_counts, color="#6366f1")
        with ch2:
            with st.container(border=True):
                st.markdown("##### Resolution Outcome")
                status_counts = df_all["status"].value_counts()
                st.bar_chart(status_counts, color="#6366f1")

        # ── Extra row: sentiment & urgency ──
        ch3, ch4 = st.columns(2)
        with ch3:
            with st.container(border=True):
                st.markdown("##### Customer Sentiment")
                if "sentiment" in df_all.columns:
                    sent_counts = df_all["sentiment"].value_counts()
                    st.bar_chart(sent_counts, color="#f59e0b")
        with ch4:
            with st.container(border=True):
                st.markdown("##### Urgency Distribution")
                if "urgency" in df_all.columns:
                    urg_counts = df_all["urgency"].value_counts()
                    st.bar_chart(urg_counts, color="#ef4444")


# ═══════════════════════════════════════════════════════════════════
#  TAB 3 — TICKET HISTORY
# ═══════════════════════════════════════════════════════════════════
with tab_logs:
    df_logs = load_logs()

    st.markdown('<div class="section-title">📋 Ticket Log Database</div>', unsafe_allow_html=True)

    if df_logs.empty:
        st.info("No tickets recorded yet. Submit a complaint to populate the log.", icon="📋")
    else:
        # ── Filters ──
        with st.container(border=True):
            f1, f2, f3, _spacer = st.columns([1, 1, 1, 3])
            with f1:
                status_filter = st.selectbox("Status", ["All", "Resolved", "Escalated"], key="log_status_filter")
            with f2:
                cat_options = ["All"] + sorted(df_logs["category"].dropna().unique().tolist()) if "category" in df_logs.columns else ["All"]
                cat_filter = st.selectbox("Category", cat_options, key="log_cat_filter")
            with f3:
                urg_options = ["All", "Low", "Medium", "High"]
                urg_filter = st.selectbox("Urgency", urg_options, key="log_urg_filter")

        filtered = df_logs.copy()
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]
        if cat_filter != "All" and "category" in filtered.columns:
            filtered = filtered[filtered["category"] == cat_filter]
        if urg_filter != "All" and "urgency" in filtered.columns:
            filtered = filtered[filtered["urgency"] == urg_filter]

        # ── Display columns (hide internal-only duplicates) ──
        display_cols = [c for c in filtered.columns if c not in ("category", "urgency", "sentiment")]

        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            column_config={
                "ticket_id": st.column_config.TextColumn("Ticket ID", width="small"),
                "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "customer_name / order_id": st.column_config.TextColumn("Customer · Order", width="medium"),
                "complaint": st.column_config.TextColumn("Complaint", width="large"),
                "category · urgency · sentiment": st.column_config.TextColumn("Classification", width="medium"),
                "kb_sources": st.column_config.TextColumn("KB Sources", width="medium"),
                "response": st.column_config.TextColumn("Response", width="large"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "escalation_reason": st.column_config.TextColumn("Escalation Reason", width="medium"),
            },
        )

        st.markdown("")
        dl1, dl2 = st.columns([1, 5])
        with dl1:
            csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥  Download CSV",
                data=csv_bytes,
                file_name=f"tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
