"""Streamlit Application for AI Customer Support Automation Desk.

Built with AutoGen v0.4+, ChromaDB vector RAG, and Streamlit.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from core.config import get_available_provider, get_model_client, LOG_FILE, KB_DIR
from core.team import process_complaint_stream, process_complaint
from tools.ticket_logger import CSV_COLUMNS

# Page configuration
st.set_page_config(
    page_title="AI Customer Support Desk",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .badge-resolved {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-escalated {
        display: inline-block;
        background-color: #c62828;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .agent-box {
        background-color: #f0f4f8;
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)


# Preset test cases from assignment PDF + custom test cases
TEST_CASES = {
    "Select a pre-filled test case...": None,
    "Test 1: Where is my order? (Standard Shipping -> Resolved)": {
        "name": "Priya Sharma",
        "order_id": "ORD-4512",
        "complaint": "Where is my order? It was supposed to arrive yesterday.",
    },
    "Test 2: Shoe return and refund request (Refund Policy -> Resolved)": {
        "name": "Alex Johnson",
        "order_id": "ORD-8821",
        "complaint": "I want to return the shoes I bought last week. How do I get a refund?",
    },
    "Test 3: Forgot password & can't log in (Account Help -> Resolved)": {
        "name": "Carlos Gomez",
        "order_id": "ORD-N/A",
        "complaint": "I forgot my password and can't log in.",
    },
    "Test 4: Broken blender received (Product FAQ / Replacement -> Resolved)": {
        "name": "Sarah Miller",
        "order_id": "ORD-3309",
        "complaint": "The blender I received is broken. Can I get a replacement?",
    },
    "Test 5: 3x card charge, angry, legal threat (Billing / Legal -> Escalated)": {
        "name": "Robert Vance",
        "order_id": "ORD-9901",
        "complaint": "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    },
    "Test 6: Custom corporate gift cards (Not in KB -> Escalated)": {
        "name": "Evelyn Wright",
        "order_id": "ORD-CORP",
        "complaint": "Do you sell gift cards for a company of 500 employees with custom branding?",
    },
    "Test 7 (Custom): Address modification within 1 hour (Shipping -> Resolved)": {
        "name": "Daniel Craig",
        "order_id": "ORD-7712",
        "complaint": "I just placed an order 20 minutes ago. How do I change my shipping address?",
    },
    "Test 8 (Custom): Suspected account breach & fraudulent orders (Security -> Escalated)": {
        "name": "Maya Patel",
        "order_id": "ORD-UNKNOWN",
        "complaint": "I believe my account has been hacked and someone placed unauthorized orders with my saved card. Block my account immediately!",
    },
}


def load_logs() -> pd.DataFrame:
    """Load tickets from CSV file."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        df = pd.read_csv(LOG_FILE)
        return df
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


# ---------------- SIDEBAR CONFIGURATION ----------------
st.sidebar.title("⚙️ Support Desk Config")

env_info = get_available_provider()
detected_provider = env_info.get("provider") or "openai"

provider_choice = st.sidebar.selectbox(
    "LLM Provider",
    options=["OpenAI", "Google Gemini", "Groq", "Custom / Ollama"],
    index=0 if detected_provider == "openai" else (1 if detected_provider == "gemini" else 2),
)

default_key = env_info.get("api_key") or ""
api_key_input = st.sidebar.text_input(
    "API Key",
    value=default_key,
    type="password",
    help="Loaded automatically from .env if present. You can also paste your key directly here.",
)

# Model selection based on provider
if provider_choice == "Google Gemini":
    default_model = "gemini-2.5-flash"
    custom_model = st.sidebar.text_input("Model Name", value=default_model)
    selected_provider_name = "gemini"
    selected_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
elif provider_choice == "Groq":
    default_model = "llama-3.3-70b-versatile"
    custom_model = st.sidebar.text_input("Model Name", value=default_model)
    selected_provider_name = "groq"
    selected_base_url = "https://api.groq.com/openai/v1"
elif provider_choice == "Custom / Ollama":
    custom_model = st.sidebar.text_input("Model Name", value="llama3.1")
    selected_provider_name = "custom"
    selected_base_url = st.sidebar.text_input("Base URL", value="http://localhost:11434/v1")
else:
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    custom_model = st.sidebar.text_input("Model Name", value=default_model)
    selected_provider_name = "openai"
    selected_base_url = None

if api_key_input:
    st.sidebar.success(f"🔑 API Key connected ({provider_choice})")
else:
    st.sidebar.warning("⚠️ No API key set. Provide one above or in .env")

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Company Knowledge Base")
with st.sidebar.expander("Inspect KB Documents"):
    kb_files = list(KB_DIR.glob("*.md"))
    for kf in kb_files:
        st.write(f"📄 **{kf.name}**")
        with st.expander(f"View content: {kf.name}", expanded=False):
            st.code(kf.read_text(encoding="utf-8")[:400] + "...", language="markdown")

if st.sidebar.button("🗑️ Clear Ticket Logs"):
    if LOG_FILE.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(LOG_FILE, index=False)
        st.sidebar.success("Tickets log reset successfully!")
        st.rerun()


# ---------------- MAIN APPLICATION ----------------
st.title("🎧 AI Customer Support Automation Desk")
st.caption("Multi-Agent Support Orchestration powered by AutoGen v0.4+ and ChromaDB Vector RAG")

tab_console, tab_logs = st.tabs(["💬 Support Console", "📊 Analytics & Ticket Logs"])

# ---------------- TAB 1: SUPPORT CONSOLE ----------------
with tab_console:
    col_input, col_status = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📝 New Customer Complaint")

        # Test case selector
        selected_case = st.selectbox(
            "Quick-Fill Assignment Test Cases",
            options=list(TEST_CASES.keys()),
        )

        preset_data = TEST_CASES.get(selected_case)
        default_customer = preset_data["name"] if preset_data else "Priya Sharma"
        default_order = preset_data["order_id"] if preset_data else "ORD-4512"
        default_complaint = preset_data["complaint"] if preset_data else ""

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            customer_name = st.text_input("Customer Name", value=default_customer)
        with col_c2:
            order_id = st.text_input("Order ID", value=default_order)

        complaint_text = st.text_area(
            "Customer Complaint / Question",
            value=default_complaint,
            height=140,
            placeholder="Describe the issue, late shipment, return request, or query...",
        )

        submit_btn = st.button("🚀 Process Complaint with Agents", type="primary", use_container_width=True)

    with col_status:
        st.subheader("🤖 Live Multi-Agent Workflow")

        if submit_btn:
            if not complaint_text.strip():
                st.error("Please enter a customer complaint before submitting.")
            elif not api_key_input.strip():
                st.error("Please configure an API Key in the sidebar or in your .env file to run the AI agents.")
            else:
                progress_container = st.container()
                with progress_container:
                    status_widget = st.status("Initializing AutoGen Multi-Agent Team...", expanded=True)

                    try:
                        client = get_model_client(
                            api_key=api_key_input,
                            model=custom_model,
                            provider=selected_provider_name,
                            base_url=selected_base_url,
                        )

                        workflow_state = {"final_data": None, "updates": []}

                        async def run_workflow():
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
                                    if src == "Query_Agent":
                                        status_widget.write("🔍 **Query Agent**: Extracted Category, Urgency, and Sentiment.")
                                    elif src == "Knowledge_Agent":
                                        status_widget.write("📚 **Knowledge Agent**: Searched ChromaDB knowledge base.")
                                    elif src == "Resolution_Agent":
                                        status_widget.write("✍️ **Resolution Agent**: Drafted grounded customer resolution.")
                                    elif src == "Log_Agent":
                                        status_widget.write("📋 **Log Agent**: Evaluated escalation rules and persisted to CSV.")
                                elif event["type"] == "final_result":
                                    workflow_state["final_data"] = event["result"]

                        # Execute async loop
                        asyncio.run(run_workflow())
                        status_widget.update(label="✅ Workflow Execution Complete!", state="complete", expanded=False)

                        st.session_state["latest_result"] = workflow_state["final_data"]
                        st.session_state["latest_updates"] = workflow_state["updates"]

                    except Exception as e:
                        status_widget.update(label="❌ Execution Failed", state="error", expanded=True)
                        st.error(f"Error during agent execution: {str(e)}")

        # Display latest result if stored in session state
        if "latest_result" in st.session_state and st.session_state["latest_result"]:
            res = st.session_state["latest_result"]

            # Visual status badge
            is_escalated = res["status"].lower() == "escalated"
            if is_escalated:
                badge_html = f'<div class="badge-escalated">🚨 ESCALATED (Handed to Human Support)</div>'
            else:
                badge_html = f'<div class="badge-resolved">✅ RESOLVED (Handled by AI Support)</div>'

            st.markdown(badge_html, unsafe_allow_html=True)

            st.markdown("### 💬 Proposed Customer Response")
            if is_escalated:
                st.warning(res["response"])
            else:
                st.success(res["response"])

            # Metadata details
            with st.expander("📌 Ticket Breakdown & Escalation Details", expanded=True):
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Ticket ID", res.get("ticket_id", "N/A"))
                mcol2.metric("Category", res.get("category", "General"))
                mcol3.metric("Urgency", res.get("urgency", "Low"))
                mcol4.metric("Sentiment", res.get("sentiment", "Neutral"))

                st.write(f"**Knowledge Sources Consulted**: `{res.get('kb_sources', 'None')}`")
                st.write(f"**Resolution Confidence**: `{res.get('confidence', 'High')}`")
                if is_escalated:
                    st.error(f"**Escalation Reason**: {res.get('escalation_reason', 'None')}")

            # Expandable timeline of individual agent messages
            if "latest_updates" in st.session_state and st.session_state["latest_updates"]:
                with st.expander("🕵️ Inspect Step-by-Step Agent Outputs", expanded=False):
                    for src, cnt in st.session_state["latest_updates"]:
                        st.markdown(f"**🤖 {src}:**")
                        st.code(cnt, language="markdown")


# ---------------- TAB 2: ANALYTICS & TICKET LOGS ----------------
with tab_logs:
    st.subheader("📊 Support Performance & Ticket Logs")

    df_logs = load_logs()

    # Metric summary cards
    total_tickets = len(df_logs)
    if total_tickets > 0:
        resolved_count = len(df_logs[df_logs["status"] == "Resolved"])
        escalated_count = len(df_logs[df_logs["status"] == "Escalated"])
        high_urgency_count = len(df_logs[df_logs["urgency"] == "High"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets", total_tickets)
        m2.metric("Resolved", f"{resolved_count} ({resolved_count/total_tickets*100:.0f}%)")
        m3.metric("Escalated", f"{escalated_count} ({escalated_count/total_tickets*100:.0f}%)")
        m4.metric("High Urgency", high_urgency_count)

        # Charts row
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Tickets by Category")
            if "category" in df_logs.columns and not df_logs["category"].empty:
                cat_counts = df_logs["category"].value_counts()
                st.bar_chart(cat_counts)
        with chart_col2:
            st.markdown("##### Resolved vs Escalated")
            status_counts = df_logs["status"].value_counts()
            st.bar_chart(status_counts)
    else:
        st.info("No tickets logged yet. Submit a complaint in the Support Console to populate logs!")

    st.markdown("---")
    st.subheader("📋 Ticket Log Database (`tickets.csv`)")

    # Filter options
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        status_filter = st.selectbox("Filter by Status", options=["All", "Resolved", "Escalated"])

    filtered_df = df_logs
    if status_filter != "All":
        filtered_df = df_logs[df_logs["status"] == status_filter]

    # Display dataframe
    st.dataframe(filtered_df, use_container_width=True)

    # Download button
    if not df_logs.empty:
        csv_data = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Ticket Log (CSV)",
            data=csv_data,
            file_name="tickets.csv",
            mime="text/csv",
        )
