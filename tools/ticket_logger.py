"""Ticket logging tool that records customer support tickets to a CSV file."""

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from core.config import LOG_FILE, LOGS_DIR

logger = logging.getLogger(__name__)
_log_lock = threading.Lock()

CSV_COLUMNS = [
    "ticket_id",
    "timestamp",
    "customer_name / order_id",
    "complaint",
    "category · urgency · sentiment",
    "kb_sources",
    "response",
    "status",
    "escalation_reason",
    "category",
    "urgency",
    "sentiment",
]


def _ensure_csv_exists() -> None:
    """Ensure logs directory and tickets.csv exist with required header row."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def get_next_ticket_id() -> str:
    """Generate the next sequential ticket identifier (e.g. TKT-0001)."""
    with _log_lock:
        _ensure_csv_exists()
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty or "ticket_id" not in df.columns:
                return "TKT-0001"
            valid_ids = df["ticket_id"].dropna().astype(str)
            numeric_ids = []
            for tid in valid_ids:
                if tid.startswith("TKT-"):
                    try:
                        numeric_ids.append(int(tid.split("-")[1]))
                    except (ValueError, IndexError):
                        pass
            next_num = max(numeric_ids, default=0) + 1
            return f"TKT-{next_num:04d}"
        except Exception:
            return "TKT-0001"


def log_ticket(
    ticket_id: str,
    customer_info: str,
    complaint: str,
    category: str,
    urgency: str,
    sentiment: str,
    kb_sources: str,
    response: str,
    status: str,
    escalation_reason: str = "None",
    timestamp: Optional[str] = None,
) -> str:
    """Logs a completed customer support ticket to the CSV database.

    Args:
        ticket_id: Ticket identifier (e.g. TKT-0001)
        customer_info: Customer name and order identifier (e.g. 'Priya · ORD-4512')
        complaint: The customer's original complaint text
        category: Classified category (e.g. Shipping, Refund, Account, Product, Billing)
        urgency: Urgency level (Low, Medium, High)
        sentiment: Customer sentiment (Calm, Neutral, Frustrated, Angry, Very Angry)
        kb_sources: Source files used from knowledge base (e.g. 'shipping_policy.md' or 'None')
        response: The proposed customer reply
        status: Final ticket status ('Resolved' or 'Escalated')
        escalation_reason: Reason for escalation, or 'None' if resolved
        timestamp: Optional timestamp string (defaults to current time YYYY-MM-DD HH:MM:SS)

    Returns:
        Confirmation string indicating successful logging.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tid_input = (ticket_id or "").strip()
    # Ensure placeholder values like 'TKT-AUTO' or empty values are replaced with real sequential IDs
    if not tid_input or tid_input in ("TKT-AUTO", "TKT-XXXX", "None", "AUTO", "TKT-0000"):
        tid = get_next_ticket_id()
    else:
        tid = tid_input
    stat = "Escalated" if "escalat" in (status or "").lower() else "Resolved"
    esc_reason = (escalation_reason or "None").strip()

    category_urgency_sentiment = f"{category.strip()} · {urgency.strip()} · {sentiment.strip()}"

    row = [
        tid,
        ts,
        customer_info.strip(),
        complaint.strip(),
        category_urgency_sentiment,
        kb_sources.strip() if kb_sources else "None",
        response.strip(),
        stat,
        esc_reason,
        category.strip(),
        urgency.strip(),
        sentiment.strip(),
    ]

    with _log_lock:
        _ensure_csv_exists()
        try:
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            logger.info("Ticket %s successfully logged to %s", tid, LOG_FILE)
            return f"Ticket {tid} logged successfully with status: {stat}."
        except Exception as e:
            logger.error("Failed to append ticket to CSV: %s", e)
            return f"Error logging ticket: {str(e)}"
