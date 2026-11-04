"""Automated Test Suite for AI Customer Support Automation Desk.

Tests knowledge base retrieval, escalation rule guardrails, ticket logging,
and multi-agent parsing against all 6 assignment test cases plus 2 custom cases.
"""

import sys
import unittest
import pandas as pd
from pathlib import Path

from tools.kb_search import search_kb
from tools.ticket_logger import log_ticket, get_next_ticket_id, CSV_COLUMNS
from core.team import parse_agent_outputs
from core.config import LOG_FILE


class TestCustomerSupportPipeline(unittest.TestCase):
    """Unit and Integration tests for support pipeline."""

    def test_01_kb_shipping_query(self):
        """Test Case 1: Standard shipping query matches shipping_policy.md."""
        query = "Where is my order? It was supposed to arrive yesterday."
        result = search_kb(query)
        self.assertIn("[Source: shipping_policy.md]", result)
        self.assertNotIn("No relevant documents found", result)

    def test_02_kb_refund_query(self):
        """Test Case 2: Shoe return inquiry matches refund_policy.md."""
        query = "I want to return the shoes I bought last week. How do I get a refund?"
        result = search_kb(query)
        self.assertIn("[Source: refund_policy.md]", result)
        self.assertNotIn("No relevant documents found", result)

    def test_03_kb_account_query(self):
        """Test Case 3: Forgot password query matches account_and_login.md."""
        query = "I forgot my password and can't log in."
        result = search_kb(query)
        self.assertIn("[Source: account_and_login.md]", result)
        self.assertNotIn("No relevant documents found", result)

    def test_04_kb_product_replacement_query(self):
        """Test Case 4: Broken blender matches product_faq.md."""
        query = "The blender I received is broken. Can I get a replacement?"
        result = search_kb(query)
        self.assertIn("[Source: product_faq.md]", result)
        self.assertNotIn("No relevant documents found", result)

    def test_05_kb_unmatched_gift_cards(self):
        """Test Case 6: Custom corporate gift cards not in KB returns no documents."""
        query = "Do you sell gift cards for a company of 500 employees with custom branding?"
        result = search_kb(query)
        self.assertIn("No relevant documents found in knowledge base.", result)

    def test_06_ticket_logger_columns(self):
        """Verify ticket_logger creates all required CSV columns from Section 7."""
        tid = get_next_ticket_id()
        msg = log_ticket(
            ticket_id=tid,
            customer_info="Test User · ORD-9999",
            complaint="Unit test complaint",
            category="Shipping",
            urgency="Low",
            sentiment="Neutral",
            kb_sources="shipping_policy.md",
            response="Your order is en route.",
            status="Resolved",
            escalation_reason="None",
        )
        self.assertIn("logged successfully", msg)
        df = pd.read_csv(LOG_FILE)
        for col in ["ticket_id", "timestamp", "customer_name / order_id", "complaint",
                    "category · urgency · sentiment", "kb_sources", "response", "status", "escalation_reason"]:
            self.assertIn(col, df.columns)

    def test_07_escalation_rules_case5_legal_billing(self):
        """Test Case 5: 3x charge, angry, legal threat must escalate."""
        complaint = "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer."
        mock_messages = [
            type("Msg", (), {"source": "Query_Agent", "content": "Category: Billing\nUrgency: High\nSentiment: Angry\nSummary: Customer was charged 3 times."})(),
            type("Msg", (), {"source": "Knowledge_Agent", "content": "No relevant documents found in knowledge base."})(),
            type("Msg", (), {"source": "Resolution_Agent", "content": "Response: We apologize for the issue...\nConfidence: Low"})(),
            type("Msg", (), {"source": "Log_Agent", "content": "Status: Escalated\nEscalation Reason: High urgency + angry + payment dispute\nLOGGED"})(),
        ]
        parsed = parse_agent_outputs(mock_messages, "Robert · ORD-9901", complaint)
        self.assertEqual(parsed["status"], "Escalated")
        self.assertIn("payment dispute", parsed["escalation_reason"].lower())

    def test_08_escalation_rules_case6_no_kb(self):
        """Test Case 6: No KB match must escalate."""
        complaint = "Do you sell gift cards for a company of 500 employees with custom branding?"
        mock_messages = [
            type("Msg", (), {"source": "Query_Agent", "content": "Category: Other\nUrgency: Low\nSentiment: Neutral\nSummary: Inquiring about custom branded gift cards."})(),
            type("Msg", (), {"source": "Knowledge_Agent", "content": "No relevant documents found in knowledge base."})(),
            type("Msg", (), {"source": "Resolution_Agent", "content": "Response: We do not offer custom branded bulk gift cards...\nConfidence: Low"})(),
            type("Msg", (), {"source": "Log_Agent", "content": "Status: Escalated\nEscalation Reason: No matching document in knowledge base\nLOGGED"})(),
        ]
        parsed = parse_agent_outputs(mock_messages, "Evelyn · ORD-CORP", complaint)
        self.assertEqual(parsed["status"], "Escalated")
        self.assertIn("no matching document", parsed["escalation_reason"].lower())

    def test_09_resolved_standard_shipping(self):
        """Test Case 1: Standard shipping inquiry resolves successfully."""
        complaint = "Where is my order? It was supposed to arrive yesterday."
        mock_messages = [
            type("Msg", (), {"source": "Query_Agent", "content": "Category: Shipping\nUrgency: Medium\nSentiment: Neutral\nSummary: Order delayed by 1 day."})(),
            type("Msg", (), {"source": "Knowledge_Agent", "content": "[Source: shipping_policy.md]\n24-48 Hour Buffer: If your scheduled delivery date passed yesterday, please allow a 24 to 48 hour buffer period."})(),
            type("Msg", (), {"source": "Resolution_Agent", "content": "Response: Please allow a 24-48 hour buffer as carriers often update status the next day.\nConfidence: High"})(),
            type("Msg", (), {"source": "Log_Agent", "content": "Status: Resolved\nEscalation Reason: None\nLOGGED"})(),
        ]
        parsed = parse_agent_outputs(mock_messages, "Priya · ORD-4512", complaint)
        self.assertEqual(parsed["status"], "Resolved")
        self.assertEqual(parsed["escalation_reason"], "None")


if __name__ == "__main__":
    unittest.main()
