"""Team orchestration module using AutoGen RoundRobinGroupChat."""

import re
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_core.models import ChatCompletionClient

from agents.query_agent import create_query_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.resolution_agent import create_resolution_agent
from agents.log_agent import create_log_agent
from tools.ticket_logger import get_next_ticket_id, log_ticket

logger = logging.getLogger(__name__)


def create_support_team(
    model_client: ChatCompletionClient,
    max_turns: int = 12,
) -> RoundRobinGroupChat:
    """Creates and configures the AutoGen customer support team.

    Sequence: Query_Agent -> Knowledge_Agent -> Resolution_Agent -> Log_Agent
    Terminates when Log_Agent outputs 'LOGGED' or max messages exceeded.
    """
    query_agent = create_query_agent(model_client)
    knowledge_agent = create_knowledge_agent(model_client)
    resolution_agent = create_resolution_agent(model_client)
    log_agent = create_log_agent(model_client)

    termination = TextMentionTermination("LOGGED") | MaxMessageTermination(max_messages=max_turns)

    team = RoundRobinGroupChat(
        participants=[query_agent, knowledge_agent, resolution_agent, log_agent],
        termination_condition=termination,
    )
    return team


def parse_agent_outputs(
    messages: list,
    customer_info: str,
    complaint: str,
) -> Dict[str, Any]:
    """Parse structured results from agent message history with resilient fallbacks."""
    result: Dict[str, Any] = {
        "customer_info": customer_info,
        "complaint": complaint,
        "category": "General",
        "urgency": "Low",
        "sentiment": "Neutral",
        "summary": complaint[:120],
        "kb_sources": "None",
        "passages": "",
        "response": "",
        "confidence": "High",
        "status": "Resolved",
        "escalation_reason": "None",
        "ticket_id": get_next_ticket_id(),
        "messages": [],
    }

    complaint_lower = complaint.lower()

    for msg in messages:
        # Determine source and content
        source = getattr(msg, "source", None) or getattr(msg, "name", "Agent")
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)

        result["messages"].append({
            "source": source,
            "content": content,
        })

        if source == "Query_Agent":
            cat_match = re.search(r"Category:\s*([A-Za-z]+)", content, re.IGNORECASE)
            if cat_match:
                result["category"] = cat_match.group(1).title()

            urg_match = re.search(r"Urgency:\s*(Low|Medium|High)", content, re.IGNORECASE)
            if urg_match:
                result["urgency"] = urg_match.group(1).title()

            sent_match = re.search(r"Sentiment:\s*(Calm|Neutral|Frustrated|Angry|Very Angry)", content, re.IGNORECASE)
            if sent_match:
                result["sentiment"] = sent_match.group(1).title()

            sum_match = re.search(r"Summary:\s*(.+)", content, re.IGNORECASE)
            if sum_match:
                result["summary"] = sum_match.group(1).strip()

        elif source == "Knowledge_Agent":
            sources = re.findall(r"\[Source:\s*([^\]]+)\]", content)
            if sources:
                result["kb_sources"] = ", ".join(dict.fromkeys(sources))
                result["passages"] = content
            elif "no relevant documents" in content.lower():
                result["kb_sources"] = "None"
                result["passages"] = "No relevant documents found in knowledge base."
            else:
                result["passages"] = content

        elif source == "Resolution_Agent":
            resp_match = re.search(r"Response:\s*(.*?)(?=\nConfidence:|$)", content, re.DOTALL | re.IGNORECASE)
            if resp_match:
                result["response"] = resp_match.group(1).strip()
            else:
                clean_content = re.sub(r"Confidence:\s*(High|Low)", "", content, flags=re.IGNORECASE).strip()
                result["response"] = clean_content

            conf_match = re.search(r"Confidence:\s*(High|Low)", content, re.IGNORECASE)
            if conf_match:
                result["confidence"] = conf_match.group(1).title()

        elif source == "Log_Agent":
            stat_match = re.search(r"Status:\s*(Resolved|Escalated)", content, re.IGNORECASE)
            if stat_match:
                result["status"] = stat_match.group(1).title()

            reason_match = re.search(r"Escalation Reason:\s*(.+)", content, re.IGNORECASE)
            if reason_match:
                result["escalation_reason"] = reason_match.group(1).strip()

    # Rule-based guardrails to strictly enforce Task 5 escalation criteria
    legal_or_fraud_terms = ["lawyer", "bank", "sue", "legal", "court", "police", "fraud", "scam", "charged three times", "three times"]
    has_legal_or_fraud = any(term in complaint_lower for term in legal_or_fraud_terms)
    is_no_kb = result["kb_sources"] == "None" or "no relevant documents" in result.get("passages", "").lower()
    is_high_urgency = result["urgency"].lower() == "high"
    is_angry = result["sentiment"].lower() in ["angry", "very angry"]
    is_low_conf = result["confidence"].lower() == "low"

    escalate_reasons = []
    if is_high_urgency:
        escalate_reasons.append("High urgency")
    if is_angry:
        escalate_reasons.append(f"Customer is {result['sentiment'].lower()}")
    if is_no_kb:
        escalate_reasons.append("No matching document in knowledge base")
    if has_legal_or_fraud:
        escalate_reasons.append("Payment dispute / legal threat / fraud inquiry")
    if is_low_conf:
        escalate_reasons.append("Low resolution confidence")

    if escalate_reasons:
        result["status"] = "Escalated"
        if result["escalation_reason"] in ("None", "", None):
            result["escalation_reason"] = " + ".join(escalate_reasons)
    else:
        result["status"] = "Resolved"
        result["escalation_reason"] = "None"

    # Fallback response if resolution agent failed
    if not result["response"]:
        if result["status"] == "Escalated":
            result["response"] = (
                "Thank you for contacting customer support. We have received your inquiry. "
                "Because your request requires specialized attention, your ticket has been "
                "escalated to a senior support specialist who will review your case and contact you shortly."
            )
        else:
            result["response"] = (
                "Thank you for contacting customer support. Your request has been processed in accordance "
                "with our store policies."
            )

    return result


async def process_complaint_stream(
    complaint: str,
    customer_name: str = "Priya",
    order_id: str = "ORD-4512",
    model_client: Optional[ChatCompletionClient] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streams each agent step live and yields the final result."""
    from core.config import get_model_client

    client = model_client or get_model_client()
    team = create_support_team(client)

    customer_info = f"{customer_name.strip()} · {order_id.strip()}"
    prompt = (
        f"Customer Name: {customer_name.strip()}\n"
        f"Order ID: {order_id.strip()}\n"
        f"Customer Info: {customer_info}\n"
        f"Complaint: {complaint.strip()}"
    )

    collected_messages = []

    async for item in team.run_stream(task=prompt):
        if isinstance(item, TextMessage):
            collected_messages.append(item)
            yield {
                "type": "agent_message",
                "source": item.source,
                "content": item.content,
            }

    parsed = parse_agent_outputs(collected_messages, customer_info, complaint)

    # Ensure logged to CSV if Log_Agent didn't call it directly
    log_ticket(
        ticket_id=parsed["ticket_id"],
        customer_info=parsed["customer_info"],
        complaint=parsed["complaint"],
        category=parsed["category"],
        urgency=parsed["urgency"],
        sentiment=parsed["sentiment"],
        kb_sources=parsed["kb_sources"],
        response=parsed["response"],
        status=parsed["status"],
        escalation_reason=parsed["escalation_reason"],
    )

    yield {
        "type": "final_result",
        "result": parsed,
    }


async def process_complaint(
    complaint: str,
    customer_name: str = "Priya",
    order_id: str = "ORD-4512",
    model_client: Optional[ChatCompletionClient] = None,
) -> Dict[str, Any]:
    """Execute complete workflow and return structured final output."""
    from core.config import get_model_client

    client = model_client or get_model_client()
    team = create_support_team(client)

    customer_info = f"{customer_name.strip()} · {order_id.strip()}"
    prompt = (
        f"Customer Name: {customer_name.strip()}\n"
        f"Order ID: {order_id.strip()}\n"
        f"Customer Info: {customer_info}\n"
        f"Complaint: {complaint.strip()}"
    )

    task_result = await team.run(task=prompt)
    messages = [msg for msg in task_result.messages if isinstance(msg, TextMessage)]
    parsed = parse_agent_outputs(messages, customer_info, complaint)

    # Ensure logged to CSV
    log_ticket(
        ticket_id=parsed["ticket_id"],
        customer_info=parsed["customer_info"],
        complaint=parsed["complaint"],
        category=parsed["category"],
        urgency=parsed["urgency"],
        sentiment=parsed["sentiment"],
        kb_sources=parsed["kb_sources"],
        response=parsed["response"],
        status=parsed["status"],
        escalation_reason=parsed["escalation_reason"],
    )

    return parsed
