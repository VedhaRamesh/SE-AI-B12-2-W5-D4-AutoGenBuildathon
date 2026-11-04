"""Query Agent: Understands the customer request, extracts category, urgency, sentiment, and summary."""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

QUERY_AGENT_SYSTEM_MESSAGE = """You are a customer query analysis expert for an online retail support desk.
Analyze the incoming customer complaint and identify the underlying problem, how urgent it is, and the customer's emotional state.

You MUST produce your output formatted strictly as follows:
Category: <Shipping | Refund | Account | Product | Billing | Other>
Urgency: <Low | Medium | High>
Sentiment: <Calm | Neutral | Frustrated | Angry | Very Angry>
Summary: <A concise, clear 1-line summary of the customer's problem>

Rules:
1. Category must be one of: Shipping, Refund, Account, Product, Billing, Other.
2. Urgency: Set to "High" if the customer mentions severe delivery delays, unauthorized charges, repeated billing, legal action, or extreme distress. Set to "Medium" for normal inquiries or requests. Set to "Low" for general questions.
3. Sentiment: Detect if the customer is Calm, Neutral, Frustrated, Angry, or Very Angry (e.g. threats of lawyers, bank chargebacks, exclamation marks).
4. Summary: Exactly one sentence capturing the essential customer issue without fluff.
"""


def create_query_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Instantiate and return the Customer Query Agent."""
    return AssistantAgent(
        name="Query_Agent",
        model_client=model_client,
        description="Understands the customer complaint, extracts category, urgency, sentiment, and summary.",
        system_message=QUERY_AGENT_SYSTEM_MESSAGE,
    )
