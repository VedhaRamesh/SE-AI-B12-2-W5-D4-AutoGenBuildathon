"""Knowledge Agent: Searches internal company documents using the search_kb tool."""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from tools.kb_search import search_kb

KNOWLEDGE_AGENT_SYSTEM_MESSAGE = """You are a dedicated research assistant for the support team.
Your sole job is to search company documents (shipping, refund, account, product policies) using the `search_kb` tool.

Instructions:
1. Read the summary, category, and context provided by the Query Agent.
2. Call the tool `search_kb(query=...)` with a targeted query reflecting the customer's core question.
3. If matching passages are retrieved, output the exact passages and explicitly include their source file name(s) (e.g., "[Source: shipping_policy.md]").
4. If `search_kb` returns "No relevant documents found in knowledge base.", output exactly:
"No relevant documents found in knowledge base."
5. Never invent or assume facts or policies that are not returned by the tool.
"""


def create_knowledge_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Instantiate and return the Knowledge Base Agent."""
    return AssistantAgent(
        name="Knowledge_Agent",
        model_client=model_client,
        tools=[search_kb],
        description="Searches company documents using search_kb and returns matching passages with file names.",
        system_message=KNOWLEDGE_AGENT_SYSTEM_MESSAGE,
    )
