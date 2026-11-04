"""Configuration management for AI Customer Support Automation."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "knowledge_base"
LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE = LOGS_DIR / "tickets.csv"

# Ensure directories exist
KB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(dotenv_path=ROOT_DIR / ".env")


def get_available_provider() -> dict:
    """Detect available LLM provider and credentials from environment."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if openai_key and not openai_key.startswith("your_"):
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            "base_url": None,
        }
    elif gemini_key and not gemini_key.startswith("your_"):
        return {
            "provider": "gemini",
            "api_key": gemini_key,
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }
    elif groq_key and not groq_key.startswith("your_"):
        return {
            "provider": "groq",
            "api_key": groq_key,
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
            "base_url": "https://api.groq.com/openai/v1",
        }

    return {
        "provider": None,
        "api_key": None,
        "model": None,
        "base_url": None,
    }


def get_model_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Instantiate and return an AutoGen ChatCompletionClient.

    Supports OpenAI, Google Gemini (via OpenAI-compatible endpoint), Groq,
    Azure, or local Ollama instances.
    """
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    env_config = get_available_provider()

    active_key = (api_key or "").strip() or env_config.get("api_key")
    active_provider = (provider or "").strip().lower() or env_config.get("provider") or "openai"
    active_model = (model or "").strip() or env_config.get("model")
    active_base_url = (base_url or "").strip() or env_config.get("base_url")

    # Select default model per provider if not specified
    if not active_model:
        if active_provider == "gemini":
            active_model = "gemini-2.5-flash"
        elif active_provider == "groq":
            active_model = "llama-3.3-70b-versatile"
        else:
            active_model = "gpt-4o-mini"

    # Select base_url per provider
    if not active_base_url:
        if active_provider == "gemini":
            active_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif active_provider == "groq":
            active_base_url = "https://api.groq.com/openai/v1"

    if not active_key:
        raise ValueError(
            "Missing LLM API Key. Please provide OPENAI_API_KEY, GEMINI_API_KEY, "
            "or GROQ_API_KEY in your .env file or input your key directly in the sidebar."
        )

    client_kwargs = {
        "model": active_model,
        "api_key": active_key,
        "temperature": 0.2,
    }
    if active_base_url:
        client_kwargs["base_url"] = active_base_url

    return OpenAIChatCompletionClient(**client_kwargs)
