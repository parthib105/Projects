"""
Shared dependencies for the Job Hunting Agent.

This module initializes the configuration, logging, LLM, and search
tool exactly once. Other core modules import from here instead of
creating their own instances.

Changing the search provider (e.g., replacing Tavily in Phase 2)
only requires editing this single file.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from config.settings import Config
from utils.logging_config import setup_logging

# ── Initialize configuration & logging (runs once on first import) ──
config = Config()
setup_logging(level=config.log_level)

# ── Fast LLM (for query generation & filtering) ──
llm = ChatGoogleGenerativeAI(
    model=config.llm_model,
    temperature=config.llm_temperature,
    google_api_key=config.google_api_key,
)

# ── Reasoning LLM (for deep evaluation & ranking) ──
reasoning_llm = ChatGoogleGenerativeAI(
    model=config.reasoning_llm_model,
    temperature=config.llm_temperature,
    google_api_key=config.google_api_key,
)

# ── Search tool (initialized if API key is present or on demand) ──
tavily_tool = None
try:
    if config._tavily_api_key and config._tavily_api_key != "your_tavily_api_key_here":
        tavily_tool = TavilySearch(
            max_results=config.max_search_results,
            api_key=config.tavily_api_key,
        )
except Exception:  # noqa: BLE001
    tavily_tool = None
