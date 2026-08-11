"""
Query generation node for the Job Hunting Agent.

Uses the Gemini LLM to generate diverse job search queries
based on the parsed resume text and user preferences.
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.dependencies import llm
from core.state import AgentState
from utils.logging_config import get_logger

logger = get_logger(__name__)


class JobSearchQueries(BaseModel):
    queries: list[str] = Field(
        description="A list of 10-15 highly specific web search queries. Each query MUST include a job title AND job-specific keywords (like 'responsibilities', 'qualifications', 'requirements', or 'apply') to ensure search engines return actual job postings rather than general articles."
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def generate_search_queries(state: AgentState) -> dict[str, Any]:
    """Generates job search queries using Gemini based on resume text and user preferences.

    Args:
        state: Current agent state containing ``resume_text`` and optional ``preferences``.

    Returns:
        Dict with ``search_queries`` key containing a list of query strings.
    """
    logger.info("NODE: GENERATING SEARCH QUERIES")
    resume_text = state.resume_text
    prefs = state.preferences

    pref_details = ""
    if prefs.target_roles:
        pref_details += f"\nTarget Roles: {', '.join(prefs.target_roles)}"
    if prefs.preferred_locations:
        pref_details += f"\nPreferred Locations: {', '.join(prefs.preferred_locations)}"
    if prefs.remote_preference != "Any":
        pref_details += f"\nWork Mode Preference: {prefs.remote_preference}"

    prompt = ChatPromptTemplate.from_template(
        """Based on the following resume text and candidate preferences, generate 10-15 highly specific job search queries for web search engines.

Resume:
{resume}

Candidate Preferences:
{preferences}"""
    )

    try:
        structured_llm = llm.with_structured_output(JobSearchQueries)
        chain = prompt | structured_llm
        result = chain.invoke({"resume": resume_text, "preferences": pref_details or "Standard candidate preferences."})
        queries_list = result.queries
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM query generation encountered an error: %s. Using fallback query generator.", e)
        roles = prefs.target_roles or ["Machine Learning Engineer", "AI Developer"]
        locations = prefs.preferred_locations or ["Remote"]
        loc_str = locations[0] if locations else "Remote"

        queries_list = []
        for role in roles:
            queries_list.extend([
                f"{role} {loc_str} job openings requirements",
                f"Senior {role} hiring qualifications apply",
                f"{role} software engineer career opportunity"
            ])

    logger.info("Done generating search queries ✅ — %d queries", len(queries_list))
    return {"search_queries": queries_list}
