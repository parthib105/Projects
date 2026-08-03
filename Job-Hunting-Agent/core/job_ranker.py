"""
Job ranking node for the Job Hunting Agent.

Uses the Gemini LLM to filter and rank job listings against
the user's resume.
"""

from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from core.state import AgentState
from core.dependencies import llm
from utils.logging_config import get_logger

logger = get_logger(__name__)


def filter_and_rank_jobs(state: AgentState) -> Dict[str, str]:
    """Uses Gemini to filter and rank the found job listings against the resume.

    Args:
        state: Current agent state containing ``resume_text`` and ``job_listings``.

    Returns:
        Dict with ``ranked_jobs`` key containing the LLM's ranked output.
    """
    logger.info("NODE: FILTERING AND RANKING JOBS")
    resume_text = state.resume_text
    listings = state.job_listings

    formatted_listings = ""
    for i, job in enumerate(listings):
        if isinstance(job, dict):
            content = job.get('content', 'No description available.')
            url = job.get('url', 'No URL available.')
            title = job.get('title', 'No title available.')
        else:
            content = str(job)
            url = 'No URL available.'
            title = 'No title available.'
        formatted_listings += f"--- Job {i+1} ---\nTitle: {title}\nURL: {url}\nDescription: {content}\n\n"

    prompt = ChatPromptTemplate.from_template(
        """You are an expert career assistant. Based on the provided resume, analyze the following job listings.
        Filter out irrelevant listings and rank the top 5 most suitable jobs.
        For each ranked job, provide the job title, its URL, and a brief (2-3 sentences) explanation of why it's a good match.

        Resume:
        {resume}

        Job Listings:
        {listings}
        """
    )
    chain = prompt | llm | StrOutputParser()
    ranked_list = chain.invoke({"resume": resume_text, "listings": formatted_listings})
    logger.info("Done ranking jobs ✅")
    return {"ranked_jobs": ranked_list}
