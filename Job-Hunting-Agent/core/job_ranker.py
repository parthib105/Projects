"""
Job ranking node for the Job Hunting Agent.

Uses Gemini reasoning LLM to filter and quantitatively rank job listings
against the candidate's resume and preferences.
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.dependencies import reasoning_llm
from core.state import AgentState, JobListing, RankedJobList
from utils.logging_config import get_logger

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def filter_and_rank_jobs(state: AgentState) -> dict[str, Any]:
    """Uses Gemini reasoning model to filter and rank job listings against the resume.

    Args:
        state: Current agent state containing ``resume_text`` and ``job_listings``.

    Returns:
        Dict with ``ranked_matches`` (List[JobMatchAnalysis]) and ``ranked_jobs`` (str summary).
    """
    logger.info("NODE: FILTERING AND RANKING JOBS")
    resume_text = state.resume_text
    listings = state.job_listings

    formatted_listings = "<job_listings>\n"
    for i, job in enumerate(listings):
        if isinstance(job, JobListing):
            j_id = job.id
            title = job.title
            company = job.company
            url = job.url
            content = job.description
        elif isinstance(job, dict):
            j_id = job.get('id', f"job_{i+1}")
            title = job.get('title', 'N/A')
            company = job.get('company', 'Unknown')
            url = job.get('url', 'N/A')
            content = job.get('content', job.get('description', 'No description available.'))
        else:
            j_id = f"job_{i+1}"
            title = 'N/A'
            company = 'Unknown'
            url = 'N/A'
            content = str(job)

        formatted_listings += (
            f'  <job_listing id="{j_id}">\n'
            f'    <title>{title}</title>\n'
            f'    <company>{company}</company>\n'
            f'    <url>{url}</url>\n'
            f'    <description>{content}</description>\n'
            f'  </job_listing>\n'
        )
    formatted_listings += "</job_listings>"

    prompt = ChatPromptTemplate.from_template(
        """You are an expert executive career advisor.
Analyze the candidate's resume wrapped inside <candidate_resume> tags against the retrieved job listings wrapped inside <job_listings> tags.

<candidate_resume>
{resume}
</candidate_resume>

{listings}

<evaluation_instructions>
1. Filter out irrelevant job listings.
2. Evaluate and rank the top 5 most suitable jobs.
3. For each job, calculate an overall match score (0-100), skills fit score (0-100), and experience fit score (0-100).
4. Provide lists of matching skills, missing skills, pros, cons, and a 2-3 sentence match rationale.
5. Provide a 2-paragraph overall executive summary of the evaluation results.
</evaluation_instructions>"""
    )

    structured_llm = reasoning_llm.with_structured_output(RankedJobList)
    chain = prompt | structured_llm

    result: RankedJobList = chain.invoke({"resume": resume_text, "listings": formatted_listings})
    ranked_matches = result.ranked_jobs

    # Generate backwards-compatible text summary for CLI output
    cli_output = f"{result.summary}\n\n"
    for idx, match in enumerate(ranked_matches, 1):
        cli_output += (
            f"Rank {idx}: {match.job_title} at {match.company} (Match Score: {match.overall_score:.0f}%)\n"
            f"URL: {match.url}\n"
            f"Match Rationale: {match.match_rationale}\n"
            f"Matching Skills: {', '.join(match.matching_skills) if match.matching_skills else 'N/A'}\n\n"
        )

    logger.info("Done ranking jobs ✅ (%d top matches evaluated)", len(ranked_matches))
    return {
        "ranked_matches": ranked_matches,
        "ranked_jobs": cli_output.strip()
    }
