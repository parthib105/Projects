
"""
Application tailoring node for the Job Hunting Agent.

Generates custom tailored resume bullet points and a personalized cover letter
for the top-ranked job match evaluated by the agent.
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
from core.state import AgentState, ApplicationMaterials, JobMatchAnalysis
from utils.logging_config import get_logger

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def tailor_application(state: AgentState) -> dict[str, Any]:
    """Generates tailored resume bullets and a cover letter for the #1 top-ranked job match.

    Args:
        state: Current agent state containing ``resume_text`` and ``ranked_matches``.

    Returns:
        Dict with ``application_materials`` (ApplicationMaterials model).
    """
    logger.info("NODE: TAILORING APPLICATION MATERIALS FOR TOP MATCH")

    ranked_matches = state.ranked_matches
    if not ranked_matches:
        logger.warning("No ranked job matches available for tailoring.")
        return {"application_materials": None}

    # Select the #1 top-ranked job match
    top_match: JobMatchAnalysis = ranked_matches[0]
    resume_text = state.resume_text

    logger.info(
        "Tailoring application for top match: '%s' at '%s' (Match Score: %.0f%%)",
        top_match.job_title,
        top_match.company,
        top_match.overall_score
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an elite executive resume writer and career coach.
Your task is to create highly customized application materials for the candidate's target job.

<candidate_resume>
{resume}
</candidate_resume>

<target_job>
Title: {job_title}
Company: {company}
Match Rationale: {match_rationale}
Matching Skills: {matching_skills}
Missing Skills: {missing_skills}
</target_job>

<instructions>
1. Generate 3 to 5 powerful tailored resume bullet points highlighting relevant candidate accomplishments formatted in Google XYZ style ("Accomplished [X] as measured by [Y], by doing [Z]").
2. Write a compelling, 3-paragraph professional cover letter addressed to the hiring team at {company} expressing strong enthusiasm and demonstrating exact skill alignment.
</instructions>"""
    )

    structured_llm = reasoning_llm.with_structured_output(ApplicationMaterials)
    chain = prompt | structured_llm

    materials: ApplicationMaterials = chain.invoke({
        "resume": resume_text,
        "job_title": top_match.job_title,
        "company": top_match.company,
        "match_rationale": top_match.match_rationale,
        "matching_skills": ", ".join(top_match.matching_skills) if top_match.matching_skills else "N/A",
        "missing_skills": ", ".join(top_match.missing_skills) if top_match.missing_skills else "N/A",
    })

    # Persist application materials to SQLite database
    from database.manager import db
    db.save_application_materials(materials)

    logger.info("Done generating application materials ✅ (Generated %d tailored bullets)", len(materials.tailored_bullets))
    return {"application_materials": materials}
