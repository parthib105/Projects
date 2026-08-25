"""
Job ranking node for the Job Hunting Agent.

Uses Gemini reasoning LLM to quantitatively rank job listings
against candidate resumes using a Map-Reduce parallel batching architecture.
"""

import asyncio
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
from core.state import AgentState, JobListing, JobMatchAnalysis, RankedJobList
from utils.logging_config import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 5


def extract_domain_indicators(text: str) -> dict[str, int]:
    """Extract domain-specific indicators from text using keyword categories.
    Returns a dictionary of domain scores.
    """
    # Define domain categories with associated keywords
    # This can be extended/customized based on observed resume/job patterns
    domain_keywords = {
        'healthcare': {
            'patient', 'clinical', 'therapy', 'rehabilitation', 'medical', 'healthcare',
            'physiotherapy', 'physician', 'nurse', 'hospital', 'clinic', 'diagnosis',
            'treatment', 'care', 'pharmacy', 'health', 'wellness'
        },
        'technology': {
            'software', 'programming', 'developer', 'engineer', 'algorithm', 'database',
            'web', 'mobile', 'api', 'framework', 'language', 'python', 'java', 'javascript',
            'sql', 'cloud', 'devops', 'agile', 'scrum', 'data', 'system', 'IT'
        },
        'business': {
            'management', 'marketing', 'sales', 'finance', 'accounting', 'hr', 'human resources',
            'strategy', 'operations', 'consulting', 'business', 'administration', 'project'
        },
        'creative': {
            'design', 'art', 'graphic', 'content', 'writing', 'copy', 'creative', 'media',
            'advertising', 'brand', 'visual', 'ux', 'ui', 'multimedia'
        },
        'education': {
            'teaching', 'education', 'training', 'curriculum', 'instruction', 'academic',
            'student', 'faculty', 'research', 'learning', 'course'
        },
        'engineering': {
            'engineering', 'mechanical', 'electrical', 'civil', 'chemical', 'manufacturing',
            'industrial', 'robotics', 'automotive', 'aerospace', 'construction'
        }
        # Add more domains as needed
    }

    text_lower = text.lower()
    domain_scores = {}

    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        domain_scores[domain] = score

    return domain_scores


def calculate_generic_domain_penalty(resume_text: str, job_description: str) -> float:
    """Calculate penalty score (0.0 to 1.0) for domain mismatch between resume and job.
    Works for any domain by comparing domain indicator profiles.
    """
    resume_domains = extract_domain_indicators(resume_text)
    job_domains = extract_domain_indicators(job_description)

    # Find dominant domain in each (domain with highest score)
    resume_dominant = max(resume_domains, key=resume_domains.get) if resume_domains else None
    job_dominant = max(job_domains, key=job_domains.get) if job_domains else None

    resume_max_score = max(resume_domains.values()) if resume_domains else 0
    job_max_score = max(job_domains.values()) if job_domains else 0

    # Require minimum signal strength to consider a domain "detected"
    min_signal_threshold = 2

    resume_is_clear = resume_dominant and resume_max_score >= min_signal_threshold
    job_is_clear = job_dominant and job_max_score >= min_signal_threshold

    # If either doesn't have clear domain signals, apply small penalty (uncertainty)
    if not resume_is_clear or not job_is_clear:
        return 0.1  # 10% penalty for unclear domain signals

    # If domains match, no penalty
    if resume_dominant == job_dominant:
        return 0.0

    # If domains differ, apply penalty based on signal strength
    # Stronger signals = higher penalty for mismatch
    avg_signal = (resume_max_score + job_max_score) / 2
    # Normalize penalty: 0.1 (weak signals) to 0.4 (strong signals)
    penalty = min(0.4, 0.1 + (avg_signal * 0.05))

    return penalty


def format_batch_xml(batch: list[JobListing]) -> str:
    """Formats a batch of JobListing objects into structured XML tags."""
    xml_output = "<job_listings>\n"
    for i, job in enumerate(batch):
        if isinstance(job, JobListing):
            j_id = job.id
            title = job.title
            company = job.company
            url = job.url
            content = job.description
        elif isinstance(job, dict):
            j_id = job.get("id", f"job_{i+1}")
            title = job.get("title", "N/A")
            company = job.get("company", "Unknown")
            url = job.get("url", "N/A")
            content = job.get("content", job.get("description", "No description available."))
        else:
            j_id = f"job_{i+1}"
            title = "N/A"
            company = "Unknown"
            url = "N/A"
            content = str(job)

        xml_output += (
            f'  <job_listing id="{j_id}">\n'
            f'    <title>{title}</title>\n'
            f'    <company>{company}</company>\n'
            f'    <url>{url}</url>\n'
            f'    <description>{content}</description>\n'
            f'  </job_listing>\n'
        )
    xml_output += "</job_listings>"
    return xml_output


async def _evaluate_batch_async(batch: list[JobListing], resume_text: str) -> RankedJobList:
    """Evaluates a single batch of job listings asynchronously using reasoning_llm."""
    formatted_listings = format_batch_xml(batch)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert executive career advisor evaluating candidate job matches.
Analyze the candidate's resume wrapped inside <candidate_resume> tags against the retrieved job listings wrapped inside <job_listings> tags.

<candidate_resume>
{resume}
</candidate_resume>

{listings}

<multi_criteria_scoring_rubric>
For each job listing in this batch, compute scores (0-100) using the following weighted multi-criteria rubric:
1. Technical Skill Fit (40% Weight): Overlap between candidate's programming languages, frameworks, AI/ML tools, and job requirements.
2. Experience & Seniority Alignment (30% Weight): Seniority level, leadership experience, and years of relevant industry experience.
3. Domain & Industry Relevance (20% Weight): Alignment of past candidate projects with employer's industry sector.
4. Work Conditions & Preferences (10% Weight): Match on remote work preferences, location, and workplace flexibility.

Calculate overall_score = (Skill_Fit * 0.40) + (Experience * 0.30) + (Domain * 0.20) + (Conditions * 0.10).
</multi_criteria_scoring_rubric>

<evaluation_instructions>
1. Filter out completely irrelevant job postings.
2. For each valid job listing, calculate:
   - overall_score (0-100) using the weighted rubric above
   - skills_match_score (0-100)
   - experience_match_score (0-100)
3. Extract matching_skills (list of matching candidate skills) and missing_skills (required skills candidate lacks).
4. Provide pros (strengths of this match), cons (potential drawbacks or gaps), and a 2-3 sentence match rationale.
5. Provide a brief executive summary of this batch's evaluation results.
</evaluation_instructions>"""
    )

    structured_llm = reasoning_llm.with_structured_output(RankedJobList)
    chain = prompt | structured_llm

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: chain.invoke({"resume": resume_text, "listings": formatted_listings})
    )


async def _map_reduce_evaluate(listings: list[JobListing], resume_text: str) -> list[JobMatchAnalysis]:
    """Map-Reduce parallel evaluation pipeline across batches of job listings."""
    # Chunk listings into batches of size BATCH_SIZE (Map Phase)
    batches = [listings[i:i + BATCH_SIZE] for i in range(0, len(listings), BATCH_SIZE)]
    logger.info("Map-Reduce: Evaluating %d listings across %d parallel batches", len(listings), len(batches))

    tasks = [_evaluate_batch_async(batch, resume_text) for batch in batches]
    results: list[RankedJobList | Exception] = await asyncio.gather(*tasks, return_exceptions=True)

    # Reduce Phase: Aggregate, deduplicate, filter, and sort top matches
    all_evaluated: list[JobMatchAnalysis] = []
    seen_job_ids: set[str] = set()

    for batch_idx, res in enumerate(results):
        if isinstance(res, RankedJobList):
            for match in res.ranked_jobs:
                if match.job_id not in seen_job_ids:
                    # Find the job in the original listings to get its description for domain penalty
                    job = next((job for job in listings if job.id == match.job_id), None)
                    if job:
                        domain_penalty = calculate_generic_domain_penalty(resume_text, job.description)
                        # Adjust the overall score by domain penalty
                        adjusted_score = match.overall_score * (1.0 - domain_penalty)
                        match.overall_score = round(adjusted_score, 1)
                        # Update rationale to note domain adjustment
                        match.match_rationale = f"{match.match_rationale} [Domain relevance adjusted]"
                    # Apply threshold after adjustment
                    if match.overall_score >= 65.0:
                        seen_job_ids.add(match.job_id)
                        all_evaluated.append(match)
        elif isinstance(res, Exception):
            logger.warning("Error evaluating batch %d: %s. Applying fallback vector scoring.", batch_idx + 1, res)

    # Fallback: if no LLM matches, return empty results (better than irrelevant matches)
    if not all_evaluated:
        logger.info("LLM ranking produced no matches. Returning empty results.")
        return []

    # Sort matches by overall_score descending
    all_evaluated.sort(key=lambda m: m.overall_score, reverse=True)
    return all_evaluated[:5]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def filter_and_rank_jobs(state: AgentState) -> dict[str, Any]:
    """Map-Reduce job evaluation node for LangGraph workflow.

    Args:
        state: Current agent state containing ``resume_text`` and ``job_listings``.

    Returns:
        Dict with ``ranked_matches`` (list[JobMatchAnalysis]) and ``ranked_jobs`` (str summary).
    """
    logger.info("NODE: FILTERING AND RANKING JOBS (MAP-REDUCE ENGINE)")
    resume_text = state.resume_text
    listings = state.job_listings

    if not listings:
        logger.warning("No job listings available to rank.")
        return {"ranked_matches": [], "ranked_jobs": "No job listings found."}

    # Execute Map-Reduce evaluation pipeline
    top_matches = asyncio.run(_map_reduce_evaluate(listings, resume_text))

    # Persist job evaluation history to SQLite database
    from database.manager import db
    db.save_job_evaluations(top_matches)

    # Generate backwards-compatible text summary for CLI output
    cli_output = f"Executive Summary: Evaluated {len(listings)} listings across Map-Reduce batches. Top {len(top_matches)} matches selected.\n\n"
    for idx, match in enumerate(top_matches, 1):
        cli_output += (
            f"Rank {idx}: {match.job_title} at {match.company} (Match Score: {match.overall_score:.0f}%)\n"
            f"URL: {match.url}\n"
            f"Match Rationale: {match.match_rationale}\n"
            f"Matching Skills: {', '.join(match.matching_skills) if match.matching_skills else 'N/A'}\n\n"
        )

    logger.info("Done Map-Reduce job ranking ✅ (%d top matches selected)", len(top_matches))
    return {
        "ranked_matches": top_matches,
        "ranked_jobs": cli_output.strip()
    }
