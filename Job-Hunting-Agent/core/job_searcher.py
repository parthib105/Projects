"""
Job search node for the Job Hunting Agent.

Searches for job listings online using search tools and returns
structured JobListing Pydantic models.
"""

import hashlib
import logging
from typing import Any
from tenacity import before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from core.dependencies import tavily_tool
from core.state import AgentState, JobListing
from utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_job_id(title: str, company: str, url: str) -> str:
    """Generates a stable unique hash ID for a job listing."""
    raw = f"{title.strip().lower()}|{company.strip().lower()}|{url.strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def search_for_jobs(state: AgentState) -> dict[str, Any]:
    """Searches for jobs online and returns structured JobListing objects.

    Args:
        state: Current agent state containing ``search_queries``.

    Returns:
        Dict with ``job_listings`` key containing a list of JobListing objects.
    """
    logger.info("NODE: SEARCHING FOR JOBS")
    queries = state.search_queries
    job_listings: list[JobListing] = []

    for query in queries:
        logger.info("Searching for: '%s'", query)
        try:
            search_results = tavily_tool.invoke({"query": query})
            logger.debug("Search results type: %s", type(search_results))

            results_to_process = []
            if isinstance(search_results, dict):
                results_to_process = search_results.get('results', [search_results])
            elif isinstance(search_results, list):
                results_to_process = search_results
            elif isinstance(search_results, str):
                results_to_process = [{"content": search_results, "url": "N/A"}]

            for res in results_to_process:
                if isinstance(res, dict):
                    content = res.get('content', res.get('snippet', ''))
                    url = res.get('url', res.get('link', 'N/A'))
                    title = res.get('title', 'N/A')
                    company = res.get('company', 'Unknown')

                    if content:
                        job_id = generate_job_id(title, company, url)
                        job_listings.append(
                            JobListing(
                                id=job_id,
                                title=title,
                                company=company,
                                url=url,
                                description=content,
                                source="Tavily"
                            )
                        )
                elif isinstance(res, str) and res.strip():
                    job_id = generate_job_id("Job Opportunity", "Unknown", "N/A")
                    job_listings.append(
                        JobListing(
                            id=job_id,
                            title="Job Opportunity",
                            company="Unknown",
                            url="N/A",
                            description=res.strip(),
                            source="Tavily"
                        )
                    )

        except Exception:
            logger.error("API error while searching for '%s', triggering retry if applicable...", query)
            raise

    if not job_listings:
        logger.warning("No results found from search tool, using fallback demo results...")
        fallback_results = [
            JobListing(
                id="fallback_1",
                title="Machine Learning Intern",
                company="Tech AI Corp",
                location="Remote",
                url="https://example.com/job1",
                description="Entry level machine learning position with focus on Python and data analysis. Requirements include programming skills and analytical thinking.",
                source="Fallback"
            ),
            JobListing(
                id="fallback_2",
                title="Software Engineering Intern",
                company="DevSoft Systems",
                location="Remote",
                url="https://example.com/job2",
                description="Software engineering internship opportunity for students with C++ and algorithm experience. Great learning environment.",
                source="Fallback"
            )
        ]
        job_listings = fallback_results

    logger.info("Collected %d valid structured job listings ✅", len(job_listings))
    return {"job_listings": job_listings}
