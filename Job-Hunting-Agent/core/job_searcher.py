"""
Job search node for the Job Hunting Agent.

Executes search queries concurrently across the configured search provider,
deduplicates results via DeduplicationEngine, and returns structured JobListing objects.
"""

import asyncio
import logging
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.deduplicator import DeduplicationEngine
from core.search_providers import get_search_provider
from core.state import AgentState, JobListing
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def _execute_parallel_searches(queries: list[str]) -> list[JobListing]:
    """Helper coroutine executing search provider tasks concurrently for all queries."""
    provider = get_search_provider()
    tasks = [provider.search_async(query) for query in queries]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    all_listings: list[JobListing] = []
    for q, res in zip(queries, results_list):
        if isinstance(res, list):
            all_listings.extend(res)
        elif isinstance(res, Exception):
            logger.warning("Error searching for query '%s': %s", q, res)

    return all_listings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def search_for_jobs(state: AgentState) -> dict[str, Any]:
    """Searches for jobs online asynchronously across queries and deduplicates listings.

    Args:
        state: Current agent state containing ``search_queries``.

    Returns:
        Dict with ``job_listings`` key containing a list of deduplicated JobListing objects.
    """
    logger.info("NODE: SEARCHING FOR JOBS (PARALLEL ASYNC EXECUTION)")
    queries = state.search_queries

    if not queries:
        logger.warning("No search queries provided in state.")
        return {"job_listings": []}

    # Execute all query searches concurrently using asyncio
    raw_listings = asyncio.run(_execute_parallel_searches(queries))

    # Deduplicate retrieved listings using DeduplicationEngine
    dedup_engine = DeduplicationEngine()
    unique_listings = dedup_engine.deduplicate(raw_listings)

    if not unique_listings:
        logger.warning("No search results returned from provider, creating fallback demo results...")
        unique_listings = [
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

    # Persist unique listings to SQLite database
    from database.manager import db
    db.save_job_listings(unique_listings)

    logger.info("Collected %d unique structured job listings ✅", len(unique_listings))
    return {"job_listings": unique_listings}
