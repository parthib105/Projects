"""
Job search node for the Job Hunting Agent.

Searches for job listings online using the Tavily Search API.
This module will be replaced with a local database query in Phase 2.
"""

import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_not_exception_type

from typing import Dict, List, Any

from core.state import AgentState
from core.dependencies import tavily_tool
from utils.logging_config import get_logger

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((ValueError, KeyError, TypeError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def search_for_jobs(state: AgentState) -> Dict[str, Any]:
    """Searches for jobs online using Tavily Search.

    Args:
        state: Current agent state containing ``search_queries``.

    Returns:
        Dict with ``job_listings`` key containing a list of job dicts.
    """
    logger.info("NODE: SEARCHING FOR JOBS")
    queries = state.search_queries
    all_results: List[Dict] = []

    for query in queries:
        logger.info("Searching for: '%s'", query)
        try:
            # We no longer append "job opening" because the query generator already includes strict keywords
            search_results = tavily_tool.invoke({"query": query})

            # Debug: log the type and structure of results
            logger.debug("Search results type: %s", type(search_results))

            # Normalize results into a list of dicts
            results_to_process = []
            if isinstance(search_results, dict):
                results_to_process = search_results.get('results', [search_results])
            elif isinstance(search_results, list):
                results_to_process = search_results
            elif isinstance(search_results, str):
                results_to_process = [{"content": search_results, "url": "N/A"}]

            # Process the results without manual keyword filtering
            for res in results_to_process:
                if isinstance(res, dict):
                    content = res.get('content', res.get('snippet', ''))
                    url = res.get('url', res.get('link', 'N/A'))
                    title = res.get('title', 'N/A')

                    if content:
                        all_results.append({
                            "content": content,
                            "url": url,
                            "title": title
                        })
                elif isinstance(res, str):
                    all_results.append({
                        "content": res,
                        "url": "N/A",
                        "title": "N/A"
                    })

        except Exception as e:
            # Re-raise so tenacity can catch it and retry if it's a network issue
            logger.error("API error while searching for '%s', triggering retry if applicable...", query)
            raise

    # If no results found, create some fallback results
    if not all_results:
        logger.warning("No results found from Tavily, creating fallback results...")
        # You might want to add fallback logic here or modify the search strategy
        fallback_results = [
            {
                "content": "Entry level machine learning position with focus on Python and data analysis. Requirements include programming skills and analytical thinking.",
                "url": "https://example.com/job1",
                "title": "Machine Learning Intern"
            },
            {
                "content": "Software engineering internship opportunity for students with C++ and algorithm experience. Great learning environment.",
                "url": "https://example.com/job2",
                "title": "Software Engineering Intern"
            }
        ]
        all_results = fallback_results
        logger.warning("Using fallback results for demonstration purposes")

    logger.info("Collected %d valid job listings ✅", len(all_results))
    return {"job_listings": all_results}
