"""
Search Provider Plug-and-Play Strategy Engine for the Job Hunting Agent.

Abstracts search providers (DuckDuckGo, Tavily, etc.) behind a unified interface
with async support for concurrent query execution and native content retrieval.
"""

import abc
import asyncio
import hashlib
import re

from core.dependencies import config, tavily_tool
from core.state import JobListing
from utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_job_id(title: str, company: str, url: str) -> str:
    """Generates a stable unique hash ID for a job listing."""
    raw = f"{title.strip().lower()}|{company.strip().lower()}|{url.strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class BaseSearchProvider(abc.ABC):
    """Abstract base class establishing the interface for all search provider adapters."""

    def __init__(self, provider_name: str) -> None:
        """Initialize the search provider with its identifying name."""
        self.provider_name = provider_name

    @abc.abstractmethod
    async def search_async(self, query: str, max_results: int = 5) -> list[JobListing]:
        """Asynchronously executes a web search query and returns structured JobListing objects.

        Args:
            query: The job search query string.
            max_results: Maximum number of search results to return.

        Returns:
            list[JobListing]: Structured job listings retrieved by the search.
        """
        raise NotImplementedError(
            f"Search provider '{self.provider_name}' must implement the 'search_async' method."
        )


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """Free open-source search provider powered by DuckDuckGo Search (no API key required)."""

    def __init__(self) -> None:
        super().__init__(provider_name="DuckDuckGo")
        try:
            from duckduckgo_search import DDGS
            self._ddgs_cls = DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search / ddgs package is required for DuckDuckGoSearchProvider. "
                "Install it using: pip install ddgs"
            )

    async def search_async(self, query: str, max_results: int = 5) -> list[JobListing]:
        """Executes a DuckDuckGo text search asynchronously in an executor thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_search, query, max_results)

    def _sync_search(self, query: str, max_results: int) -> list[JobListing]:
        listings: list[JobListing] = []
        clean_q = re.sub(r'["\']', '', query).strip()
        queries_to_try = [clean_q]

        # If query has too many keywords, try a simplified query fallback
        words = clean_q.split()
        if len(words) > 5:
            queries_to_try.append(" ".join(words[:4]))

        try:
            with self._ddgs_cls() as ddgs:
                for q in queries_to_try:
                    results = list(ddgs.text(q, max_results=max_results))
                    if results:
                        for res in results:
                            title = res.get("title", "N/A")
                            url = res.get("href", res.get("link", "N/A"))
                            snippet = res.get("body", res.get("snippet", ""))

                            if snippet:
                                job_id = generate_job_id(title, "Unknown", url)
                                listings.append(
                                    JobListing(
                                        id=job_id,
                                        title=title,
                                        company="Unknown",
                                        url=url,
                                        description=snippet,
                                        source="DuckDuckGo"
                                    )
                                )
                        break
        except Exception as e:  # noqa: BLE001
            logger.warning("DuckDuckGo search error for query '%s': %s", query, e)

        return listings


class TavilySearchProvider(BaseSearchProvider):
    """Search provider powered by Tavily Search API."""

    def __init__(self) -> None:
        super().__init__(provider_name="Tavily")

    async def search_async(self, query: str, max_results: int = 5) -> list[JobListing]:
        """Executes a Tavily search asynchronously in an executor thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_search, query, max_results)

    def _sync_search(self, query: str, max_results: int) -> list[JobListing]:
        listings: list[JobListing] = []
        try:
            search_results = tavily_tool.invoke({"query": query})

            results_to_process = []
            if isinstance(search_results, dict):
                results_to_process = search_results.get("results", [search_results])
            elif isinstance(search_results, list):
                results_to_process = search_results
            elif isinstance(search_results, str):
                results_to_process = [{"content": search_results, "url": "N/A"}]

            for res in results_to_process:
                if isinstance(res, dict):
                    content = res.get("content", res.get("snippet", ""))
                    url = res.get("url", res.get("link", "N/A"))
                    title = res.get("title", "N/A")
                    company = res.get("company", "Unknown")

                    if content:
                        job_id = generate_job_id(title, company, url)
                        listings.append(
                            JobListing(
                                id=job_id,
                                title=title,
                                company=company,
                                url=url,
                                description=content,
                                source="Tavily"
                            )
                        )
        except Exception as e:  # noqa: BLE001
            logger.warning("Tavily search error for query '%s': %s", query, e)

        return listings


def get_search_provider(provider_name: str | None = None) -> BaseSearchProvider:
    """Factory function returning the configured search provider instance.

    Args:
        provider_name: Optional override for search provider name.

    Returns:
        BaseSearchProvider: Configured search provider instance.
    """
    import os
    selected_name = provider_name or os.getenv("SEARCH_PROVIDER", config.search_provider)
    name = selected_name.lower()

    if name == "duckduckgo":
        logger.info("Initializing search provider: DuckDuckGo (Free Open-Source)")
        return DuckDuckGoSearchProvider()
    elif name == "tavily":
        logger.info("Initializing search provider: Tavily API")
        return TavilySearchProvider()
    else:
        logger.warning("Unknown search provider '%s', defaulting to DuckDuckGo", name)
        return DuckDuckGoSearchProvider()
