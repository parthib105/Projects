"""
Search Provider Plug-and-Play Strategy Engine for the Job Hunting Agent.

Abstracts search providers (DuckDuckGo, Tavily, etc.) behind a unified interface
with async support for concurrent query execution and native content retrieval.
"""

import abc
import asyncio
import hashlib
import os
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


class AdzunaSearchProvider(BaseSearchProvider):
    """Free structured job search API from Adzuna."""

    def __init__(self):
        super().__init__(provider_name="Adzuna")
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")
        if not self.app_id or not self.app_key:
            logger.warning("Adzuna API credentials not found. Set ADZUNA_APP_ID and ADZUNA_APP_KEY.")

    async def search_async(self, query: str, max_results: int = 5) -> list[JobListing]:
        """Executes an Adzuna job search asynchronously."""
        if not self.app_id or not self.app_key:
            logger.warning("Adzuna credentials missing, returning empty results")
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_search, query, max_results)

    def _sync_search(self, query: str, max_results: int) -> list[JobListing]:
        """Sync implementation of Adzuna search."""
        listings = []
        try:
            # Adzuna API endpoint (example for US - adjust for other countries)
            country = "us"  # Could be made configurable
            url = f"http://api.adzuna.com/v1/api/jobs/{country}/search/1"

            params = {
                'app_id': self.app_id,
                'app_key': self.app_key,
                'results_per_page': min(max_results, 50),  # Adzuna max is 50
                'what': query,
                'content-type': 'application/json'
            }

            import requests
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                for res in results:
                    title = res.get('title', 'N/A')
                    company = res.get('company', {}).get('display_name', 'Unknown')
                    location = res.get('location', {}).get('display_area', 'Unknown')
                    url = res.get('redirect_url', 'N/A')
                    description = res.get('description', '')
                    salary_min = res.get('salary_min')
                    salary_max = res.get('salary_max')

                    # Enhance description with salary info if available
                    if salary_min and salary_max:
                        description += f"\n\nSalary: ${salary_min:,.0f} - ${salary_max:,.0f}"
                    elif salary_min:
                        description += f"\n\nSalary: ${salary_min:,.0f}+"

                    if description:
                        job_id = generate_job_id(title, company, url)
                        listings.append(
                            JobListing(
                                id=job_id,
                                title=title,
                                company=company,
                                location=location,
                                url=url,
                                description=description,
                                source="Adzuna"
                            )
                        )
            else:
                logger.warning(f"Adzuna API error: {response.status_code} - {response.text}")

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Adzuna search error for query '{query}': {e}")

        return listings


def get_search_provider(provider_name: str | None = None) -> BaseSearchProvider:
    """Factory function returning the configured search provider instance.

    Args:
        provider_name: Optional override for search provider name.

    Returns:
        BaseSearchProvider: Configured search provider instance.
    """
    selected_name = provider_name or os.getenv("SEARCH_PROVIDER", config.search_provider)
    name = selected_name.lower()

    if name == "duckduckgo":
        logger.info("Initializing search provider: DuckDuckGo (Free Open-Source)")
        return DuckDuckGoSearchProvider()
    elif name == "tavily":
        logger.info("Initializing search provider: Tavily API")
        return TavilySearchProvider()
    elif name == "adzuna":
        logger.info("Initializing search provider: Adzuna (Structured Job API)")
        return AdzunaSearchProvider()
    else:
        logger.warning("Unknown search provider '%s', defaulting to DuckDuckGo", name)
        return DuckDuckGoSearchProvider()
