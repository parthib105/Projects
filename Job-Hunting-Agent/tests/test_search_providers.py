import asyncio
import pytest
from core.search_providers import BaseSearchProvider, DuckDuckGoSearchProvider, TavilySearchProvider, get_search_provider


def test_get_search_provider_factory(monkeypatch):
    """Verify factory returns appropriate search provider instances."""
    monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
    provider = get_search_provider()
    assert isinstance(provider, DuckDuckGoSearchProvider)
    assert provider.provider_name == "DuckDuckGo"

    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    provider_tavily = get_search_provider()
    assert isinstance(provider_tavily, TavilySearchProvider)
    assert provider_tavily.provider_name == "Tavily"

    # Direct explicit parameter testing
    assert isinstance(get_search_provider("duckduckgo"), DuckDuckGoSearchProvider)
    assert isinstance(get_search_provider("tavily"), TavilySearchProvider)


def test_duckduckgo_search_provider_execution():
    """Verify DuckDuckGoSearchProvider runs search_async and returns JobListing objects."""
    async def _test():
        provider = DuckDuckGoSearchProvider()
        results = await provider.search_async("Python Engineer job openings", max_results=3)
        assert isinstance(results, list)
        if results:
            assert hasattr(results[0], "title")
            assert hasattr(results[0], "url")
            assert hasattr(results[0], "description")
            assert results[0].source == "DuckDuckGo"

    asyncio.run(_test())
