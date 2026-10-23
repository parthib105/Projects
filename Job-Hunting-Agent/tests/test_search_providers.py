import asyncio
from unittest.mock import MagicMock, patch
import pytest

from core.search_providers import (
    AdzunaSearchProvider,
    BaseSearchProvider,
    DuckDuckGoSearchProvider,
    TavilySearchProvider,
    get_search_provider,
)


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

    monkeypatch.setenv("SEARCH_PROVIDER", "adzuna")
    provider_adzuna = get_search_provider()
    assert isinstance(provider_adzuna, AdzunaSearchProvider)
    assert provider_adzuna.provider_name == "Adzuna"

    # Direct explicit parameter testing
    assert isinstance(get_search_provider("duckduckgo"), DuckDuckGoSearchProvider)
    assert isinstance(get_search_provider("tavily"), TavilySearchProvider)
    assert isinstance(get_search_provider("adzuna"), AdzunaSearchProvider)


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


def test_adzuna_search_provider_missing_credentials(monkeypatch):
    """Adzuna provider should gracefully return empty results if credentials are missing."""
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)

    provider = AdzunaSearchProvider()
    results = asyncio.run(provider.search_async("Machine Learning", max_results=3))
    assert results == []


def test_adzuna_search_provider_mocked_success(monkeypatch):
    """Adzuna provider should parse structured job results and format salary."""
    monkeypatch.setenv("ADZUNA_APP_ID", "mock_app_id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "mock_app_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Senior Data Scientist",
                "company": {"display_name": "HealthAI Corp"},
                "location": {"display_area": ["Remote", "USA"]},
                "redirect_url": "https://adzuna.com/job/123",
                "description": "Design and train deep learning models for healthcare diagnostics.",
                "salary_min": 120000,
                "salary_max": 160000,
            }
        ]
    }

    with patch("requests.get", return_value=mock_response):
        provider = AdzunaSearchProvider()
        results = asyncio.run(provider.search_async("Data Scientist", max_results=2))

        assert len(results) == 1
        listing = results[0]
        assert listing.title == "Senior Data Scientist"
        assert listing.company == "HealthAI Corp"
        assert listing.source == "Adzuna"
        assert "Salary: $120,000 - $160,000" in listing.description


def test_adzuna_search_provider_error_handling(monkeypatch):
    """Adzuna provider should handle non-200 or request exceptions gracefully."""
    monkeypatch.setenv("ADZUNA_APP_ID", "mock_app_id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "mock_app_key")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("requests.get", return_value=mock_response):
        provider = AdzunaSearchProvider()
        results = asyncio.run(provider.search_async("Software Engineer", max_results=2))
        assert results == []
