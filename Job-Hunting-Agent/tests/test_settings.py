"""
Unit tests for configuration management and validation.
"""

import pytest

from config.settings import Config, ConfigurationError


def test_config_defaults(monkeypatch):
    """Test that default values are populated correctly when environment variables are valid."""
    monkeypatch.setenv("GOOGLE_API_KEY", "valid_google_key")
    monkeypatch.setenv("TAVILY_API_KEY", "valid_tavily_key")

    cfg = Config()
    assert cfg.google_api_key == "valid_google_key"
    assert cfg.tavily_api_key == "valid_tavily_key"
    assert cfg.llm_model == "gemini-2.0-flash"
    assert cfg.reasoning_llm_model == "gemini-1.5-pro"
    assert cfg.llm_temperature == 0.6
    assert cfg.max_search_results == 5


def test_config_placeholder_validation(monkeypatch):
    """Test that default placeholder API keys trigger ConfigurationError."""
    monkeypatch.setenv("GOOGLE_API_KEY", "your_google_api_key_here")
    monkeypatch.setenv("TAVILY_API_KEY", "your_tavily_api_key_here")

    with pytest.raises(ConfigurationError) as exc_info:
        Config()

    assert "GOOGLE_API_KEY environment variable is required" in str(exc_info.value)
    assert "TAVILY_API_KEY environment variable is required" in str(exc_info.value)


def test_config_temperature_validation(monkeypatch):
    """Test that out-of-range temperature triggers ConfigurationError."""
    monkeypatch.setenv("GOOGLE_API_KEY", "valid_google_key")
    monkeypatch.setenv("TAVILY_API_KEY", "valid_tavily_key")
    monkeypatch.setenv("LLM_TEMPERATURE", "1.5")

    with pytest.raises(ConfigurationError) as exc_info:
        Config()

    assert "LLM_TEMPERATURE must be between 0.0 and 1.0" in str(exc_info.value)
