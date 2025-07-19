"""
Configuration management for the Job Hunting Agent.

This module handles all configuration settings including:
- Environment variable validation
- API key management
- Application settings with defaults
- Configuration validation and error handling
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """
    Centralized configuration management for the Job Hunting Agent.
    
    This class handles loading and validating all configuration settings
    from environment variables, providing a single source of truth for
    application configuration.
    
    Attributes:
        google_api_key: Google Gemini API key for LLM operations
        tavily_api_key: Tavily API key for job search operations
        llm_temperature: Temperature setting for LLM responses (0.0-1.0)
        max_search_results: Maximum number of search results per query
        llm_model: Google Gemini model name to use
    """
    
    def __init__(self) -> None:
        """
        Initialize configuration by loading environment variables.
        
        Raises:
            ConfigurationError: If required environment variables are missing
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize logger
        self._logger = logging.getLogger(__name__)
        
        # Load and validate configuration
        self._load_config()
        self.validate_config()
        
        self._logger.info("Configuration loaded successfully")
    
    def _load_config(self) -> None:
        """Load configuration values from environment variables."""
        # Required API keys
        self._google_api_key = os.getenv("GOOGLE_API_KEY")
        self._tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Optional settings with defaults
        self._llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.6"))
        self._max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        self._llm_model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        
        # Logging level
        self._log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    @property
    def google_api_key(self) -> str:
        """
        Get the Google API key for Gemini LLM.
        
        Returns:
            str: The Google API key
            
        Raises:
            ConfigurationError: If the API key is not configured
        """
        if not self._google_api_key:
            raise ConfigurationError(
                "Google API key is required. Please set the GOOGLE_API_KEY environment variable."
            )
        return self._google_api_key
    
    @property
    def tavily_api_key(self) -> str:
        """
        Get the Tavily API key for job search.
        
        Returns:
            str: The Tavily API key
            
        Raises:
            ConfigurationError: If the API key is not configured
        """
        if not self._tavily_api_key:
            raise ConfigurationError(
                "Tavily API key is required. Please set the TAVILY_API_KEY environment variable."
            )
        return self._tavily_api_key
    
    @property
    def llm_temperature(self) -> float:
        """
        Get the LLM temperature setting.
        
        Returns:
            float: Temperature value between 0.0 and 1.0
        """
        return self._llm_temperature
    
    @property
    def max_search_results(self) -> int:
        """
        Get the maximum number of search results per query.
        
        Returns:
            int: Maximum search results count
        """
        return self._max_search_results
    
    @property
    def llm_model(self) -> str:
        """
        Get the Google Gemini model name.
        
        Returns:
            str: Model name (e.g., 'gemini-2.5-pro')
        """
        return self._llm_model
    
    @property
    def log_level(self) -> str:
        """
        Get the logging level.
        
        Returns:
            str: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        return self._log_level
    
    def validate_config(self) -> None:
        """
        Validate all configuration settings.
        
        Raises:
            ConfigurationError: If any configuration is invalid
        """
        errors = []
        
        # Validate required API keys
        if not self._google_api_key:
            errors.append(
                "GOOGLE_API_KEY environment variable is required. "
                "Get your API key from https://makersuite.google.com/app/apikey"
            )
        
        if not self._tavily_api_key:
            errors.append(
                "TAVILY_API_KEY environment variable is required. "
                "Get your API key from https://tavily.com/"
            )
        
        # Validate temperature range
        if not (0.0 <= self._llm_temperature <= 1.0):
            errors.append(
                f"LLM_TEMPERATURE must be between 0.0 and 1.0, got {self._llm_temperature}"
            )
        
        # Validate max search results
        if self._max_search_results <= 0:
            errors.append(
                f"MAX_SEARCH_RESULTS must be positive, got {self._max_search_results}"
            )
        
        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self._log_level not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}, got {self._log_level}"
            )
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ConfigurationError(error_message)
    
    def get_env_template(self) -> str:
        """
        Get a template for the .env file with all required variables.
        
        Returns:
            str: Template content for .env file
        """
        return """# Job Hunting Agent Configuration
    

# Copy this template to .env and fill in your API keys

# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional Settings (with defaults)
LLM_TEMPERATURE=0.6
MAX_SEARCH_RESULTS=5
LLM_MODEL=gemini-2.5-pro
LOG_LEVEL=INFO
"""
