"""
Centralized logging configuration for the Job Hunting Agent.

This module provides consistent logging setup across all components
of the application, with support for different log levels and formats.
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors if supported."""
        # Add color to levelname
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    colored_output: bool = True
) -> None:
    """
    Set up centralized logging configuration for the Job Hunting Agent.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
        colored_output: Whether to use colored output for different log levels
    
    Example:
        >>> setup_logging(level="DEBUG", colored_output=True)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Choose formatter based on colored_output preference
    if colored_output:
        formatter = ColoredFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to appropriate levels
    logging.getLogger("job_hunting_agent").setLevel(numeric_level)
    
    # Suppress verbose logging from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


class ProgressLogger:
    """
    Utility class for logging progress through multi-step operations.
    
    This class helps track progress through the job hunting workflow
    with consistent formatting and timing information.
    """
    
    def __init__(self, logger: logging.Logger, total_steps: int):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance to use for output
            total_steps: Total number of steps in the operation
        """
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
    
    def log_step(self, step_name: str, details: Optional[str] = None) -> None:
        """
        Log completion of a step.
        
        Args:
            step_name: Name of the completed step
            details: Optional additional details about the step
        """
        self.current_step += 1
        progress = f"[{self.current_step}/{self.total_steps}]"
        
        message = f"{progress} {step_name}"
        if details:
            message += f" - {details}"
        
        self.logger.info(message)
    
    def log_completion(self, summary: Optional[str] = None) -> None:
        """
        Log completion of the entire operation.
        
        Args:
            summary: Optional summary of the completed operation
        """
        message = f"✅ Operation completed ({self.total_steps} steps)"
        if summary:
            message += f" - {summary}"
        
        self.logger.info(message)


# Pre-configured logger for the package
package_logger = logging.getLogger("job_hunting_agent")
