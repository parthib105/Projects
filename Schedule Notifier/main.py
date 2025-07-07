#!/usr/bin/env python3
"""
Schedule Notifier - Main Entry Point
A comprehensive academic schedule management and WhatsApp notification system.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ui.cli import cli
from config import config, is_development, get_log_level
import logging

# Set up logging
logging.basicConfig(
    level=getattr(logging, get_log_level()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the schedule notifier."""
    try:
        # Ensure data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Run CLI
        cli()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if is_development():
            raise
        else:
            print(f"❌ An error occurred: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()