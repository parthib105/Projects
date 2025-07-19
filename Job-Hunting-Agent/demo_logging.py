# demo_logging.py
from utils.logging_config import setup_logging, get_logger

# Step 1: Set up logging (do this once at the start of your app)
setup_logging(level="INFO", colored_output=True)

# Step 2: Get a logger for your module
logger = get_logger(__name__)

# Step 3: Use the logger
def demo_basic_logging():
    logger.debug("This won't show (DEBUG level is below INFO)")
    logger.info("✅ Application started")
    logger.warning("⚠️  This is a warning")
    logger.error("❌ This is an error")
    
    # Simulate some work
    logger.info("Processing resume...")
    logger.info("Generating search queries...")
    logger.info("Searching for jobs...")
    logger.info("🎉 Job search completed!")

demo_basic_logging()
