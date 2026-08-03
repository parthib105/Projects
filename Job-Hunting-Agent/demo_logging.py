"""
demo_logging.py — Interactive walkthrough of utils/logging_config.py

Run this file to see exactly what each component does:
    python demo_logging.py

It covers:
  1. setup_logging()  — How the centralized logger is configured
  2. get_logger()     — How individual modules get their own named logger
  3. Log levels       — DEBUG, INFO, WARNING, ERROR, CRITICAL and filtering
  4. ColoredFormatter — How ANSI color codes are injected into output
  5. ProgressLogger   — How multi-step workflows report progress
  6. Library silencing — How noisy third-party loggers are suppressed
"""

import sys
import logging

# ─────────────────────────────────────────────────────────────────────
# DEMO 1: What setup_logging() actually does under the hood
# ─────────────────────────────────────────────────────────────────────
def demo_1_setup_logging():
    """
    setup_logging() does 5 things in sequence:

    1. Converts the string level ("INFO") to a numeric constant (20).
       Python's log levels:  DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
       Any message with a level BELOW the configured level is silently dropped.

    2. Builds a format string that controls what each log line looks like.
       Default (with timestamp):
           "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
           → "2026-08-03 19:00:00 - __main__ - INFO - Hello world"

       Without timestamp:
           "%(name)s - %(levelname)s - %(message)s"
           → "__main__ - INFO - Hello world"

    3. Wraps that format string in a ColoredFormatter (or plain Formatter).
       ColoredFormatter injects ANSI escape codes around the level name:
           "\\033[32mINFO\\033[0m"  → terminal renders "INFO" in green
       Colors: DEBUG=Cyan, INFO=Green, WARNING=Yellow, ERROR=Red, CRITICAL=Magenta

    4. Attaches a StreamHandler (→ stdout) to the ROOT logger.
       The root logger is the parent of ALL loggers. By configuring it once,
       every logger created with get_logger() or logging.getLogger() inherits
       the same handler, format, and level — no per-module setup needed.

    5. Silences noisy third-party libraries (urllib3, requests, httpx)
       by setting their loggers to WARNING, so their DEBUG/INFO spam is hidden.
    """
    from utils.logging_config import setup_logging, get_logger

    print("=" * 70)
    print("DEMO 1: setup_logging() — Centralized configuration")
    print("=" * 70)

    # Call setup with DEBUG level so we can see ALL messages
    setup_logging(level="DEBUG", colored_output=True)
    logger = get_logger("demo_1")

    print("\n→ Logger name: 'demo_1'")
    print(f"→ Logger effective level: {logging.getLevelName(logger.getEffectiveLevel())}")
    print(f"→ Root logger handlers: {logging.getLogger().handlers}")
    print(f"→ Formatter class: {type(logging.getLogger().handlers[0].formatter).__name__}")
    print()


# ─────────────────────────────────────────────────────────────────────
# DEMO 2: Log levels and how filtering works
# ─────────────────────────────────────────────────────────────────────
def demo_2_log_levels():
    """
    Each log call has a severity level. Only messages at or above the
    configured level are shown. Everything below is silently discarded.

    Level hierarchy (low → high):
        DEBUG (10) → INFO (20) → WARNING (30) → ERROR (40) → CRITICAL (50)

    If setup_logging(level="WARNING"), then:
        ✗ logger.debug("...")    — DROPPED (10 < 30)
        ✗ logger.info("...")     — DROPPED (20 < 30)
        ✓ logger.warning("...")  — SHOWN   (30 >= 30)
        ✓ logger.error("...")    — SHOWN   (40 >= 30)
        ✓ logger.critical("...") — SHOWN   (50 >= 30)
    """
    from utils.logging_config import setup_logging, get_logger

    print("\n" + "=" * 70)
    print("DEMO 2: Log levels — What gets shown vs. dropped")
    print("=" * 70)

    # --- Round 1: Level = DEBUG (show everything) ---
    print("\n--- Level = DEBUG (all 5 messages appear) ---")
    setup_logging(level="DEBUG")
    logger = get_logger("demo_2")

    logger.debug("🔍 DEBUG: Detailed diagnostic info (e.g., variable values)")
    logger.info("ℹ️  INFO: Normal operation milestones (e.g., 'Resume parsed')")
    logger.warning("⚠️  WARNING: Something unexpected but not fatal (e.g., 'Using fallback')")
    logger.error("❌ ERROR: Something failed (e.g., 'API call failed')")
    logger.critical("🔥 CRITICAL: Application cannot continue")

    # --- Round 2: Level = WARNING (only 3 messages appear) ---
    print("\n--- Level = WARNING (DEBUG and INFO are now hidden) ---")
    setup_logging(level="WARNING")

    logger.debug("🔍 This DEBUG message is DROPPED — you won't see it")
    logger.info("ℹ️  This INFO message is also DROPPED")
    logger.warning("⚠️  WARNING still shows")
    logger.error("❌ ERROR still shows")
    logger.critical("🔥 CRITICAL still shows")


# ─────────────────────────────────────────────────────────────────────
# DEMO 3: get_logger() and the %(name)s field
# ─────────────────────────────────────────────────────────────────────
def demo_3_named_loggers():
    """
    get_logger(name) is just a thin wrapper around logging.getLogger(name).

    The 'name' parameter becomes %(name)s in the log output format.
    By convention, each module passes __name__ so you can trace which
    file produced each log line.

    Example:
        In core/resume_parser.py → logger = get_logger(__name__)
        Output: "2026-08-03 ... - core.resume_parser - INFO - Parsing..."
                                   ^^^^^^^^^^^^^^^^^^^
                                   This tells you the source module.

    All named loggers inherit the root logger's handler (set by setup_logging),
    so you configure once and every module automatically gets the same format.
    """
    from utils.logging_config import setup_logging, get_logger

    print("\n" + "=" * 70)
    print("DEMO 3: Named loggers — Tracing which module produced a message")
    print("=" * 70)
    print()

    setup_logging(level="INFO")

    # Simulate loggers from different modules
    parser_logger = get_logger("core.resume_parser")
    search_logger = get_logger("core.job_searcher")
    ranker_logger = get_logger("core.job_ranker")

    parser_logger.info("Parsing resume PDF...")
    search_logger.info("Searching for 'ML Engineer' jobs...")
    ranker_logger.info("Ranking 15 job listings...")

    print("\n→ Notice how %(name)s changes per logger, showing you the source module.")


# ─────────────────────────────────────────────────────────────────────
# DEMO 4: ColoredFormatter internals
# ─────────────────────────────────────────────────────────────────────
def demo_4_colored_formatter():
    """
    ColoredFormatter extends logging.Formatter by overriding format().

    Before calling the parent's format(), it wraps record.levelname with
    ANSI escape codes:
        record.levelname = "\\033[32mINFO\\033[0m"
                            ^^^^^^^^      ^^^^^^^
                            green start   color reset

    ANSI codes only work in terminals that support them (checked via isatty()).
    If output is piped to a file, colors are skipped to avoid garbled text.

    Color mapping:
        DEBUG    → \\033[36m (Cyan)
        INFO     → \\033[32m (Green)
        WARNING  → \\033[33m (Yellow)
        ERROR    → \\033[31m (Red)
        CRITICAL → \\033[35m (Magenta)
    """
    from utils.logging_config import ColoredFormatter

    print("\n" + "=" * 70)
    print("DEMO 4: ColoredFormatter — How colors are injected")
    print("=" * 70)

    # Show the raw ANSI codes
    colors = {
        'DEBUG':    '\033[36m',
        'INFO':     '\033[32m',
        'WARNING':  '\033[33m',
        'ERROR':    '\033[31m',
        'CRITICAL': '\033[35m',
    }
    reset = '\033[0m'

    print("\nRaw ANSI color codes used by ColoredFormatter:")
    for level, code in colors.items():
        print(f"  {code}{level}{reset}  ←  {repr(code)}{level}{repr(reset)}")

    print(f"\n→ Terminal support check: sys.stderr.isatty() = {sys.stderr.isatty()}")
    print("→ If False (e.g., piped to a file), colors are skipped automatically.\n")


# ─────────────────────────────────────────────────────────────────────
# DEMO 5: ProgressLogger — Tracking multi-step workflows
# ─────────────────────────────────────────────────────────────────────
def demo_5_progress_logger():
    """
    ProgressLogger is a helper class that wraps a logger and adds
    step-counting to multi-step operations.

    Usage:
        progress = ProgressLogger(logger, total_steps=4)
        progress.log_step("Parse resume")       → [1/4] Parse resume
        progress.log_step("Generate queries")    → [2/4] Generate queries
        progress.log_step("Search jobs")         → [3/4] Search jobs
        progress.log_step("Rank results")        → [4/4] Rank results
        progress.log_completion("Found 5 jobs")  → ✅ Operation completed (4 steps) - Found 5 jobs

    This is how Job_hunting_agent.py could track its LangGraph workflow
    progress in a cleaner way than individual logger.info() calls.
    """
    from utils.logging_config import setup_logging, get_logger, ProgressLogger

    print("\n" + "=" * 70)
    print("DEMO 5: ProgressLogger — Step-by-step workflow tracking")
    print("=" * 70)
    print()

    setup_logging(level="INFO")
    logger = get_logger("demo_5.workflow")

    # Simulate the Job Hunting Agent's 4-step workflow
    progress = ProgressLogger(logger, total_steps=4)

    progress.log_step("Parse resume", details="Extracted 847 words from resume.pdf")
    progress.log_step("Generate queries", details="Created 12 search queries")
    progress.log_step("Search jobs", details="Found 38 raw listings")
    progress.log_step("Rank results", details="Selected top 5 matches")
    progress.log_completion(summary="Top 5 jobs ready for review")


# ─────────────────────────────────────────────────────────────────────
# DEMO 6: Library silencing — Why urllib3/requests/httpx are suppressed
# ─────────────────────────────────────────────────────────────────────
def demo_6_library_silencing():
    """
    When you set level="DEBUG", you see everything — including verbose
    internal logs from libraries like urllib3 ("Starting new HTTPS
    connection...") and httpx. These are noisy and unhelpful.

    setup_logging() handles this by explicitly setting those loggers
    to WARNING level:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    This means YOUR debug messages appear, but library spam does not.
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Library silencing — Suppressing third-party noise")
    print("=" * 70)

    # Show the silenced loggers and their levels
    silenced = ["urllib3", "requests", "httpx"]
    print("\nAfter setup_logging(), these loggers are pinned to WARNING:")
    for name in silenced:
        lib_logger = logging.getLogger(name)
        level_name = logging.getLevelName(lib_logger.getEffectiveLevel())
        print(f"  logging.getLogger({name!r}).level = {level_name}")

    print("\n→ Result: Your DEBUG messages show up. Library internals don't.\n")


# ─────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        Logging Deep-Dive: What utils/logging_config.py Does          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    demo_1_setup_logging()
    demo_2_log_levels()
    demo_3_named_loggers()
    demo_4_colored_formatter()
    demo_5_progress_logger()
    demo_6_library_silencing()

    print("=" * 70)
    print("✅ All demos complete! You now understand every piece of")
    print("   utils/logging_config.py and how Job_hunting_agent.py uses it.")
    print("=" * 70)
