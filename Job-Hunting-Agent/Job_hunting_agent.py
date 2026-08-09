"""
Job Hunting Agent — CLI Entry Point

This is the command-line interface for running the Job Hunting Agent.
All core logic lives in the ``core/`` package. This file simply
invokes the compiled LangGraph workflow with a resume path.

Usage:
    python Job_hunting_agent.py
"""

from core import app
from utils.logging_config import get_logger

logger = get_logger(__name__)


import os

if __name__ == "__main__":
    # Using a raw string (r"...") or forward slashes is safer for file paths on Windows
    resume_file = r"./database/Parthib_CV_for_ML.pdf"
    if not os.path.exists(resume_file):
        resume_file = "sample_resume.txt"

    inputs = {"resume_path": resume_file}

    final_state = app.invoke(inputs)

    logger.info("JOB SEARCH COMPLETE ✅")
    print("\n" + "=" * 70)
    print("RANKED JOB MATCHES & SUMMARY:")
    print("=" * 70)
    print(final_state["ranked_jobs"])