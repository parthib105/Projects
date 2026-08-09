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

    config = {"configurable": {"thread_id": "session_1"}}
    final_state = app.invoke(inputs, config=config)

    logger.info("JOB SEARCH COMPLETE ✅")
    print("\n" + "=" * 70)
    print("RANKED JOB MATCHES & SUMMARY:")
    print("=" * 70)
    print(final_state.get("ranked_jobs", ""))

    materials = final_state.get("application_materials")
    if materials:
        print("\n" + "=" * 70)
        print(f"TAILORED APPLICATION MATERIALS FOR TOP MATCH ({materials.job_title} at {materials.company}):")
        print("=" * 70)
        print("\n--- Tailored Resume Bullets ---")
        for bullet in materials.tailored_bullets:
            print(f"• {bullet}")
        print("\n--- Customized Cover Letter ---")
        print(materials.cover_letter)