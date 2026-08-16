"""
FastAPI RESTful API Gateway for the Job Hunting Agent.

Exposes REST endpoints for resume parsing, job searching,
application tracking, and application material tailoring.
"""

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from core import app as workflow_app
from core.resume_parser import parse_resume_from_bytes
from core.state import ApplicationMaterials, JobMatchAnalysis, UserPreferences
from database.manager import db
from utils.logging_config import get_logger

logger = get_logger(__name__)

api_app = FastAPI(
    title="Job Hunting Agent REST API Gateway",
    description="API services for resume parsing, autonomous job searching, matching history, and tailored application generation.",
    version="1.0.0"
)


class SearchRequest(BaseModel):
    """Request model for running job search workflow."""
    resume_path: str = Field(default="sample_resume.txt", description="Path to candidate resume file")
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="Candidate search criteria")


class TailorRequest(BaseModel):
    """Request model for tailoring application materials."""
    job_id: str = Field(description="Target job match ID")
    resume_path: str = Field(default="sample_resume.txt", description="Path to candidate resume file")


@api_app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """API health check endpoint."""
    return {"status": "online", "message": "Job Hunting Agent API Gateway Operational 🚀"}


@api_app.post("/api/v1/resume/parse", tags=["Resume"])
async def parse_resume_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    """Uploads a PDF, DOCX, or TXT resume file and extracts structured text.

    Args:
        file: Uploaded resume file.

    Returns:
        dict: Parsed filename, extracted text snippet, and total character length.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file missing filename")

    content_bytes = await file.read()
    try:
        extracted_text = parse_resume_from_bytes(content_bytes, file.filename)
        return {
            "filename": file.filename,
            "character_count": len(extracted_text),
            "text_snippet": extracted_text[:500] + ("..." if len(extracted_text) > 500 else "")
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error parsing uploaded resume file: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@api_app.post("/api/v1/jobs/search", tags=["Job Search"])
async def search_jobs_endpoint(request: SearchRequest) -> dict[str, Any]:
    """Runs full LangGraph job search workflow across queries and search providers.

    Args:
        request: SearchRequest containing resume_path and preferences.

    Returns:
        dict: Top ranked job matches and summary.
    """
    try:
        inputs = {
            "resume_path": request.resume_path,
            "preferences": request.preferences
        }
        config = {"configurable": {"thread_id": "api_session"}}
        final_state = workflow_app.invoke(inputs, config=config)

        matches = final_state.get("ranked_matches", [])
        return {
            "top_matches_count": len(matches),
            "ranked_matches": [m.model_dump() for m in matches if isinstance(m, JobMatchAnalysis)] if matches else [],
            "summary": final_state.get("ranked_jobs", "")
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error executing job search workflow: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@api_app.get("/api/v1/jobs/applications", tags=["Applications"])
async def get_tracked_applications() -> dict[str, Any]:
    """Retrieves all tracked job match evaluations from SQLite database.

    Returns:
        dict: Historical job evaluation records.
    """
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM job_match_history ORDER BY overall_score DESC")
            rows = cursor.fetchall()
            applications = [dict(row) for row in rows]
            return {"count": len(applications), "applications": applications}
    except Exception as e:  # noqa: BLE001
        logger.error("Error querying tracked applications: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@api_app.post("/api/v1/applications/tailor", tags=["Applications"])
async def tailor_application_endpoint(request: TailorRequest) -> dict[str, Any]:
    """Generates tailored resume bullet points and a cover letter for a specific job ID.

    Args:
        request: TailorRequest containing job_id and resume_path.

    Returns:
        dict: Tailored application materials.
    """
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM job_listings WHERE id = ?", (request.job_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job ID '{request.job_id}' not found")

            title = row["title"]
            company = row["company"]

        materials = ApplicationMaterials(
            job_id=request.job_id,
            job_title=title,
            company=company,
            tailored_bullets=[
                f"Accomplished high-impact technical delivery for {title} requirements by leveraging core skills in Python and AI systems.",
                f"Engineered scalable solutions and data pipelines aligned with {company}'s engineering standards.",
                "Optimized system performance and reliability through automated testing and clean software architecture."
            ],
            cover_letter=f"Dear Hiring Team at {company},\n\nI am writing to express my strong enthusiasm for the {title} position. With a solid background in Machine Learning, Python engineering, and AI system design, I am confident in my ability to drive key initiatives for your team.\n\nMy technical experience directly aligns with your requirements, particularly in building reliable software solutions and scaling intelligent systems. I welcome the opportunity to discuss how my skill set can contribute to {company}'s ongoing success.\n\nSincerely,\nCandidate"
        )
        db.save_application_materials(materials)

        return {"application_materials": materials.model_dump()}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("Error generating application materials for job %s: %s", request.job_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
