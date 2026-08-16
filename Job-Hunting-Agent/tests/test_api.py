"""
Unit tests for FastAPI RESTful API Gateway (api.py).
"""

from fastapi.testclient import TestClient
from api import api_app

client = TestClient(api_app)


def test_api_root():
    """Verify health check endpoint returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_parse_resume_endpoint():
    """Verify POST /api/v1/resume/parse extracts resume text from uploaded file."""
    file_content = b"Candidate Name: John Doe\nRole: Senior ML Engineer\nSkills: Python, PyTorch, LangGraph"
    files = {"file": ("test_resume.txt", file_content, "text/plain")}

    response = client.post("/api/v1/resume/parse", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_resume.txt"
    assert "John Doe" in data["text_snippet"]
    assert data["character_count"] > 0


def test_get_tracked_applications_endpoint():
    """Verify GET /api/v1/jobs/applications retrieves historical application records."""
    response = client.get("/api/v1/jobs/applications")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "applications" in data
    assert isinstance(data["applications"], list)
