"""
Unit tests for application tailoring node and ApplicationMaterials state.
"""

from core.state import AgentState, ApplicationMaterials, JobMatchAnalysis
from core.tailor import tailor_application


def test_tailor_application_no_matches():
    """Verify tailor_application returns None when no ranked matches exist in state."""
    state = AgentState(
        resume_text="Candidate resume content",
        ranked_matches=[]
    )
    result = tailor_application(state)
    assert result["application_materials"] is None


def test_application_materials_model_initialization():
    """Verify ApplicationMaterials Pydantic model initialization."""
    materials = ApplicationMaterials(
        job_id="job_101",
        job_title="Senior AI Engineer",
        company="Open Innovation Corp",
        tailored_bullets=[
            "Accomplished 40% latency reduction by optimizing LangGraph workflow execution.",
            "Architected multi-agent LLM systems with 95% accuracy in production."
        ],
        cover_letter="Dear Hiring Manager,\n\nI am writing to express my strong interest in the Senior AI Engineer role..."
    )

    assert materials.job_id == "job_101"
    assert materials.company == "Open Innovation Corp"
    assert len(materials.tailored_bullets) == 2
    assert "Dear Hiring Manager" in materials.cover_letter
