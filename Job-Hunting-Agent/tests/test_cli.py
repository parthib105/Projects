"""
Unit tests for cli.py interactive preference prompter and export engine.
"""

from cli import export_application_materials, prompt_user_preferences, sanitize_filename
from core.state import ApplicationMaterials


def test_sanitize_filename():
    """Verify filename sanitization strips special characters and spaces."""
    assert sanitize_filename("Senior ML Engineer / AI Lead") == "Senior_ML_Engineer_AI_Lead"
    assert sanitize_filename("Acme Corp, Inc.!") == "Acme_Corp_Inc"


def test_export_application_materials(tmp_path):
    """Verify exporting application materials creates formatted Markdown file."""
    materials = ApplicationMaterials(
        job_id="test_job_99",
        job_title="Staff AI Engineer",
        company="Global Innovation Corp",
        tailored_bullets=[
            "Accomplished 35% latency reduction by optimizing LangGraph workflow execution.",
            "Architected scalable Python microservices supporting 10k daily requests."
        ],
        cover_letter="Dear Hiring Team at Global Innovation Corp,\n\nI am writing to express my strong interest..."
    )

    exported_path = export_application_materials(materials, export_dir=str(tmp_path))

    assert exported_path.exists()
    content = exported_path.read_text(encoding="utf-8")
    assert "Staff AI Engineer" in content
    assert "Global Innovation Corp" in content
    assert "Accomplished 35% latency reduction" in content
    assert "Dear Hiring Team" in content


def test_prompt_user_preferences_non_interactive():
    """Verify non-interactive user preference prompting returns valid preferences."""
    path, prefs, provider = prompt_user_preferences(non_interactive=True)

    assert path == "sample_resume.txt"
    assert "Machine Learning Engineer" in prefs.target_roles
    assert "Remote" in prefs.preferred_locations
    assert provider.lower() in ("duckduckgo", "tavily")
