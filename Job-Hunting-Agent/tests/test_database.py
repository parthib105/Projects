"""
Unit tests for database module, SQLite caching, and database persistence.
"""

from database.manager import JobDatabase, JobListing, JobMatchAnalysis, ApplicationMaterials


def test_database_initialization(tmp_path):
    """Verify JobDatabase creates tables cleanly on a temporary database path."""
    db_file = tmp_path / "test_agent.db"
    db_instance = JobDatabase(db_path=db_file)

    assert db_file.exists()


def test_database_job_listings_caching(tmp_path):
    """Verify job listings insertion and URL canonical checking in database."""
    db_file = tmp_path / "test_agent.db"
    db_instance = JobDatabase(db_path=db_file)

    listings = [
        JobListing(
            id="db_job_1",
            title="Senior Python Architect",
            company="DataScale Labs",
            url="https://datascale.com/jobs/1?utm_medium=email",
            description="Role description 1",
            source="Test"
        )
    ]

    inserted = db_instance.save_job_listings(listings)
    assert inserted == 1

    # Verify tracking parameter stripped URL is cached as seen
    assert db_instance.is_url_seen("https://datascale.com/jobs/1")
    assert db_instance.is_url_seen("https://datascale.com/jobs/1?utm_medium=email")

    # Duplicate insertion should be ignored
    dup_inserted = db_instance.save_job_listings(listings)
    assert dup_inserted == 0


def test_database_evaluations_and_materials_persistence(tmp_path):
    """Verify job evaluations and application materials persistence in database."""
    db_file = tmp_path / "test_agent.db"
    db_instance = JobDatabase(db_path=db_file)

    evaluations = [
        JobMatchAnalysis(
            job_id="db_job_1",
            job_title="Senior Python Architect",
            company="DataScale Labs",
            url="https://datascale.com/jobs/1",
            overall_score=94.0,
            skills_match_score=95.0,
            experience_match_score=90.0,
            matching_skills=["Python", "SQL", "Architecture"],
            missing_skills=[],
            match_rationale="Strong technical fit.",
            pros=["Remote"],
            cons=[]
        )
    ]
    db_instance.save_job_evaluations(evaluations)

    materials = ApplicationMaterials(
        job_id="db_job_1",
        job_title="Senior Python Architect",
        company="DataScale Labs",
        tailored_bullets=["Accomplished X by doing Y"],
        cover_letter="Dear Hiring Manager..."
    )
    db_instance.save_application_materials(materials)

    with db_instance._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT overall_score FROM job_match_history WHERE job_id = ?", ("db_job_1",))
        row = cursor.fetchone()
        assert row["overall_score"] == 94.0

        cursor.execute("SELECT cover_letter FROM application_materials WHERE job_id = ?", ("db_job_1",))
        mat_row = cursor.fetchone()
        assert "Dear Hiring Manager" in mat_row["cover_letter"]
