"""
Unit tests for structured Pydantic models in AgentState.
"""

from core.state import AgentState, JobListing, JobMatchAnalysis, UserPreferences


def test_agent_state_initialization():
    """Verify structured Pydantic models initialize correctly with defaults."""
    state = AgentState(
        resume_path="/path/to/resume.pdf",
        preferences=UserPreferences(
            target_roles=["ML Engineer", "AI Developer"],
            preferred_locations=["San Francisco", "Remote"],
            remote_preference="Remote"
        ),
        job_listings=[
            JobListing(
                id="job123",
                title="Staff AI Engineer",
                company="Open AI Lab",
                url="https://example.com/job123",
                description="Seeking AI Engineer skilled in Python and LLMs."
            )
        ],
        ranked_matches=[
            JobMatchAnalysis(
                job_id="job123",
                job_title="Staff AI Engineer",
                company="Open AI Lab",
                url="https://example.com/job123",
                overall_score=92.5,
                skills_match_score=95.0,
                experience_match_score=90.0,
                matching_skills=["Python", "LLMs", "LangChain"],
                missing_skills=["PyTorch"],
                match_rationale="Strong technical fit for candidate experience.",
                pros=["Remote work", "High equity"],
                cons=["Fast-paced startup environment"]
            )
        ]
    )

    assert state.preferences.target_roles == ["ML Engineer", "AI Developer"]
    assert state.job_listings[0].company == "Open AI Lab"
    assert state.ranked_matches[0].overall_score == 92.5
    assert "Python" in state.ranked_matches[0].matching_skills
