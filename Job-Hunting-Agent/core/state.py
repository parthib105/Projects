"""
Shared agent state and domain models for the Job Hunting Agent.

Defines structured Pydantic models for user preferences, job listings,
job match evaluations, and the shared agent state.
"""

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """Candidate search preferences."""
    target_roles: list[str] = Field(default_factory=list, description="Target job titles/roles")
    preferred_locations: list[str] = Field(default_factory=list, description="Target geographical locations")
    remote_preference: str = Field(default="Any", description="Remote, Hybrid, Onsite, or Any")
    min_salary: int | None = Field(default=None, description="Minimum annual salary threshold")
    experience_level: str = Field(default="Mid-Level", description="Target experience level (Entry, Mid, Senior, Lead)")


class JobListing(BaseModel):
    """Structured representation of a single job listing."""
    id: str = Field(description="Unique identifier or hash of the listing")
    title: str = Field(description="Job title")
    company: str = Field(default="Unknown", description="Hiring company name")
    location: str = Field(default="Remote / Unspecified", description="Job location")
    url: str = Field(default="N/A", description="Direct job URL or source link")
    description: str = Field(description="Raw snippet or full job description")
    source: str = Field(default="Tavily", description="Search provider source")
    posted_date: str | None = Field(default=None, description="Job posting date if available")


class JobMatchAnalysis(BaseModel):
    """Quantitative evaluation and ranking analysis for a job match."""
    job_id: str = Field(description="Matches JobListing.id")
    job_title: str = Field(description="Job title")
    company: str = Field(default="Unknown", description="Company name")
    url: str = Field(default="N/A", description="Application link")
    overall_score: float = Field(ge=0.0, le=100.0, description="Overall match score (0-100)")
    skills_match_score: float = Field(ge=0.0, le=100.0, description="Skills fit score (0-100)")
    experience_match_score: float = Field(ge=0.0, le=100.0, description="Experience alignment score (0-100)")
    matching_skills: list[str] = Field(default_factory=list, description="Candidate skills matching job requirements")
    missing_skills: list[str] = Field(default_factory=list, description="Key missing skills or gaps")
    match_rationale: str = Field(description="Concise 2-3 sentence summary of why this is a good match")
    pros: list[str] = Field(default_factory=list, description="Key advantages of this opportunity")
    cons: list[str] = Field(default_factory=list, description="Potential drawbacks or gaps")


class RankedJobList(BaseModel):
    """Collection of evaluated and ranked job opportunities."""
    ranked_jobs: list[JobMatchAnalysis] = Field(default_factory=list, description="Top ranked job matches sorted by score")
    summary: str = Field(default="", description="Executive summary of ranking results")


class ApplicationMaterials(BaseModel):
    """Tailored resume bullets and cover letter generated for top job match."""
    job_id: str = Field(description="Target job match ID")
    job_title: str = Field(description="Target job title")
    company: str = Field(description="Hiring company name")
    tailored_bullets: list[str] = Field(default_factory=list, description="3-5 tailored resume bullet points highlighting relevant achievements")
    cover_letter: str = Field(description="Professional tailored cover letter formatted for application")


class AgentState(BaseModel):
    """Shared state flowing through the LangGraph workflow.

    Attributes:
        resume_path: Filesystem path to the candidate resume file.
        resume_text: Extracted plain-text content of the resume.
        preferences: User search preferences and criteria.
        search_queries: Generated job search queries.
        job_listings: Structured list of retrieved job postings.
        ranked_matches: Structured list of evaluated and ranked job matches.
        ranked_jobs: Plain-text summary for backwards compatibility.
        application_materials: Tailored resume bullets and cover letter.
        professional_terms: Extracted professional/technical terms from the resume.
    """
    resume_path: str = Field(default="")
    resume_text: str = Field(default="")
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    search_queries: list[str] = Field(default_factory=list)
    job_listings: list[JobListing] = Field(default_factory=list)
    ranked_matches: list[JobMatchAnalysis] = Field(default_factory=list)
    ranked_jobs: str = Field(default="", description="Plain-text summary for CLI output compatibility")
    application_materials: ApplicationMaterials | None = Field(default=None, description="Tailored application materials for top job match")
    professional_terms: list[str] = Field(default_factory=list, description="Extracted professional/technical terms from the resume")

