"""
Unit tests for the Map-Reduce job ranker and domain penalty calculation.
"""

from core.job_ranker import (
    calculate_generic_domain_penalty,
    extract_domain_indicators,
    format_batch_xml,
)
from core.state import JobListing


def test_extract_domain_indicators():
    """Verify keyword extraction accurately profiles domain categories."""
    healthcare_text = "Experienced clinical physiotherapist diagnosing patient care, rehabilitation therapy in hospital clinic."
    scores = extract_domain_indicators(healthcare_text)
    assert scores.get("healthcare", 0) >= 5
    assert scores.get("technology", 0) == 0

    tech_text = "Senior Python software engineer building scalable cloud microservices, APIs, algorithms, and SQL database systems."
    tech_scores = extract_domain_indicators(tech_text)
    assert tech_scores.get("technology", 0) >= 5
    assert tech_scores.get("healthcare", 0) == 0


def test_calculate_generic_domain_penalty_matching_domain():
    """Matching domains should incur 0.0 penalty."""
    resume_text = "Senior physiotherapist with extensive patient clinical care and therapy rehabilitation experience in hospital."
    job_desc = "Seeking a licensed physiotherapist for patient therapy and clinical care in wellness clinic."
    penalty = calculate_generic_domain_penalty(resume_text, job_desc)
    assert penalty == 0.0


def test_calculate_generic_domain_penalty_mismatched_domain():
    """Mismatched domains with strong signals should receive substantial penalty (>= 0.25)."""
    healthcare_resume = "Clinical physiotherapist diagnosing patient care, medical therapy, rehabilitation in hospital."
    tech_job = "Python software engineer to develop backend cloud microservices, REST APIs, databases, and algorithms."
    penalty = calculate_generic_domain_penalty(healthcare_resume, tech_job)
    assert penalty >= 0.25


def test_calculate_generic_domain_penalty_weak_signals():
    """Ambiguous or weak domain signals incur a mild uncertainty penalty of 0.10."""
    neutral_resume = "Hardworking professional with communication skills and punctuality."
    neutral_job = "Looking for a coordinator to assist with office duties."
    penalty = calculate_generic_domain_penalty(neutral_resume, neutral_job)
    assert penalty == 0.10


def test_format_batch_xml():
    """Verify XML tag formatting for Map-Reduce LLM prompt."""
    listings = [
        JobListing(
            id="job_001",
            title="Data Scientist",
            company="AI Labs",
            location="Remote",
            url="https://example.com/job1",
            description="Deep learning and machine learning systems.",
            source="DuckDuckGo",
        )
    ]
    xml = format_batch_xml(listings)
    assert '<job_listings>' in xml
    assert '<job_listing id="job_001">' in xml
    assert '<title>Data Scientist</title>' in xml
    assert '<company>AI Labs</company>' in xml
    assert '</job_listings>' in xml
