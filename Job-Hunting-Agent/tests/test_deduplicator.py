"""
Unit tests for URL normalization and DeduplicationEngine.
"""

from core.deduplicator import DeduplicationEngine, normalize_key, normalize_url
from core.state import JobListing


def test_normalize_url_strips_tracking():
    """Verify tracking parameters like utm_source, gclid, and ref are stripped."""
    raw_url = "https://jobs.example.com/view?id=123&utm_source=linkedin&gclid=xyz987&ref=search#top"
    normalized = normalize_url(raw_url)

    assert "utm_source" not in normalized
    assert "gclid" not in normalized
    assert "ref=" not in normalized
    assert "id=123" in normalized
    assert normalized == "https://jobs.example.com/view?id=123"


def test_normalize_key():
    """Verify string normalization for title and company matching."""
    assert normalize_key("  Senior ML Engineer,  Inc. ") == "senior ml engineer inc"
    assert normalize_key("PyTorch / GenAI Specialist") == "pytorch genai specialist"


def test_deduplication_engine_filters_duplicates():
    """Verify DeduplicationEngine filters duplicate URLs and company+title combinations."""
    listings = [
        JobListing(
            id="1",
            title="Senior Python Developer",
            company="Acme Corp",
            url="https://acme.com/jobs/1?utm_source=google",
            description="Role 1"
        ),
        JobListing(
            id="2",
            title="Senior Python Developer",
            company="Acme Corp",
            url="https://acme.com/jobs/1?utm_source=twitter",
            description="Duplicate URL and key"
        ),
        JobListing(
            id="3",
            title="Senior Python Developer",
            company="Acme Corp",
            url="https://other-board.com/jobs/99",
            description="Duplicate title/company key on different board"
        ),
        JobListing(
            id="4",
            title="Lead AI Architect",
            company="Acme Corp",
            url="https://acme.com/jobs/2",
            description="Unique role"
        )
    ]

    engine = DeduplicationEngine()
    deduped = engine.deduplicate(listings)

    assert len(deduped) == 2
    assert deduped[0].id == "1"
    assert deduped[1].id == "4"
