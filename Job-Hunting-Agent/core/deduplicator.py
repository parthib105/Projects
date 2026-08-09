"""
Canonical Job Deduplication Engine for the Job Hunting Agent.

Normalizes job URLs by removing tracking parameters and deduplicates
job listings based on canonical URLs and normalized company/title keys.
"""

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from core.state import JobListing
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Query parameters to strip for canonical URL normalization
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "reference", "s_cid", "session_id", "sid", "trk"
}


def normalize_url(url: str) -> str:
    """Strips tracking query parameters from a URL to produce its canonical form.

    Args:
        url: Original job listing URL.

    Returns:
        str: Canonical URL stripped of tracking parameters.
    """
    if not url or url.upper() == "N/A":
        return "N/A"

    try:
        parsed = urlparse(url)
        # Parse query string and remove tracking parameters
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        filtered_pairs = [pair for pair in query_pairs if pair[0].lower() not in TRACKING_PARAMS]
        new_query = urlencode(filtered_pairs)

        # Remove trailing slash from path for consistency
        path = parsed.path.rstrip("/")

        canonical_parts = (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            new_query,
            ""  # Strip fragment anchors
        )
        return urlunparse(canonical_parts)
    except Exception as e:  # noqa: BLE001
        logger.debug("URL normalization error for '%s': %s", url, e)
        return url.strip().rstrip("/")


def normalize_key(text: str) -> str:
    """Lowercases, removes punctuation, and normalizes spaces for fuzzy matching.

    Args:
        text: Raw text string (title or company).

    Returns:
        str: Normalized key string.
    """
    if not text:
        return ""
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    # Collapse multiple spaces into a single space
    return re.sub(r"\s+", " ", cleaned).strip()


class DeduplicationEngine:
    """High-performance deduplication engine for job listings."""

    def __init__(self) -> None:
        """Initialize DeduplicationEngine with empty tracking sets."""
        self._seen_urls: set[str] = set()
        self._seen_keys: set[str] = set()

    def reset(self) -> None:
        """Reset internal tracking sets for a new search run."""
        self._seen_urls.clear()
        self._seen_keys.clear()

    def deduplicate(self, listings: list[JobListing]) -> list[JobListing]:
        """Deduplicates a list of JobListing objects.

        Deduplication rules:
        1. Canonical URL matching: If a canonical URL has been seen, discard duplicate.
        2. Composite key matching: If normalized (company_name + job_title) has been seen, discard duplicate.

        Args:
            listings: Raw retrieved job listings.

        Returns:
            list[JobListing]: Filtered list of unique JobListing objects.
        """
        unique_listings: list[JobListing] = []
        discarded_count = 0

        for job in listings:
            canonical_url = normalize_url(job.url)
            norm_title = normalize_key(job.title)
            norm_company = normalize_key(job.company)

            # Create a composite entity key (company + title)
            composite_key = f"{norm_company}|{norm_title}"

            # Check 1: Canonical URL duplication
            if canonical_url != "N/A" and canonical_url in self._seen_urls:
                discarded_count += 1
                logger.debug("Discarding duplicate URL: %s", canonical_url)
                continue

            # Check 2: Composite key duplication (only if valid title exists)
            if norm_title and composite_key in self._seen_keys:
                discarded_count += 1
                logger.debug("Discarding duplicate job key: %s", composite_key)
                continue

            # Mark as seen
            if canonical_url != "N/A":
                self._seen_urls.add(canonical_url)
            if norm_title:
                self._seen_keys.add(composite_key)

            # Update job URL with canonical URL
            job.url = canonical_url if canonical_url != "N/A" else job.url
            unique_listings.append(job)

        if discarded_count > 0:
            logger.info("Deduplication Engine: Filtered out %d duplicate listings ✅", discarded_count)
        else:
            logger.info("Deduplication Engine: All %d listings were unique ✅", len(unique_listings))

        return unique_listings
