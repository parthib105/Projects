"""
Term extraction node for the Job Hunting Agent.
"""

import re
from typing import Any

from core.state import AgentState
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to load spaCy model for NLP-based term extraction
_nlp = None
try:
    import spacy
    # Load the small English model, disabling NER for speed (we don't need entity recognition)
    _nlp = spacy.load("en_core_web_sm", disable=["ner"])
except (ImportError, OSError):
    _nlp = None
    logger.warning("spaCy or en_core_web_sm model not available. Falling back to heuristic term extraction.")


def _extract_professional_terms_heuristic(resume_text: str) -> list[str]:
    """Extract professional/technical terms from resume using simple heuristics.
    Works across domains by looking for capitalized terms, technical phrases, etc."""
    # Look for terms that appear to be professional/job-related
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', resume_text)

    # Also look for quoted terms or hyphenated technical terms
    quoted_terms = re.findall(r'"([^"]*)"', resume_text)
    hyphenated = re.findall(r'\b\w+-\w+\b', resume_text)

    # Combine and filter
    all_terms = set(words + quoted_terms + hyphenated)

    # Remove common non-professional words
    stop_words = {'The', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
    professional_terms = [term for term in all_terms if term not in stop_words and len(term) > 2]

    # Return top terms by frequency or just first N
    return professional_terms[:10]


def extract_terms_node(state: AgentState) -> dict[str, Any]:
    """Extracts professional terms from the resume text and updates the state.

    Args:
        state: Current agent state containing ``resume_text``.

    Returns:
        Dict with ``professional_terms`` key containing a list of term strings.
    """
    logger.info("NODE: EXTRACTING PROFESSIONAL TERMS")
    resume_text = state.resume_text

    if not resume_text:
        logger.warning("Resume text is empty, returning empty professional terms.")
        return {"professional_terms": []}

    if _nlp is not None:
        # Use spaCy
        doc = _nlp(resume_text)
        # Extract noun chunks (multi-word terms)
        noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
        # Extract proper nouns (single-word terms like company names, technologies)
        proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
        # Combine and deduplicate
        all_terms = set(noun_chunks + proper_nouns)
    else:
        # Fallback to heuristic method
        all_terms = set(_extract_professional_terms_heuristic(resume_text))

    # Remove common non-professional words
    stop_words = {'The', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
    # Filter out stop words and short terms (length <= 2)
    professional_terms = [
        term for term in all_terms
        if term not in stop_words and len(term) > 2
    ]

    # Return top terms by length (descending) or just first N?
    # We'll sort by length descending to get more meaningful longer terms first, then take first 10.
    professional_terms.sort(key=len, reverse=True)
    return {"professional_terms": professional_terms[:10]}
