"""
Semantic Vector Search Engine for the Job Hunting Agent.

Uses GoogleGenerativeAIEmbeddings to embed candidate resume sections
and job descriptions into high-dimensional vector space and computes
cosine similarity matching scores.
"""

import math
import re

from core.dependencies import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two float vectors in pure Python.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = math.sqrt(sum(a * a for a in vec1))
    norm_vec2 = math.sqrt(sum(b * b for b in vec2))

    if norm_vec1 == 0.0 or norm_vec2 == 0.0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


def fallback_text_similarity(text1: str, text2: str) -> float:
    """Fallback Jaccard word set similarity score if embedding API is unavailable."""
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


class VectorSearchEngine:
    """Semantic vector search and embedding similarity engine."""

    def __init__(self) -> None:
        """Initialize GoogleGenerativeAIEmbeddings model."""
        self._embeddings = None
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if config.google_api_key and config.google_api_key != "your_google_api_key_here":
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=config.google_api_key
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not initialize GoogleGenerativeAIEmbeddings: %s", e)

    def embed_text(self, text: str) -> list[float]:
        """Generates embedding vector for input text string.

        Args:
            text: Text content to embed.

        Returns:
            list[float]: Embedding vector float list.
        """
        if not text or not self._embeddings:
            return []

        try:
            return self._embeddings.embed_query(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("Vector embedding API error: %s", e)
            return []

    def compute_similarity(self, resume_text: str, job_description: str) -> float:
        """Computes semantic similarity score (0.0 to 100.0) between resume and job description.

        Args:
            resume_text: Extracted candidate resume content.
            job_description: Job posting description snippet.

        Returns:
            float: Similarity score normalized between 0.0 and 100.0.
        """
        if not resume_text or not job_description:
            return 0.0

        vec_resume = self.embed_text(resume_text[:2000])  # Chunk resume text
        vec_job = self.embed_text(job_description[:2000])   # Chunk job text

        if vec_resume and vec_job:
            sim = cosine_similarity(vec_resume, vec_job)
            # Map cosine similarity (-1.0 to 1.0) to percentage (0.0 to 100.0)
            percentage = max(0.0, min(100.0, (sim + 1.0) / 2.0 * 100.0))
            logger.debug("Computed vector cosine similarity score: %.2f%%", percentage)
            return percentage

        # Fallback text similarity if embeddings fail or API quota reached
        fallback_sim = fallback_text_similarity(resume_text, job_description) * 100.0
        logger.debug("Used fallback text similarity score: %.2f%%", fallback_sim)
        return fallback_sim


# Global vector search engine instance
vector_engine = VectorSearchEngine()
