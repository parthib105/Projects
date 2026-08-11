"""
Unit tests for core/vector_search.py vector math and similarity engine.
"""

from core.vector_search import VectorSearchEngine, cosine_similarity, fallback_text_similarity


def test_cosine_similarity_math():
    """Verify pure Python cosine similarity calculations."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert cosine_similarity(v1, v2) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v3) == 0.0


def test_fallback_text_similarity():
    """Verify text word overlap similarity matching."""
    t1 = "Senior Machine Learning Engineer PyTorch"
    t2 = "Machine Learning Engineer PyTorch LangChain"
    score = fallback_text_similarity(t1, t2)
    assert score > 0.40


def test_vector_search_engine_compute():
    """Verify VectorSearchEngine similarity score computation."""
    engine = VectorSearchEngine()
    resume = "Python Developer with experience in FastApi, PostgreSQL, and Docker."
    job = "Hiring Software Engineer skilled in Python, FastApi, and SQL."

    sim_score = engine.compute_similarity(resume, job)
    assert 0.0 <= sim_score <= 100.0
