"""
Unit tests for resume parsing and text cleaning.
"""

import pytest

from core.resume_parser import clean_text, parse_resume_from_bytes


def test_clean_text_preserves_unicode():
    """Verify clean_text preserves international Unicode characters, accents, and symbols."""
    raw_text = "Parthib Ghosh — Equip Lead €100k+  \t\n\n\nSkills: Python, C++ & ML."
    cleaned = clean_text(raw_text)

    assert "Equip" in cleaned
    assert "€100k+" in cleaned
    assert "—" in cleaned
    assert "\n\n" in cleaned  # Paragraph break preserved
    assert "\t" not in cleaned


def test_parse_txt_resume():
    """Verify in-memory TXT resume parsing."""
    txt_content = b"John Doe\nSoftware Engineer\nPython, FastApi, Docker"
    extracted = parse_resume_from_bytes(txt_content, "resume.txt")

    assert "John Doe" in extracted
    assert "Software Engineer" in extracted
    assert "Docker" in extracted


def test_unsupported_file_extension():
    """Verify ValueError raised for unsupported file types."""
    with pytest.raises(ValueError) as exc_info:
        parse_resume_from_bytes(b"content", "resume.xyz")

    assert "Unsupported file extension" in str(exc_info.value)
