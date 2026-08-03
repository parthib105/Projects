"""
Resume parsing node for the Job Hunting Agent.

Extracts plain text from a PDF resume file.
"""

import PyPDF2
from typing import Dict

from core.state import AgentState
from utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_resume(state: AgentState) -> Dict[str, str]:
    """Parses the resume PDF to extract text.

    Args:
        state: Current agent state containing ``resume_path``.

    Returns:
        Dict with ``resume_text`` key containing the extracted text.

    Raises:
        FileNotFoundError: If the resume file does not exist.
        ValueError: If the PDF is empty or unreadable.
    """
    logger.info("NODE: PARSING RESUME")
    path = state.resume_path
    try:
        with open(path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text: str = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            if not text:
                raise ValueError("PDF content could not be extracted or is empty!")
        logger.info("Done parsing resume ✅")
        return {"resume_text": text}
    except FileNotFoundError:
        logger.error("Resume file not found at path: %s", path)
        raise FileNotFoundError(f"Resume file not found at path: {path}")
