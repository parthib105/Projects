"""
Resume parsing node for the Job Hunting Agent.

Extracts plain text from PDF, DOCX, and TXT files.
Includes helper methods to process files directly from memory bytes,
which ensures compatibility with future FastAPI endpoints.
"""

import io
import re
import os
import pypdf
import docx
from typing import Dict

from core.state import AgentState
from utils.logging_config import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Removes excessive whitespace and unprintable characters to save LLM tokens."""
    # Remove non-ascii characters and weird unicode spaces
    text = text.encode("ascii", "ignore").decode()
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_resume_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extracts text from an in-memory file buffer based on its extension.

    Args:
        file_bytes: The raw bytes of the file.
        filename: The original filename (used to determine extension).

    Returns:
        The extracted and cleaned plain text.

    Raises:
        ValueError: If the file type is unsupported or extraction fails.
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    file_obj = io.BytesIO(file_bytes)

    if ext == ".pdf":
        pdf_reader = pypdf.PdfReader(file_obj)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif ext == ".docx":
        doc = docx.Document(file_obj)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        text = file_bytes.decode("utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Expected .pdf, .docx, or .txt")

    if not text.strip():
        raise ValueError(f"Content could not be extracted from {filename} or the file is empty.")

    return clean_text(text)


def parse_resume(state: AgentState) -> Dict[str, str]:
    """Parses the resume file from disk (LangGraph node).

    Args:
        state: Current agent state containing ``resume_path``.

    Returns:
        Dict with ``resume_text`` key containing the extracted text.
    """
    logger.info("NODE: PARSING RESUME")
    path = state.resume_path
    
    if not os.path.exists(path):
        logger.error("Resume file not found at path: %s", path)
        raise FileNotFoundError(f"Resume file not found at path: {path}")

    filename = os.path.basename(path)
    
    with open(path, "rb") as f:
        file_bytes = f.read()
        
    logger.info("Extracting text from %s...", filename)
    extracted_text = parse_resume_from_bytes(file_bytes, filename)
    
    logger.info("Done parsing resume ✅ (Length: %d characters)", len(extracted_text))
    return {"resume_text": extracted_text}
