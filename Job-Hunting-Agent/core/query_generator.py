"""
Query generation node for the Job Hunting Agent.

Uses the Gemini LLM to generate diverse job search queries
based on the parsed resume text.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from core.state import AgentState
from core.dependencies import llm
from utils.logging_config import get_logger

logger = get_logger(__name__)


class JobSearchQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of 10-15 highly specific web search queries. Each query MUST include a job title AND job-specific keywords (like 'responsibilities', 'qualifications', 'requirements', or 'apply') to ensure search engines return actual job postings rather than general articles."
    )


def generate_search_queries(state: AgentState) -> Dict[str, Any]:
    """Generates job search queries using Gemini based on resume text.

    Args:
        state: Current agent state containing ``resume_text``.

    Returns:
        Dict with ``search_queries`` key containing a list of query strings.
    """
    logger.info("NODE: GENERATING SEARCH QUERIES")
    resume_text = state.resume_text
    
    prompt = ChatPromptTemplate.from_template(
        """Based on the following resume text, generate highly specific job search queries designed for a web search engine.
        
        Resume:
        {resume}"""
    )
    
    structured_llm = llm.with_structured_output(JobSearchQueries)
    chain = prompt | structured_llm
    
    result = chain.invoke({"resume": resume_text})
    queries_list = result.queries
    
    logger.info("Done generating search queries ✅ — %d queries", len(queries_list))
    return {"search_queries": queries_list}
