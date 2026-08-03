"""
LangGraph workflow definition for the Job Hunting Agent.

This module wires together all four nodes (parse → generate → search → rank)
into a compiled LangGraph application.
"""

from langgraph.graph import StateGraph, END

from core.state import AgentState
from core.resume_parser import parse_resume
from core.query_generator import generate_search_queries
from core.job_searcher import search_for_jobs
from core.job_ranker import filter_and_rank_jobs
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Build the graph ──
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("parse_resume", parse_resume)
workflow.add_node("generate_queries", generate_search_queries)
workflow.add_node("search_jobs", search_for_jobs)
workflow.add_node("rank_jobs", filter_and_rank_jobs)

# Define edges (the flow of control)
workflow.set_entry_point("parse_resume")
workflow.add_edge("parse_resume", "generate_queries")
workflow.add_edge("generate_queries", "search_jobs")
workflow.add_edge("search_jobs", "rank_jobs")
workflow.add_edge("rank_jobs", END)

# Compile the graph into a runnable app
app = workflow.compile()

logger.info("LangGraph workflow compiled successfully ✅")
