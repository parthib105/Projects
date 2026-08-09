"""
LangGraph workflow definition for the Job Hunting Agent.

This module wires together all four nodes (parse → generate → search → rank)
into a compiled LangGraph application.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from core.job_ranker import filter_and_rank_jobs
from core.job_searcher import search_for_jobs
from core.query_generator import generate_search_queries
from core.resume_parser import parse_resume
from core.state import AgentState
from core.tailor import tailor_application
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Build the graph ──
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("parse_resume", parse_resume)
workflow.add_node("generate_queries", generate_search_queries)
workflow.add_node("search_jobs", search_for_jobs)
workflow.add_node("rank_jobs", filter_and_rank_jobs)
workflow.add_node("tailor_application", tailor_application)

# Define edges (the flow of control)
workflow.set_entry_point("parse_resume")
workflow.add_edge("parse_resume", "generate_queries")
workflow.add_edge("generate_queries", "search_jobs")
workflow.add_edge("search_jobs", "rank_jobs")
workflow.add_edge("rank_jobs", "tailor_application")
workflow.add_edge("tailor_application", END)

# Instantiate memory checkpointer for state persistence
checkpointer = MemorySaver()

# Compile the graph into a runnable app with checkpointer
app = workflow.compile(checkpointer=checkpointer)

logger.info("LangGraph workflow compiled successfully with MemorySaver checkpointer ✅")
