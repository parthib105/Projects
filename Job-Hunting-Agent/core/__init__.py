"""
Core package for the Job Hunting Agent.

Provides the compiled LangGraph workflow application and the shared
agent state type for convenience imports::

    from core import app, AgentState
"""

from core.workflow import app
from core.state import AgentState

__all__ = ["app", "AgentState"]
