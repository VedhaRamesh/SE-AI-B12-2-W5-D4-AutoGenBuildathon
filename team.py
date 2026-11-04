"""Root team orchestration re-export for AutoGen multi-agent system."""

from core.team import (
    create_support_team,
    process_complaint,
    process_complaint_stream,
    parse_agent_outputs,
)

__all__ = [
    "create_support_team",
    "process_complaint",
    "process_complaint_stream",
    "parse_agent_outputs",
]
