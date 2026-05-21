"""
core/protocol — Dizel Tool Communication Protocol.

Provides structured tool invocation, registration, dispatch,
and audit logging across all agents.

Usage:
    from core.protocol import ToolRegistry, ToolDispatcher

    registry = ToolRegistry()
    registry.register("web_search", my_search_fn, description="Search the web")

    dispatcher = ToolDispatcher(registry)
    result = dispatcher.invoke("task-1", "dizel", "web_search", {"query": "hello"})
"""

from .schema import ToolMessage, ToolStatus
from .registry import ToolRegistry, ToolSpec
from .dispatcher import ToolDispatcher
from .logger import ProtocolLogger


__all__ = [
    "ToolMessage",
    "ToolStatus",
    "ToolRegistry",
    "ToolSpec",
    "ToolDispatcher",
    "ProtocolLogger",
]
