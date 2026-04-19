"""
core/tool_state.py — Shared state object for tool-augmented requests.

Created fresh per request. Tracks which tools are active and holds
context gathered by each tool pipeline before generation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ToolState:
    # Active toggles (mapped from UI context IDs)
    web_search_enabled: bool = False
    deep_think_enabled: bool = False
    parse_files_enabled: bool = False

    # Request data
    user_input: str = ""
    uploaded_files: List[str] = field(default_factory=list)

    # Populated after tool execution
    web_results: str = ""
    file_context: str = ""

    # Status tracking
    processing_status: str = "idle"

    @classmethod
    def from_ui(cls, active_contexts: set, user_text: str, files: list = None) -> "ToolState":
        """Build a ToolState from InputPanel's active_contexts set."""
        return cls(
            web_search_enabled="web" in active_contexts,
            deep_think_enabled="deep" in active_contexts,
            parse_files_enabled="files" in active_contexts,
            uploaded_files=files or [],
            user_input=user_text,
        )

    def has_active_tools(self) -> bool:
        """True if any tool pipeline needs to run before generation."""
        return (
            self.web_search_enabled
            or self.deep_think_enabled
            or (self.parse_files_enabled and bool(self.uploaded_files))
        )

    def has_preprocessing(self) -> bool:
        """True if web search or file parsing needs to run (excludes Deep Think)."""
        return (
            self.web_search_enabled
            or (self.parse_files_enabled and bool(self.uploaded_files))
        )
