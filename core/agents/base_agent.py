"""
base_agent.py — Abstract interface and result type for Dict / Lily agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentResult:
    """Structured output returned by any agent."""
    source: str                          # "Dict" or "Lily"
    file_name: str = ""
    file_type: str = ""
    description: str = ""
    details: List[str] = field(default_factory=list)
    notes: str = ""
    raw_text: str = ""                   # full extracted text (Lily)
    error: Optional[str] = None          # non-None if processing failed

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_context_block(self) -> str:
        """Render as a text block for the prompt builder."""
        tag = "DICT - IMAGE ANALYSIS" if self.source == "Dict" else "LILY - FILE ANALYSIS"
        lines = [f"[{tag}]"]
        if self.file_name:
            lines.append(f"File: {self.file_name}")
        if self.file_type:
            lines.append(f"Type: {self.file_type}")
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.details:
            lines.append("Details:")
            for d in self.details:
                lines.append(f"- {d}")
        if self.raw_text:
            lines.append("Extracted Text:")
            lines.append(self.raw_text)
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class BaseAgent(ABC):
    """Every agent converts some input into clean text for Dizel."""

    @abstractmethod
    def process(self, input_path: str, **kwargs) -> AgentResult:
        """Process a file/image and return structured text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
