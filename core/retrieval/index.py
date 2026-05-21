"""
core/retrieval/index.py — Modular indexing interface.

Abstract base for different index types (full-text, metadata, semantic).
Each retrievable data type can use its own index strategy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class IndexEntry:
    """A single indexed item."""
    id: str
    content: str
    source: str                                    # "session" | "memory" | "file" | "task" | "tool" | "decision"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    score: float = 0.0                             # computed at search time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
        }


class BaseIndex(ABC):
    """Abstract interface for indexing backends."""

    @abstractmethod
    def add(self, entry: IndexEntry):
        """Add or update an entry in the index."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[IndexEntry]:
        """Search the index and return ranked results."""

    @abstractmethod
    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return total indexed entries."""

    @abstractmethod
    def clear(self):
        """Remove all entries."""

    def add_batch(self, entries: List[IndexEntry]):
        """Add multiple entries. Default: sequential."""
        for entry in entries:
            self.add(entry)
