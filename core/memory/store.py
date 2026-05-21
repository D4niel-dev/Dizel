"""
core/memory/store.py — Abstract MemoryStore interface.

All concrete stores (working, episodic, semantic) implement this
interface so the rest of the system can query memory uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .schema import MemoryItem, MemoryType


@dataclass
class MemoryQuery:
    """Structured query for retrieving memory items."""
    text: Optional[str] = None             # free-text search
    types: Optional[List[MemoryType]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_confidence: float = 0.0
    since: Optional[datetime] = None       # only items after this time
    limit: int = 20
    include_expired: bool = False
    metadata_filters: Dict[str, str] = field(default_factory=dict)


class MemoryStore(ABC):
    """Abstract interface for all memory backends."""

    @abstractmethod
    def write(self, item: MemoryItem) -> str:
        """Store a memory item. Returns the item ID."""

    @abstractmethod
    def read(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve a single item by ID. Returns None if not found."""

    @abstractmethod
    def query(self, q: MemoryQuery) -> List[MemoryItem]:
        """Search for items matching the query. Returns sorted by relevance."""

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Remove a single item. Returns True if it existed."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of items in this store."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all items. Returns number of items removed."""

    def write_batch(self, items: List[MemoryItem]) -> List[str]:
        """Store multiple items. Default: sequential writes."""
        return [self.write(item) for item in items]

    def prune_expired(self) -> int:
        """Remove expired items. Default: scan and delete."""
        all_items = self.query(MemoryQuery(include_expired=True, limit=10_000))
        removed = 0
        for item in all_items:
            if item.is_expired:
                self.delete(item.id)
                removed += 1
        return removed
