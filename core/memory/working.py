"""
core/memory/working.py — In-memory working memory (session-scoped).

Fast, volatile storage for the current conversation context.
Cleared when the session ends. No disk persistence.
"""

from datetime import datetime
from typing import Dict, List, Optional

from .schema import MemoryItem, MemoryType
from .store import MemoryQuery, MemoryStore


class WorkingMemory(MemoryStore):
    """Session-scoped in-memory store. Fast reads, no persistence."""

    def __init__(self, max_items: int = 500):
        self._items: Dict[str, MemoryItem] = {}
        self._max_items = max_items

    def write(self, item: MemoryItem) -> str:
        item.type = MemoryType.WORKING
        # Evict oldest if at capacity
        if len(self._items) >= self._max_items:
            self._evict_oldest()
        self._items[item.id] = item
        return item.id

    def read(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def query(self, q: MemoryQuery) -> List[MemoryItem]:
        results = list(self._items.values())

        # Filter expired
        if not q.include_expired:
            results = [m for m in results if not m.is_expired]

        # Filter by source
        if q.sources:
            results = [m for m in results if m.source in q.sources]

        # Filter by tags
        if q.tags:
            tag_set = set(q.tags)
            results = [m for m in results if tag_set & set(m.tags)]

        # Filter by confidence
        if q.min_confidence > 0:
            results = [m for m in results if m.confidence >= q.min_confidence]

        # Filter by time
        if q.since:
            results = [m for m in results if m.timestamp >= q.since]

        # Filter by metadata
        for key, val in q.metadata_filters.items():
            results = [m for m in results if m.metadata.get(key) == val]

        # Text search (simple substring)
        if q.text:
            lower = q.text.lower()
            results = [m for m in results if lower in m.content.lower()]

        # Sort by recency (newest first)
        results.sort(key=lambda m: m.timestamp, reverse=True)

        return results[:q.limit]

    def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        return n

    # ── Session helpers ───────────────────────────────────────────────────

    def get_conversation_context(self, last_n: int = 10) -> List[MemoryItem]:
        """Retrieve the most recent conversation items."""
        q = MemoryQuery(
            tags=["conversation"],
            limit=last_n,
        )
        return self.query(q)

    def add_message(self, role: str, content: str, agent: str = "user") -> str:
        """Convenience: store a conversation turn."""
        item = MemoryItem(
            type=MemoryType.WORKING,
            source=agent,
            content=content,
            tags=["conversation", role],
            metadata={"role": role},
        )
        return self.write(item)

    def snapshot(self) -> List[Dict]:
        """Return all items as dicts (for debugging / export)."""
        return [m.to_dict() for m in self._items.values()]

    # ── Internal ──────────────────────────────────────────────────────────

    def _evict_oldest(self):
        """Remove the oldest item to make room."""
        if not self._items:
            return
        oldest_id = min(self._items, key=lambda k: self._items[k].timestamp)
        del self._items[oldest_id]
