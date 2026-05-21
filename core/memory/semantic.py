"""
core/memory/semantic.py — Long-term semantic memory with indexing.

Stores stable knowledge: user preferences, project decisions,
reusable facts. Uses JSON-backed storage with simple TF-IDF
similarity for retrieval (no heavy vector DB dependencies).
"""

import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set

from .schema import MemoryItem, MemoryType
from .store import MemoryQuery, MemoryStore

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "memory", "semantic.json"
)

# Simple stopwords for TF-IDF (kept minimal)
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "this", "that", "it", "i", "you", "he",
    "she", "we", "they", "my", "your", "his", "her", "its", "our",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, split, remove stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class SemanticMemory(MemoryStore):
    """JSON-backed semantic memory with lightweight TF-IDF search."""

    def __init__(self, file_path: Optional[str] = None):
        self._path = file_path or os.path.normpath(_DEFAULT_PATH)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._items: Dict[str, MemoryItem] = {}
        self._idf_cache: Dict[str, float] = {}
        self._dirty = False
        self._load()

    # ── MemoryStore interface ─────────────────────────────────────────────

    def write(self, item: MemoryItem) -> str:
        item.type = MemoryType.SEMANTIC
        self._items[item.id] = item
        self._dirty = True
        self._invalidate_idf()
        self._save()
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

        # Semantic ranking via TF-IDF similarity
        if q.text:
            scored = self._rank_by_similarity(q.text, results)
            # Update relevance scores on the items
            for item, score in scored:
                item.relevance_score = score
            results = [item for item, _ in scored if _ > 0.0]
        else:
            # Default: sort by confidence then recency
            results.sort(key=lambda m: (m.confidence, m.timestamp.timestamp()), reverse=True)

        return results[:q.limit]

    def delete(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            self._dirty = True
            self._invalidate_idf()
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        self._dirty = True
        self._invalidate_idf()
        self._save()
        return n

    # ── Semantic-specific helpers ─────────────────────────────────────────

    def store_knowledge(self, content: str, tags: List[str],
                        source: str = "dizel", confidence: float = 0.85) -> str:
        """Convenience: store a piece of reusable knowledge."""
        item = MemoryItem(
            type=MemoryType.SEMANTIC,
            source=source,
            content=content,
            tags=["knowledge"] + tags,
            confidence=confidence,
        )
        return self.write(item)

    def store_preference(self, key: str, value: str, source: str = "user") -> str:
        """Store a user preference as semantic memory."""
        item = MemoryItem(
            type=MemoryType.SEMANTIC,
            source=source,
            content=f"{key}: {value}",
            tags=["preference"],
            metadata={"pref_key": key, "pref_value": value},
            confidence=1.0,
        )
        return self.write(item)

    def find_similar(self, text: str, top_k: int = 5) -> List[MemoryItem]:
        """Find the most semantically similar items to a query string."""
        return self.query(MemoryQuery(text=text, limit=top_k))

    # ── TF-IDF ranking ────────────────────────────────────────────────────

    def _rank_by_similarity(self, query_text: str,
                            candidates: List[MemoryItem]) -> List[tuple]:
        """Rank candidates by TF-IDF cosine similarity to query."""
        if not candidates:
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return [(m, 0.0) for m in candidates]

        # Build IDF over all items (cached)
        self._ensure_idf()
        query_tf = Counter(query_tokens)

        scored = []
        for item in candidates:
            doc_tokens = _tokenize(item.content)
            if not doc_tokens:
                scored.append((item, 0.0))
                continue

            doc_tf = Counter(doc_tokens)
            # Cosine similarity using TF-IDF weights
            score = self._cosine_tfidf(query_tf, doc_tf)
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _ensure_idf(self):
        if self._idf_cache:
            return
        n_docs = max(len(self._items), 1)
        doc_freq: Counter = Counter()
        for item in self._items.values():
            unique_tokens = set(_tokenize(item.content))
            for token in unique_tokens:
                doc_freq[token] += 1
        self._idf_cache = {
            token: math.log(n_docs / (1 + freq))
            for token, freq in doc_freq.items()
        }

    def _cosine_tfidf(self, tf_a: Counter, tf_b: Counter) -> float:
        all_terms = set(tf_a) | set(tf_b)
        dot = 0.0
        mag_a = 0.0
        mag_b = 0.0
        for term in all_terms:
            idf = self._idf_cache.get(term, 1.0)
            wa = tf_a.get(term, 0) * idf
            wb = tf_b.get(term, 0) * idf
            dot += wa * wb
            mag_a += wa * wa
            mag_b += wb * wb
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))

    def _invalidate_idf(self):
        self._idf_cache.clear()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                item = MemoryItem.from_dict(d)
                self._items[item.id] = item
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, start fresh

    def _save(self):
        if not self._dirty:
            return
        data = [item.to_dict() for item in self._items.values()]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._dirty = False
