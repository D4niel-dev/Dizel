"""
core/retrieval/search.py — Hybrid search (BM25 + cosine similarity).

Lightweight, local-first search that combines keyword matching
(BM25-inspired) with TF-IDF cosine similarity for ranking.
No external dependencies required.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set

from .index import BaseIndex, IndexEntry


_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "this", "that", "it", "i", "you", "he",
    "she", "we", "they", "my", "your", "his", "her", "its", "our",
    "do", "does", "did", "will", "would", "could", "should", "can",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class HybridIndex(BaseIndex):
    """
    In-memory hybrid search index combining BM25 keyword scoring
    with TF-IDF cosine similarity.
    """

    def __init__(self, bm25_weight: float = 0.6, semantic_weight: float = 0.4):
        self._entries: Dict[str, IndexEntry] = {}
        self._doc_tokens: Dict[str, List[str]] = {}     # id → tokens
        self._doc_freq: Counter = Counter()               # token → doc count
        self._avg_dl: float = 0.0                         # average doc length
        self._bm25_w = bm25_weight
        self._semantic_w = semantic_weight

    # ── BaseIndex interface ───────────────────────────────────────────────

    def add(self, entry: IndexEntry):
        # Remove old version if exists
        if entry.id in self._entries:
            self._remove_from_index(entry.id)

        self._entries[entry.id] = entry
        tokens = _tokenize(entry.content)
        self._doc_tokens[entry.id] = tokens

        # Update document frequencies
        for token in set(tokens):
            self._doc_freq[token] += 1

        # Update average document length
        total_tokens = sum(len(t) for t in self._doc_tokens.values())
        self._avg_dl = total_tokens / max(len(self._entries), 1)

    def search(self, query: str, limit: int = 10) -> List[IndexEntry]:
        if not self._entries:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            # No useful query tokens — return by recency
            entries = sorted(self._entries.values(),
                             key=lambda e: e.timestamp, reverse=True)
            return entries[:limit]

        scored: List[tuple] = []
        n_docs = len(self._entries)

        for doc_id, entry in self._entries.items():
            doc_tokens = self._doc_tokens.get(doc_id, [])
            if not doc_tokens:
                continue

            bm25 = self._bm25_score(query_tokens, doc_tokens, n_docs)
            cosine = self._cosine_score(query_tokens, doc_tokens)
            combined = (self._bm25_w * bm25) + (self._semantic_w * cosine)

            entry.score = round(combined, 4)
            scored.append((entry, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    def remove(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            return False
        self._remove_from_index(entry_id)
        return True

    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self._doc_tokens.clear()
        self._doc_freq.clear()
        self._avg_dl = 0.0

    # ── Scoring ───────────────────────────────────────────────────────────

    def _bm25_score(self, query_tokens: List[str],
                    doc_tokens: List[str], n_docs: int,
                    k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 scoring for keyword relevance."""
        dl = len(doc_tokens)
        doc_tf = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            df = self._doc_freq.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(self._avg_dl, 1)))
            score += idf * tf_norm

        return score

    def _cosine_score(self, query_tokens: List[str],
                      doc_tokens: List[str]) -> float:
        """TF-IDF cosine similarity."""
        q_tf = Counter(query_tokens)
        d_tf = Counter(doc_tokens)
        all_terms = set(q_tf) | set(d_tf)
        n_docs = max(len(self._entries), 1)

        dot = mag_q = mag_d = 0.0
        for term in all_terms:
            idf = math.log(n_docs / (1 + self._doc_freq.get(term, 0)))
            wq = q_tf.get(term, 0) * idf
            wd = d_tf.get(term, 0) * idf
            dot += wq * wd
            mag_q += wq * wq
            mag_d += wd * wd

        if mag_q == 0 or mag_d == 0:
            return 0.0
        return dot / (math.sqrt(mag_q) * math.sqrt(mag_d))

    def _remove_from_index(self, entry_id: str):
        tokens = self._doc_tokens.pop(entry_id, [])
        for token in set(tokens):
            self._doc_freq[token] = max(0, self._doc_freq[token] - 1)
        del self._entries[entry_id]
