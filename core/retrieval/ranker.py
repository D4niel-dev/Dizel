"""
core/retrieval/ranker.py — Recency + relevance + confidence scoring.

Provides a unified ranking function to combine multiple signals
into a final score for retrieval.
"""

import math
from datetime import datetime
from typing import List

from .index import IndexEntry


class Ranker:
    """Combines multiple signals (relevance, recency, confidence) into a final score."""

    def __init__(self, relevance_w: float = 0.5, recency_w: float = 0.3, confidence_w: float = 0.2):
        self.w_rel = relevance_w
        self.w_rec = recency_w
        self.w_conf = confidence_w

    def rank(self, entries: List[IndexEntry], query_time: datetime = None) -> List[IndexEntry]:
        """Rank entries based on combined signals."""
        if not entries:
            return []
            
        now = query_time or datetime.utcnow()
        
        # We need to normalize relevance scores to 0-1 range to combine them properly.
        max_rel = max((e.score for e in entries), default=0.0)
        
        scored_entries = []
        for entry in entries:
            # Normalize relevance (if max_rel > 0)
            norm_rel = (entry.score / max_rel) if max_rel > 0 else 0.0
            
            # Calculate recency score (exponential decay)
            # Half-life of roughly 30 days (in seconds: 30 * 24 * 60 * 60 = 2592000)
            age_seconds = (now - entry.timestamp).total_seconds()
            age_seconds = max(0, age_seconds)
            recency_score = math.exp(-age_seconds / 2592000.0)
            
            # Get confidence (if available in metadata, else default 1.0)
            confidence = entry.metadata.get("confidence", 1.0)
            
            # Combined score
            final_score = (
                (self.w_rel * norm_rel) + 
                (self.w_rec * recency_score) + 
                (self.w_conf * confidence)
            )
            
            # Update the score (can store old score in metadata if needed)
            entry.metadata["raw_relevance"] = entry.score
            entry.score = final_score
            scored_entries.append((entry, final_score))
            
        # Sort by final score descending
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored_entries]
