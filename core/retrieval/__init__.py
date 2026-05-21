"""
core/retrieval — Dizel Knowledge Retrieval Layer.

Provides local-first, dependency-free hybrid search across
all retrievable ecosystem data using BM25 and semantic vectors.

Usage:
    from core.retrieval import HybridIndex, IndexEntry, Ranker

    index = HybridIndex()
    index.add(IndexEntry(id="1", content="Python is cool", source="file"))
    
    results = index.search("python")
    results = Ranker().rank(results)
"""

from .index import BaseIndex, IndexEntry
from .search import HybridIndex
from .ranker import Ranker
from .summarizer import Summarizer

__all__ = [
    "BaseIndex",
    "IndexEntry",
    "HybridIndex",
    "Ranker",
    "Summarizer",
]
