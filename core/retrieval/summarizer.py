"""
core/retrieval/summarizer.py — Auto-summarize long items.

Extracts the most important parts of a text to fit within
a token budget without losing critical information.
"""

import re
from typing import List


class Summarizer:
    """Lightweight summarizer for long retrieval results."""

    def __init__(self, max_chars: int = 1000):
        self.max_chars = max_chars

    def summarize(self, text: str, query: str = "") -> str:
        """
        Summarize text. If query is provided, extracts sentences
        most relevant to the query. Otherwise, extracts a general summary.
        """
        if len(text) <= self.max_chars:
            return text

        sentences = self._split_sentences(text)
        if not sentences:
            return text[:self.max_chars]

        if query:
            return self._query_focused_summary(sentences, query)
        else:
            return self._extractive_summary(sentences)

    def _split_sentences(self, text: str) -> List[str]:
        """Simple regex-based sentence splitter."""
        # Split on . ! ? followed by space and uppercase
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        return re.split(pattern, text.strip())

    def _query_focused_summary(self, sentences: List[str], query: str) -> str:
        """Score sentences by keyword overlap with query."""
        query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_words:
            return self._extractive_summary(sentences)

        scored = []
        for i, sent in enumerate(sentences):
            sent_words = set(re.findall(r"[a-z0-9]+", sent.lower()))
            overlap = len(query_words & sent_words)
            
            # Boost early sentences slightly to maintain context
            position_boost = 1.0 / (i + 1)
            score = overlap + position_boost
            
            scored.append((sent, score, i))

        # Sort by score, take top, then sort back by original order
        scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        char_count = 0
        for sent, score, idx in scored:
            if char_count + len(sent) > self.max_chars and char_count > 0:
                break
            selected.append((sent, idx))
            char_count += len(sent)
            
        selected.sort(key=lambda x: x[1])
        return " ... ".join(s for s, i in selected)

    def _extractive_summary(self, sentences: List[str]) -> str:
        """Take beginning and end of text if no query provided."""
        if not sentences:
            return ""
            
        result = []
        char_count = 0
        
        # Always take the first sentence
        first = sentences[0]
        result.append(first)
        char_count += len(first)
        
        if len(sentences) > 1:
            # Try to add the last sentence if it fits
            last = sentences[-1]
            if char_count + len(last) + 5 <= self.max_chars:
                if len(sentences) > 2:
                    result.append("...")
                result.append(last)
                
        return " ".join(result)
