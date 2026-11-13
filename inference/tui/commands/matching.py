"""Small fuzzy scorer shared by the slash drawer and status hints.

Lower is better. Exact and prefix matches beat substring matches, which beat
gapped subsequences ("smz" -> summarize). Aliases participate equally.
"""

from __future__ import annotations

from typing import Iterable, Optional

INFINITY = float("inf")


def subsequence_gaps(text: str, query: str) -> Optional[int]:
    cursor = 0
    gaps = 0
    last = -1
    for char in query:
        found = text.find(char, cursor)
        if found < 0:
            return None
        if last >= 0:
            gaps += found - last - 1
        last = found
        cursor = found + 1
    return gaps


def match_score(text: str, query: str) -> float:
    if not query:
        return 0.0
    text = text.lower()
    if text == query:
        return 0.0
    if text.startswith(query):
        return 1.0 + (len(text) - len(query)) * 0.01
    index = text.find(query)
    if index >= 0:
        return 3.0 + index * 0.1
    gaps = subsequence_gaps(text, query)
    return 10.0 + gaps if gaps is not None else INFINITY


def command_score(name: str, aliases: Iterable[str], query: str) -> float:
    return min([match_score(name, query)] + [match_score(alias, query) for alias in aliases])
