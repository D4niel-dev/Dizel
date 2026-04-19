"""
core/tools/web_search.py — Web search via DuckDuckGo.

Provides a simple search_web() function that returns formatted
search results as a string, ready for prompt injection.

Dependency: pip install duckduckgo-search
Lazy-imported so the app doesn't crash if not installed.
"""

import traceback

MAX_RESULTS = 5
MAX_CHARS = 2000
TIMEOUT = 10


def search_web(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.

    Returns a formatted string of top results, or an empty string on failure.
    Each result includes title, snippet, and source URL.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print(
            "[web_search] duckduckgo-search not installed. "
            "Install with: pip install duckduckgo-search"
        )
        return ""

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        if not raw:
            return ""

        parts = []
        total_chars = 0

        for i, r in enumerate(raw, 1):
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()

            entry = f"[{i}] {title}\n{body}\nSource: {href}"

            if total_chars + len(entry) > MAX_CHARS:
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(parts)

    except Exception as e:
        print(f"[web_search] Search failed: {e}")
        traceback.print_exc()
        return ""


def is_available() -> bool:
    """Check if the duckduckgo-search library is installed."""
    try:
        from duckduckgo_search import DDGS  # noqa: F401
        return True
    except ImportError:
        return False
