"""
summarizer.py — Truncate or summarize text that exceeds Dizel's context window.
"""

# Rough estimate: 1 token ≈ 4 characters for English text
MAX_CONTEXT_CHARS = 6000   # ~1500 tokens, leaves room for system + history
HEAD_RATIO = 0.7           # keep 70% from the start
TAIL_RATIO = 0.2           # keep 20% from the end (10% is the marker)


def estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return len(text) // 4


def needs_truncation(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> bool:
    return len(text) > max_chars


def truncate_text(
    text: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple:
    """
    Truncate long text, keeping head + tail with a marker in between.

    Returns (truncated_text, was_truncated, original_chars, final_chars).
    """
    if not needs_truncation(text, max_chars):
        return text, False, len(text), len(text)

    original_len = len(text)
    head_size = int(max_chars * HEAD_RATIO)
    tail_size = int(max_chars * TAIL_RATIO)

    marker = (
        f"\n\n[... {original_len - head_size - tail_size:,} characters trimmed ...]\n\n"
    )

    truncated = text[:head_size] + marker + text[-tail_size:]
    return truncated, True, original_len, len(truncated)


def summarize_if_needed(
    text: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple:
    """
    Primary entry point. For now, this just truncates.
    Future: use Dizel itself to generate a summary.

    Returns (processed_text, note_string).
    """
    result, was_truncated, orig, final = truncate_text(text, max_chars)

    if was_truncated:
        note = (
            f"Content trimmed from {orig:,} to {final:,} characters "
            f"(~{orig // 4:,} → ~{final // 4:,} tokens)."
        )
        return result, note

    return result, ""
