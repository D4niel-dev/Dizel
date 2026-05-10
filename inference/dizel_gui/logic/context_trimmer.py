"""
dizel_ui/logic/context_trimmer.py
─────────────────────────────────
Smart context window management for Dizel.

Trims old conversation history when context budget is exceeded,
preserving recent messages and any turns with attachments/tool data.
"""

from typing import List, Dict, Callable, Optional


def estimate_tokens(text: str) -> int:
    """
    Fast token count estimate without running the tokenizer.
    Approximation: ~4 characters per token for English text.
    """
    return max(1, len(text) // 4)


def estimate_history_tokens(history: List[Dict]) -> int:
    """Estimate total token count across all messages in history."""
    total = 0
    for msg in history:
        # Role token overhead (~5 tokens per message for role + end markers)
        total += 5
        total += estimate_tokens(msg.get("content", ""))
    return total


def _has_attachments(msg: Dict) -> bool:
    """Check if a message has tool output, attachments, or image data."""
    if msg.get("attachments"):
        return True
    content = msg.get("content", "")
    # Preserve messages that contain tool/file/image context markers
    if any(marker in content for marker in ["[file:", "[image:", "[search:", "[tool:"]):
        return True
    return False


def trim_context_if_needed(
    history: List[Dict],
    max_context_tokens: int,
    system_prompt_tokens: int = 50,
) -> tuple:
    """
    Trim conversation history to fit within the context token budget.

    Strategy:
    1. Keep the most recent messages.
    2. Never drop messages with attachments/tool context.
    3. Drop oldest user+assistant pairs first.
    4. If still over budget after aggressive trimming, prepend a summary note.

    Parameters
    ----------
    history : list of message dicts ({"role": ..., "content": ...})
    max_context_tokens : maximum tokens allowed for the conversation
    system_prompt_tokens : estimated tokens used by the system prompt

    Returns
    -------
    (trimmed_history, num_dropped_messages)
    """
    if not history:
        return history, 0

    available = max_context_tokens - system_prompt_tokens
    current_tokens = estimate_history_tokens(history)

    if current_tokens <= available:
        return list(history), 0

    # Work with a mutable copy
    trimmed = list(history)
    dropped = 0

    # Phase 1: Drop oldest non-attachment pairs
    while estimate_history_tokens(trimmed) > available and len(trimmed) > 2:
        # Find the oldest message that doesn't have attachments
        drop_idx = None
        for i in range(len(trimmed) - 2):  # never touch the last 2
            if not _has_attachments(trimmed[i]):
                drop_idx = i
                break

        if drop_idx is None:
            # All remaining messages have attachments — can't drop more safely
            break

        trimmed.pop(drop_idx)
        dropped += 1

    # Phase 2: If still over budget, do a hard summary
    if estimate_history_tokens(trimmed) > available and len(trimmed) > 4:
        # Keep the last 4 messages, summarize what was dropped
        kept = trimmed[-4:]
        extra_dropped = len(trimmed) - 4
        dropped += extra_dropped

        summary_note = {
            "role": "system",
            "content": f"[Earlier conversation context was summarized. {dropped} messages were trimmed to fit the context window.]",
        }
        trimmed = [summary_note] + kept

    return trimmed, dropped


def summarize_old_context(history: List[Dict], max_messages: int = 6) -> Optional[str]:
    """
    Generate a concise text summary of older conversation messages.

    This is a lightweight extractive summary (no model call).
    It picks the first sentence of each older message to give
    the model a hint of what was discussed previously.
    """
    if len(history) <= max_messages:
        return None

    old_messages = history[:-max_messages]
    snippets = []

    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        if not content or role == "system":
            continue

        # Take first sentence or first 80 chars
        first_sentence = content.split(".")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."
        snippets.append(f"{role}: {first_sentence}")

    if not snippets:
        return None

    return "Previous context: " + " | ".join(snippets[:5])
