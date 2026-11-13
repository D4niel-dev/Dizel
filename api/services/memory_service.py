"""Conversation summarising and recall.

`summarize_old_context` already existed in the PySide6 logic layer but only that
GUI ever called it — the API chat path never did. These are thin wrappers so the
desktop app can use the same extractive summariser.
"""

from typing import Any, Dict, List

from dizel_core import history_manager
from dizel_core.context_trimmer import summarize_old_context


class MemoryService:
    @staticmethod
    def summarize(messages: List[Dict[str, Any]]) -> str:
        """
        Extractive summary of a conversation. No model call, so it's fast and
        deterministic — it can't hallucinate a topic that wasn't discussed.
        """
        clean = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        return summarize_old_context(clean, max_messages=2) or ""

    @staticmethod
    def recent(limit: int = 3) -> List[Dict[str, Any]]:
        """
        The most recent conversations with a short human line each.

        Deliberately *not* `summarize_old_context`: that produces a prompt
        fragment ("Previous context: user: … | assistant: …") meant for the model,
        and reading it in the UI is noise. Here the useful line is where the
        conversation ended — the last reply.
        """
        out: List[Dict[str, Any]] = []

        for entry in history_manager.list_sessions():
            if entry.get("archived"):
                continue

            data = history_manager.load_session(entry["id"]) or {}
            messages = data.get("messages", [])
            if not messages:
                continue

            last = next(
                (
                    m.get("content", "")
                    for m in reversed(messages)
                    if m.get("role") == "assistant" and m.get("content", "").strip()
                ),
                "",
            )
            flat = " ".join(str(last or entry.get("preview", "")).split())
            snippet = flat[:87].rstrip() + "…" if len(flat) > 90 else flat
            if not snippet:
                continue

            out.append(
                {
                    "id": entry["id"],
                    "title": entry.get("title", "Untitled"),
                    "summary": snippet,
                    "created": entry.get("created", ""),
                }
            )
            if len(out) >= limit:
                break

        return out
