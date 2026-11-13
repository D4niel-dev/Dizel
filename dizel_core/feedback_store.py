from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable

_LOCK = threading.Lock()
MAX_NOTE = 2000
MAX_RESPONSE = 20000
MAX_PROMPT = 20000
MAX_REASONS = 8
MAX_REASON = 80


def feedback_path() -> Path:
    root = Path(
        os.environ.get("DIZEL_DATA_DIR")
        or Path(__file__).resolve().parent.parent / "data"
    )
    return root / "feedback" / "feedback.jsonl"


def record_feedback(
    *,
    rating: str,
    session_id: str = "",
    message_key: str = "",
    reasons: Iterable[str] = (),
    note: str = "",
    response: str = "",
    prompt: str = "",
    model: str = "",
    provider: str = "",
) -> Path:
    """Append one feedback event to the local JSONL file.

    Append-only keeps a failed write from corrupting earlier feedback. A later
    event for the same message key supersedes an earlier one when the data is
    read for training or review.
    """
    if rating not in ("up", "down", "removed"):
        raise ValueError("rating must be up, down, or removed")

    clean_reasons = [
        " ".join(str(reason).split())[:MAX_REASON]
        for reason in reasons
        if str(reason).strip()
    ][:MAX_REASONS]
    entry = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rating": rating,
        "session_id": str(session_id or "")[:120],
        "message_key": str(message_key or "")[:120],
        "reasons": clean_reasons,
        "note": str(note or "").strip()[:MAX_NOTE],
        "response": str(response or "")[:MAX_RESPONSE],
        "prompt": str(prompt or "")[:MAX_PROMPT],
        "model": str(model or "")[:120],
        "provider": str(provider or "")[:80],
    }

    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return path
