"""
dizel_ui/logic/history_manager.py
──────────────────────────────────
Manages saving and loading chat sessions to/from disk.

Each session is stored as a JSON file under:
    dizel_ui/history/<timestamp>_<title>.json

Format:
{
    "id":       "20260314_143022",
    "title":    "First message trimmed to 40 chars",
    "created":  "2026-03-14T14:30:22",
    "messages": [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]
}
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional


# Default directory for chat history files (relative to this file's parent)
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)           # dizel_ui/
HISTORY_DIR = os.path.join(_ROOT, "history")


def _ensure_dir() -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _safe_filename(text: str, max_len: int = 40) -> str:
    """Turn arbitrary text into a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:max_len] or "untitled"


# ── Public API ────────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_session(
    messages:   List[Dict],
    session_id: str = None,
    title:      str = None,
    pinned:     bool = None,
) -> str:
    """
    Save a conversation to disk.

    Parameters
    ----------
    messages   : list of {"role": ..., "content": ...} dicts
    session_id : reuse an existing id to overwrite that session
    title      : human-readable title; auto-derived from first user msg if omitted

    Returns
    -------
    The session_id used (useful when creating a new session).
    """
    _ensure_dir()

    if session_id is None:
        session_id = new_session_id()

    # Derive title from first user message if not provided
    if title is None:
        for msg in messages:
            if msg.get("role") == "user":
                title = msg["content"][:45].strip()
                break
        title = title or "New Chat"

    is_pinned = False
    existing_data = load_session(session_id)
    if existing_data:
        is_pinned = existing_data.get("pinned", False)
        # Preserve original creation time if available
        created_time = existing_data.get("created", datetime.now().isoformat(timespec="seconds"))
    else:
        created_time = datetime.now().isoformat(timespec="seconds")
    
    if pinned is not None:
        is_pinned = pinned

    data = {
        "id":       session_id,
        "title":    title,
        "created":  created_time,
        "messages": messages,
        "pinned":   is_pinned,
    }

    slug = _safe_filename(title)
    fname = f"{session_id}_{slug}.json"
    path  = os.path.join(HISTORY_DIR, fname)

    # Remove any previous file for this session_id (title may have changed)
    for existing in os.listdir(HISTORY_DIR):
        if existing.startswith(session_id) and existing.endswith(".json"):
            os.remove(os.path.join(HISTORY_DIR, existing))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return session_id


def load_session(session_id: str) -> Optional[Dict]:
    """Load a session by id.  Returns None if not found."""
    _ensure_dir()
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def list_sessions() -> List[Dict]:
    """
    Return all saved sessions sorted newest-first.

    Each entry: {"id": ..., "title": ..., "created": ..., "preview": ...}
    """
    _ensure_dir()
    sessions = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Build a one-line preview from the first assistant message
            preview = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "assistant":
                    preview = msg["content"][:60].replace("\n", " ")
                    break
            sessions.append({
                "id":      data.get("id", ""),
                "title":   data.get("title", "Untitled"),
                "created": data.get("created", ""),
                "preview": preview,
                "pinned":  data.get("pinned", False),
            })
        except Exception:
            continue

    sessions.sort(key=lambda s: (not s.get("pinned", False), s["created"]), reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session file.  Returns True if deleted."""
    _ensure_dir()
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            os.remove(os.path.join(HISTORY_DIR, fname))
            return True
    return False


def toggle_pin_session(session_id: str) -> bool:
    """Toggle pin state of a session and return the new status."""
    _ensure_dir()
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                new_pinned = not data.get("pinned", False)
                data["pinned"] = new_pinned
                
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return new_pinned
            except Exception:
                pass
    return False


def delete_all_sessions() -> int:
    """Delete all sessions.  Returns count deleted."""
    _ensure_dir()
    count = 0
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith(".json"):
            os.remove(os.path.join(HISTORY_DIR, fname))
            count += 1
    return count
