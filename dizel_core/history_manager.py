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
_ROOT      = os.path.dirname(_HERE)           # repository root
# Sessions live in data/, not inside the package, so they survive refactors.
# Packaged builds can point this at Electron's user-data directory.
_DATA_ROOT = os.path.expanduser(
    os.environ.get("DIZEL_DATA_DIR") or os.path.join(_ROOT, "data")
)
HISTORY_DIR = os.path.join(_DATA_ROOT, "history")


def _ensure_dir() -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _safe_filename(text: str, max_len: int = 40) -> str:
    """Turn arbitrary text into a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:max_len] or "untitled"


_FILE_BLOCK = re.compile(r"<file\b[^>]*>.*?</file>", re.DOTALL | re.IGNORECASE)


def _without_file_blocks(text: str) -> str:
    """Drop attached-file payloads before a title is read from the text.

    The GUI folds each attachment into the prompt as a <file> block, so the first
    user turn of an attachment-only message is 6,000 characters of file contents —
    which produced sidebar titles like `<file name="ENT403 - PC…`.
    """
    return _FILE_BLOCK.sub("", text or "").strip()


_GREETINGS = {
    "hi", "hi there", "hello", "hello there", "hey", "hey there", "yo", "sup",
    "test", "testing", "hii", "hiii", "good morning", "good evening",
}


def _is_thin(title: str) -> bool:
    """True for a title that tells you nothing: a greeting, or one short word."""
    text = (title or "").strip()
    if not text or text == "New Chat":
        return True
    if text.lower().strip(" .!?,") in _GREETINGS:
        return True
    # A title that starts with an attachment block was derived before file
    # payloads were stripped — re-derive it rather than leave it in the sidebar.
    if text.lower().startswith("<file"):
        return True
    return len(text.split()) < 3


def title_from_messages(messages: List[Dict], max_len: int = 48, created: str = "") -> str:
    """A title for a conversation, preferring one that says what it was about.

    Taking the first user message alone gives a sidebar full of "Hi!" — the
    greeting is rarely the subject. Walk the user's turns until one says
    something, then fall back to the assistant, then to the date. It should never
    hand back a greeting.
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        candidate = derive_title(
            _without_file_blocks(message.get("content", "")), max_len
        )
        if not _is_thin(candidate):
            return candidate

    for message in messages:
        if message.get("role") == "assistant":
            reply = derive_title(
                _without_file_blocks(message.get("content", "")), max_len
            )
            if not _is_thin(reply):
                return reply
            break

    # Nothing quotable. A dated "Chat" beats a sidebar of identical "Hi!" rows.
    if created:
        try:
            stamp = datetime.fromisoformat(created).strftime("%d %b")
            return f"Chat · {stamp}"
        except Exception:
            pass
    return "Chat"


def derive_title(text: str, max_len: int = 48) -> str:
    """
    A readable title from the first user message.

    The old rule was `content[:45]`, which cut mid-word ("Explain what a vari")
    and kept markdown noise. This takes the first line, strips the common markup
    and breaks at a word boundary.
    """
    line = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    if not line:
        return "New Chat"

    line = re.sub(r"^[#>\-*\s]+", "", line)          # leading heading/bullet marks
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # [label](url) -> label
    line = re.sub(r"[*_`]+", "", line)               # emphasis and code ticks
    line = re.sub(r"\s+", " ", line).strip()

    if not line:
        return "New Chat"
    if len(line) <= max_len:
        return line

    cut = line[:max_len].rsplit(" ", 1)[0].rstrip(" ,.;:-")
    return f"{cut or line[:max_len].rstrip()}…"


# ── Public API ────────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_session(
    messages:   List[Dict],
    session_id: str = None,
    title:      str = None,
    pinned:     bool = None,
    tags:       List[str] = None,
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

    existing_data = load_session(session_id)

    # A title the user typed is final. One the app derived can still improve —
    # the first save happens before there is a reply to learn anything from, so
    # "Hi!" would otherwise be permanent.
    if title is None:
        existing_title = (existing_data or {}).get("title") or ""
        auto = (existing_data or {}).get("title_auto", True)
        if existing_title and not auto:
            title = existing_title
            title_is_auto = False
        else:
            title = title_from_messages(messages)
            title_is_auto = True
    else:
        title_is_auto = False

    is_pinned = False
    existing_tags = []
    if existing_data:
        is_pinned = existing_data.get("pinned", False)
        existing_tags = existing_data.get("tags", [])
        # Preserve original creation time if available
        created_time = existing_data.get("created", datetime.now().isoformat(timespec="seconds"))
    else:
        created_time = datetime.now().isoformat(timespec="seconds")
    
    if pinned is not None:
        is_pinned = pinned

    if tags is not None:
        existing_tags = list(set(existing_tags + tags))

    data = {
        "id":       session_id,
        "title":    title,
        "title_auto": title_is_auto,
        "created":  created_time,
        "messages": messages,
        "pinned":   is_pinned,
        "tags":     existing_tags,
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
            title = data.get("title", "Untitled")
            # An auto title that still says "Hi!" is re-derived here rather than
            # rewritten to disk — listing is a read, and the next save persists it.
            if data.get("title_auto", True) and _is_thin(title):
                title = title_from_messages(
                    data.get("messages", []), created=data.get("created", "")
                )
            sessions.append({
                "id":      data.get("id", ""),
                "title":   title,
                "created": data.get("created", ""),
                "preview": preview,
                "pinned":  data.get("pinned", False),
                "archived": data.get("archived", False),
                "tags":    data.get("tags", []),
            })
        except Exception:
            continue

    sessions.sort(key=lambda s: (not s.get("pinned", False), s["created"]), reverse=True)
    return sessions


def add_tag_to_session(session_id: str, tag: str) -> bool:
    """Add a single tag to a session by ID."""
    _ensure_dir()
    tag = tag.strip().lower()
    if tag.startswith("#"):
        tag = tag[1:]
    
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                tags = set(data.get("tags", []))
                tags.add(tag)
                data["tags"] = list(tags)
                
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception:
                pass
    return False


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


def toggle_archive_session(session_id: str) -> bool:
    """Toggle archived state of a session and return the new status."""
    _ensure_dir()
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                new_archived = not data.get("archived", False)
                data["archived"] = new_archived

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return new_archived
            except Exception:
                pass
    return False


def truncate_session(session_id: str, keep: int) -> bool:
    """
    Drop every message after the first `keep` ones.

    Used by edit-and-resend: rewinding the file before the new text is sent means
    the model never sees the message that was just replaced.
    """
    _ensure_dir()
    if keep < 0:
        return False

    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                messages = data.get("messages", [])
                if keep >= len(messages):
                    return True
                data["messages"] = messages[:keep]
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception:
                return False
    return False


def rename_session(session_id: str, title: str) -> bool:
    """Rename a session's title.  Returns True if the session was found."""
    _ensure_dir()
    clean = (title or "").strip()[:80]
    if not clean:
        return False

    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith(session_id) and fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["title"] = clean
                data["title_auto"] = False
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception:
                return False
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
