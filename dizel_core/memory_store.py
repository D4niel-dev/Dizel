"""Central per-project memory for Dizel.

Entries live in the app data directory, keyed by the resolved project path, so
nothing is written into the user's repository. Writes are small and atomic;
the store enforces entry and character caps rather than growing without bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
MEMORY_DIR = os.path.join(
    os.path.expanduser(os.environ.get("DIZEL_DATA_DIR") or os.path.join(_ROOT, "data")),
    "memory",
)

MAX_ENTRIES = 100
MAX_CHARS = 12000

PathLike = Union[str, Path, None]


def _root(project_root: PathLike = None) -> Path:
    return Path(project_root or os.getcwd()).expanduser().resolve()


def _path(root: Path) -> Path:
    key = hashlib.sha1(os.path.normcase(str(root)).encode("utf-8")).hexdigest()
    return Path(MEMORY_DIR) / f"{key}.json"


def _load(root: Path) -> dict:
    try:
        data = json.loads(_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return data
    return {"version": 1, "project": str(root), "entries": []}


def _save(root: Path, data: dict) -> None:
    directory = Path(MEMORY_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(root)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, target)


def entries(project_root: PathLike = None) -> List[dict]:
    return list(_load(_root(project_root)).get("entries", []))


def add(
    text: str, project_root: PathLike = None, source: str = "user"
) -> dict:
    content = str(text or "").strip()
    if not content:
        raise ValueError("Memory text is empty.")
    root = _root(project_root)
    data = _load(root)
    rows = data["entries"]
    if len(rows) >= MAX_ENTRIES:
        raise ValueError(
            f"Project memory is full ({MAX_ENTRIES} entries) — remove some first."
        )
    used = sum(len(str(row.get("text", ""))) for row in rows)
    if used + len(content) > MAX_CHARS:
        raise ValueError(
            f"Project memory is full ({MAX_CHARS:,} characters) — remove some first."
        )
    entry = {
        "id": uuid.uuid4().hex[:8],
        "text": content,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(source or "user"),
    }
    rows.append(entry)
    _save(root, data)
    return entry


def remove(entry_id: str, project_root: PathLike = None) -> bool:
    """Remove by id, or by a 1-based index for the CLI hand."""
    root = _root(project_root)
    data = _load(root)
    rows = data["entries"]
    selector = str(entry_id or "").strip()
    keep = [row for row in rows if str(row.get("id")) != selector]
    if len(keep) == len(rows):
        try:
            index = int(selector) - 1
        except (TypeError, ValueError):
            return False
        if not 0 <= index < len(rows):
            return False
        keep = rows[:index] + rows[index + 1 :]
    data["entries"] = keep
    _save(root, data)
    return True


def clear(project_root: PathLike = None) -> int:
    root = _root(project_root)
    data = _load(root)
    count = len(data["entries"])
    data["entries"] = []
    _save(root, data)
    return count
