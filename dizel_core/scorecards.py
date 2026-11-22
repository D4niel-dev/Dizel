"""Local scorecards for checkpoint evaluations.

A card is only shown while the file it describes is unchanged: the size and
mtime it was measured at are stored beside it, and a modified checkpoint
invalidates the badge instead of advertising a stale result.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DATA_ROOT = os.path.expanduser(
    os.environ.get("DIZEL_DATA_DIR") or os.path.join(_ROOT, "data")
)
SCORECARDS_DIR = os.path.join(_DATA_ROOT, "scorecards")


def fingerprint(path: str) -> str:
    normalized = os.path.normcase(os.path.abspath(path))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _card_path(path: str) -> str:
    return os.path.join(SCORECARDS_DIR, f"{fingerprint(path)}.json")


def save(path: str, card: dict) -> dict:
    """Store a card for `path`, stamping it with the file it measured."""
    resolved = os.path.abspath(path)
    stat = os.stat(resolved)
    record = dict(card)
    record.update(
        {
            "path": resolved,
            "name": os.path.basename(resolved),
            "size": int(stat.st_size),
            "mtime": int(stat.st_mtime),
            "fingerprint": fingerprint(resolved),
        }
    )
    os.makedirs(SCORECARDS_DIR, exist_ok=True)
    with open(_card_path(resolved), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return record


def load(path: str, fresh: bool = True) -> Optional[dict]:
    """The card for `path`, or None when missing, unreadable, or stale."""
    resolved = os.path.abspath(path)
    try:
        with open(_card_path(resolved), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if fresh:
        try:
            stat = os.stat(resolved)
        except OSError:
            return None
        if int(stat.st_size) != record.get("size") or int(stat.st_mtime) != record.get(
            "mtime"
        ):
            return None
    return record


def summary(card: dict) -> dict:
    """The subset the model switcher and compact reports need."""
    keys = (
        "passed",
        "total",
        "pass_rate",
        "avg_score",
        "tok_per_s",
        "evaluated_at",
        "by_category",
    )
    return {key: card.get(key) for key in keys}


def report(card: dict) -> str:
    """A plain-text report for panels and the CLI."""
    name = card.get("name") or card.get("checkpoint") or "checkpoint"
    passed = card.get("passed", 0)
    total = card.get("total", 0)
    avg = float(card.get("avg_score") or 0)
    speed = float(card.get("tok_per_s") or 0)
    lines = [
        f"Scorecard — {name}",
        "",
        f"{passed} / {total} cases passed · avg {avg:.2f}",
        f"speed ~{speed:.1f} tok/s · evaluated {card.get('evaluated_at', 'unknown')}",
        "",
        "By category:",
    ]
    for category, score in sorted((card.get("by_category") or {}).items()):
        lines.append(f"  {category:<20} {float(score):.2f}")
    return "\n".join(lines)
