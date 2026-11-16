"""Shared policy for standing write trust.

Both UIs and the sidecar decide auto-approval here so the tiers, the root
bound, and the never-auto-delete rule cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

from dizel_core.config_manager import ConfigManager
from dizel_core.tools import FileChange

WRITE_TOOLS = frozenset({"write_file", "edit_file"})

ALLOW_ONCE = "allow_once"
ALWAYS_SESSION = "always_session"
ALWAYS_PERSISTENT = "always_persistent"
DENY = "deny"
DECISIONS = frozenset({ALLOW_ONCE, ALWAYS_SESSION, ALWAYS_PERSISTENT, DENY})


def _in_root(path: str, root: Path) -> bool:
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def should_auto_approve(
    change: FileChange,
    session_id: str,
    *,
    session_grants: set[str],
    persistent: bool,
    root: Path,
) -> bool:
    if change.tool not in WRITE_TOOLS:
        return False
    if not (persistent or (bool(session_id) and session_id in session_grants)):
        return False
    return _in_root(change.path, root)


def approved_for(decision: str) -> bool:
    return decision in DECISIONS and decision != DENY


def record_session_grant(grants: set[str], session_id: str) -> None:
    if session_id:
        grants.add(session_id)


def revoke_session_grant(grants: set[str], session_id: str) -> None:
    grants.discard(session_id)


def persistent_enabled() -> bool:
    try:
        cfg = ConfigManager.load()
        return bool(cfg.get("tools", {}).get("always_allow_writes", False))
    except Exception:
        return False


def set_persistent(enabled: bool) -> None:
    cfg = ConfigManager.load()
    if not isinstance(cfg.get("tools"), dict):
        cfg["tools"] = {}
    cfg["tools"]["always_allow_writes"] = bool(enabled)
    ConfigManager.save(cfg)
