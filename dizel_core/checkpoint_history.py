"""Recently loaded checkpoints, newest first, shared by both clients."""

from __future__ import annotations

from pathlib import Path

from dizel_core.config_manager import ConfigManager

KEY = "checkpoint_history"
LIMIT = 8


def _resolved(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


def _entries(cfg: dict) -> list[str]:
    raw = cfg.get(KEY)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item.strip()]


def record(path: str) -> None:
    clean = str(path or "").strip()
    if not clean:
        return
    resolved = _resolved(clean)
    cfg = ConfigManager.load()
    history = [entry for entry in _entries(cfg) if _resolved(entry) != resolved]
    history.insert(0, resolved)
    cfg[KEY] = history[:LIMIT]
    ConfigManager.save(cfg)


def recent(limit: int = LIMIT) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in _entries(ConfigManager.load()):
        resolved = _resolved(entry)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered[:limit]
