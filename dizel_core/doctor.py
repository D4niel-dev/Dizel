"""Fast, offline runtime checks shared by every surface.

Nothing here touches the network, loads model weights, or prints a secret.
The report is a list of rows so the clients can render it plainly and the
launcher can merge the JSON into its install report.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dizel_core import config_manager, history_manager

CORE_MODULES = ("fastapi", "uvicorn", "pydantic", "httpx")
STATUSES = ("ok", "warn", "fail")


@dataclass(frozen=True)
class Check:
    id: str
    label: str
    status: str
    detail: str


def _settings() -> Tuple[Optional[dict], Optional[str]]:
    path = Path(config_manager.SETTINGS_FILE)
    if not path.exists():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return None, f"cannot read settings: {error.strerror or error}"
    except json.JSONDecodeError as error:
        return None, f"invalid JSON (line {error.lineno}, column {error.colno}): {error.msg}"
    if not isinstance(data, dict):
        return None, "settings must be a JSON object"
    return data, None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _python() -> Check:
    required = (3, 10)
    version = platform.python_version()
    if sys.version_info >= required:
        return Check("python", "Python", "ok", version)
    return Check("python", "Python", "fail", f"{version} — Dizel needs 3.10+")


def _core_deps() -> Check:
    missing = [name for name in CORE_MODULES if not _has_module(name)]
    if missing:
        return Check("core_deps", "Core packages", "fail", "missing: " + ", ".join(missing))
    return Check("core_deps", "Core packages", "ok", ", ".join(CORE_MODULES))


def _llama_cpp() -> Check:
    if _has_module("llama_cpp"):
        return Check("llama_cpp", "llama.cpp", "ok", "installed — GGUF checkpoints can load")
    return Check(
        "llama_cpp",
        "llama.cpp",
        "warn",
        "not installed — GGUF checkpoints need it; torch checkpoints still load",
    )


def _torch() -> Check:
    if _has_module("torch"):
        return Check("torch", "PyTorch", "ok", "installed — .pt checkpoints can load")
    return Check(
        "torch",
        "PyTorch",
        "warn",
        "not installed — local .pt checkpoints need it",
    )


def _settings_check(data, error) -> Check:
    if error:
        return Check("settings", "Settings", "fail", error)
    if data == {}:
        return Check("settings", "Settings", "ok", "defaults (no settings file yet)")
    return Check("settings", "Settings", "ok", "settings.json parses")


def _usage_state() -> Check:
    path = Path(config_manager.SETTINGS_FILE).with_name("usage_state.json")
    if not path.exists():
        return Check("usage_state", "Usage state", "ok", "not created yet")
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return Check("usage_state", "Usage state", "warn", f"cannot read: {error.strerror or error}")
    except json.JSONDecodeError:
        return Check(
            "usage_state",
            "Usage state",
            "warn",
            "invalid JSON — the usage window resets on next launch",
        )
    return Check("usage_state", "Usage state", "ok", "parses")


def _config_dir() -> Check:
    directory = Path(config_manager.SETTINGS_FILE).parent
    if not directory.exists():
        return Check("config_dir", "Config folder", "ok", "will be created on first save")
    try:
        handle, name = tempfile.mkstemp(dir=directory, prefix=".doctor-", suffix=".tmp")
        os.close(handle)
        os.unlink(name)
    except OSError as error:
        return Check(
            "config_dir",
            "Config folder",
            "fail",
            f"not writable: {error.strerror or error}",
        )
    return Check("config_dir", "Config folder", "ok", "writable")


def _history() -> Check:
    directory = Path(history_manager.HISTORY_DIR)
    if not directory.exists():
        return Check("history", "Chat history", "ok", "no sessions yet")
    try:
        sessions = history_manager.list_sessions()
    except Exception as error:  # a corrupt store must not take the app down
        return Check("history", "Chat history", "fail", f"cannot list sessions: {error}")
    count = len(sessions)
    return Check(
        "history", "Chat history", "ok", f"{count} saved session{'s' if count != 1 else ''}"
    )


def _checkpoints(settings) -> Check:
    entries = [str(path) for path in (settings.get("checkpoint_history") or [])]
    if not entries:
        return Check("checkpoints", "Checkpoints", "ok", "none recorded yet")
    present = sum(1 for path in entries if Path(path).exists())
    status = "ok" if present == len(entries) else "warn"
    return Check(
        "checkpoints",
        "Checkpoints",
        status,
        f"{present} of {len(entries)} recent checkpoint(s) on disk",
    )


def _provider(settings) -> Check:
    router = settings.get("api_router") or {}
    provider = str(router.get("provider") or "local")
    if provider == "local":
        return Check("provider", "Provider", "ok", "local model — no API key needed")
    if router.get("api_key"):
        return Check("provider", "Provider", "ok", f"{provider} key configured")
    return Check(
        "provider",
        "Provider",
        "warn",
        f"{provider} has no API key — set one in Settings",
    )


def run_checks() -> List[Check]:
    settings, error = _settings()
    rows = [_python(), _core_deps(), _llama_cpp(), _torch(), _settings_check(settings, error)]
    rows.append(_usage_state())
    rows.append(_config_dir())
    rows.append(_history())
    if error:
        rows.append(Check("checkpoints", "Checkpoints", "warn", "skipped — settings invalid"))
        rows.append(Check("provider", "Provider", "warn", "skipped — settings invalid"))
    else:
        rows.append(_checkpoints(settings))
        rows.append(_provider(settings))
    return rows


def overall_status(checks: List[Check]) -> str:
    if any(row.status == "fail" for row in checks):
        return "fail"
    if any(row.status == "warn" for row in checks):
        return "warn"
    return "ok"


def summary_text(checks: Optional[List[Check]] = None) -> str:
    rows = checks if checks is not None else run_checks()
    counts = {status: sum(1 for row in rows if row.status == status) for status in STATUSES}
    lines = [
        f"Runtime doctor — {counts['ok']} ok · {counts['warn']} warn · {counts['fail']} fail",
        "",
    ]
    markers = {"ok": "[ok]", "warn": "[warn]", "fail": "[fail]"}
    for row in rows:
        lines.append(f"{markers[row.status]:<7}{row.label} — {row.detail}")
    return "\n".join(lines)


def as_dict(checks: Optional[List[Check]] = None) -> dict:
    rows = checks if checks is not None else run_checks()
    counts = {status: sum(1 for row in rows if row.status == status) for status in STATUSES}
    return {
        "checks": [asdict(row) for row in rows],
        "counts": counts,
        "status": overall_status(rows),
    }
