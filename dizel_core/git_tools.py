"""Read-only git helpers for the review and commit commands.

Only `rev-parse`, `status`, and `diff` are ever run. Nothing here stages,
commits, pushes, or writes to the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import tools

MAX_DIFF_CHARS = 60000
TIMEOUT = 20

PathLike = Union[str, Path, None]


def _git(root: Path, args: List[str]):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def _top(project_root: PathLike = None) -> tuple[Optional[Path], str]:
    root = Path(project_root or tools._project_root())
    try:
        result = _git(root, ["rev-parse", "--show-toplevel"])
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"git is unavailable: {error}"
    if result.returncode != 0:
        return None, "Not a git repository."
    return Path(result.stdout.strip()), ""


def _branch(top: Path) -> str:
    try:
        name = _git(top, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        name = ""
    return name or "unknown"


def _files(top: Path) -> List[Dict[str, str]]:
    result = _git(top, ["status", "--porcelain=v1", "-uall"])
    rows: List[Dict[str, str]] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        rows.append(
            {
                "status": line[:2].strip(),
                "index": line[0],
                "worktree": line[1],
                "path": line[3:].strip(),
            }
        )
    return rows


def _cap(diff: str) -> tuple[str, bool]:
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS].rstrip() + "\n\n…[diff truncated]", True
    return diff, False


def _raw_diff(top: Path, staged: bool) -> str:
    if staged:
        cached = _git(top, ["diff", "--cached", "--no-color"]).stdout
        return cached if cached.strip() else ""
    pending = _git(top, ["diff", "--no-color"]).stdout
    cached = _git(top, ["diff", "--cached", "--no-color"]).stdout
    parts = []
    if cached.strip():
        parts.append("Staged changes:\n" + cached)
    if pending.strip():
        parts.append("Unstaged changes:\n" + pending)
    return "\n\n".join(parts)


def _change(project_root: PathLike = None, staged: bool = False) -> Dict[str, Any]:
    top, error = _top(project_root)
    if top is None:
        return {
            "available": False,
            "branch": "",
            "files": [],
            "diff": "",
            "truncated": False,
            "error": error,
        }
    files = _files(top)
    if staged:
        files = [row for row in files if row["index"] not in (" ", "?")]
    raw = _raw_diff(top, staged)
    if not raw.strip():
        message = (
            "Nothing staged to commit — stage the files you want first."
            if staged
            else "No working-tree changes to review."
        )
        return {
            "available": True,
            "branch": _branch(top),
            "files": files,
            "diff": "",
            "truncated": False,
            "error": message,
        }
    diff, truncated = _cap(raw)
    return {
        "available": True,
        "branch": _branch(top),
        "files": files,
        "diff": diff,
        "truncated": truncated,
        "error": "",
    }


def working_diff(project_root: PathLike = None) -> Dict[str, Any]:
    return _change(project_root, staged=False)


def staged_diff(project_root: PathLike = None) -> Dict[str, Any]:
    return _change(project_root, staged=True)


def _file_lines(change: Dict[str, Any]) -> str:
    rows = change.get("files") or []
    if not rows:
        return "  (none)"
    return "\n".join(f"  {row['status'] or '?'} {row['path']}" for row in rows)


def review_prompt(change: Dict[str, Any]) -> str:
    return (
        "Review this working-tree diff. Be concrete: call out bugs, regressions, "
        "risky assumptions, and missing tests, with file references. Do not "
        "rewrite the code unless asked.\n\n"
        f"Branch: {change.get('branch') or 'unknown'}\n"
        f"Changed files:\n{_file_lines(change)}\n\n"
        f"Diff:\n{change.get('diff', '')}"
    )


def commit_prompt(change: Dict[str, Any]) -> str:
    return (
        "Write one commit message for these staged changes. Match the "
        "repository's existing commit style. Return only the commit message — "
        "no code fences, no explanation.\n\n"
        f"Staged files:\n{_file_lines(change)}\n\n"
        f"Diff:\n{change.get('diff', '')}"
    )
