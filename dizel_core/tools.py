"""Tools the model can call.

Small local models have no native function calling, so the protocol is textual and
deliberately blunt — a tag the model can learn from one example:

    <tool name="read_file">{"path": "README.md", "start": 1, "end": 40}</tool>

The harness finds those, runs them, and feeds the result back for another round.

Design rules, all of which exist to keep a 1.6B model from making a mess:

* **Reviewed writes.** `write_file` and `edit_file` change files only after the
  calling surface shows a diff and the user approves it. `delete_file` moves a
  file to the recycle bin, but only after an explicit per-call approval — it is
  never auto-approved by any trust setting. Nothing executes a shell command; a
  model with arbitrary command execution is a vulnerability with a chat interface.
* **Bounded.** Every tool caps its output. An unbounded result would blow the
  2,048-token window in one call and take the conversation with it.
* **Off by default.** The prompt block costs tokens on every request, so tools are
  opt-in per session (`set_enabled`).
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import operator
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List
from urllib.parse import quote_plus

MAX_FILE_CHARS = 6000
MAX_ENTRIES = 60
MAX_MATCHES = 40
MAX_SCAN_FILES = 400
MAX_FETCH_CHARS = 3000
MAX_WRITE_CHARS = 120_000
MAX_DIFF_CHARS = 80_000
MAX_UI_RESULT_CHARS = 14_000
MAX_UI_CONTENT_CHARS = 20_000

BLOCKED_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".pt", ".gguf", ".safetensors",
    ".zip", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".mp4", ".wav", ".pdf", ".model",
}
BLOCKED_PARTS = {"node_modules", ".git", "__pycache__", ".venv", "dist", ".next"}


# ── Session toggle ───────────────────────────────────────────────────────────
_ENABLED = False


def set_enabled(value: bool) -> None:
    global _ENABLED
    _ENABLED = bool(value)


def is_enabled() -> bool:
    return _ENABLED


@dataclass(frozen=True)
class FileChange:
    """One reviewed file change; the approval modal renders this."""

    tool: str
    path: str
    action: str
    diff: str
    additions: int
    deletions: int


@dataclass
class Tool:
    name: str
    description: str
    args: Dict[str, str]
    run: Callable[..., str]
    requires_approval: bool = False

    def schema(self) -> str:
        args = ", ".join(f'"{k}": {v}' for k, v in self.args.items())
        return f'<tool name="{self.name}">{{{args}}}</tool>  — {self.description}'


_UI_META = threading.local()


def _ui_begin() -> None:
    _UI_META.current = {}


def _ui_record(**values: Any) -> None:
    meta = getattr(_UI_META, "current", None)
    if isinstance(meta, dict):
        meta.update({key: value for key, value in values.items() if value is not None})


def _ui_end() -> Dict[str, Any]:
    meta = getattr(_UI_META, "current", None)
    _UI_META.current = None
    return meta if isinstance(meta, dict) else {}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _skip(path: Path) -> bool:
    if any(part in BLOCKED_PARTS for part in path.parts):
        return True
    return path.suffix.lower() in BLOCKED_SUFFIXES


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ── Files ────────────────────────────────────────────────────────────────────
def _read_file(path: str, start: int = 1, end: int = 0) -> str:
    target = Path(path).expanduser()
    if not target.exists():
        return f"No such file: {path}"
    if not target.is_file():
        return f"Not a file: {path}"
    if target.suffix.lower() in BLOCKED_SUFFIXES:
        return f"Refusing to read {target.suffix} files — binary or archive."
    try:
        text = _read_text(target)
    except OSError as error:
        return f"Could not read {path}: {error}"

    lines = text.splitlines()
    total = len(lines)
    # Ranges matter: a model reading a 900-line file wastes the whole window on
    # the first tool call and then cannot fit the answer.
    lo = max(1, int(start or 1))
    hi = int(end) if end else min(total, lo + 199)
    hi = max(lo, min(hi, total))
    chunk = "\n".join(lines[lo - 1 : hi])

    if len(chunk) > MAX_FILE_CHARS:
        chunk = chunk[:MAX_FILE_CHARS] + "\n[truncated]"
    header = f"{path} — lines {lo}-{hi} of {total}\n"
    return header + (chunk or "(empty)")


def _list_dir(path: str = ".", pattern: str = "") -> str:
    target = Path(path).expanduser()
    if not target.exists():
        return f"No such directory: {path}"
    if not target.is_dir():
        return f"Not a directory: {path}"

    entries = [e for e in target.iterdir() if not _skip(e)]
    if pattern:
        needle = pattern.lower().replace("*", "")
        entries = [e for e in entries if needle in e.name.lower()]
    entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

    rows = []
    for entry in entries[:MAX_ENTRIES]:
        if entry.is_dir():
            rows.append(f"  {entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            rows.append(f"  {entry.name}  ({size:,} bytes)")
    if len(entries) > MAX_ENTRIES:
        rows.append(f"  … {len(entries) - MAX_ENTRIES} more")
    return "\n".join(rows) or "(empty directory)"


def _find_files(pattern: str = "*", path: str = ".") -> str:
    root = Path(path).expanduser()
    if not root.exists():
        return f"No such directory: {path}"
    matches = []
    for candidate in root.rglob(pattern or "*"):
        if _skip(candidate) or not candidate.is_file():
            continue
        matches.append(candidate)
        if len(matches) >= MAX_MATCHES:
            break
    if not matches:
        return f"No files matching {pattern!r} under {path}."
    return "\n".join(f"  {m}" for m in matches)


def _grep(pattern: str, path: str = ".", glob: str = "*") -> str:
    root = Path(path).expanduser()
    if not root.exists():
        return f"No such path: {path}"
    try:
        rx = re.compile(pattern)
    except re.error as error:
        return f"Bad pattern: {error}"

    targets = [root] if root.is_file() else [
        p for p in root.rglob(glob or "*") if p.is_file() and not _skip(p)
    ]
    hits: List[str] = []
    scanned = 0
    for file in targets[:MAX_SCAN_FILES]:
        scanned += 1
        try:
            for number, line in enumerate(_read_text(file).splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{file}:{number}: {line.strip()[:140]}")
                    if len(hits) >= MAX_MATCHES:
                        break
        except (OSError, UnicodeDecodeError):
            continue
        if len(hits) >= MAX_MATCHES:
            break

    if not hits:
        return f"No matches for {pattern!r} ({scanned} files scanned)."
    tail = "" if len(hits) < MAX_MATCHES else "\n[stopped at the match limit]"
    return "\n".join(hits) + tail


def _file_info(path: str) -> str:
    target = Path(path).expanduser()
    if not target.exists():
        return f"No such path: {path}"
    stat = target.stat()
    stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%d %B %Y %H:%M")
    if target.is_dir():
        count = sum(1 for _ in target.iterdir())
        return f"{path}\ndirectory · {count} entries · modified {stamp}"
    try:
        lines = len(_read_text(target).splitlines())
    except (OSError, UnicodeDecodeError):
        lines = 0
    return f"{path}\n{stat.st_size:,} bytes · {lines:,} lines · modified {stamp}"


# ── Reviewed writes ──────────────────────────────────────────────────────────
def _project_root() -> Path:
    return Path.cwd().resolve()


def _resolve_write(path: str) -> tuple[Path | None, str]:
    raw = str(path or "").strip().strip("\"'")
    if not raw:
        return None, "No path given."
    root = _project_root()
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = root / target
    try:
        target = target.resolve(strict=False)
    except OSError as error:
        return None, f"Invalid path: {error}"
    try:
        relative = target.relative_to(root)
    except ValueError:
        return None, "File changes are limited to the current project directory."
    if any(part in BLOCKED_PARTS for part in relative.parts):
        return None, "Refusing to write generated, binary, or VCS internals."
    if target.suffix.lower() in BLOCKED_SUFFIXES:
        return None, "Refusing to write binary or archive files."
    if target.exists() and not target.is_file():
        return None, "Not a regular file."
    return target, ""


def _read_for_edit(path: Path) -> tuple[str | None, str]:
    if not path.exists():
        return "", ""
    try:
        if path.stat().st_size > MAX_WRITE_CHARS:
            return None, "File is too large to edit safely."
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read(), ""
    except OSError as error:
        return None, f"Could not read file: {error}"


def _diff_change(before: str, after: str, display: str) -> tuple[str | None, int, int, str]:
    before_lines = before.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    after_lines = after.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{display}",
            tofile=f"b/{display}",
            lineterm="",
        )
    )
    diff = "\n".join(lines)
    additions = sum(
        1 for line in lines if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in lines if line.startswith("-") and not line.startswith("---")
    )
    if len(diff) > MAX_DIFF_CHARS:
        return None, additions, deletions, "Diff is too large to review safely."
    return diff, additions, deletions, ""


def _approved(
    change: FileChange,
    approve: Callable[[FileChange], bool] | None,
    verb: str = "Write",
) -> tuple[bool, str]:
    if approve is None:
        return False, f"{verb} refused: this surface cannot approve file changes."
    try:
        if approve(change):
            return True, ""
    except Exception as error:
        return False, f"{verb} refused: approval failed ({error})."
    return False, f"{verb} cancelled."


def _atomic_write(target: Path, text: str) -> None:
    parent = target.parent
    if not parent.is_dir():
        raise OSError(f"Parent directory does not exist: {parent}")
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".dizel.tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_file(
    path: str,
    content: str,
    approve: Callable[[FileChange], bool] | None = None,
) -> str:
    target, error = _resolve_write(path)
    if target is None:
        return f"[write_file] {error}"
    content = str(content or "")
    if len(content) > MAX_WRITE_CHARS:
        return f"[write_file] Content is too large ({len(content):,} chars)."

    display = target.relative_to(_project_root()).as_posix()
    existed = target.exists()
    before, read_error = _read_for_edit(target)
    if before is None:
        return f"[write_file] {read_error}"
    if existed and before == content:
        return f"[write_file] No change: {display}"

    diff, additions, deletions, diff_error = _diff_change(before, content, display)
    if diff is None:
        return f"[write_file] {diff_error}"
    change = FileChange(
        "write_file",
        display,
        "overwrite" if existed else "create",
        diff,
        additions,
        deletions,
    )
    _ui_record(
        action=change.action,
        diff=change.diff,
        additions=change.additions,
        deletions=change.deletions,
    )
    allowed, reason = _approved(change, approve)
    if not allowed:
        return f"[write_file] {reason}"
    try:
        _atomic_write(target, content)
    except OSError as write_error:
        return f"[write_file] Failed: {write_error}"
    verb = "Overwrote" if existed else "Created"
    return f"[write_file] {verb} {display} (+{additions}/-{deletions})"


def _edit_file(
    path: str,
    old: str,
    new: str,
    approve: Callable[[FileChange], bool] | None = None,
) -> str:
    target, error = _resolve_write(path)
    if target is None:
        return f"[edit_file] {error}"
    old = str(old or "")
    new = str(new or "")
    if not old:
        return "[edit_file] The old snippet is empty."
    if old == new:
        return "[edit_file] No change requested."
    if not target.exists():
        return f"[edit_file] No such file: {path}"

    before, read_error = _read_for_edit(target)
    if before is None:
        return f"[edit_file] {read_error}"

    old_normalized = old.replace("\r\n", "\n").replace("\r", "\n")
    new_normalized = new.replace("\r\n", "\n").replace("\r", "\n")
    if "\r\n" in before:
        old_match = old_normalized.replace("\n", "\r\n")
        replacement = new_normalized.replace("\n", "\r\n")
    else:
        old_match = old_normalized
        replacement = new_normalized

    first = before.find(old_match)
    if first < 0:
        return "[edit_file] Old snippet was not found; read the file again."
    if before.find(old_match, first + 1) >= 0:
        return "[edit_file] Old snippet occurs more than once; make it unique."
    after = before[:first] + replacement + before[first + len(old_match) :]

    display = target.relative_to(_project_root()).as_posix()
    diff, additions, deletions, diff_error = _diff_change(before, after, display)
    if diff is None:
        return f"[edit_file] {diff_error}"
    change = FileChange("edit_file", display, "edit", diff, additions, deletions)
    _ui_record(
        action=change.action,
        diff=change.diff,
        additions=change.additions,
        deletions=change.deletions,
    )
    allowed, reason = _approved(change, approve)
    if not allowed:
        return f"[edit_file] {reason}"
    try:
        _atomic_write(target, after)
    except OSError as write_error:
        return f"[edit_file] Failed: {write_error}"
    return f"[edit_file] Edited {display} (+{additions}/-{deletions})"


# ── Reviewed delete ──────────────────────────────────────────────────────────
def _resolve_delete(path: str) -> tuple[Path | None, str]:
    raw = str(path or "").strip().strip("\"'")
    if not raw:
        return None, "No path given."
    root = _project_root()
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = root / target
    try:
        target = target.resolve(strict=False)
    except OSError as error:
        return None, f"Invalid path: {error}"
    try:
        relative = target.relative_to(root)
    except ValueError:
        return None, "File changes are limited to the current project directory."
    if any(part in BLOCKED_PARTS for part in relative.parts):
        return None, "Refusing to delete VCS or generated internals."
    if not target.exists():
        return None, f"No such file: {raw}"
    if not target.is_file():
        return None, "Not a regular file."
    return target, ""


def _delete_file(
    path: str,
    approve: Callable[[FileChange], bool] | None = None,
) -> str:
    target, error = _resolve_delete(path)
    if target is None:
        return f"[delete_file] {error}"

    display = target.relative_to(_project_root()).as_posix()
    before, _read_error = _read_for_edit(target)
    if before is None:
        before = ""
    diff, _additions, deletions, _diff_error = _diff_change(before, "", display)
    # A removal diff of a binary is replacement-character noise; show none.
    if diff is None or target.suffix.lower() in BLOCKED_SUFFIXES:
        diff = ""
    change = FileChange("delete_file", display, "delete", diff, 0, deletions)
    _ui_record(action="delete", diff=diff, additions=0, deletions=deletions)
    # Delete is always a per-call decision; no trust tier may short-circuit it.
    allowed, reason = _approved(change, approve, verb="Delete")
    if not allowed:
        return f"[delete_file] {reason}"
    try:
        from send2trash import send2trash
    except ImportError:
        return "[delete_file] Failed: send2trash is not installed."
    try:
        send2trash(str(target))
    except Exception as error:
        return f"[delete_file] Failed: {error}"
    return f"[delete_file] Deleted {display} (recycle bin)"


# ── Web ──────────────────────────────────────────────────────────────────────
def _web_search(query: str, max_results: int = 4) -> str:
    try:
        from core.tools.web_search import search_web
    except Exception as error:
        return f"Web search unavailable: {error}"
    try:
        # search_web returns formatted text (or "" on failure), not dicts.
        results = search_web(query, max_results=max_results)
    except Exception as error:
        return f"Search failed: {error}"
    return results.strip() if results else "No results."


def _fetch_url(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        return "Only http(s) URLs are allowed."
    try:
        import httpx
    except Exception as error:
        return f"httpx unavailable: {error}"
    try:
        response = httpx.get(url, timeout=12.0, follow_redirects=True)
        response.raise_for_status()
        html = response.text
    except Exception as error:
        return f"Fetch failed: {error}"

    # Strip what carries no meaning to the model, then collapse whitespace.
    html = re.sub(r"(?is)<(script|style|nav|footer)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + "\n[truncated]"
    return text or "(no readable text)"


# ── Maths ────────────────────────────────────────────────────────────────────
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCS:
            raise ValueError(f"{node.func.id} is not allowed")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("only arithmetic is allowed")


def _calc(expression: str) -> str:
    # eval() would let the model reach the whole interpreter; an AST whitelist
    # gives it a calculator and nothing else.
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval(tree.body)
    except Exception as error:
        return f"Could not evaluate {expression!r}: {error}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


# ── Misc ─────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%A, %d %B %Y — %H:%M")


def _system_info() -> str:
    import platform
    import shutil

    try:
        import psutil

        memory = psutil.virtual_memory()
        ram = f"{memory.used / 1e9:.1f} / {memory.total / 1e9:.1f} GB"
        cpu = f"{psutil.cpu_percent(interval=0.3):.0f}%"
    except Exception:
        ram, cpu = "unknown", "unknown"

    disk = shutil.disk_usage(Path.home())
    return (
        f"{platform.system()} {platform.release()} · Python {platform.python_version()}\n"
        f"cpu {cpu} · ram {ram}\n"
        f"disk {disk.free / 1e9:.0f} GB free of {disk.total / 1e9:.0f} GB"
    )


REGISTRY: Dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool("read_file", "read a text file, optionally a line range",
             {"path": "string", "start": "int?", "end": "int?"}, _read_file),
        Tool("list_dir", "list a directory, optionally filtered",
             {"path": "string?", "pattern": "string?"}, _list_dir),
        Tool("find_files", "find files by name pattern",
             {"pattern": "string", "path": "string?"}, _find_files),
        Tool("grep", "search file contents for a regex",
             {"pattern": "string", "path": "string?", "glob": "string?"}, _grep),
        Tool("file_info", "size, line count and modified time of a path",
             {"path": "string"}, _file_info),
        Tool(
            "write_file",
            "create or overwrite a project file; the user reviews a diff and approves",
            {"path": "string", "content": "string"},
            _write_file,
            requires_approval=True,
        ),
        Tool(
            "edit_file",
            "replace one unique snippet in a project file; the user approves the diff",
            {"path": "string", "old": "string", "new": "string"},
            _edit_file,
            requires_approval=True,
        ),
        Tool(
            "delete_file",
            "delete a project file (moved to the recycle bin); the user must approve every deletion",
            {"path": "string"},
            _delete_file,
            requires_approval=True,
        ),
        Tool("web_search", "search the web and return the top results",
             {"query": "string"}, _web_search),
        Tool("fetch_url", "fetch a web page and return its text",
             {"url": "string"}, _fetch_url),
        Tool("calc", "evaluate an arithmetic expression",
             {"expression": "string"}, _calc),
        Tool("now", "the current local date and time", {}, _now),
        Tool("system_info", "os, cpu, memory and disk of this machine", {}, _system_info),
    ]
}

# Tools contributed by external runtimes (MCP). Kept separate so a session
# without MCP pays nothing and a config reload can drop them wholesale.
_EXTERNAL: Dict[str, "Tool"] = {}


def set_external(entries: Dict[str, "Tool"]) -> None:
    global _EXTERNAL
    _EXTERNAL = dict(entries or {})


def catalog(include_writes: bool = False) -> Dict[str, "Tool"]:
    merged = {
        name: tool
        for name, tool in REGISTRY.items()
        if include_writes or not tool.requires_approval
    }
    for name in sorted(_EXTERNAL):
        if name not in merged:
            merged[name] = _EXTERNAL[name]
    return merged


TOOL_PROMPT = """\
You can use tools. When you need one, reply with ONLY this and nothing else:

<tool name="NAME">{"arg": "value"}</tool>

Available tools:
{tools}

After a tool runs you will be given its output and asked to continue. Use a tool
only when the answer genuinely depends on it — for anything you already know, just
answer. Never guess a file's contents when you can read it.
"""

_CALL = re.compile(r"<tool\s+name=[\"']([\w_]+)[\"']\s*>(.*?)</tool>", re.DOTALL)


def parse_calls(text: str) -> List[Dict[str, Any]]:
    """Every tool call in a reply, in order."""
    calls = []
    for match in _CALL.finditer(text or ""):
        name = match.group(1)
        raw = match.group(2).strip()
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            # A malformed call is worth surfacing, not silently dropping.
            calls.append({"name": name, "args": {}, "error": f"could not parse arguments: {raw[:80]}"})
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "args": args})
    return calls


def display_detail(call: Dict[str, Any]) -> str:
    """The one-line target shown by either surface's tool row."""
    name = str(call.get("name") or "?")
    args = call.get("args") or {}
    path = str(args.get("path") or "").strip()
    if name == "read_file" and path:
        start = args.get("start")
        end = args.get("end")
        if start or end:
            return f"{path}:{start or 1}-{end or '…'}"
        return path
    if name in ("list_dir", "file_info") and path:
        return path
    if name == "find_files":
        return " ".join(
            part for part in (str(args.get("pattern") or "*"), path) if part
        )
    if name == "grep":
        return " ".join(
            part for part in (str(args.get("pattern") or ""), path or ".") if part
        )
    if name == "web_search":
        return str(args.get("query") or "")
    if name == "fetch_url":
        return str(args.get("url") or "")
    if name == "calc":
        return str(args.get("expression") or "")
    if name in ("write_file", "edit_file", "delete_file"):
        return path
    return ", ".join(f"{key}={value}" for key, value in args.items())


def _clip_ui(value: Any, limit: int) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n…", True


def ui_payload(call: Dict[str, Any], outcome: str = "") -> Dict[str, Any]:
    """What the transcript needs to explain a tool call after it ran.

    The model receives the full outcome; a surface gets a bounded copy plus the
    fields its detail view knows how to render (path, query, written content,
    diff, sources).
    """
    name = str(call.get("name") or "?")
    args = call.get("args") or {}
    result, result_truncated = _clip_ui(outcome, MAX_UI_RESULT_CHARS)
    payload: Dict[str, Any] = {
        "name": name,
        "detail": display_detail(call),
        "result": result,
        "result_truncated": result_truncated,
    }

    path = str(args.get("path") or "").strip()
    if path:
        payload["path"] = path
    if name == "grep":
        payload["query"] = str(args.get("pattern") or "")
        payload["glob"] = str(args.get("glob") or "")
    elif name == "find_files":
        payload["query"] = str(args.get("pattern") or "")
    elif name == "web_search":
        query = str(args.get("query") or "").strip()
        payload["query"] = query
        if query:
            payload["search_url"] = f"https://duckduckgo.com/?q={quote_plus(query)}"
    elif name == "fetch_url":
        payload["url"] = str(args.get("url") or "")

    if name == "write_file":
        content, truncated = _clip_ui(args.get("content"), MAX_UI_CONTENT_CHARS)
        payload["content"] = content
        payload["content_truncated"] = truncated
    elif name == "edit_file":
        old, old_truncated = _clip_ui(args.get("old"), MAX_UI_CONTENT_CHARS)
        new, new_truncated = _clip_ui(args.get("new"), MAX_UI_CONTENT_CHARS)
        payload["old"] = old
        payload["new"] = new
        payload["replacement_truncated"] = old_truncated or new_truncated

    meta = call.get("_ui") or {}
    for key in ("action", "diff", "additions", "deletions"):
        if key not in meta:
            continue
        if key == "diff":
            diff, truncated = _clip_ui(meta[key], MAX_UI_RESULT_CHARS)
            payload["diff"] = diff
            payload["diff_truncated"] = truncated
        else:
            payload[key] = meta[key]

    return payload


def outcome_failed(name: str, outcome: str) -> bool:
    """Whether a tool's returned text represents a failed call for UI status."""
    text = (outcome or "").strip()
    prefix = f"[{name}] "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.startswith(
        (
            "No such",
            "Not a ",
            "Refusing",
            "Could not",
            "Bad pattern",
            "Fetch failed",
            "Search failed",
            "Only http",
            "Write cancelled",
            "Write refused",
            "Delete cancelled",
            "Delete refused",
            "Failed:",
            "Old snippet",
            "File is too large",
            "Diff is too large",
            "File changes are limited",
            "Content is too large",
            "No path",
            "Invalid path",
            "Parent directory",
            "MCP call failed",
        )
    )


def execute(
    call: Dict[str, Any],
    approve: Callable[[FileChange], bool] | None = None,
) -> str:
    """Run one parsed call and return text to feed back to the model."""
    name = call.get("name", "")
    if call.get("error"):
        return f"[{name}] {call['error']}"
    tool = catalog(include_writes=True).get(name)
    if tool is None:
        available = ", ".join(sorted(catalog()))
        return f"[{name}] No such tool. Available: {available}."
    args = call.get("args", {})
    _ui_begin()
    try:
        if tool.requires_approval:
            if approve is None:
                outcome = (
                    f"[{name}] Write refused: this surface cannot approve "
                    "file changes."
                )
            else:
                outcome = tool.run(**args, approve=approve)
        else:
            outcome = tool.run(**args)
    except TypeError as error:
        outcome = f"[{name}] Bad arguments: {error}"
    except Exception as error:
        outcome = f"[{name}] Failed: {error}"
    finally:
        meta = _ui_end()
        if meta:
            call["_ui"] = meta
    return outcome


def prompt_block(allow_writes: bool = False) -> str:
    """The instructions to inject when tools are enabled."""
    listing = "\n".join(
        f"  {tool.schema()}" for tool in catalog(include_writes=allow_writes).values()
    )
    # Not str.format: the protocol example contains literal braces.
    return TOOL_PROMPT.replace("{tools}", listing)
