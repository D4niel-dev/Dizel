"""Tools the model can call.

Small local models have no native function calling, so the protocol is textual and
deliberately blunt — a tag the model can learn from one example:

    <tool name="read_file">{"path": "README.md", "start": 1, "end": 40}</tool>

The harness finds those, runs them, and feeds the result back for another round.

Design rules, all of which exist to keep a 1.6B model from making a mess:

* **Read-only.** Nothing here writes, deletes, or executes a shell command. A model
  that can run arbitrary commands on the user's machine is a vulnerability with a
  chat interface.
* **Bounded.** Every tool caps its output. An unbounded result would blow the
  2,048-token window in one call and take the conversation with it.
* **Off by default.** The prompt block costs tokens on every request, so tools are
  opt-in per session (`set_enabled`).
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

MAX_FILE_CHARS = 6000
MAX_ENTRIES = 60
MAX_MATCHES = 40
MAX_SCAN_FILES = 400
MAX_FETCH_CHARS = 3000

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


@dataclass
class Tool:
    name: str
    description: str
    args: Dict[str, str]
    run: Callable[..., str]

    def schema(self) -> str:
        args = ", ".join(f'"{k}": {v}' for k, v in self.args.items())
        return f'<tool name="{self.name}">{{{args}}}</tool>  — {self.description}'


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


def catalog() -> Dict[str, "Tool"]:
    merged = dict(REGISTRY)
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


def execute(call: Dict[str, Any]) -> str:
    """Run one parsed call and return text to feed back to the model."""
    name = call.get("name", "")
    if call.get("error"):
        return f"[{name}] {call['error']}"
    tool = catalog().get(name)
    if tool is None:
        return f"[{name}] No such tool. Available: {', '.join(sorted(catalog()))}."
    try:
        return tool.run(**call.get("args", {}))
    except TypeError as error:
        return f"[{name}] Bad arguments: {error}"
    except Exception as error:
        return f"[{name}] Failed: {error}"


def prompt_block() -> str:
    """The instructions to inject when tools are enabled."""
    listing = "\n".join(f"  {tool.schema()}" for tool in catalog().values())
    # Not str.format: the protocol example contains literal braces.
    return TOOL_PROMPT.replace("{tools}", listing)
