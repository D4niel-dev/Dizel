# MCP Client + Workspaces Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MCP client (stdio + Streamable HTTP) usable from both the GUI and the TUI, and give the GUI Workspaces surfaces a full motion pass.

**Architecture:** A shared `dizel_core/mcp/` package owns connections on a background asyncio thread, adapts MCP tools into the existing textual tool registry (so the existing 3-round agent loop and tool notices work unchanged), and persists servers under `mcp.servers` in `settings.json`. The FastAPI sidecar exposes `/mcp/*` routes for the GUI; the TUI uses the same package in-process via a `/mcp` modal. Part B is CSS/React-only motion on the Workspaces popover and secondary panels.

**Tech Stack:** Python 3.13, official `mcp` SDK (fastmcp fixture), FastAPI sidecar, Electron/React/TypeScript GUI (tsc + Vite), Textual TUI, pytest.

**Spec:** `specs/2026-11-08-mcp-workspaces-design.md`

## Implementation refinements to the spec

- Secret edits: the GUI never receives `env`/`headers` values. The PUT handler merges stored values for keys whose submitted value is empty, so "leave blank to keep" works.
- Library removal animates opacity/scale (transform-only) instead of a grid-row collapse — no layout animation.
- The manager does not poll-apply; only startup, PUT, `/mcp` edits, and tests call `apply()`/`apply_async()`.

## Global Constraints

- Tests must isolate config: set `DIZEL_CONFIG_DIR` to a temp dir or monkeypatch `dizel_core.config_manager.SETTINGS_FILE`/`_DATA_DIR`. Never read the repo `.env`, never write the maintainer's real `.dizel/settings.json`.
- Tool names are namespaced `{server_id}__{tool}` lowercased, `[a-z0-9_]` only; because the call regex is `[\w_]+` (`dizel_core/tools.py:342`), names containing `-` or dots must be sanitized before registration.
- Per-server enable toggle is the only gate; no per-call approval UI.
- All new motion uses the existing `--ease` / `--dur` tokens and must be neutralized by the existing `:root[data-motion="off"]` and `prefers-reduced-motion` blocks (`inference/gui/src/styles/app.css:3931-3947`). Animate transform/opacity only.
- Do not commit. The maintainer commits when they ask; each task ends with a verification checkpoint (tests/build) instead.
- Keep GUI copy honest: "Enabled servers start with the app", "Leave blank to keep the stored value".

---

### Task 1: MCP config schema and defaults

**Files:**
- Create: `dizel_core/mcp/__init__.py` (empty)
- Create: `dizel_core/mcp/config.py`
- Modify: `dizel_core/config_manager.py` (add `mcp` default inside `_DEFAULTS`, after `"api_router"` block ~line 119)
- Test: `tests/test_mcp_config.py`

**Interfaces:**
- Produces: `MCPConfigError(str)`; `validate_servers(raw: Any) -> list[dict]` (strict, raises); `load(cfg: dict) -> list[dict]` (lenient, skips bad entries); `save(servers: list[dict]) -> list[dict]` (normalizes then read-modify-write `settings.json`).
- Normalized server keys: `id, name, transport, enabled` plus `command, args, env` (stdio) or `url, headers` (http).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_config.py
import pytest

from dizel_core.mcp import config as mcp_config


def test_stdio_server_normalizes_with_defaults():
    servers = mcp_config.validate_servers(
        [{"name": "My Files", "transport": "stdio", "command": "npx", "args": ["-y", "server"]}]
    )
    assert servers == [
        {
            "id": "my-files",
            "name": "My Files",
            "transport": "stdio",
            "enabled": True,
            "command": "npx",
            "args": ["-y", "server"],
            "env": {},
        }
    ]


def test_http_server_normalizes():
    servers = mcp_config.validate_servers(
        [{"id": "linear", "name": "Linear", "transport": "http", "url": "https://mcp.linear.app/mcp", "headers": {"Authorization": "Bearer x"}, "enabled": False}]
    )
    assert servers[0]["id"] == "linear"
    assert servers[0]["headers"] == {"Authorization": "Bearer x"}
    assert servers[0]["enabled"] is False


def test_duplicate_ids_get_suffixes():
    servers = mcp_config.validate_servers(
        [
            {"name": "Same Name", "command": "a"},
            {"name": "Same Name", "command": "b"},
        ]
    )
    assert [s["id"] for s in servers] == ["same-name", "same-name-2"]


def test_stdio_requires_command():
    with pytest.raises(mcp_config.MCPConfigError):
        mcp_config.validate_servers([{"name": "x", "transport": "stdio"}])


def test_http_requires_http_url():
    with pytest.raises(mcp_config.MCPConfigError):
        mcp_config.validate_servers([{"name": "x", "transport": "http", "url": "ftp://nope"}])


def test_unknown_transport_rejected():
    with pytest.raises(mcp_config.MCPConfigError):
        mcp_config.validate_servers([{"name": "x", "transport": "smtp"}])


def test_load_skips_invalid_entries():
    cfg = {
        "mcp": {
            "servers": [
                {"name": "Good", "command": "npx"},
                {"name": "Bad", "transport": "http"},
                "not-a-dict",
            ]
        }
    }
    assert [s["name"] for s in mcp_config.load(cfg)] == ["Good"]


def test_load_without_section_is_empty():
    assert mcp_config.load({}) == []


def test_save_roundtrip(tmp_path, monkeypatch):
    import dizel_core.config_manager as cm

    monkeypatch.setattr(cm, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "SETTINGS_FILE", str(tmp_path / "settings.json"))

    saved = mcp_config.save([{"name": "Echo", "command": "python"}])
    assert saved[0]["id"] == "echo"
    assert cm.ConfigManager.load()["mcp"]["servers"][0]["command"] == "python"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mcp_config.py -q`
Expected: FAIL (no module `dizel_core.mcp`).

- [ ] **Step 3: Implement**

```python
# dizel_core/mcp/__init__.py
"""MCP client support: config, connections, and the tool bridge."""

# dizel_core/mcp/config.py
"""Normalize and persist the `mcp.servers` section of settings.json."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from dizel_core.config_manager import ConfigManager


class MCPConfigError(ValueError):
    """A server entry that cannot be used."""


_TRANSPORTS = {"stdio", "http"}
_SLUG = re.compile(r"[^a-z0-9_-]+")


def _slugify(value: str) -> str:
    slug = _SLUG.sub("-", str(value).strip().lower()).strip("-_")
    return slug[:32] or "server"


def normalize(raw: Any, seen: Set[str] | None = None) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise MCPConfigError("server entry must be an object")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise MCPConfigError("name is required")
    transport = str(raw.get("transport") or "stdio").strip().lower()
    if transport not in _TRANSPORTS:
        raise MCPConfigError(f"unknown transport: {transport}")

    server_id = _slugify(raw.get("id") or name)
    if seen is not None:
        base, suffix = server_id, 2
        while server_id in seen:
            server_id = f"{base}-{suffix}"
            suffix += 1
        seen.add(server_id)

    entry: Dict[str, Any] = {
        "id": server_id,
        "name": name[:60],
        "transport": transport,
        "enabled": bool(raw.get("enabled", True)),
    }

    if transport == "stdio":
        command = str(raw.get("command") or "").strip()
        if not command:
            raise MCPConfigError("command is required for stdio servers")
        args = raw.get("args") or []
        if not isinstance(args, list):
            raise MCPConfigError("args must be a list")
        env = raw.get("env") or {}
        if not isinstance(env, dict):
            raise MCPConfigError("env must be an object")
        entry["command"] = command
        entry["args"] = [str(value) for value in args]
        entry["env"] = {str(key): str(value) for key, value in env.items()}
    else:
        url = str(raw.get("url") or "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise MCPConfigError("url must start with http:// or https://")
        headers = raw.get("headers") or {}
        if not isinstance(headers, dict):
            raise MCPConfigError("headers must be an object")
        entry["url"] = url
        entry["headers"] = {str(key): str(value) for key, value in headers.items()}

    return entry


def validate_servers(raw_servers: Any) -> List[Dict[str, Any]]:
    if raw_servers is None:
        return []
    if not isinstance(raw_servers, list):
        raise MCPConfigError("servers must be a list")
    seen: Set[str] = set()
    return [normalize(item, seen) for item in raw_servers]


def load(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lenient read: bad entries are skipped, never fatal."""
    section = cfg.get("mcp") if isinstance(cfg, dict) else None
    raw = section.get("servers") if isinstance(section, dict) else None
    if not isinstance(raw, list):
        return []
    seen: Set[str] = set()
    servers: List[Dict[str, Any]] = []
    for item in raw:
        try:
            servers.append(normalize(item, seen))
        except MCPConfigError as error:
            print(f"[dizel-mcp] skipping server: {error}", flush=True)
    return servers


def save(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = validate_servers(servers)
    cfg = ConfigManager.load()
    cfg["mcp"] = {"servers": normalized}
    ConfigManager.save(cfg)
    return normalized
```

Add to `ConfigManager._DEFAULTS` (after the `"api_router"` entry):

```python
        "mcp": {
            "servers": [],
        },
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mcp_config.py -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint**

Run: `git status --short dizel_core/mcp dizel_core/config_manager.py tests/test_mcp_config.py` and confirm only the intended files changed. Do not commit.

---

### Task 2: Tool registry seam and catalog consumers

**Files:**
- Modify: `dizel_core/tools.py` (add `_EXTERNAL`, `set_external`, `catalog`; use in `execute`/`prompt_block`)
- Modify: `api/routes/chat.py:31-44` (`tools_state`, `tools_toggle` use `tools.catalog()`)
- Test: `tests/test_tools_catalog.py`

**Interfaces:**
- Produces: `tools.set_external(tools: dict[str, Tool]) -> None`; `tools.catalog() -> dict[str, Tool]` (builtins first, external sorted by name, builtins win on collision).
- Consumes: `Tool` dataclass (`tools.py:59-68`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_catalog.py
from dizel_core import tools


def _external(name: str):
    return tools.Tool(name, "read a file", {"path": "string"}, lambda path: f"read {path}")


def test_catalog_includes_external_and_executes():
    tools.set_external({"fs__read": _external("fs__read")})
    try:
        assert "fs__read" in tools.catalog()
        assert tools.execute({"name": "fs__read", "args": {"path": "x"}}) == "read x"
        assert "fs__read" in tools.prompt_block()
    finally:
        tools.set_external({})


def test_builtin_wins_name_collision():
    tools.set_external({"read_file": _external("read_file")})
    try:
        assert tools.catalog()["read_file"].description != "read a file"
    finally:
        tools.set_external({})


def test_cleared_external_disappears():
    tools.set_external({"x__y": _external("x__y")})
    tools.set_external({})
    assert "x__y" not in tools.catalog()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_tools_catalog.py -q`
Expected: FAIL (`set_external` missing).

- [ ] **Step 3: Implement**

In `dizel_core/tools.py`, after the `REGISTRY` definition (line 327):

```python
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
```

Change `execute()` (line 368) and `prompt_block()` (line 381) to use the merged view:

```python
    tool = catalog().get(name)
    if tool is None:
        return f"[{name}] No such tool. Available: {', '.join(sorted(catalog()))}."
```

```python
    listing = "\n".join(f"  {tool.schema()}" for tool in catalog().values())
```

In `api/routes/chat.py`, replace both `sorted(tools.REGISTRY)` occurrences (lines 36 and 44) with `sorted(tools.catalog())`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_tools_catalog.py -q`
Expected: pass.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/test_tools_catalog.py tests/test_mcp_config.py -q` — all pass.

---

### Task 3: MCP manager (connections, discovery, calls)

**Files:**
- Create: `dizel_core/mcp/manager.py`
- Create: `tests/fixtures/fake_mcp_server.py`
- Test: `tests/test_mcp_manager.py`

**Interfaces:**
- Produces: `MCPManager` with `apply(servers: list[dict]) -> None`, `apply_async(servers) -> None`, `test(server_id: str) -> dict` (raises `KeyError`), `call(server_id: str, tool: str, arguments: dict) -> str`, `status() -> list[dict]` rows `{id, state, error, tools}` where `tools` is the SDK's tool objects, `tools() -> list[{server, tools}]` for ready servers only, `shutdown() -> None`; module-level `get_manager() -> MCPManager` singleton.
- Consumes: normalized server dicts from Task 1.

- [ ] **Step 1: Write the fixture server**

```python
# tests/fixtures/fake_mcp_server.py
"""A tiny stdio MCP server used by the manager tests."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("dizel-test")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool()
def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


if __name__ == "__main__":
    server.run()
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_mcp_manager.py
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from dizel_core.mcp.manager import MCPManager

FIXTURE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"

SERVER = {
    "id": "fake",
    "name": "Fake",
    "transport": "stdio",
    "enabled": True,
    "command": sys.executable,
    "args": [str(FIXTURE)],
    "env": {},
}


def _manager() -> MCPManager:
    manager = MCPManager()
    manager.apply([SERVER])
    return manager


def test_connects_and_lists_tools():
    manager = _manager()
    try:
        state = manager.status()[0]
        assert state["id"] == "fake"
        assert state["state"] == "ready", state
        names = {tool.name for tool in state["tools"]}
        assert {"echo", "add", "explode"} <= names
    finally:
        manager.shutdown()


def test_calls_a_tool():
    manager = _manager()
    try:
        assert manager.call("fake", "echo", {"text": "hi"}) == "echo: hi"
        assert manager.call("fake", "add", {"a": 2, "b": 3}) == "5"
    finally:
        manager.shutdown()


def test_tool_error_becomes_text():
    manager = _manager()
    try:
        result = manager.call("fake", "explode", {})
        assert result.startswith("Error:")
    finally:
        manager.shutdown()


def test_disabled_server_does_not_spawn():
    manager = MCPManager()
    try:
        manager.apply([{**SERVER, "enabled": False}])
        assert manager.status()[0]["state"] == "disabled"
        assert manager.tools() == []
    finally:
        manager.shutdown()


def test_unknown_server_call_returns_message():
    manager = MCPManager()
    try:
        assert "not connected" in manager.call("nope", "echo", {})
    finally:
        manager.shutdown()


def test_missing_command_is_an_error_state():
    manager = MCPManager()
    try:
        manager.apply([{**SERVER, "command": "definitely-not-a-command-xyz"}])
        assert manager.status()[0]["state"] == "error"
    finally:
        manager.shutdown()


def test_shutdown_leaves_no_orphan():
    import psutil

    manager = _manager()
    assert manager.status()[0]["state"] == "ready"
    manager.shutdown()

    deadline = time.time() + 5
    while time.time() < deadline:
        alive = [
            proc
            for proc in psutil.process_iter(["cmdline"])
            if "fake_mcp_server" in " ".join(proc.info["cmdline"] or [])
        ]
        if not alive:
            return
        time.sleep(0.2)
    pytest.fail("fixture server process survived shutdown")
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_mcp_manager.py -q`
Expected: FAIL (no `dizel_core.mcp.manager`).

- [ ] **Step 4: Implement the manager**

```python
# dizel_core/mcp/manager.py
"""MCP client connections on one background event loop.

Public methods are synchronous so the chat thread can call them directly;
all async work is marshalled onto a dedicated daemon thread.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
from contextlib import AsyncExitStack
from typing import Any, Dict, List

try:  # optional dependency: the app still runs without MCP support installed
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    MCP_AVAILABLE = True
except Exception:  # pragma: no cover - only when the package is absent
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamablehttp_client = None
    MCP_AVAILABLE = False

CONNECT_TIMEOUT = 15.0
CALL_TIMEOUT = 60.0


def _idle() -> Dict[str, Any]:
    return {"state": "disabled", "error": "", "tools": []}


class _Session:
    def __init__(self, stack: AsyncExitStack, session: Any, tools: List[Any]) -> None:
        self.stack = stack
        self.session = session
        self.tools = tools


class MCPManager:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: Dict[str, _Session] = {}
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._states: Dict[str, Dict[str, Any]] = {}
        self._guard = threading.RLock()

    # ── loop plumbing ────────────────────────────────────────────────
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._thread is None or not self._thread.is_alive():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="dizel-mcp", daemon=True
            )
            self._thread.start()
        return self._loop

    def _run(self, coro: Any, timeout: float) -> Any:
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    # ── public, synchronous ─────────────────────────────────────────
    def apply(self, servers: List[Dict[str, Any]]) -> None:
        with self._guard:
            wanted = {server["id"]: server for server in servers}
            for server_id in list(self._sessions):
                if server_id not in wanted or wanted[server_id] != self._servers.get(server_id):
                    self._run(self._close(server_id), CONNECT_TIMEOUT)
                    self._states.pop(server_id, None)
            self._servers = wanted

            for server in servers:
                server_id = server["id"]
                if not server.get("enabled"):
                    self._states[server_id] = _idle()
                    continue
                if server_id in self._sessions:
                    continue
                self._run(self._start(server), CONNECT_TIMEOUT * 2 + 2)

            for server_id in list(self._states):
                if server_id not in wanted:
                    self._states.pop(server_id, None)
        self._refresh_bridge()

    def apply_async(self, servers: List[Dict[str, Any]]) -> None:
        threading.Thread(
            target=self.apply, args=(servers,), name="dizel-mcp-apply", daemon=True
        ).start()

    def test(self, server_id: str) -> Dict[str, Any]:
        with self._guard:
            server = self._servers.get(server_id)
            if server is None:
                raise KeyError(server_id)
            self._run(self._close(server_id), CONNECT_TIMEOUT)
            self._states.pop(server_id, None)
            if server.get("enabled"):
                self._run(self._start(server), CONNECT_TIMEOUT * 2 + 2)
            else:
                self._states[server_id] = _idle()
            state = self._states.get(server_id, {})
        self._refresh_bridge()
        return {"id": server_id, **state}

    def call(self, server_id: str, tool: str, arguments: Dict[str, Any]) -> str:
        with self._guard:
            session = self._sessions.get(server_id)
            if session is None and server_id in self._servers:
                server = self._servers[server_id]
                if server.get("enabled"):
                    self._run(self._start(server), CONNECT_TIMEOUT * 2 + 2)
                    session = self._sessions.get(server_id)
            if session is None:
                state = self._states.get(server_id, {})
                reason = state.get("error") or state.get("state", "unknown")
                return f"MCP server '{server_id}' is not connected: {reason}"
            try:
                result = self._run(
                    session.session.call_tool(tool, arguments), CALL_TIMEOUT + 5
                )
            except Exception as error:
                return f"MCP call failed: {error}"
        parts = [_content_text(item) for item in (result.content or [])]
        text = "\n".join(part for part in parts if part)
        if result.isError:
            return "Error: " + (text or "tool reported an error")
        return text or "(no output)"

    def status(self) -> List[Dict[str, Any]]:
        with self._guard:
            return [
                {"id": server_id, **self._states.get(server_id, _idle())}
                for server_id in self._servers
            ]

    def tools(self) -> List[Dict[str, Any]]:
        with self._guard:
            rows = []
            for server_id, state in self._states.items():
                if state.get("state") != "ready":
                    continue
                rows.append({"server": server_id, "tools": state.get("tools", [])})
            return rows

    def shutdown(self) -> None:
        with self._guard:
            for server_id in list(self._sessions):
                try:
                    self._run(self._close(server_id), CONNECT_TIMEOUT)
                except Exception:
                    pass
            self._sessions.clear()
            self._states.clear()
            self._servers.clear()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._refresh_bridge()

    # ── async internals ─────────────────────────────────────────────
    async def _start(self, server: Dict[str, Any]) -> None:
        server_id = server["id"]
        self._states[server_id] = {"state": "starting", "error": "", "tools": []}
        if not MCP_AVAILABLE:
            self._states[server_id] = {
                "state": "error",
                "error": "mcp package is not installed",
                "tools": [],
            }
            return
        stack = AsyncExitStack()
        try:
            if server["transport"] == "stdio":
                env = {**os.environ, **server.get("env", {})}
                command = shutil.which(server["command"]) or server["command"]
                params = StdioServerParameters(
                    command=command, args=server.get("args", []), env=env
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params)
                )
            else:
                read_stream, write_stream, *_ = await stack.enter_async_context(
                    streamablehttp_client(
                        server["url"], headers=server.get("headers") or {}
                    )
                )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
            listed = await asyncio.wait_for(
                session.list_tools(), timeout=CONNECT_TIMEOUT
            )
            self._sessions[server_id] = _Session(stack, session, list(listed.tools))
            self._states[server_id] = {
                "state": "ready",
                "error": "",
                "tools": list(listed.tools),
            }
        except Exception as error:
            try:
                await stack.aclose()
            except Exception:
                pass
            self._sessions.pop(server_id, None)
            self._states[server_id] = {
                "state": "error",
                "error": str(error),
                "tools": [],
            }

    async def _close(self, server_id: str) -> None:
        session = self._sessions.pop(server_id, None)
        if session is None:
            return
        try:
            await session.stack.aclose()
        except Exception:
            pass

    def _refresh_bridge(self) -> None:
        try:
            from dizel_core.mcp import bridge

            bridge.refresh(self)
        except Exception as error:
            print(f"[dizel-mcp] bridge refresh failed: {error}", flush=True)


def _content_text(item: Any) -> str:
    kind = getattr(item, "type", "")
    if kind == "text":
        return getattr(item, "text", "") or ""
    if kind == "image":
        return f"[image: {getattr(item, 'mimeType', 'unknown')}]"
    if kind == "audio":
        return "[audio omitted]"
    if kind == "resource":
        resource = getattr(item, "resource", None)
        return f"[resource: {getattr(resource, 'uri', '')}]"
    return f"[{kind or 'content'} omitted]"


_manager: MCPManager | None = None


def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
```

Note: `_refresh_bridge` imports `bridge` which Task 4 creates. Until Task 4 lands, the import fails and is swallowed with a printed warning — tests still pass.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_mcp_manager.py -q`
Expected: pass (the manager tests spawn the fixture server with `sys.executable`).

- [ ] **Step 6: Checkpoint**

Run: `python -m pytest tests/test_mcp_manager.py tests/test_tools_catalog.py tests/test_mcp_config.py -q` — all pass.

---

### Task 4: MCP tool bridge

**Files:**
- Create: `dizel_core/mcp/bridge.py`
- Test: `tests/test_mcp_bridge.py`

**Interfaces:**
- Produces: `bridge.refresh(manager) -> dict[str, Tool]` (registers ready-server tools via `tools.set_external`); helpers `tool_name(server_id, name) -> str`, `arg_map(input_schema) -> dict[str, str]`.
- Consumes: `manager.tools()` / `manager.call(...)` (Task 3); `tools.Tool` / `tools.set_external` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_bridge.py
from types import SimpleNamespace

from dizel_core import tools
from dizel_core.mcp import bridge


def _spec(name, description="does a thing", schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema
        or {
            "type": "object",
            "properties": {"path": {"type": "string"}, "lines": {"type": ["integer", "null"]}},
            "required": ["path"],
        },
    )


class FakeManager:
    def __init__(self, rows, result="ok"):
        self.rows = rows
        self.calls = []
        self._result = result

    def tools(self):
        return self.rows

    def call(self, server_id, tool, arguments):
        self.calls.append((server_id, tool, arguments))
        return self._result


def test_names_are_namespaced_and_sanitized():
    manager = FakeManager([{"server": "My Server", "tools": [_spec("Read File")]}])
    built = bridge.refresh(manager)
    try:
        assert list(built) == ["my_server__read_file"]
    finally:
        tools.set_external({})


def test_schema_rendering_marks_optional_args():
    manager = FakeManager([{"server": "fs", "tools": [_spec("read")]}])
    bridge.refresh(manager)
    try:
        text = tools.catalog()["fs__read"].schema()
        assert '"path": "string"' in text
        assert '"lines": "integer?"' in text
        assert "does a thing" in text
    finally:
        tools.set_external({})


def test_runner_dispatches_to_manager():
    manager = FakeManager([{"server": "fs", "tools": [_spec("read")]}], result="file body")
    bridge.refresh(manager)
    try:
        assert tools.execute({"name": "fs__read", "args": {"path": "a.txt"}}) == "file body"
        assert manager.calls == [("fs", "read", {"path": "a.txt"})]
    finally:
        tools.set_external({})


def test_duplicate_sanitized_names_get_suffix():
    rows = [
        {"server": "my server", "tools": [_spec("tool")]},
        {"server": "my-server", "tools": [_spec("tool")]},
    ]
    built = bridge.refresh(FakeManager(rows))
    try:
        assert sorted(built) == ["my_server__tool", "my_server__tool_2"]
    finally:
        tools.set_external({})


def test_error_state_servers_contribute_nothing():
    assert bridge.refresh(FakeManager([])) == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mcp_bridge.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Implement**

```python
# dizel_core/mcp/bridge.py
"""Adapt ready MCP tools into the textual tool registry."""

from __future__ import annotations

import re
from typing import Any, Dict

from dizel_core import tools as tool_registry

_NAME = re.compile(r"[^a-z0-9_]+")


def _clean(value: str) -> str:
    return _NAME.sub("_", str(value).lower()).strip("_") or "tool"


def tool_name(server_id: str, name: str) -> str:
    return f"{_clean(server_id)}__{_clean(name)}"


def arg_map(input_schema: Any) -> Dict[str, str]:
    if not isinstance(input_schema, dict):
        return {}
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = set(input_schema.get("required") or [])
    args: Dict[str, str] = {}
    for key, spec in properties.items():
        kind = "string"
        if isinstance(spec, dict):
            declared = spec.get("type")
            if isinstance(declared, list):
                declared = next((t for t in declared if t != "null"), "string")
            kind = str(declared or "string")
        args[str(key)] = kind if key in required else f"{kind}?"
    return args


def _description(spec: Any) -> str:
    text = " ".join(str(getattr(spec, "description", "") or "").split())
    return text[:150]


def _runner(manager: Any, server_id: str, tool: str):
    def run(**kwargs: Any) -> str:
        return manager.call(server_id, tool, kwargs)

    return run


def build(manager: Any) -> Dict[str, tool_registry.Tool]:
    result: Dict[str, tool_registry.Tool] = {}
    for row in manager.tools():
        server_id = row["server"]
        for spec in row["tools"]:
            name = tool_name(server_id, spec.name)
            while name in result or name in tool_registry.REGISTRY:
                name += "_2"
            result[name] = tool_registry.Tool(
                name=name,
                description=_description(spec),
                args=arg_map(getattr(spec, "inputSchema", None)),
                run=_runner(manager, server_id, spec.name),
            )
    return result


def refresh(manager: Any) -> Dict[str, tool_registry.Tool]:
    built = build(manager)
    tool_registry.set_external(built)
    return built
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mcp_bridge.py -q`
Expected: pass.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/test_mcp_bridge.py tests/test_mcp_manager.py -q` — all pass (manager now refreshes the bridge).

---

### Task 5: Sidecar routes and lifecycle

**Files:**
- Create: `api/routes/mcp.py`
- Modify: `api/main.py` (import router, `include_router`, add `PUT` to CORS, startup/shutdown hooks)
- Test: `tests/test_mcp_routes.py`

**Interfaces:**
- Produces: `GET /api/mcp/servers` → `{servers: [public server]}` where public = config minus `env`/`headers`, plus `env_keys`, `header_keys`, `state`, `error`, `tools: [{name, description}]`; `PUT /api/mcp/servers` body `{servers: [...]}` (422 on invalid, secrets merged, manager reload); `POST /api/mcp/servers/{id}/test`.
- Consumes: Tasks 1–4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_routes.py
import pytest
from fastapi.testclient import TestClient

import dizel_core.config_manager as cm


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_put_then_get_roundtrip_without_secret_values(client):
    response = client.put(
        "/api/mcp/servers",
        json={
            "servers": [
                {
                    "id": "linear",
                    "name": "Linear",
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "headers": {"Authorization": "Bearer secret"},
                    "enabled": False,
                }
            ]
        },
    )
    assert response.status_code == 200, response.text
    server = response.json()["servers"][0]
    assert server["id"] == "linear"
    assert server["state"] == "disabled"
    assert server["header_keys"] == ["Authorization"]
    assert "headers" not in server and "env" not in server

    again = client.get("/api/mcp/servers").json()["servers"][0]
    assert again["header_keys"] == ["Authorization"]
    assert "Bearer secret" not in str(again)


def test_blank_secret_value_keeps_stored_value(client, tmp_path):
    cm.ConfigManager.save(
        {
            "mcp": {
                "servers": [
                    {
                        "id": "linear",
                        "name": "Linear",
                        "transport": "http",
                        "url": "https://mcp.linear.app/mcp",
                        "headers": {"Authorization": "Bearer kept"},
                        "enabled": False,
                    }
                ]
            }
        }
    )
    client.put(
        "/api/mcp/servers",
        json={
            "servers": [
                {
                    "id": "linear",
                    "name": "Linear",
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "headers": {"Authorization": ""},
                    "enabled": False,
                }
            ]
        },
    )
    stored = cm.ConfigManager.load()["mcp"]["servers"][0]
    assert stored["headers"]["Authorization"] == "Bearer kept"


def test_missing_field_keeps_stored_secret(client, tmp_path):
    """A toggle/delete round-trip sends public objects without env/headers."""
    cm.ConfigManager.save(
        {
            "mcp": {
                "servers": [
                    {
                        "id": "linear",
                        "name": "Linear",
                        "transport": "http",
                        "url": "https://mcp.linear.app/mcp",
                        "headers": {"Authorization": "Bearer kept"},
                        "enabled": False,
                    }
                ]
            }
        }
    )
    client.put(
        "/api/mcp/servers",
        json={
            "servers": [
                {
                    "id": "linear",
                    "name": "Linear",
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "enabled": True,
                }
            ]
        },
    )
    stored = cm.ConfigManager.load()["mcp"]["servers"][0]
    assert stored["headers"]["Authorization"] == "Bearer kept"


def test_invalid_config_returns_422(client):
    response = client.put("/api/mcp/servers", json={"servers": [{"name": "x", "transport": "http"}]})
    assert response.status_code == 422


def test_test_route_unknown_server_404(client):
    assert client.post("/api/mcp/servers/nope/test").status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mcp_routes.py -q`
Expected: FAIL (404 on `/api/mcp/servers`).

- [ ] **Step 3: Implement the route**

```python
# api/routes/mcp.py
"""MCP server configuration and live status for the desktop app."""

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from api.services.config_service import ConfigService
from dizel_core.mcp import config as mcp_config
from dizel_core.mcp.manager import get_manager

router = APIRouter(prefix="/mcp", tags=["mcp"])

_SECRET_FIELDS = ("env", "headers")


def _load() -> list[dict]:
    return mcp_config.load(ConfigService.get_config())


def _merge_secrets(payload: list[dict], previous: list[dict]) -> list[dict]:
    """Keep stored secrets unless a non-empty value replaces them.

    Public server objects never carry env/headers, so a toggle or delete
    round-trip arrives without those fields — carry them over wholesale.
    """
    previous_by_id = {server["id"]: server for server in previous}
    merged: list[dict] = []
    for entry in payload:
        if not isinstance(entry, dict):
            merged.append(entry)
            continue
        entry = dict(entry)
        old = previous_by_id.get(entry.get("id"))
        if old:
            for field in _SECRET_FIELDS:
                if field not in entry:
                    if old.get(field):
                        entry[field] = dict(old[field])
                    continue
                values = entry.get(field) or {}
                old_values = old.get(field) or {}
                entry[field] = {
                    key: (value if value else old_values.get(key, ""))
                    for key, value in values.items()
                }
        merged.append(entry)
    return merged


def _public(servers: list[dict]) -> Dict[str, Any]:
    live = {row["id"]: row for row in get_manager().status()}
    rows = []
    for server in servers:
        state = live.get(
            server["id"],
            {"state": "starting" if server.get("enabled") else "disabled", "error": "", "tools": []},
        )
        public = {key: value for key, value in server.items() if key not in _SECRET_FIELDS}
        public["env_keys"] = sorted(server.get("env", {}))
        public["header_keys"] = sorted(server.get("headers", {}))
        public["state"] = state.get("state", "disabled")
        public["error"] = state.get("error", "")
        public["tools"] = [
            {"name": tool.name, "description": getattr(tool, "description", "") or ""}
            for tool in state.get("tools", [])
        ]
        rows.append(public)
    return {"servers": rows}


@router.get("/servers")
def list_servers() -> Dict[str, Any]:
    return _public(_load())


@router.put("/servers")
def set_servers(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    previous = _load()
    try:
        merged = _merge_secrets(payload.get("servers") or [], previous)
        saved = mcp_config.save(merged)
    except mcp_config.MCPConfigError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    get_manager().apply_async(saved)
    return _public(saved)


@router.post("/servers/{server_id}/test")
def test_server(server_id: str) -> Dict[str, Any]:
    try:
        state = get_manager().test(server_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"unknown server: {server_id}") from error
    match = next(
        (row for row in _public(_load())["servers"] if row["id"] == server_id), None
    )
    return {"server": match, "state": state.get("state"), "error": state.get("error", "")}
```

In `api/main.py`:

- Import: add `mcp,` to the `from api.routes import (...)` list (alphabetical, after `health,`).
- Register: after the `chat` router line, add `app.include_router(mcp.router, prefix="/api", tags=["mcp"])`.
- CORS: `allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]`.
- Lifecycle, after the existing `_announce` startup handler:

```python
@app.on_event("startup")
def _start_mcp() -> None:
    from api.services.config_service import ConfigService
    from dizel_core.mcp import config as mcp_config
    from dizel_core.mcp.manager import get_manager

    get_manager().apply_async(mcp_config.load(ConfigService.get_config()))


@app.on_event("shutdown")
def _stop_mcp() -> None:
    from dizel_core.mcp.manager import get_manager

    get_manager().shutdown()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mcp_routes.py -q`
Expected: pass. Note the test server entries are disabled, so nothing spawns.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/test_mcp_routes.py tests/test_mcp_manager.py -q` — all pass.

---

### Task 6: GUI API client and types

**Files:**
- Modify: `inference/gui/src/services/api.ts` (append an MCP section before `// ── Voice` or at the end)

**Interfaces:**
- Produces: types `McpToolInfo`, `McpServerInfo`, `McpServerState`; functions `mcpServers()`, `setMcpServers(servers)`, `testMcpServer(id)`.

- [ ] **Step 1: Add the client**

```ts
// ── MCP servers ─────────────────────────────────────────────────────────────

export type McpServerState = "disabled" | "starting" | "ready" | "error";

export interface McpToolInfo {
  name: string;
  description: string;
}

export interface McpServerInfo {
  id: string;
  name: string;
  transport: "stdio" | "http";
  enabled: boolean;
  command?: string;
  args?: string[];
  url?: string;
  env_keys: string[];
  header_keys: string[];
  state: McpServerState;
  error: string;
  tools: McpToolInfo[];
}

export const mcpServers = () =>
  request<{ servers: McpServerInfo[] }>("/mcp/servers");

export const setMcpServers = (servers: unknown[]) =>
  request<{ servers: McpServerInfo[] }>("/mcp/servers", {
    method: "PUT",
    body: JSON.stringify({ servers }),
  });

export const testMcpServer = (id: string) =>
  request<{ server: McpServerInfo | null; state: McpServerState; error: string }>(
    `/mcp/servers/${encodeURIComponent(id)}/test`,
    { method: "POST" },
  );
```

- [ ] **Step 2: Typecheck**

Run: `cd inference/gui && npm run typecheck`
Expected: pass.

- [ ] **Step 3: Checkpoint**

Run: `cd inference/gui && npm run typecheck` once more after any import cleanup; confirm no errors.

---

### Task 7: GUI Settings → MCP tab

**Files:**
- Create: `inference/gui/src/components/settingsUi.tsx` (extract `Section`, `Switch`, `Row` from SettingsDialog unchanged)
- Create: `inference/gui/src/components/McpSettings.tsx`
- Modify: `inference/gui/src/components/SettingsDialog.tsx` (remove the now-shared local components, import them; add MCP tab + content branch + footer guard)
- Modify: `inference/gui/src/styles/app.css` (MCP styles after the settings styles, ~line 4000, and motion block in Task 9's file)
- Verify: typecheck/build + a headless capture (commands below)

**Interfaces:**
- Consumes: `McpServerInfo` and the three API functions (Task 6); `set-section`, `set-input`, `set-textarea`, `set-switch` styles.
- Produces: `MCP` tab reachable from the settings nav; server CRUD saving immediately via the API.

- [ ] **Step 1: Extract shared settings primitives**

Move `Section` (`SettingsDialog.tsx:94-116`), `Switch` (`120-141`) and `Row` (`143-167`) verbatim into `settingsUi.tsx` with `export` on each, and re-import them in SettingsDialog:

```tsx
// inference/gui/src/components/settingsUi.tsx
import type { ReactNode } from "react";

export function Section({ title, hint, action, children }: {...}) { /* verbatim */ }
export function Switch({ checked, onChange, label }: {...}) { /* verbatim */ }
export function Row({ label, hint, children, onClick, spec }: {...}) { /* verbatim */ }
```

In SettingsDialog, delete the three local definitions and add:

```tsx
import { Row, Section, Switch } from "./settingsUi";
```

- [ ] **Step 2: Create the MCP section component**

```tsx
// inference/gui/src/components/McpSettings.tsx
import { useCallback, useEffect, useRef, useState } from "react";

import { Icon } from "./Icon";
import { Section, Switch } from "./settingsUi";
import * as api from "../services/api";
import type { McpServerInfo, McpServerState } from "../services/api";

const STATE_LABEL: Record<McpServerState, string> = {
  ready: "Connected",
  starting: "Starting…",
  error: "Error",
  disabled: "Off",
};

interface Draft {
  id?: string;
  name: string;
  transport: "stdio" | "http";
  command: string;
  args: string;
  env: string;
  url: string;
  headers: string;
  enabled: boolean;
}

const EMPTY: Draft = {
  id: undefined,
  name: "",
  transport: "stdio",
  command: "",
  args: "",
  env: "",
  url: "",
  headers: "",
  enabled: true,
};

function draftFrom(server: McpServerInfo): Draft {
  return {
    id: server.id,
    name: server.name,
    transport: server.transport,
    command: server.command ?? "",
    args: (server.args ?? []).join("\n"),
    env: server.env_keys.join("\n"),
    url: server.url ?? "",
    headers: server.header_keys.join("\n"),
    enabled: server.enabled,
  };
}

function parseKv(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const eq = trimmed.indexOf("=");
    if (eq === 0) continue;
    if (eq < 0) {
      out[trimmed] = "";
      continue;
    }
    out[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
  }
  return out;
}

export function McpSettings() {
  const [servers, setServers] = useState<McpServerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const alive = useRef(true);

  const load = useCallback(async () => {
    try {
      const data = await api.mcpServers();
      if (alive.current) {
        setServers(data.servers);
        setError("");
      }
    } catch (err) {
      if (alive.current) setError((err as Error).message);
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    void load();
    const timer = window.setInterval(() => void load(), 2000);
    return () => {
      alive.current = false;
      window.clearInterval(timer);
    };
  }, [load]);

  const save = async (next: Ddraft[]) => {
    setBusy(true);
    setError("");
    try {
      const data = await api.setMcpServers(next);
      setServers(data.servers);
      setDraft(null);
      setNotice("Saved.");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const payloadFor = (entry: McpServerInfo | null, draftValue: Draft) => {
    const base: Record<string, unknown> = {
      id: entry?.id,
      name: draftValue.name.trim(),
      transport: draftValue.transport,
      enabled: draftValue.enabled,
    };
    if (draftValue.transport === "stdio") {
      base.command = draftValue.command.trim();
      base.args = draftValue.args
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      base.env = parseKv(draftValue.env);
    } else {
      base.url = draftValue.url.trim();
      base.headers = parseKv(draftValue.headers);
    }
    return base;
  };

  const commitDraft = () => {
    if (!draft) return;
    if (!draft.name.trim()) {
      setError("A name is required.");
      return;
    }
    const current = draft.id ? servers.find((s) => s.id === draft.id) ?? null : null;
    const payload = payloadFor(current, draft);
    const next = current
      ? servers.map((s) => (s.id === current.id ? { ...s, ...payload } : s))
      : [...servers, payload];
    void save(next);
  };

  const toggle = (server: McpServerInfo) => {
    void save(
      servers.map((s) => (s.id === server.id ? { ...s, enabled: !s.enabled } : s)),
    );
  };

  const remove = (server: McpServerInfo) => {
    setConfirmId(null);
    void save(servers.filter((s) => s.id !== server.id));
  };

  const test = async (server: McpServerInfo) => {
    setBusy(true);
    try {
      const result = await api.testMcpServer(server.id);
      setNotice(
        result.state === "ready"
          ? `${server.name}: connected.`
          : `${server.name}: ${result.error || result.state}.`,
      );
      await load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Section
        title="MCP servers"
        hint="Enabled servers start with the app. Their tools appear when the model's tools switch is on."
        action={
          <button className="btn small" onClick={() => setDraft({ ...EMPTY })} disabled={busy}>
            <Icon name="plus" size={12} />
            Add server
          </button>
        }
      >
        {loading && <div className="set-note">Reading servers…</div>}
        {!loading && servers.length === 0 && (
          <div className="set-note">
            No MCP servers yet. Add one to give the model extra tools — filesystem, GitHub,
            a database, anything that speaks MCP.
          </div>
        )}
        <div className="mcp-list">
          {servers.map((server) => (
            <div className="mcp-card" key={server.id}>
              <div className="mcp-card-head">
                <span className="mcp-dot" data-state={server.state} />
                <span className="mcp-card-name">{server.name}</span>
                <span className="mcp-badge">{server.transport}</span>
                <span className="mcp-card-tools">
                  {server.state === "ready"
                    ? `${server.tools.length} tool${server.tools.length === 1 ? "" : "s"}`
                    : STATE_LABEL[server.state]}
                </span>
                <span className="mcp-card-actions">
                  <button className="sub-icon-btn" title="Test connection" onClick={() => void test(server)} disabled={busy}>
                    <Icon name="refresh" size={12} />
                  </button>
                  <button className="sub-icon-btn" title="Edit" onClick={() => setDraft(draftFrom(server))} disabled={busy}>
                    <Icon name="settings" size={12} />
                  </button>
                  <button className="sub-icon-btn" title="Remove" onClick={() => setConfirmId(server.id)} disabled={busy}>
                    <Icon name="trash" size={12} />
                  </button>
                </span>
                <Switch
                  checked={server.enabled}
                  onChange={() => toggle(server)}
                  label={`Enable ${server.name}`}
                />
              </div>
              {server.error && <div className="mcp-error">{server.error}</div>}
              {confirmId === server.id && (
                <div className="mcp-confirm">
                  <span>Remove “{server.name}” and its tools?</span>
                  <button className="btn small" onClick={() => setConfirmId(null)}>
                    Cancel
                  </button>
                  <button className="btn small danger" onClick={() => remove(server)}>
                    Remove
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
        {notice && <div className="set-note">{notice}</div>}
        {error && <div className="set-error">{error}</div>}
      </Section>

      {draft && (
        <Section
          title={draft.id ? `Edit ${draft.name || "server"}` : "New server"}
          hint="Values are stored in settings.json on this machine. Leave a value blank to keep the stored one."
        >
          <div className="mcp-form">
            <label>
              Name
              <input
                className="set-input"
                value={draft.name}
                onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                spellCheck={false}
              />
            </label>
            <label>
              Transport
              <select
                className="set-select"
                value={draft.transport}
                onChange={(event) =>
                  setDraft({ ...draft, transport: event.target.value as Draft["transport"] })
                }
              >
                <option value="stdio">stdio — spawn a local command</option>
                <option value="http">http — Streamable HTTP endpoint</option>
              </select>
            </label>
            {draft.transport === "stdio" ? (
              <>
                <label>
                  Command
                  <input
                    className="set-input"
                    value={draft.command}
                    placeholder="npx"
                    onChange={(event) => setDraft({ ...draft, command: event.target.value })}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Arguments, one per line
                  <textarea
                    className="set-textarea"
                    rows={3}
                    value={draft.args}
                    placeholder={"-y\n@modelcontextprotocol/server-filesystem\nC:\\work"}
                    onChange={(event) => setDraft({ ...draft, args: event.target.value })}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Environment, KEY=VALUE per line
                  <textarea
                    className="set-textarea"
                    rows={2}
                    value={draft.env}
                    onChange={(event) => setDraft({ ...draft, env: event.target.value })}
                    spellCheck={false}
                  />
                </label>
              </>
            ) : (
              <>
                <label>
                  URL
                  <input
                    className="set-input"
                    value={draft.url}
                    placeholder="https://mcp.example.com/mcp"
                    onChange={(event) => setDraft({ ...draft, url: event.target.value })}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Headers, KEY=VALUE per line
                  <textarea
                    className="set-textarea"
                    rows={2}
                    value={draft.headers}
                    onChange={(event) => setDraft({ ...draft, headers: event.target.value })}
                    spellCheck={false}
                  />
                </label>
              </>
            )}
            <div className="mcp-form-actions">
              <button className="btn" onClick={() => setDraft(null)}>
                Cancel
              </button>
              <button className="btn primary" onClick={commitDraft} disabled={busy}>
                Save server
              </button>
            </div>
          </div>
        </Section>
      )}
    </>
  );
}
```

(Fix the typo `Ddraft` → `Draft` when writing the file.)

- [ ] **Step 3: Wire the tab into SettingsDialog**

- `TABS`: `["Chat", "MCP", "Model", "Appearance", "Speech", "Shortcuts", "About"]`.
- `TAB_ICONS`: `MCP: "globe"`.
- `TAB_HINTS`: `MCP: "External tool servers the model can call"`.
- Import at top: `import { McpSettings } from "./McpSettings";`
- Content branch after the `tab === "Chat"` block ends (before `tab === "Model"`):

```tsx
                {tab === "MCP" && <McpSettings />}
```

- Footer reset guard (line ~1710): add `|| tab === "MCP"` to the disabled condition.
- MCP edits save immediately through its own API, so `Save & close` needs no MCP fields.

- [ ] **Step 4: Add styles**

Append in `app.css` near the settings styles:

```css
/* ── MCP settings ──────────────────────────────────────────────────────── */

.mcp-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}
.mcp-card {
  display: grid;
  gap: 6px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-panel);
}
.mcp-card-head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.mcp-card-name {
  font-weight: 500;
}
.mcp-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-dim);
  flex-shrink: 0;
}
.mcp-dot[data-state="ready"] {
  background: #34d399;
}
.mcp-dot[data-state="starting"] {
  background: #f59e0b;
  animation: mcp-pulse 1.2s ease-in-out infinite;
}
.mcp-dot[data-state="error"] {
  background: var(--danger);
}
@keyframes mcp-pulse {
  50% {
    opacity: 0.35;
  }
}
.mcp-badge {
  font-size: 11px;
  padding: 1px 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text-secondary);
}
.mcp-card-tools {
  font-size: 12px;
  color: var(--text-dim);
}
.mcp-card-actions {
  display: flex;
  gap: 2px;
  margin-left: auto;
}
.mcp-error {
  font-size: 12px;
  color: var(--danger);
}
.mcp-confirm {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  color: var(--text-secondary);
}
.mcp-form {
  display: grid;
  gap: 10px;
  width: 100%;
}
.mcp-form label {
  display: grid;
  gap: 5px;
  font-size: 12.5px;
  color: var(--text-secondary);
}
.mcp-form .set-input,
.mcp-form .set-select,
.mcp-form .set-textarea {
  width: 100%;
}
.mcp-form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
```

- [ ] **Step 5: Verify build**

Run: `cd inference/gui && npm run build`
Expected: clean tsc + vite build.

- [ ] **Step 6: Headless capture (verification)**

```bash
TMP=$(mktemp -d)
mkdir -p "$TMP/config"
cat > "$TMP/config/settings.json" <<'JSON'
{"mcp": {"servers": [{"id": "demo", "name": "Demo server", "transport": "http", "url": "https://example.com/mcp", "enabled": false, "headers": {}}]}}
JSON
cat > "$TMP/open-mcp.js" <<'JS'
setTimeout(() => {
  const btn = [...document.querySelectorAll("button")].find((b) =>
    ((b.getAttribute("aria-label") || b.title || "").toLowerCase().includes("settings")));
  if (btn) btn.click();
}, 800);
setTimeout(() => {
  const tab = [...document.querySelectorAll(".set-nav-item")].find((b) => b.textContent.trim() === "MCP");
  if (tab) tab.click();
}, 1800);
JS
cd inference/gui
DIZEL_CONFIG_DIR="$TMP/config" DIZEL_SHOT="$TMP/mcp-tab.png" DIZEL_SHOT_DELAY=6000 \
  DIZEL_SHOT_JS_FILE="$TMP/open-mcp.js" npx electron .
```

Inspect `$TMP/mcp-tab.png`: the MCP tab shows the Demo server card with a dim status dot, "http" badge, Off state, and Add server. If the settings button selector misses, adjust the JS to click the actual settings trigger and re-run.

- [ ] **Step 7: Checkpoint**

`npm run build` passes and the capture shows the MCP tab correctly.

---

### Task 8: TUI `/mcp` command, modal, and lifecycle

**Files:**
- Create: `inference/tui/panels/modals/mcp_modal.py`
- Create: `inference/tui/commands/builtins/mcp.py`
- Create: `scripts/snap_tui_mcp.py`
- Modify: `inference/tui/panels/modals/__init__.py` (export `McpModal`)
- Modify: `inference/tui/commands/builtins/__init__.py` (import + register)
- Modify: `inference/tui/app.py` (start manager after mount, stop on unmount)
- Test: `tests/test_mcp_tui_helpers.py`

**Interfaces:**
- Produces: `row_text(server: dict, state: dict) -> str`, `parse_kv_lines(text: str) -> dict`, `format_kv_lines(mapping: dict) -> str`, `McpModal`, `McpFormModal`, `/mcp` command.
- Consumes: Tasks 1–4, `ConfirmModal`/`plain_text`/`set_error` from `inference/tui/panels/modals/common.py`, `TextArea`/`OptionList`/`Input`.

- [ ] **Step 1: Write the failing helper tests**

```python
# tests/test_mcp_tui_helpers.py
from inference.tui.panels.modals.mcp_modal import (
    format_kv_lines,
    parse_kv_lines,
    row_text,
)


def test_row_text_states():
    server = {"id": "fs", "name": "Files", "transport": "stdio", "enabled": True}
    assert row_text(server, {"state": "ready", "tools": [1, 2]}).startswith("●  Files")
    assert "2 tools" in row_text(server, {"state": "ready", "tools": [1, 2]})
    assert row_text(server, {"state": "disabled"}).startswith("○  Files")


def test_parse_kv_lines_ignores_blank_and_bare_keys():
    assert parse_kv_lines("A=1\n\nB = two \nC\n=D") == {"A": "1", "B": "two", "C": ""}


def test_format_kv_lines_roundtrip():
    assert format_kv_lines({"A": "1", "B": "2"}) == "A=1\nB=2"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mcp_tui_helpers.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Implement the modal**

```python
# inference/tui/panels/modals/mcp_modal.py
"""List, add, edit, test and toggle MCP servers from the TUI."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from inference.tui.panels.modals.common import ConfirmModal, plain_text, set_error

_GLYPHS = {"ready": "●", "starting": "◐", "error": "✕", "disabled": "○"}


def row_text(server: dict, state: dict) -> str:
    glyph = _GLYPHS.get(str(state.get("state", "disabled")), "○")
    count = len(state.get("tools", []) or [])
    tools = f"{count} tool" + ("" if count == 1 else "s")
    return f"{glyph}  {server['name']}  ·  {server['transport']}  ·  {tools}"


def parse_kv_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        key, sep, value = trimmed.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip() if sep else ""
    return values


def format_kv_lines(mapping: dict) -> str:
    return "\n".join(f"{key}={value}" for key, value in (mapping or {}).items())


class McpFormModal(ModalScreen[dict | None]):
    """Add or edit one server; dismisses with a raw payload."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, server: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._server = dict(server) if server else {}

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card mcp-form-card"):
            with Horizontal(classes="modal-head"):
                title = "Edit MCP server" if self._server else "Add MCP server"
                yield Static(plain_text(title), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static(plain_text("name"), classes="field-label")
            yield Input(value=self._server.get("name", ""), id="mcp-name", classes="modal-input")
            yield Static(plain_text("transport: stdio or http"), classes="field-label")
            yield Input(value=self._server.get("transport", "stdio"), id="mcp-transport", classes="modal-input")
            yield Static(plain_text("stdio command"), classes="field-label")
            yield Input(value=self._server.get("command", ""), id="mcp-command", classes="modal-input")
            yield Static(plain_text("stdio args, one per line"), classes="field-label")
            yield TextArea("\n".join(self._server.get("args", [])), id="mcp-args")
            yield Static(plain_text("stdio env, KEY=VALUE per line"), classes="field-label")
            yield TextArea(format_kv_lines(self._server.get("env", {})), id="mcp-env")
            yield Static(plain_text("http url"), classes="field-label")
            yield Input(value=self._server.get("url", ""), id="mcp-url", classes="modal-input")
            yield Static(plain_text("http headers, KEY=VALUE per line"), classes="field-label")
            yield TextArea(format_kv_lines(self._server.get("headers", {})), id="mcp-headers")
            yield Static("", classes="modal-error")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="mcp-cancel")
                yield Button("Save", id="mcp-save", variant="primary")

    def on_mount(self) -> None:
        self.query_one(".modal-error", Static).styles.display = "none"
        self.query_one("#mcp-name", Input).focus()

    @on(Button.Pressed, "#mcp-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#mcp-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        from dizel_core.mcp import config as mcp_config

        transport = self.query_one("#mcp-transport", Input).value.strip().lower()
        payload: dict[str, Any] = {
            "id": self._server.get("id"),
            "name": self.query_one("#mcp-name", Input).value.strip(),
            "transport": transport,
            "enabled": self._server.get("enabled", True),
        }
        if transport == "stdio":
            payload["command"] = self.query_one("#mcp-command", Input).value.strip()
            payload["args"] = [
                line.strip()
                for line in self.query_one("#mcp-args", TextArea).text.splitlines()
                if line.strip()
            ]
            payload["env"] = parse_kv_lines(self.query_one("#mcp-env", TextArea).text)
        else:
            payload["url"] = self.query_one("#mcp-url", Input).value.strip()
            payload["headers"] = parse_kv_lines(self.query_one("#mcp-headers", TextArea).text)

        try:
            normalized = mcp_config.validate_servers([payload])[0]
        except mcp_config.MCPConfigError as error:
            set_error(self, str(error))
            return
        self.dismiss(normalized)

    def action_cancel(self) -> None:
        self.dismiss(None)


class McpModal(ModalScreen[None]):
    """The /mcp surface: list servers, toggle, test, add, edit, delete."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close"),
        Binding("space", "toggle", "Toggle"),
        Binding("t", "test", "Test"),
        Binding("a", "add", "Add"),
        Binding("e", "edit", "Edit"),
        Binding("d", "delete", "Delete"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card mcp-card"):
            with Horizontal(classes="modal-head"):
                yield Static(plain_text("MCP servers"), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static(
                plain_text(
                    "Enabled servers start with the app. Tools appear when the model's tools switch is on."
                ),
                classes="modal-sub",
            )
            yield OptionList(classes="modal-list")
            yield Static(
                "space toggle · t test · a add · e edit · d delete · esc close",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        self._servers: list[dict] = []
        self._states: dict[str, dict] = {}
        self._reload()
        self.set_interval(1.5, self._reload)

    def _reload(self) -> None:
        from dizel_core.config_manager import ConfigManager
        from dizel_core.mcp import config as mcp_config
        from dizel_core.mcp.manager import get_manager

        self._servers = mcp_config.load(ConfigManager.load())
        self._states = {row["id"]: row for row in get_manager().status()}
        options = self.query_one(OptionList)
        previous = None
        if options.highlighted is not None and options.option_count:
            previous = options.get_option_at_index(options.highlighted).id
        options.clear_options()
        for server in self._servers:
            state = self._states.get(
                server["id"],
                {"state": "starting" if server.get("enabled") else "disabled", "tools": []},
            )
            options.add_option(Option(plain_text(row_text(server, state)), id=server["id"]))
        if previous:
            for index in range(options.option_count):
                if options.get_option_at_index(index).id == previous:
                    options.highlighted = index
                    break

    def _selected(self) -> dict | None:
        options = self.query_one(OptionList)
        if options.highlighted is None:
            return None
        option = options.get_option_at_index(options.highlighted)
        return next((s for s in self._servers if s["id"] == option.id), None)

    def _persist(self, servers: list[dict]) -> None:
        from dizel_core.mcp import config as mcp_config
        from dizel_core.mcp.manager import get_manager

        try:
            saved = mcp_config.save(servers)
        except mcp_config.MCPConfigError as error:
            self.notify(str(error), severity="error")
            return
        get_manager().apply_async(saved)
        self._servers = saved
        self._reload()

    def action_toggle(self) -> None:
        server = self._selected()
        if server is None:
            return
        servers = [dict(entry) for entry in self._servers]
        for entry in servers:
            if entry["id"] == server["id"]:
                entry["enabled"] = not entry.get("enabled")
        self._persist(servers)

    def action_test(self) -> None:
        server = self._selected()
        if server is None:
            return
        from dizel_core.mcp.manager import get_manager

        try:
            state = get_manager().test(server["id"])
        except KeyError:
            self.notify("Save the server first, then test it.", severity="warning")
            return
        message = f"{server['name']}: {state.get('state')}"
        if state.get("error"):
            message += f" — {state['error']}"
        self.notify(message, severity="information" if state.get("state") == "ready" else "warning")
        self._reload()

    def action_add(self) -> None:
        self.app.push_screen(McpFormModal(), self._on_form)

    def action_edit(self) -> None:
        server = self._selected()
        if server is not None:
            self.app.push_screen(McpFormModal(server), self._on_form)

    def _on_form(self, payload: dict | None) -> None:
        if not payload:
            return
        servers = [dict(entry) for entry in self._servers]
        existing = next(
            (index for index, entry in enumerate(servers) if entry["id"] == payload.get("id")),
            None,
        )
        if existing is None:
            servers.append(payload)
        else:
            servers[existing] = payload
        self._persist(servers)

    def action_delete(self) -> None:
        server = self._selected()
        if server is None:
            return

        def _confirm(ok: bool) -> None:
            if ok:
                self._persist([s for s in self._servers if s["id"] != server["id"]])

        self.app.push_screen(
            ConfirmModal(
                "Remove server",
                f"Remove “{server['name']}” and its tools?",
                confirm_label="Remove",
            ),
            _confirm,
        )

    def action_close_modal(self) -> None:
        self.dismiss(None)
```

- [ ] **Step 4: Register the command**

```python
# inference/tui/commands/builtins/mcp.py
from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class McpCommand(Command):
    name = "mcp"
    help_text = "Manage MCP servers and see the tools they expose."
    category = "Tools"
    usage = "/mcp"
    palette_hint = "/mcp"
    examples = ["/mcp"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from inference.tui.panels.modals import McpModal

        app.push_screen(McpModal())
        return ""
```

In `inference/tui/commands/builtins/__init__.py`: add `from .mcp import McpCommand` and `registry.register(McpCommand())` after `ToolsCommand`.

In `inference/tui/panels/modals/__init__.py`: add `from inference.tui.panels.modals.mcp_modal import McpModal` and add `"McpModal"` to `__all__`.

- [ ] **Step 5: App lifecycle**

In `inference/tui/app.py`, inside `on_mount` next to the other `call_after_refresh` calls (~line 232):

```python
        self.call_after_refresh(self._start_mcp)
```

Add methods near `_restore_resumed_session` (line 243):

```python
    def _start_mcp(self) -> None:
        from dizel_core.config_manager import ConfigManager
        from dizel_core.mcp import config as mcp_config
        from dizel_core.mcp.manager import get_manager

        get_manager().apply_async(mcp_config.load(ConfigManager.load()))

    def _stop_mcp(self) -> None:
        from dizel_core.mcp.manager import get_manager

        get_manager().shutdown()

    def on_unmount(self) -> None:
        self._stop_mcp()
```

If an `on_unmount` already exists, call `self._stop_mcp()` at its end instead of adding a second definition.

- [ ] **Step 6: Snapshot script**

```python
# scripts/snap_tui_mcp.py
"""Headless snapshot of the /mcp modal."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.tui.app import DizelCMDApp
from inference.tui.panels.modals import McpModal

OUT = Path(__file__).parent / ".tui-mcp-snapshot.svg"


async def main() -> None:
    app = DizelCMDApp()
    async with app.run_test(size=(150, 42)) as pilot:
        await pilot.pause()
        app.push_screen(McpModal())
        await pilot.pause()
        OUT.write_text(app.export_screenshot(), encoding="utf-8")
        print("wrote", OUT)


asyncio.run(main())
```

- [ ] **Step 7: Run tests and snapshot**

```bash
python -m pytest tests/test_mcp_tui_helpers.py -q
```
Expected: pass.

```bash
TMP=$(mktemp -d)
cat > "$TMP/settings.json" <<'JSON'
{"mcp": {"servers": [{"id": "demo", "name": "Demo server", "transport": "http", "url": "https://example.com/mcp", "enabled": false, "headers": {}}]}}
JSON
DIZEL_CONFIG_DIR="$TMP" python scripts/snap_tui_mcp.py
```

Open `scripts/.tui-mcp-snapshot.svg` (or convert to PNG) and confirm: title "MCP servers", one row `○  Demo server · http · 0 tools`, footer bindings. Delete the SVG afterwards unless the maintainer wants it kept.

- [ ] **Step 8: Checkpoint**

`python -m pytest tests/test_mcp_tui_helpers.py -q` passes and the snapshot renders the list row correctly.

---

### Task 9: Workspaces motion pass

**Files:**
- Modify: `inference/gui/src/components/SecondarySidebar.tsx` (`key` on `.sub-body`, Library removal state)
- Modify: `inference/gui/src/styles/app.css` (motion block after `.sub-body`, hover lift near `.sub-row`, gem leaving near `.prompt-gem`, meter sweep near `.meter`, popover icon nudge near `.popover-item`)
- Verify: build + captures with motion on/off

- [ ] **Step 1: Remount the panel per view**

In `SecondarySidebar.tsx:408`, change:

```tsx
      <div className="sub-body">
```
to
```tsx
      <div className="sub-body" key={view}>
```

- [ ] **Step 2: Library removal transition**

In `LibraryView` (SecondarySidebar.tsx:222-326): add state and a delayed remove:

```tsx
  const [removing, setRemoving] = useState<string | null>(null);
```

Add beside the other handlers:

```tsx
  const removeWithMotion = (id: string) => {
    setRemoving(id);
    window.setTimeout(() => {
      removePrompt(id);
      setRemoving(null);
    }, 200);
  };
```

Change the gem wrapper class and the delete handler:

```tsx
          <div key={prompt.id} className={removing === prompt.id ? "prompt-gem is-leaving" : "prompt-gem"}>
```

```tsx
                onClick={() => removeWithMotion(prompt.id)}
```

- [ ] **Step 3: Add the motion CSS**

After the `.sub-body { ... }` block (app.css:2308-2315):

```css
/* ── Workspaces motion ─────────────────────────────────────────────────── */

.sub-body {
  animation: ws-swap 180ms var(--ease) both;
}

@keyframes ws-swap {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@keyframes ws-rise {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card) {
  animation: ws-rise 220ms var(--ease) both;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(2) {
  animation-delay: 28ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(3) {
  animation-delay: 56ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(4) {
  animation-delay: 84ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(5) {
  animation-delay: 112ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(6) {
  animation-delay: 140ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(7) {
  animation-delay: 168ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(8) {
  animation-delay: 196ms;
}
.sub-body :is(.sub-row, .prompt-gem, .sub-empty, .sub-group, .stat-card):nth-child(n + 9) {
  animation-delay: 224ms;
}
```

Extend `.sub-row` (line 2352) with lift and press:

```css
.sub-row {
  transition:
    background var(--dur) var(--ease),
    transform var(--dur) var(--ease);
}
.sub-row:hover {
  transform: translateY(-1px);
}
.sub-row:active {
  transform: scale(0.985);
}
```

Add next to `.prompt-gem` (line 2462):

```css
.prompt-gem.is-leaving {
  opacity: 0;
  transform: scale(0.96);
  transition:
    opacity 200ms var(--ease),
    transform 200ms var(--ease);
  pointer-events: none;
}
```

After `.meter > span` (line 4999):

```css
.subsidebar .meter > span {
  transform-origin: left center;
  animation: meter-sweep 600ms cubic-bezier(0.22, 0.7, 0.3, 1) both;
}

@keyframes meter-sweep {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
}
```

Near `.popover-item:hover` (line 1942):

```css
.popover-item .mask-icon,
.popover-item svg {
  transition: transform var(--dur) var(--ease);
}
.popover-item:hover .mask-icon,
.popover-item:hover svg {
  transform: translateX(1px) scale(1.05);
}
```

The existing global kill switches (`app.css:3931-3947`) already zero every animation and transition above; no extra rules are needed.

- [ ] **Step 4: Build**

Run: `cd inference/gui && npm run build`
Expected: clean.

- [ ] **Step 5: Capture with motion on and off**

```bash
TMP=$(mktemp -d)
cat > "$TMP/open-workspaces.js" <<'JS'
setTimeout(() => {
  const item = [...document.querySelectorAll(".nav-item")].find((b) => (b.textContent || "").includes("Workspaces"));
  if (item) item.click();
}, 800);
setTimeout(() => {
  const entry = [...document.querySelectorAll(".popover-item")].find((b) => (b.textContent || "").includes("Monitor"));
  if (entry) entry.click();
}, 1400);
JS
cd inference/gui
DIZEL_CONFIG_DIR="$TMP/config" DIZEL_SHOT="$TMP/ws-motion.png" DIZEL_SHOT_DELAY=5000 \
  DIZEL_SHOT_JS_FILE="$TMP/open-workspaces.js" npx electron .
DIZEL_CONFIG_DIR="$TMP/config" DIZEL_SHOT="$TMP/ws-static.png" DIZEL_SHOT_DELAY=5000 \
  DIZEL_SHOT_JS="document.documentElement.dataset.motion='off';" \
  DIZEL_SHOT_JS_FILE="$TMP/open-workspaces.js" npx electron .
```

Inspect both captures: the Monitor panel shows meters filled in both; motion-on should look identical after settling (animations end at their final state). Mid-animation correctness is covered by the CSS (transform-only, both fill mode).

- [ ] **Step 6: Checkpoint**

Build passes; both captures show the Monitor workspace correctly with no layout shift.

---

### Task 10: Dependency, full verification, cleanup

**Files:**
- Modify: `requirements.txt` (add MCP block after the tools block, line ~52)

- [ ] **Step 1: Add the dependency**

```text
# ── MCP client (external tool servers) ───────────────────────────────────
mcp>=1.27.0
```

- [ ] **Step 2: Run the full Python suite**

Run: `python -m pytest tests/ -q`
Expected: all pass. If an unrelated pre-existing test fails, record it and confirm it also fails on a clean checkout before this branch (do not "fix" unrelated tests).

- [ ] **Step 3: Build the GUI**

Run: `cd inference/gui && npm run build`
Expected: clean.

- [ ] **Step 4: End-to-end MCP sanity (in-process)**

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from dizel_core.mcp.manager import MCPManager
from dizel_core import tools
from dizel_core.mcp import bridge

fixture = Path("tests/fixtures/fake_mcp_server.py").resolve()
manager = MCPManager()
manager.apply([{
    "id": "fake", "name": "Fake", "transport": "stdio", "enabled": True,
    "command": sys.executable, "args": [str(fixture)], "env": {},
}])
try:
    print("state:", manager.status()[0]["state"])
    print("registry:", sorted(bridge.build(manager)))
    tools.set_external(bridge.build(manager))
    print("call:", tools.execute({"name": "fake__echo", "args": {"text": "hello"}}))
finally:
    manager.shutdown()
PY
```

Expected: `state: ready`, registry contains `fake__add`, `fake__echo`, `fake__explode`, call prints `echo: hello`. Then confirm no `fake_mcp_server` process remains (`psutil` scan or Task Manager).

- [ ] **Step 5: Docs touch-up**

Check `inference/tui/commands/builtins/help.py` lists commands from the registry automatically (it does, via `registry`), so `/mcp` needs no manual help edit. If any hardcoded command list exists, add `/mcp` there.

- [ ] **Step 6: Final checkpoint**

Report: pytest results, GUI build result, the MCP sanity output, and the two capture paths. Leave all changes uncommitted for the maintainer's review.

---

## Self-review checklist (run before executing)

- Spec coverage: config (T1), registry seam + `/chat/tools` (T2), manager lifecycle (T3), bridge/naming (T4), sidecar routes/lifespan/CORS (T5), GUI API (T6), GUI tab + secrets hint (T7), TUI command/modal/lifecycle (T8), Workspaces motion (T9), dependency + verification (T10).
- Type consistency: `Tool(name, description, args, run)`; `manager.status() -> [{id, state, error, tools}]`; `bridge.refresh(manager)`; `McpServerInfo.env_keys/header_keys`; `McpModal` exported via `panels.modals`.
- No placeholders: every new file has complete code; every command is exact.
