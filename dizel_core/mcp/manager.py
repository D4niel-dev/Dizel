"""MCP client connections on one background event loop.

Public methods are synchronous so the chat thread can call them directly;
all async work is marshalled onto a dedicated daemon thread.

Every server is owned by a single long-lived coroutine on that loop: it enters
the transport and session contexts, reports readiness, waits for its stop
event, and exits the contexts again in the same task. The anyio task groups
inside the MCP client reject context exits from a different task, so a
connection must never be opened or closed straight from a caller task.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import os
import shutil
import threading
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
CLOSE_TIMEOUT = 5.0
LOOP_JOIN_TIMEOUT = 5.0
# Tool output lands in the model's prompt, so it is capped like built-in tools
# instead of trusting a server to stay small.
MAX_OUTPUT_CHARS = 6000
MAX_PART_CHARS = 4000
TRUNCATION_MARKER = "\n[truncated]"
# A paginating server could otherwise keep a connect suspended indefinitely.
MAX_TOOL_PAGES = 20


def _idle() -> Dict[str, Any]:
    return {"state": "disabled", "error": "", "tools": []}


def _settled(state: str, error: str = "", tools: Any = None) -> Dict[str, Any]:
    return {"state": state, "error": error, "tools": list(tools or [])}


class _Handle:
    """One server connection, owned by a single coroutine on the loop."""

    def __init__(self, server: Dict[str, Any]) -> None:
        self.server = server
        self.stop = asyncio.Event()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.future: Any = None
        self.session: Any = None
        self.closing = False
        self.close_error = ""


class MCPManager:
    def __init__(self, errlog: Any = None) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._handles: Dict[str, _Handle] = {}
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._states: Dict[str, Dict[str, Any]] = {}
        self._attempted: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, Any] = {}
        self._guard = threading.RLock()
        self._apply_lock = threading.Lock()
        self._closed = False
        # Applied-server snapshots are coalesced by one serialized worker; an
        # older snapshot can never overtake a newer one.
        self._pending_servers: List[Dict[str, Any]] | None = None
        self._pending_gen = 0
        self._apply_thread: threading.Thread | None = None
        self._pending_cond = threading.Condition(self._guard)
        # Textual draws to the same stderr a spawned server inherits, so the
        # TUI routes child output to a file through this handle.
        self._errlog = errlog
        self._owns_errlog = False
        # Global preferences from the MCP settings tab. Defaults mirror the
        # module constants above so behavior is unchanged until configured.
        self._connect_timeout = CONNECT_TIMEOUT
        self._call_timeout = CALL_TIMEOUT
        self._max_output_chars = MAX_OUTPUT_CHARS
        self._retry_calls = True

    # ── loop plumbing ────────────────────────────────────────────────
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise RuntimeError("MCP manager is shut down")
        if self._loop is None or self._thread is None or not self._thread.is_alive():
            old = self._loop
            if old is not None and not old.is_closed():
                try:
                    old.close()
                except Exception:
                    pass
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="dizel-mcp", daemon=True
            )
            thread.start()
            self._loop = loop
            self._thread = thread
        return self._loop

    def _is_closed(self) -> bool:
        with self._guard:
            return self._closed

    def _lock_for(self, server_id: str) -> Any:
        with self._guard:
            lock = self._locks.get(server_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[server_id] = lock
            return lock

    def set_errlog(self, stream: Any, owned: bool = False) -> None:
        """Route spawned-server stderr to ``stream``; ``None`` keeps the SDK default.

        ``owned`` marks a stream the manager itself opened, so ``shutdown()``
        closes it. Caller-owned streams (including the constructor argument)
        are left open.
        """
        with self._guard:
            if self._closed:
                # shutdown() already ran (e.g. a repeat TUI mount); never own
                # a stream nobody will close.
                if owned and stream is not None:
                    _close_stream(stream)
                return
            previous = self._errlog if self._owns_errlog else None
            self._errlog = stream
            self._owns_errlog = bool(owned) and stream is not None
        if previous is not None and previous is not stream:
            _close_stream(previous)

    def configure(self, options: Dict[str, Any]) -> None:
        """Apply global MCP preferences (timeouts, retry, output cap).

        Values are clamped to sane ranges so a hand-edited settings.json can
        never make a call hang forever or truncate tool output to nothing.
        Takes effect on the next connect or call; live connections are kept.
        """

        def _number(key: str, default: float, lo: float, hi: float) -> float:
            try:
                value = float(options.get(key, default))
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, value))

        with self._guard:
            self._connect_timeout = _number("connect_timeout", CONNECT_TIMEOUT, 5.0, 60.0)
            self._call_timeout = _number("call_timeout", CALL_TIMEOUT, 10.0, 300.0)
            self._max_output_chars = int(
                _number("max_output_chars", MAX_OUTPUT_CHARS, 1000, 32000)
            )
            self._retry_calls = options.get("retry_failed_calls", True) is not False

    # ── connection lifecycle ────────────────────────────────────────
    def _spawn(self, server: Dict[str, Any]) -> _Handle | None:
        """Register a new owner coroutine for ``server`` on the loop."""
        server_id = server["id"]
        with self._guard:
            if self._closed:
                return None
            handle = self._handles.get(server_id)
            if handle is not None:
                return handle
            loop = self._ensure_loop()
            handle = _Handle(server)
            self._handles[server_id] = handle
            self._states[server_id] = _settled("starting")
            self._attempted[server_id] = server
            coro = self._serve(handle)
            try:
                handle.future = asyncio.run_coroutine_threadsafe(coro, loop)
            except Exception as error:
                coro.close()
                self._handles.pop(server_id, None)
                self._states[server_id] = _settled("error", str(error))
                return None
        return handle

    def _wait_connected(self, handle: _Handle, timeout: float) -> bool:
        """Block until the owner settles; called without ``_guard`` held."""
        server_id = handle.server["id"]
        with self._guard:
            if self._closed or handle.closing:
                return False
        if not handle.ready.wait(timeout):
            with self._guard:
                handle.closing = True
            self._settle(handle, "error", f"timed out after {timeout:g}s")
            if handle.future is not None:
                handle.future.cancel()
            if handle.finished.wait(CLOSE_TIMEOUT):
                with self._guard:
                    if self._handles.get(server_id) is handle:
                        self._handles.pop(server_id, None)
            return False
        with self._guard:
            return (
                self._handles.get(server_id) is handle
                and not handle.closing
                and handle.session is not None
            )

    def _stop_handle(self, server_id: str, timeout: float = CLOSE_TIMEOUT) -> None:
        """Ask the owner coroutine to exit and wait for it (task-affine close)."""
        with self._guard:
            handle = self._handles.get(server_id)
            if handle is None:
                return
            handle.closing = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(handle.stop.set)
            except RuntimeError:
                pass
        if not handle.finished.wait(timeout):
            if handle.future is not None:
                handle.future.cancel()
            if not handle.finished.wait(timeout):
                if (
                    handle.future is not None
                    and handle.future.done()
                    and not handle.future.cancelled()
                ):
                    try:
                        exception = handle.future.exception()
                    except Exception:
                        exception = None
                    if exception is not None:
                        handle.close_error = (
                            handle.close_error
                            or str(exception)
                            or exception.__class__.__name__
                        )
                with self._guard:
                    if self._handles.get(server_id) is handle:
                        self._states[server_id] = _settled(
                            "error", "timed out while closing the connection"
                        )
                return
        if (
            not handle.close_error
            and handle.future is not None
            and handle.future.done()
            and not handle.future.cancelled()
        ):
            try:
                exception = handle.future.exception()
            except Exception:
                exception = None
            if exception is not None:
                handle.close_error = str(exception) or exception.__class__.__name__
        with self._guard:
            if self._handles.get(server_id) is not handle:
                return
            self._handles.pop(server_id, None)
            if server_id not in self._servers:
                return
            current = self._states.get(server_id, {}).get("state")
            if handle.close_error:
                self._states[server_id] = _settled(
                    "error", f"close failed: {handle.close_error}"
                )
            elif current in ("ready", "disabled"):
                self._states[server_id] = _settled("error", "connection closed")

    def _reconnect(self, server_id: str) -> _Handle | None:
        """Close the current owner (if any) and start one fresh connection."""
        with self._guard:
            if self._closed:
                return None
            server = self._servers.get(server_id)
        if server is None or not server.get("enabled"):
            return None
        self._stop_handle(server_id)
        handle = self._spawn(server)
        if handle is None or not self._wait_connected(handle, self._connect_timeout * 2 + 2):
            return None
        return handle

    # ── public, synchronous ─────────────────────────────────────────
    def apply(self, servers: List[Dict[str, Any]]) -> None:
        with self._apply_lock:
            self._apply(servers)

    def _apply(self, servers: List[Dict[str, Any]]) -> None:
        with self._guard:
            if self._closed:
                return
            wanted = {server["id"]: server for server in servers}
            self._servers = wanted
            to_close = {
                server_id
                for server_id, handle in self._handles.items()
                if server_id not in wanted or wanted[server_id] != handle.server
            }
            for server_id in list(self._states):
                if server_id not in wanted:
                    self._states.pop(server_id, None)
            for server_id in list(self._attempted):
                if server_id not in wanted:
                    self._attempted.pop(server_id, None)

            disabled = set()
            to_start = []
            for server in wanted.values():
                server_id = server["id"]
                if not server.get("enabled"):
                    disabled.add(server_id)
                    self._attempted.pop(server_id, None)
                    self._states[server_id] = _idle()
                    continue
                handle = self._handles.get(server_id)
                if server_id not in to_close and handle is not None:
                    continue
                if (
                    server_id not in to_close
                    and handle is None
                    and self._attempted.get(server_id) == server
                ):
                    # Same config as the last failed attempt: do not retry it
                    # implicitly.  Only test()/call() retry explicitly.
                    continue
                self._attempted[server_id] = server
                self._states[server_id] = _settled("starting")
                to_start.append(server)

        # The same per-server lock guards test()/call(); without it an apply
        # could delete a handle that a concurrent test is re-spawning, or
        # stop a connection the test just brought up.
        for server_id in to_close:
            with self._lock_for(server_id):
                self._stop_handle(server_id)
        with self._guard:
            if self._closed:
                return
            for server_id in disabled:
                if server_id in self._servers:
                    self._states[server_id] = _idle()

        for server in to_start:
            server_id = server["id"]
            with self._lock_for(server_id):
                with self._guard:
                    if self._closed:
                        return
                    current = self._servers.get(server_id)
                    if current is None or not current.get("enabled"):
                        continue
                    existing = self._handles.get(server_id)
                    if existing is not None and not existing.closing:
                        # test()/call() already brought this one up.
                        continue
                handle = self._spawn(server)
                if handle is None:
                    if self._is_closed():
                        return
                    continue
                self._wait_connected(handle, self._connect_timeout * 2 + 2)

        if not self._is_closed():
            self._refresh_bridge()

    def apply_async(self, servers: List[Dict[str, Any]]) -> None:
        """Queue a snapshot for the single serialized apply worker.

        Only the newest snapshot matters: a burst of calls coalesces into one
        apply, and an older snapshot can never overwrite a newer one.
        """
        with self._guard:
            if self._closed:
                return
            self._pending_servers = list(servers)
            self._pending_gen += 1
            self._pending_cond.notify()
            thread = self._apply_thread
            if thread is None or not thread.is_alive():
                self._apply_thread = threading.Thread(
                    target=self._apply_worker,
                    name="dizel-mcp-apply",
                    daemon=True,
                )
                self._apply_thread.start()

    def _apply_worker(self) -> None:
        """Apply the latest pending snapshot until no newer generation arrives."""
        while True:
            with self._guard:
                while self._pending_servers is None and not self._closed:
                    self._pending_cond.wait()
                if self._closed:
                    self._apply_thread = None
                    return
                servers = self._pending_servers
                generation = self._pending_gen
                self._pending_servers = None
            try:
                self.apply(servers)
            except Exception as error:
                print(f"[dizel-mcp] apply failed: {error}", flush=True)
            with self._guard:
                if self._closed or generation == self._pending_gen:
                    # No newer snapshot arrived while this one was applied.
                    self._apply_thread = None
                    return

    def test(self, server_id: str) -> Dict[str, Any]:
        lock = self._lock_for(server_id)
        with lock:
            # Re-check under the per-server lock: apply() may have removed the
            # server while this caller waited.  Never re-spawn a deleted one.
            with self._guard:
                if self._closed:
                    raise RuntimeError("MCP manager is shut down")
                server = self._servers.get(server_id)
                if server is None:
                    raise KeyError(server_id)
            self._stop_handle(server_id)
            if not server.get("enabled"):
                with self._guard:
                    self._attempted.pop(server_id, None)
                    if server_id in self._servers:
                        self._states[server_id] = _idle()
            else:
                handle = self._spawn(server)
                if handle is not None:
                    self._wait_connected(handle, self._connect_timeout * 2 + 2)
        with self._guard:
            if self._closed:
                state = _settled("error", "MCP manager is shut down")
            else:
                state = dict(self._states.get(server_id, _idle()))
        if not self._is_closed():
            self._refresh_bridge()
        return {"id": server_id, **state}

    def call(self, server_id: str, tool: str, arguments: Dict[str, Any]) -> str:
        with self._guard:
            if self._closed:
                return "MCP manager is shut down"
            server = self._servers.get(server_id)
            enabled = bool(server and server.get("enabled"))

        lock = self._lock_for(server_id)
        with lock:
            handle = self._live_handle(server_id)
            if handle is None:
                if not enabled:
                    return self._not_connected(server_id)
                handle = self._reconnect(server_id)
                if handle is None:
                    return self._not_connected(server_id)
            try:
                result = self._call_tool(handle.session, tool, arguments)
            except concurrent.futures.TimeoutError:
                return f"MCP call failed: timed out after {self._call_timeout:g}s"
            except Exception as error:
                handle = None
                if enabled and self._retry_calls:
                    handle = self._reconnect(server_id)
                if handle is None:
                    return f"MCP call failed: {error}"
                try:
                    result = self._call_tool(handle.session, tool, arguments)
                except concurrent.futures.TimeoutError:
                    return f"MCP call failed: timed out after {self._call_timeout:g}s"
                except Exception as retry_error:
                    return f"MCP call failed: {retry_error}"

        parts = [_cap_part(_content_text(item)) for item in (result.content or [])]
        text = "\n".join(part for part in parts if part)
        if result.isError:
            text = "Error: " + (text or "tool reported an error")
        elif not text:
            text = "(no output)"
        return _cap_output(text, self._max_output_chars)

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
                server = self._servers.get(server_id)
                if server is None or not server.get("enabled"):
                    # A removed/disabled server must not keep its tools
                    # registered until the next apply.
                    continue
                if state.get("state") != "ready":
                    continue
                rows.append({"server": server_id, "tools": state.get("tools", [])})
            return rows

    def configured(self) -> List[Dict[str, Any]]:
        """A copy of the applied server config, for cross-process reconcile."""
        with self._guard:
            return [copy.deepcopy(server) for server in self._servers.values()]

    def shutdown(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self._pending_servers = None
            self._pending_cond.notify_all()
            server_ids = list(self._handles)
            errlog = self._errlog if self._owns_errlog else None
            self._errlog = None
            self._owns_errlog = False
        for server_id in server_ids:
            self._stop_handle(server_id)
        self._stop_loop()
        with self._guard:
            self._handles.clear()
            self._servers.clear()
            self._states.clear()
            self._attempted.clear()
        if errlog is not None:
            _close_stream(errlog)
        self._refresh_bridge()

    # ── loop shutdown ───────────────────────────────────────────────
    def _stop_loop(self) -> None:
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None
        if loop is None:
            return
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._drain_and_stop, loop)
            except RuntimeError:
                pass
        if thread is not None:
            thread.join(timeout=LOOP_JOIN_TIMEOUT)
            if thread.is_alive():
                print(
                    "[dizel-mcp] shutdown: loop thread is still alive after "
                    f"{LOOP_JOIN_TIMEOUT:g}s",
                    flush=True,
                )
                return
        if not loop.is_closed():
            try:
                loop.close()
            except Exception:
                pass

    @staticmethod
    def _drain_and_stop(loop: asyncio.AbstractEventLoop) -> None:
        """Run on the loop thread: cancel every remaining task, then stop."""

        async def _drain() -> None:
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            loop.stop()

        try:
            loop.create_task(_drain())
        except Exception:
            loop.stop()

    # ── async internals, owned by one task per server ───────────────
    async def _serve(self, handle: _Handle) -> None:
        server = handle.server
        try:
            if handle.stop.is_set():
                return
            if not MCP_AVAILABLE:
                self._settle(handle, "error", "mcp package is not installed")
                return
            if server.get("transport") == "stdio":
                env = {**os.environ, **server.get("env", {})}
                command = shutil.which(server["command"]) or server["command"]
                params = StdioServerParameters(
                    command=command, args=server.get("args", []), env=env
                )
                errlog = self._errlog
                client = (
                    stdio_client(params, errlog=errlog)
                    if errlog is not None
                    else stdio_client(params)
                )
                async with client as (read, write):
                    async with ClientSession(read, write) as session:
                        await self._drive(handle, session)
            else:
                async with streamablehttp_client(
                    server["url"], headers=server.get("headers") or {}
                ) as (read, write, *_):
                    async with ClientSession(read, write) as session:
                        await self._drive(handle, session)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_failure(handle, error)
        finally:
            handle.session = None
            handle.ready.set()
            handle.finished.set()

    async def _drive(self, handle: _Handle, session: Any) -> None:
        await asyncio.wait_for(session.initialize(), timeout=self._connect_timeout)
        tools = await self._list_all_tools(session)
        if handle.stop.is_set():
            return
        handle.session = session
        self._settle(handle, "ready", "", tools)
        handle.ready.set()
        await handle.stop.wait()

    async def _list_all_tools(self, session: Any) -> List[Any]:
        """Collect every tools/list page, following ``nextCursor``."""
        tools: List[Any] = []
        cursor: str | None = None
        for _ in range(MAX_TOOL_PAGES):
            page = await asyncio.wait_for(
                session.list_tools(cursor=cursor), timeout=self._connect_timeout
            )
            tools.extend(list(page.tools or []))
            cursor = getattr(page, "nextCursor", None)
            if not cursor:
                return tools
        print(
            f"[dizel-mcp] tool list stopped after {MAX_TOOL_PAGES} pages",
            flush=True,
        )
        return tools

    def _settle(
        self, handle: _Handle, state: str, error: str = "", tools: Any = None
    ) -> None:
        server_id = handle.server["id"]
        with self._guard:
            if self._closed or self._handles.get(server_id) is not handle:
                return
            self._states[server_id] = _settled(state, error, tools)

    def _record_failure(self, handle: _Handle, error: Exception) -> None:
        message = str(error) or error.__class__.__name__
        if handle.closing:
            handle.close_error = message
            return
        self._settle(handle, "error", message)

    # ── helpers ─────────────────────────────────────────────────────
    def _live_handle(self, server_id: str) -> _Handle | None:
        with self._guard:
            handle = self._handles.get(server_id)
            if handle is None or handle.closing or handle.session is None:
                return None
            return handle

    def _not_connected(self, server_id: str) -> str:
        with self._guard:
            state = self._states.get(server_id)
        reason = "unknown"
        if state is not None:
            reason = state.get("error") or state.get("state", "unknown")
        return f"MCP server '{server_id}' is not connected: {reason}"

    def _call_tool(self, session: Any, tool: str, arguments: Dict[str, Any]) -> Any:
        with self._guard:
            if self._closed:
                raise RuntimeError("MCP manager is shut down")
            loop = self._ensure_loop()
        coro = session.call_tool(tool, arguments)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            coro.close()
            raise
        try:
            return future.result(timeout=self._call_timeout + 5)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise

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


def _cap_part(text: str) -> str:
    if len(text) > MAX_PART_CHARS:
        return text[:MAX_PART_CHARS] + TRUNCATION_MARKER
    return text


def _cap_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) > limit:
        return text[:limit] + TRUNCATION_MARKER
    return text


def _close_stream(stream: Any) -> None:
    try:
        stream.close()
    except Exception:
        pass


_manager: MCPManager | None = None


def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
