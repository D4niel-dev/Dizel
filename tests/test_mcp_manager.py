import asyncio
import io
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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


def _fixture_processes() -> list:
    import psutil

    return [
        proc
        for proc in psutil.process_iter(["name", "cmdline"])
        if "python" in (proc.info["name"] or "").lower()
        and "fake_mcp_server" in " ".join(proc.info["cmdline"] or [])
    ]


def _fixture_pids() -> set:
    return {proc.pid for proc in _fixture_processes()}


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


def test_repeated_apply_is_idempotent():
    manager = _manager()
    try:
        before = _fixture_pids()
        assert before, "fixture server did not start"
        manager.apply([SERVER])
        assert _fixture_pids() == before
        assert manager.status()[0]["state"] == "ready"
    finally:
        manager.shutdown()


def test_apply_async_then_shutdown_leaves_no_orphan():
    manager = MCPManager()
    manager.apply_async([SERVER])
    manager.shutdown()

    deadline = time.time() + 5
    while time.time() < deadline:
        loop_threads = [
            thread for thread in threading.enumerate() if thread.name == "dizel-mcp"
        ]
        if not _fixture_pids() and not loop_threads:
            return
        time.sleep(0.05)
    pytest.fail(
        f"orphans after shutdown: pids={_fixture_pids()} "
        f"threads={[thread.name for thread in threading.enumerate()]}"
    )


def test_test_route_recovers_and_reports_clean_error():
    manager = _manager()
    try:
        info = manager.test("fake")
        assert info["state"] == "ready", info
        assert info["error"] == ""
        assert manager.status()[0]["state"] == "ready"
    finally:
        manager.shutdown()


def test_call_reconnects_after_transport_failure():
    manager = _manager()
    try:
        processes = _fixture_processes()
        assert processes, "fixture server did not start"
        for proc in processes:
            proc.kill()
        for proc in processes:
            proc.wait(timeout=5)
        assert manager.call("fake", "echo", {"text": "hi"}) == "echo: hi"
        assert manager.status()[0]["state"] == "ready"
    finally:
        manager.shutdown()


# ── Fix wave: output cap, pagination, errlog, serialized apply ─────────────


class _FakeText:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResult:
    def __init__(self, parts: list) -> None:
        self.content = parts
        self.isError = False


class _FakeSession:
    def __init__(self, parts: list) -> None:
        self._parts = parts

    async def call_tool(self, tool: str, arguments: dict) -> _FakeResult:
        return _FakeResult(self._parts)


def _inject_session(manager: MCPManager, server: dict, session) -> None:
    """Wire a fake live session into a manager without spawning anything."""
    from dizel_core.mcp.manager import _Handle, _settled

    handle = _Handle(dict(server))
    handle.session = session
    with manager._guard:
        manager._servers = {server["id"]: dict(server)}
        manager._handles[server["id"]] = handle
        manager._states[server["id"]] = _settled("ready")


def _drop_handles(manager: MCPManager) -> None:
    with manager._guard:
        manager._handles.clear()


def test_call_caps_large_output():
    from dizel_core.mcp.manager import (
        MAX_OUTPUT_CHARS,
        MAX_PART_CHARS,
        TRUNCATION_MARKER,
    )

    manager = MCPManager()
    _inject_session(
        manager, SERVER, _FakeSession([_FakeText("a" * 10000), _FakeText("b" * 10000)])
    )
    try:
        result = manager.call("fake", "dump", {})
    finally:
        _drop_handles(manager)
        manager.shutdown()
    assert result.endswith(TRUNCATION_MARKER)
    assert len(result) == MAX_OUTPUT_CHARS + len(TRUNCATION_MARKER)
    # Each part is capped first (with its own marker), then the join is capped.
    prefix = "a" * MAX_PART_CHARS + TRUNCATION_MARKER + "\n"
    assert result.startswith(prefix + "b" * (MAX_OUTPUT_CHARS - len(prefix)))


class _FakeLister:
    def __init__(self, pages: list) -> None:
        self._pages = pages
        self.cursors: list = []

    async def list_tools(self, cursor=None) -> SimpleNamespace:
        self.cursors.append(cursor)
        return self._pages[len(self.cursors) - 1]


def test_list_all_tools_follows_next_cursor():
    manager = MCPManager()
    page_one = SimpleNamespace(
        tools=[SimpleNamespace(name="one")], nextCursor="page-2"
    )
    page_two = SimpleNamespace(tools=[SimpleNamespace(name="two")], nextCursor=None)
    lister = _FakeLister([page_one, page_two])
    try:
        tools = asyncio.run(manager._list_all_tools(lister))
    finally:
        manager.shutdown()
    assert [tool.name for tool in tools] == ["one", "two"]
    assert lister.cursors == [None, "page-2"]


def test_list_all_tools_stops_at_page_cap():
    from dizel_core.mcp.manager import MAX_TOOL_PAGES

    class _Endless:
        def __init__(self) -> None:
            self.calls = 0

        async def list_tools(self, cursor=None) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(
                tools=[SimpleNamespace(name=f"t{self.calls}")],
                nextCursor=f"c{self.calls}",
            )

    manager = MCPManager()
    endless = _Endless()
    try:
        tools = asyncio.run(manager._list_all_tools(endless))
    finally:
        manager.shutdown()
    assert endless.calls == MAX_TOOL_PAGES
    assert len(tools) == MAX_TOOL_PAGES


def test_owned_errlog_is_closed_on_shutdown(tmp_path):
    manager = MCPManager()
    handle = open(tmp_path / "mcp.log", "a", encoding="utf-8")
    manager.set_errlog(handle, owned=True)
    manager.shutdown()
    assert handle.closed


def test_caller_owned_errlog_is_not_closed():
    stream = io.StringIO()
    manager = MCPManager(errlog=stream)
    try:
        assert manager._errlog is stream
        manager.apply([{**SERVER, "enabled": False}])
    finally:
        manager.shutdown()
    assert not stream.closed


def test_apply_async_burst_leaves_no_orphan_after_shutdown():
    manager = MCPManager()
    manager.apply_async([SERVER])
    manager.apply_async([])
    manager.shutdown()

    deadline = time.time() + 5
    while time.time() < deadline:
        apply_threads = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("dizel-mcp")
        ]
        if not _fixture_pids() and not apply_threads:
            return
        time.sleep(0.05)
    pytest.fail(
        f"orphans after shutdown: pids={_fixture_pids()} "
        f"threads={[thread.name for thread in threading.enumerate()]}"
    )


def _test_error(manager: MCPManager, server_id: str = "fake"):
    try:
        manager.test(server_id)
    except BaseException as error:  # noqa: BLE001 - captured for the assertion
        return error
    return None


def test_test_after_removal_raises_keyerror_instead_of_respawning():
    manager = MCPManager()
    try:
        manager.apply([{**SERVER, "enabled": False}])
        errors: list = []
        lock = manager._lock_for("fake")
        with lock:
            worker = threading.Thread(
                target=lambda: errors.append(_test_error(manager))
            )
            worker.start()
            # Remove the server while the test call waits for the lock; it
            # must re-check after acquiring and bail instead of re-spawning.
            with manager._guard:
                manager._servers.pop("fake", None)
                manager._states.pop("fake", None)
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert errors and isinstance(errors[0], KeyError)
    finally:
        manager.shutdown()


def test_apply_async_latest_snapshot_wins_without_overlap():
    class _BlockingManager(MCPManager):
        def __init__(self) -> None:
            super().__init__()
            self.applied: list = []
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def _apply(self, servers) -> None:
            self.applied.append([server["id"] for server in servers])
            if len(self.applied) == 1:
                self.first_started.set()
                self.release_first.wait(5)

    server = {"id": "alpha", "name": "Alpha", "enabled": False}
    manager = _BlockingManager()
    try:
        manager.apply_async([server])
        assert manager.first_started.wait(5), "apply worker never started"
        # Queue a newer snapshot while the first apply is still running.
        manager.apply_async([])
        manager.release_first.set()
        deadline = time.time() + 5
        while time.time() < deadline and len(manager.applied) < 2:
            time.sleep(0.01)
        assert manager.applied == [["alpha"], []]
    finally:
        manager.release_first.set()
        manager.shutdown()


def test_tools_ignores_removed_and_disabled_servers():
    manager = MCPManager()
    try:
        manager.apply([{**SERVER, "enabled": False}])
        with manager._guard:
            # Even a stale ready row must not contribute tools.
            manager._states["fake"] = {
                "state": "ready",
                "error": "",
                "tools": ["echo"],
            }
            manager._states["ghost"] = {
                "state": "ready",
                "error": "",
                "tools": ["x"],
            }
        assert manager.tools() == []
    finally:
        manager.shutdown()
