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
