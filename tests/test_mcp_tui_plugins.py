import asyncio

import pytest

import dizel_core.mcp.manager as mcp_manager
from dizel_core import tools as tool_registry
from inference.tui.commands.builtins.plugins import PluginsCommand
from inference.tui.commands.parser import parse_command


class _Spec:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.inputSchema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }


class _Manager:
    def __init__(self, servers, states, result="Created LIN-42"):
        self._servers = servers
        self._states = states
        self.result = result
        self.calls = []

    def configured(self):
        return self._servers

    def status(self):
        return self._states

    def call(self, server_id, tool, arguments):
        self.calls.append((server_id, tool, arguments))
        return self.result


class _App:
    def __init__(self):
        self.prompts = []

    def insert_prompt(self, text):
        self.prompts.append(text)


SERVERS = [
    {"id": "linear", "name": "Linear", "enabled": True},
    {"id": "notes", "name": "Notes", "enabled": False},
]
STATES = [
    {
        "id": "linear",
        "state": "ready",
        "error": "",
        "tools": [
            _Spec("create_issue", "Create an issue\nwith a longer body."),
            _Spec("search_issues", "Search issues."),
        ],
    },
    {"id": "notes", "state": "disabled", "error": "", "tools": []},
]


@pytest.fixture()
def run(monkeypatch):
    def _run(raw, manager=None, app=None, tools_on=False):
        manager = manager or _Manager(SERVERS, STATES)
        app = app or _App()
        monkeypatch.setattr(mcp_manager, "get_manager", lambda: manager)
        monkeypatch.setattr(tool_registry, "_ENABLED", tools_on)
        result = asyncio.run(
            PluginsCommand().execute(app, parse_command(raw))
        )
        return result, manager, app

    return _run


def test_plugins_is_registered_as_a_tools_command():
    from inference.tui.commands.builtins import register_builtins
    from inference.tui.commands.registry import registry

    register_builtins()
    command = registry.lookup("plugins")
    assert isinstance(command, PluginsCommand)
    assert command.category == "Tools"


def test_listing_shows_tools_states_and_switch(run):
    text, _, _ = run("/plugins")
    assert text.startswith("Plugins · 2 tool(s)")
    assert "Model tools: off" in text
    assert "LINEAR  ·  connected  ·  2 tool(s)" in text
    assert "linear__create_issue" in text
    assert "Create an issue with a longer body." in text
    assert "NOTES  ·  off" in text
    assert "/plugins run <server> <tool>" in text


def test_listing_without_servers_points_at_mcp(run):
    result, _, _ = run("/plugins", manager=_Manager([], []))
    assert "No plugin servers configured" in result
    assert "/mcp" in result


def test_listing_reports_connection_errors(run):
    manager = _Manager(
        [{"id": "browser", "name": "Browser", "enabled": True}],
        [
            {
                "id": "browser",
                "state": "error",
                "error": "connect ECONNREFUSED",
                "tools": [],
            }
        ],
    )
    text, _, _ = run("/plugins", manager=manager)
    assert "BROWSER  ·  error — connect ECONNREFUSED" in text


def test_run_preserves_json_spacing_and_negative_numbers(run):
    text, manager, _ = run(
        '/plugins run Linear Create_Issue {"title": "Fix login", "n": -1}'
    )
    assert manager.calls == [
        ("linear", "create_issue", {"title": "Fix login", "n": -1})
    ]
    assert text.startswith("linear__create_issue →")
    assert "Created LIN-42" in text


def test_run_accepts_a_namespaced_tool_name(run):
    _, manager, _ = run('/plugins run linear linear__search_issues {"q": "x"}')
    assert manager.calls == [("linear", "search_issues", {"q": "x"})]


def test_run_without_json_sends_empty_arguments(run):
    _, manager, _ = run("/plugins run linear search_issues")
    assert manager.calls == [("linear", "search_issues", {})]


def test_prompt_inserts_a_compact_model_call(run):
    text, manager, app = run(
        '/plugins prompt linear create_issue {"title": "Fix login"}'
    )
    assert manager.calls == []
    assert app.prompts == [
        '<tool name="linear__create_issue">{"title":"Fix login"}</tool>'
    ]
    assert "Model tools are off" in text
    assert "/tools tools on" in text


def test_prompt_omits_the_switch_warning_when_tools_are_on(run):
    text, _, app = run(
        '/plugins prompt linear create_issue {"title": "Fix login"}', tools_on=True
    )
    assert app.prompts and "linear__create_issue" in app.prompts[0]
    assert "Model tools are off" not in text


def test_prompt_warns_when_the_server_is_not_connected(run):
    manager = _Manager(
        [{"id": "browser", "name": "Browser", "enabled": True}],
        [{"id": "browser", "state": "error", "error": "down", "tools": []}],
    )
    text, _, app = run(
        '/plugins prompt browser snapshot {"url": "x"}',
        manager=manager,
        tools_on=True,
    )
    assert app.prompts == [
        '<tool name="browser__snapshot">{"url":"x"}</tool>'
    ]
    assert "browser is not connected" in text
    assert "Model tools are off" not in text


def test_unknown_server_lists_configured_ids(run):
    text, manager, _ = run('/plugins run missing x {}')
    assert "Unknown server: missing" in text
    assert "linear, notes" in text
    assert manager.calls == []


def test_unknown_tool_lists_available_names(run):
    text, manager, _ = run('/plugins run linear missing {}')
    assert "Unknown tool for linear: missing" in text
    assert "create_issue, search_issues" in text
    assert manager.calls == []


def test_not_ready_server_still_tries_the_call(run):
    manager = _Manager(
        [{"id": "browser", "name": "Browser", "enabled": True}],
        [{"id": "browser", "state": "error", "error": "down", "tools": []}],
        result="MCP server 'browser' is not connected: down",
    )
    text, _, _ = run("/plugins run browser snapshot {}", manager=manager)
    assert manager.calls == [("browser", "snapshot", {})]
    assert "not connected" in text


def test_invalid_json_is_rejected(run):
    text, manager, _ = run('/plugins run linear create_issue {"title": }')
    assert "not valid JSON" in text
    assert manager.calls == []


def test_non_object_json_is_rejected(run):
    text, manager, _ = run("/plugins run linear create_issue [1, 2]")
    assert "must be a JSON object" in text
    assert manager.calls == []


def test_missing_arguments_print_usage(run):
    text, _, _ = run("/plugins run linear")
    assert text.startswith("Usage: /plugins")
    text, _, _ = run("/plugins unknown")
    assert text.startswith("Usage: /plugins")
