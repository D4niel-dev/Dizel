from inference.tui.panels.modals.mcp_modal import (
    format_kv_lines,
    merge_form_payload,
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


def _server(server_id: str, command: str, enabled: bool) -> dict:
    return {
        "id": server_id,
        "name": server_id.title(),
        "transport": "stdio",
        "command": command,
        "enabled": enabled,
    }


def test_add_collision_does_not_overwrite_existing_server():
    servers = [_server("demo", "original", False)]
    payload = {
        "id": "demo",
        "name": "Demo",
        "transport": "stdio",
        "command": "replacement",
        "enabled": True,
    }
    merged, error = merge_form_payload(servers, payload, edit_id=None)
    assert merged is None
    assert "demo" in error
    assert servers == [_server("demo", "original", False)]


def test_add_with_unused_id_appends():
    servers = [_server("demo", "original", False)]
    payload = {
        "id": "other",
        "name": "Other",
        "transport": "stdio",
        "command": "new",
        "enabled": True,
    }
    merged, error = merge_form_payload(servers, payload, edit_id=None)
    assert error == ""
    assert [entry["id"] for entry in merged] == ["demo", "other"]
    assert merged[1]["command"] == "new"


def test_edit_replaces_only_the_original_row():
    servers = [
        _server("demo", "original", False),
        _server("demo-2", "untouched", True),
    ]
    payload = {
        "id": "demo",
        "name": "Demo renamed",
        "transport": "stdio",
        "command": "updated",
        "enabled": False,
    }
    merged, error = merge_form_payload(servers, payload, edit_id="demo")
    assert error == ""
    assert merged[0]["command"] == "updated"
    assert merged[0]["name"] == "Demo renamed"
    assert merged[1] == servers[1]


def test_edit_of_a_vanished_row_saves_nothing():
    servers = [_server("demo", "original", False)]
    payload = _server("demo", "updated", False)
    merged, error = merge_form_payload(servers, payload, edit_id="gone")
    assert merged is None
    assert "no longer" in error
    assert servers == [_server("demo", "original", False)]
