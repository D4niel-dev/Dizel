import pytest
from fastapi.testclient import TestClient

import dizel_core.config_manager as cm
import dizel_core.mcp.manager as mcp_manager


@pytest.fixture(autouse=True)
def _fresh_manager(monkeypatch):
    """TestClient exit shuts the process-wide manager down for good."""
    monkeypatch.setattr(mcp_manager, "_manager", None)


@pytest.fixture()
def apply_calls(monkeypatch):
    """Record apply_async calls so no test opens a real MCP connection."""
    calls = []

    def _record(self, servers):
        calls.append(list(servers))

    monkeypatch.setattr(mcp_manager.MCPManager, "apply_async", _record)
    return calls


@pytest.fixture()
def client(tmp_path, monkeypatch, apply_calls):
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
    response = client.put(
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
    assert response.status_code == 200, response.text
    assert response.json()["servers"][0]["header_keys"] == ["Authorization"]
    stored = cm.ConfigManager.load()["mcp"]["servers"][0]
    assert stored["headers"]["Authorization"] == "Bearer kept"


def test_missing_field_keeps_stored_secret(client, tmp_path, apply_calls):
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
    response = client.put(
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
    assert response.status_code == 200, response.text
    stored = cm.ConfigManager.load()["mcp"]["servers"][0]
    assert stored["headers"]["Authorization"] == "Bearer kept"
    assert stored["enabled"] is True
    assert any(call and call[0]["id"] == "linear" for call in apply_calls)


def test_merge_secrets_absent_blank_and_new_value():
    from api.routes.mcp import _merge_secrets

    previous = [
        {
            "id": "absent",
            "headers": {"Authorization": "Bearer stored"},
            "env": {"TOKEN": "stored token"},
        },
        {"id": "blank", "headers": {"Authorization": "Bearer stored"}},
        {"id": "new", "headers": {"Authorization": "Bearer stored"}},
    ]
    merged = _merge_secrets(
        [
            {"id": "absent"},
            {"id": "blank", "headers": {"Authorization": ""}},
            {"id": "new", "headers": {"Authorization": "Bearer fresh"}},
        ],
        previous,
    )
    assert merged[0]["headers"] == {"Authorization": "Bearer stored"}
    assert merged[0]["env"] == {"TOKEN": "stored token"}
    assert merged[1]["headers"] == {"Authorization": "Bearer stored"}
    assert merged[2]["headers"] == {"Authorization": "Bearer fresh"}


def test_invalid_config_returns_422(client):
    response = client.put("/api/mcp/servers", json={"servers": [{"name": "x", "transport": "http"}]})
    assert response.status_code == 422


@pytest.mark.parametrize("body", [{}, {"servers": None}, {"serverz": []}])
def test_put_without_servers_list_returns_422_and_keeps_config(client, body):
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
    before = cm.ConfigManager.load()["mcp"]["servers"]
    response = client.put("/api/mcp/servers", json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == "servers must be a list"
    assert cm.ConfigManager.load()["mcp"]["servers"] == before


def test_test_route_unknown_server_404(client):
    assert client.post("/api/mcp/servers/nope/test").status_code == 404


def test_put_malformed_headers_returns_422_and_keeps_config(client):
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
    before = cm.ConfigManager.load()["mcp"]["servers"]
    response = client.put(
        "/api/mcp/servers",
        json={
            "servers": [
                {
                    "id": "linear",
                    "name": "Linear",
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "headers": "NOT_A_DICT",
                    "enabled": False,
                }
            ]
        },
    )
    assert response.status_code == 422
    assert "must be an object" in response.json()["detail"]
    assert cm.ConfigManager.load()["mcp"]["servers"] == before


def test_get_servers_reconciles_config_edited_by_another_process(
    client, apply_calls
):
    manager = mcp_manager.get_manager()
    manager.apply(
        [
            {
                "id": "alpha",
                "name": "Alpha",
                "transport": "http",
                "url": "https://alpha.example.com/mcp",
                "headers": {},
                "enabled": False,
            }
        ]
    )
    apply_calls.clear()
    # Simulate the TUI writing settings.json while the sidecar runs.
    cm.ConfigManager.save(
        {
            "mcp": {
                "servers": [
                    {
                        "id": "beta",
                        "name": "Beta",
                        "transport": "http",
                        "url": "https://beta.example.com/mcp",
                        "enabled": False,
                    }
                ]
            }
        }
    )
    response = client.get("/api/mcp/servers")
    assert response.status_code == 200
    assert response.json()["servers"][0]["id"] == "beta"
    assert apply_calls, "GET did not reconcile the file config"
    assert [server["id"] for server in apply_calls[-1]] == ["beta"]


class _ToolSpec:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


def _save_server(server_id="linear", enabled=True):
    cm.ConfigManager.save(
        {
            "mcp": {
                "servers": [
                    {
                        "id": server_id,
                        "name": server_id.title(),
                        "transport": "http",
                        "url": f"https://{server_id}.example.com/mcp",
                        "enabled": enabled,
                    }
                ]
            }
        }
    )


def test_tools_catalog_lists_connected_tools_with_schemas(client, monkeypatch):
    from dizel_core import tools as tool_registry
    from dizel_core.mcp import config as mcp_config

    monkeypatch.setattr(tool_registry, "_ENABLED", True)
    _save_server(enabled=False)
    manager = mcp_manager.get_manager()
    manager.apply(mcp_config.load(cm.ConfigManager.load()))
    manager._states["linear"] = {
        "state": "ready",
        "error": "",
        "tools": [
            _ToolSpec(
                "create_issue",
                "Create an issue\n\nLonger explanation follows.",
                {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            )
        ],
    }

    response = client.get("/api/mcp/tools")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tools_enabled"] is True
    server = body["servers"][0]
    assert server["id"] == "linear"
    assert server["state"] == "ready"
    tool = server["tools"][0]
    assert tool["name"] == "create_issue"
    assert tool["namespaced"] == "linear__create_issue"
    assert tool["description"] == "Create an issue Longer explanation follows."
    assert tool["input_schema"]["required"] == ["title"]
    assert "env" not in server and "headers" not in server


def test_tools_catalog_shows_disabled_servers_without_tools(client, monkeypatch):
    _save_server(enabled=False)
    response = client.get("/api/mcp/tools")
    assert response.status_code == 200, response.text
    server = response.json()["servers"][0]
    assert server["state"] == "disabled"
    assert server["tools"] == []


def test_call_tool_posts_arguments_and_returns_output(client, monkeypatch):
    _save_server()
    calls = []

    def _record(self, server_id, tool, arguments):
        calls.append((server_id, tool, arguments))
        return "Created LIN-42"

    monkeypatch.setattr(mcp_manager.MCPManager, "call", _record)
    response = client.post(
        "/api/mcp/servers/linear/tools/create_issue",
        json={"arguments": {"title": "Fix login"}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == "Created LIN-42"
    assert body["namespaced"] == "linear__create_issue"
    assert calls == [("linear", "create_issue", {"title": "Fix login"})]


def test_call_tool_allows_tool_names_with_path_characters(client, monkeypatch):
    _save_server()
    seen = []

    def _record(self, server_id, tool, arguments):
        seen.append(tool)
        return "ok"

    monkeypatch.setattr(mcp_manager.MCPManager, "call", _record)
    response = client.post(
        "/api/mcp/servers/linear/tools/repo.read-file",
        json={"arguments": {}},
    )
    assert response.status_code == 200, response.text
    assert seen == ["repo.read-file"]


def test_call_tool_unknown_server_404(client):
    assert (
        client.post(
            "/api/mcp/servers/missing/tools/anything", json={"arguments": {}}
        ).status_code
        == 404
    )


def test_call_tool_rejects_non_object_arguments(client, monkeypatch):
    _save_server()
    response = client.post(
        "/api/mcp/servers/linear/tools/create_issue",
        json={"arguments": ["not", "an", "object"]},
    )
    assert response.status_code == 422


def test_get_servers_does_not_reapply_matching_config(client, apply_calls):
    from dizel_core.mcp import config as mcp_config

    cm.ConfigManager.save(
        {
            "mcp": {
                "servers": [
                    {
                        "id": "gamma",
                        "name": "Gamma",
                        "transport": "http",
                        "url": "https://gamma.example.com/mcp",
                        "enabled": False,
                    }
                ]
            }
        }
    )
    manager = mcp_manager.get_manager()
    manager.apply(mcp_config.load(cm.ConfigManager.load()))
    apply_calls.clear()
    response = client.get("/api/mcp/servers")
    assert response.status_code == 200
    assert response.json()["servers"][0]["id"] == "gamma"
    assert apply_calls == []
