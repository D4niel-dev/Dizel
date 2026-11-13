"""The generic config API must never ship MCP secret values (spec A6/A9)."""

import pytest
from fastapi.testclient import TestClient

import dizel_core.config_manager as cm
import dizel_core.mcp.manager as mcp_manager

SECRET = "sk-live-SECRET-123"

SERVER = {
    "id": "linear",
    "name": "Linear",
    "transport": "http",
    "url": "https://mcp.example.com/mcp",
    "headers": {"Authorization": f"Bearer {SECRET}"},
    "enabled": False,
}


@pytest.fixture(autouse=True)
def _fresh_manager(monkeypatch):
    """TestClient exit shuts the process-wide manager down for good."""
    monkeypatch.setattr(mcp_manager, "_manager", None)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(mcp_manager.MCPManager, "apply_async", lambda self, servers: None)
    cm.ConfigManager.save({"mcp": {"servers": [SERVER]}, "checkpoint": "x.pt"})
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_get_config_redacts_mcp_secret_values(client):
    response = client.get("/api/config/")
    assert response.status_code == 200
    assert SECRET not in response.text
    server = response.json()["mcp"]["servers"][0]
    # Key names survive so clients know what is stored; values do not.
    assert server["headers"] == {"Authorization": ""}
    assert server["id"] == "linear"


def test_post_config_response_redacts_and_never_writes_mcp(client):
    evil = {
        "servers": [
            {
                "id": "evil",
                "name": "Evil",
                "transport": "http",
                "url": "https://evil.example.com/mcp",
                "headers": {"Authorization": "stolen"},
                "enabled": False,
            }
        ]
    }
    response = client.post("/api/config/", json={"mcp": evil, "checkpoint": "y.pt"})
    assert response.status_code == 200
    assert "stolen" not in response.text
    assert SECRET not in response.text
    assert response.json()["mcp"]["servers"][0]["headers"] == {"Authorization": ""}

    stored = cm.ConfigManager.load()
    assert stored["mcp"]["servers"][0]["id"] == "linear"
    assert stored["mcp"]["servers"][0]["headers"] == {
        "Authorization": f"Bearer {SECRET}"
    }
    # Non-MCP settings still update normally.
    assert stored["checkpoint"] == "y.pt"
