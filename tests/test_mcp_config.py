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
