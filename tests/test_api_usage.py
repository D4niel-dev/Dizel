"""The usage window follows the active provider (capacity and counters)."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

import api.services.chat_service as cs


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import dizel_core.config_manager as cm

    monkeypatch.setattr(cm, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "_usage_manager", None)
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _provider_manager(slug: str, cfg=None):
    from dizel_core.chat_manager import ChatManager

    manager = ChatManager.__new__(ChatManager)
    manager._backend_lock = threading.RLock()
    manager._provider_slug = slug
    manager._api_provider = None
    manager._api_model = ""
    manager._backend = None
    manager._cfg = cfg or {"api_router": {}}
    return manager


def _use_manager(monkeypatch, manager):
    monkeypatch.setattr(
        cs.ChatService, "get_manager", classmethod(lambda cls: manager)
    )


def test_usage_follows_the_active_provider(client, monkeypatch):
    manager = _provider_manager("openai")
    _use_manager(monkeypatch, manager)

    body = client.get("/api/chat/usage").json()
    assert body["provider"] == "openai"
    assert body["capacity"] == 131072

    manager._provider_slug = "anthropic"
    body = client.get("/api/chat/usage").json()
    assert body["provider"] == "anthropic"
    assert body["capacity"] == 200000


def test_usage_buckets_are_independent_per_provider(client, monkeypatch):
    manager = _provider_manager("openai")
    _use_manager(monkeypatch, manager)

    usage = cs.get_usage_manager()
    usage.activate("openai", 131072)
    usage.charge_text(" ".join(["word"] * 100))

    assert client.get("/api/chat/usage").json()["used"] == 10

    manager._provider_slug = "local"
    assert client.get("/api/chat/usage").json()["used"] == 0


def test_playground_usage_honors_provider_overrides(client, monkeypatch):
    from dizel_core.config_manager import ConfigManager

    cfg = ConfigManager.load()
    cfg.setdefault("api_router", {})["provider_limits"] = {"openai": 4096}
    ConfigManager.save(cfg)

    manager = _provider_manager("openai", cfg=ConfigManager.load())
    _use_manager(monkeypatch, manager)

    assert client.get("/api/chat/usage").json()["capacity"] == 4096


def test_providers_report_their_context_limits(client, monkeypatch):
    body = client.get("/api/providers").json()
    by_slug = {provider["slug"]: provider for provider in body}
    assert by_slug["openai"]["context_limit"] == 131072
    assert by_slug["anthropic"]["context_limit"] == 200000
    assert by_slug["local"]["context_limit"] is None


def test_providers_report_a_saved_override(client, monkeypatch):
    from dizel_core.config_manager import ConfigManager

    cfg = ConfigManager.load()
    cfg.setdefault("api_router", {})["provider_limits"] = {"openai": 4096}
    ConfigManager.save(cfg)

    body = client.get("/api/providers").json()
    by_slug = {provider["slug"]: provider for provider in body}
    assert by_slug["openai"]["context_limit_override"] == 4096
