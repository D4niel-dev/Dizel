"""Checkpoint load history: newest first, deduped, capped, shared storage."""

from __future__ import annotations

import sys
import types

import pytest

from dizel_core import checkpoint_history
from dizel_core.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    import dizel_core.config_manager as config_manager

    monkeypatch.setattr(config_manager, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))


def test_record_adds_newest_first_and_dedupes(tmp_path):
    a = str(tmp_path / "a.pt")
    b = str(tmp_path / "b.gguf")
    for name in (a, b):
        open(name, "w").close()

    checkpoint_history.record(a)
    checkpoint_history.record(b)
    checkpoint_history.record(a)

    assert checkpoint_history.recent() == [a, b]


def test_record_caps_the_history(tmp_path):
    paths = []
    for index in range(10):
        path = str(tmp_path / f"m{index}.pt")
        open(path, "w").close()
        paths.append(path)
        checkpoint_history.record(path)

    recent = checkpoint_history.recent()
    assert len(recent) == checkpoint_history.LIMIT
    assert recent == list(reversed(paths))[: checkpoint_history.LIMIT]


def test_record_stores_the_resolved_path(tmp_path, monkeypatch):
    target = tmp_path / "model.pt"
    target.write_text("")

    monkeypatch.chdir(tmp_path)
    checkpoint_history.record("model.pt")

    assert checkpoint_history.recent() == [str(target.resolve())]


def test_recent_survives_a_corrupt_history_value(tmp_path):
    cfg = ConfigManager.load()
    cfg["checkpoint_history"] = "not-a-list"
    ConfigManager.save(cfg)

    assert checkpoint_history.recent() == []

    checkpoint_history.record(str(tmp_path / "fresh.pt"))
    assert checkpoint_history.recent() == [str((tmp_path / "fresh.pt").resolve())]


def test_record_ignores_an_empty_path():
    checkpoint_history.record("")
    checkpoint_history.record("   ")
    assert checkpoint_history.recent() == []


def test_recent_keeps_paths_whose_file_is_gone(tmp_path):
    gone = str(tmp_path / "deleted.pt")
    checkpoint_history.record(gone)
    assert checkpoint_history.recent() == [gone]


def _stub_backend_module(monkeypatch, backend_cls) -> None:
    module = types.ModuleType("dizel_core.model_backend")
    for name in ("NativeBackend", "GGUFBackend", "SafeTensorsBackend"):
        setattr(module, name, backend_cls)
    monkeypatch.setitem(sys.modules, "dizel_core.model_backend", module)


def test_load_model_records_a_successful_load(tmp_path, monkeypatch):
    from dizel_core import chat_manager

    recorded = []
    monkeypatch.setattr(
        "dizel_core.checkpoint_history.record", lambda path: recorded.append(path)
    )
    monkeypatch.setattr(chat_manager.os.path, "exists", lambda path: True)

    class FakeBackend:
        def load(self, path, device, report, **kwargs):
            pass

        def close(self):
            pass

    _stub_backend_module(monkeypatch, FakeBackend)

    manager = chat_manager.ChatManager()
    manager.load_model(str(tmp_path / "model.pt"), device="cpu")

    assert recorded == [str(tmp_path / "model.pt")]


def test_load_model_failure_is_not_recorded(tmp_path, monkeypatch):
    from dizel_core import chat_manager

    recorded = []
    monkeypatch.setattr(
        "dizel_core.checkpoint_history.record", lambda path: recorded.append(path)
    )
    monkeypatch.setattr(chat_manager.os.path, "exists", lambda path: True)

    class FakeBackend:
        def load(self, path, device, report, **kwargs):
            raise RuntimeError("boom")

        def close(self):
            pass

    _stub_backend_module(monkeypatch, FakeBackend)

    manager = chat_manager.ChatManager()
    with pytest.raises(RuntimeError):
        manager.load_model(str(tmp_path / "model.pt"), device="cpu")

    assert recorded == []
