"""The in-app scorecard job: runs the core suite on the loaded model only."""

from __future__ import annotations

import threading
import time

import pytest

from api.services.scorecard_service import ScorecardService
from dizel_core import scorecards


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecards, "SCORECARDS_DIR", str(tmp_path / "scorecards"))
    ScorecardService.reset()
    yield
    ScorecardService.reset()


def checkpoint(tmp_path):
    path = tmp_path / "Dizel-1.5-SFT.gguf"
    path.write_bytes(b"weights")
    return str(path)


class FakeManager:
    is_ready = True
    is_generating = False
    active_provider_slug = "local"
    system_prompt = "keep me"

    def __init__(self, path):
        self.checkpoint_path = path
        self.history = [{"role": "user", "content": "keep"}]

    def get_history(self):
        return list(self.history)

    def load_history(self, messages):
        self.history = list(messages)

    def send_message(
        self, user_text, attachments, on_token, on_done, on_error, *args, **kwargs
    ):
        on_token("ok")
        on_done("ok")


class SlowManager(FakeManager):
    def send_message(
        self, user_text, attachments, on_token, on_done, on_error, *args, **kwargs
    ):
        def worker():
            time.sleep(0.05)
            on_done("ok")

        threading.Thread(target=worker, daemon=True).start()


def run_to_completion(timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = ScorecardService.state()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError("scorecard run did not finish")


def test_run_scores_the_loaded_model_and_keeps_chat_untouched(tmp_path, monkeypatch):
    from api.services.chat_service import ChatService

    manager = FakeManager(checkpoint(tmp_path))
    monkeypatch.setattr(ChatService, "get_manager", classmethod(lambda cls: manager))

    started = ScorecardService.start()
    assert started["started"] is True
    state = run_to_completion()

    assert state["error"] is None
    assert state["done"] == state["total"] == 15
    assert state["scorecard"]["total"] == 15
    assert scorecards.load(manager.checkpoint_path) is not None
    assert manager.history == [{"role": "user", "content": "keep"}]
    assert manager.system_prompt == "keep me"


def test_provider_models_are_refused(tmp_path, monkeypatch):
    from api.services.chat_service import ChatService

    manager = FakeManager(checkpoint(tmp_path))
    manager.active_provider_slug = "openai"
    monkeypatch.setattr(ChatService, "get_manager", classmethod(lambda cls: manager))

    result = ScorecardService.start()

    assert result["started"] is False
    assert "local checkpoint" in result["error"].lower()


def test_missing_checkpoint_is_refused(monkeypatch):
    from api.services.chat_service import ChatService

    manager = FakeManager("")
    manager.checkpoint_path = ""
    monkeypatch.setattr(ChatService, "get_manager", classmethod(lambda cls: manager))

    result = ScorecardService.start()

    assert result["started"] is False
    assert "checkpoint" in result["error"].lower()


def test_generating_manager_is_refused(tmp_path, monkeypatch):
    from api.services.chat_service import ChatService

    manager = FakeManager(checkpoint(tmp_path))
    manager.is_generating = True
    monkeypatch.setattr(ChatService, "get_manager", classmethod(lambda cls: manager))

    result = ScorecardService.start()

    assert result["started"] is False
    assert "generat" in result["error"].lower()


def test_second_start_is_refused(tmp_path, monkeypatch):
    from api.services.chat_service import ChatService

    manager = SlowManager(checkpoint(tmp_path))
    monkeypatch.setattr(ChatService, "get_manager", classmethod(lambda cls: manager))

    assert ScorecardService.start()["started"] is True
    second = ScorecardService.start()
    assert second["started"] is False
    assert "already" in second["error"].lower()

    run_to_completion()
