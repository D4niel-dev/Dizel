"""The SSE tool loop emits a diff and resumes after the GUI approval POST."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest import mock

import pytest

from api.services.chat_service import ChatService
from dizel_core import config_manager
from dizel_core.tools import FileChange


@pytest.fixture(autouse=True)
def _isolated_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    ChatService._trusted_sessions.clear()


class FakeUsage:
    max_capacity = 1024
    usage = 0.0
    is_exhausted = False
    needs_compaction = False
    compaction_count = 0
    MAX_COMPACTIONS = 5
    COMPACTION_THRESHOLD = 84

    def set_capacity(self, value):
        pass

    def activate(self, provider, capacity=None):
        pass

    def charge_text(self, text):
        pass

    def charge_tool_calls(self, count):
        pass

    def time_until_reset(self):
        return "0m"


class FakeManager:
    is_ready = True
    is_generating = False
    is_stopping = False
    effective_ctx_len = 1024
    active_provider_slug = "local"
    system_prompt = ""
    model_info = {}
    session_id = ""

    def new_session(self):
        pass

    def load_history(self, messages):
        pass

    def get_history(self):
        return []

    def count_tokens(self, text):
        return 0

    def stop_generation(self):
        pass

    def send_with_tools(
        self,
        user_text,
        attachments,
        on_token,
        on_done,
        on_error,
        on_tool=None,
        on_tool_start=None,
        approve_tool=None,
        on_thinking=None,
        continuation=False,
    ):
        call = {
            "name": "edit_file",
            "args": {"path": "README.md", "old": "a", "new": "b"},
        }
        if on_tool_start:
            on_tool_start(call)
        change = FileChange(
            "edit_file",
            "README.md",
            "edit",
            "--- a/README.md\n+++ b/README.md",
            1,
            1,
        )
        approved = bool(approve_tool(change)) if approve_tool else False
        if on_tool:
            on_tool(
                call,
                "[edit_file] Edited README.md (+1/-1)"
                if approved
                else "[edit_file] Write cancelled.",
            )
        on_done("approved" if approved else "cancelled")


class ThreadedFakeManager(FakeManager):
    """Streams from a worker thread, the way the real manager does.

    The real ChatManager.send_with_tools starts a generation thread and
    returns immediately; FakeManager calls back synchronously, which hid
    the fact that the stream's end sentinel must be tied to the terminal
    callback rather than to send_with_tools returning.
    """

    def send_with_tools(
        self, user_text, attachments, on_token, on_done, on_error, **kwargs
    ):
        def worker():
            time.sleep(0.05)
            on_token("Hello")
            on_done("Hello world")

        threading.Thread(target=worker, daemon=True).start()


def test_stream_delivers_threaded_generation_reply():
    manager = ThreadedFakeManager()
    frames = []

    with mock.patch.object(ChatService, "get_manager", return_value=manager), mock.patch(
        "api.services.chat_service.get_usage_manager", return_value=FakeUsage()
    ), mock.patch(
        "api.services.session_service.SessionService.create_or_update_session",
        return_value=None,
    ):
        async def collect():
            async for frame in ChatService.stream_chat("hello"):
                data = json.loads(frame["data"])
                frames.append(data)
                if data["type"] in ("done", "error"):
                    break

        asyncio.run(collect())

    types = [frame["type"] for frame in frames]
    assert types == ["start", "token", "done"]
    assert frames[-1]["text"] == "Hello world"


def test_stream_emits_tool_and_waits_for_approval():
    manager = FakeManager()
    frames = []

    with mock.patch.object(ChatService, "get_manager", return_value=manager), mock.patch(
        "api.services.chat_service.get_usage_manager", return_value=FakeUsage()
    ), mock.patch(
        "api.services.session_service.SessionService.create_or_update_session",
        return_value=None,
    ):
        async def collect():
            async for frame in ChatService.stream_chat("edit the readme"):
                data = json.loads(frame["data"])
                frames.append(data)
                if data["type"] == "approval":
                    assert ChatService.resolve_tool_approval(data["id"], "allow_once")
                if data["type"] in ("done", "error"):
                    break

        asyncio.run(collect())

    types = [(frame["type"], frame.get("phase")) for frame in frames]
    assert ("start", None) in types
    assert ("tool", "start") in types
    approval = next(frame for frame in frames if frame["type"] == "approval")
    assert approval["path"] == "README.md"
    assert "+b" in approval["diff"] or "+++ b/README.md" in approval["diff"]
    assert ("tool", "done") in types
    assert frames[-1]["type"] == "done"
    assert frames[-1]["text"] == "approved"
