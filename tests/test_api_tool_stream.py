"""The SSE tool loop emits a diff and resumes after the GUI approval POST."""

from __future__ import annotations

import asyncio
import json
from unittest import mock

from api.services.chat_service import ChatService
from dizel_core.tools import FileChange


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
                    assert ChatService.resolve_tool_approval(data["id"], True)
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
