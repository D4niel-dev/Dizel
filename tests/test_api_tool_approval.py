"""The GUI approval bridge blocks the tool thread until the client answers."""

from __future__ import annotations

import threading
import time

import pytest

from api.services.chat_service import ChatService
from dizel_core import config_manager
from dizel_core.tools import FileChange


@pytest.fixture(autouse=True)
def _isolated_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    ChatService._trusted_sessions.clear()


def _change() -> FileChange:
    return FileChange(
        "edit_file",
        "README.md",
        "edit",
        "+new",
        1,
        0,
    )


def test_approval_bridge_waits_for_the_gui_decision():
    events = []
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault(
            "approved",
            ChatService._request_tool_approval(_change(), events.append, 7),
        ),
        daemon=True,
    )
    thread.start()

    for _ in range(100):
        if events:
            break
        time.sleep(0.01)

    assert events
    event = events[0]
    assert event["type"] == "approval"
    assert event["tool_id"] == 7
    assert event["path"] == "README.md"
    assert thread.is_alive()

    assert ChatService.resolve_tool_approval(event["id"], "allow_once") is True
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result["approved"] is True


def test_cancel_resolves_pending_approvals_as_rejected():
    events = []
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault(
            "approved",
            ChatService._request_tool_approval(_change(), events.append),
        ),
        daemon=True,
    )
    thread.start()
    for _ in range(100):
        if events:
            break
        time.sleep(0.01)

    ChatService.cancel_tool_approvals()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result["approved"] is False
    assert ChatService.resolve_tool_approval(events[0]["id"], "allow_once") is False
