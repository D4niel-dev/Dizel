"""Sidecar trust: auto-approval audit, grant recording, trust state, routes."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi import HTTPException

from api.routes.chat import ToolApprovalRequest, tools_approve
from api.services.chat_service import ChatService
from dizel_core import config_manager, tool_trust
from dizel_core.tools import FileChange


@pytest.fixture(autouse=True)
def _isolated_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    ChatService._trusted_sessions.clear()
    ChatService._approvals.clear()


def _change(tool: str = "edit_file", action: str = "edit") -> FileChange:
    return FileChange(tool, "README.md", action, "+new", 1, 0)


def _run_gate(change, session_id=""):
    """Run one approval request on a thread; return (events, result, thread)."""
    events = []
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "approved",
            ChatService._request_tool_approval(change, events.append, 7, session_id),
        ),
        daemon=True,
    )
    thread.start()
    for _ in range(200):
        if events or "approved" in result:
            break
        time.sleep(0.01)
    return events, result, thread


def test_persistent_trust_auto_approves_with_an_audit_notice():
    tool_trust.set_persistent(True)
    events, result, thread = _run_gate(_change(), session_id="s1")
    thread.join(timeout=2)
    assert result["approved"] is True
    assert events == [
        {"type": "notice", "text": "Auto-approved edit_file README.md (persistent)"}
    ]


def test_session_grant_auto_approves_only_the_granted_session():
    ChatService._trusted_sessions.add("s1")
    events, result, thread = _run_gate(_change(), session_id="s1")
    thread.join(timeout=2)
    assert result["approved"] is True
    assert events[0]["type"] == "notice"
    assert "(session)" in events[0]["text"]

    events2, result2, thread2 = _run_gate(_change(), session_id="s2")
    assert events2 and events2[0]["type"] == "approval"
    assert thread2.is_alive()
    ChatService.resolve_tool_approval(events2[0]["id"], "deny")
    thread2.join(timeout=2)
    assert result2["approved"] is False


def test_delete_is_never_auto_approved_even_with_every_trust_tier():
    tool_trust.set_persistent(True)
    ChatService._trusted_sessions.add("s1")
    events, result, thread = _run_gate(
        _change("delete_file", "delete"), session_id="s1"
    )
    assert events and events[0]["type"] == "approval"
    ChatService.resolve_tool_approval(events[0]["id"], "allow_once")
    thread.join(timeout=2)
    assert result["approved"] is True  # approved manually, never automatically


def test_resolve_always_session_records_the_grant():
    events, result, thread = _run_gate(_change(), session_id="s9")
    assert ChatService.resolve_tool_approval(events[0]["id"], "always_session") is True
    thread.join(timeout=2)
    assert result["approved"] is True
    assert "s9" in ChatService._trusted_sessions


def test_resolve_always_persistent_writes_the_config():
    events, result, thread = _run_gate(_change(), session_id="s1")
    ChatService.resolve_tool_approval(events[0]["id"], "always_persistent")
    thread.join(timeout=2)
    assert result["approved"] is True
    assert tool_trust.persistent_enabled() is True


def test_resolve_deny_grants_nothing():
    events, result, thread = _run_gate(_change(), session_id="s1")
    ChatService.resolve_tool_approval(events[0]["id"], "deny")
    thread.join(timeout=2)
    assert result["approved"] is False
    assert ChatService._trusted_sessions == set()
    assert tool_trust.persistent_enabled() is False


def test_trust_state_and_revoke():
    tool_trust.set_persistent(True)
    ChatService._trusted_sessions.add("s1")
    assert ChatService.trust_state("s1") == {"persistent": True, "session_trusted": True}
    assert ChatService.trust_state("s2") == {"persistent": True, "session_trusted": False}
    assert ChatService.trust_state(None) == {"persistent": True, "session_trusted": False}
    ChatService.revoke_session_trust("s1")
    assert ChatService.trust_state("s1")["session_trusted"] is False


def test_approve_route_rejects_unknown_decisions_and_ids():
    with pytest.raises(HTTPException) as bad:
        tools_approve(ToolApprovalRequest(approval_id="x", decision="bogus"))
    assert bad.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        tools_approve(ToolApprovalRequest(approval_id="x", decision="allow_once"))
    assert missing.value.status_code == 404


def _pending_entry(session_id="s1", tool="edit_file"):
    entry = {
        "event": threading.Event(),
        "approved": False,
        "session_id": session_id,
        "tool": tool,
        "claimed": False,
    }
    ChatService._approvals["a1"] = entry
    return entry


def test_cancel_claims_entries_so_a_late_resolve_cannot_approve():
    entry = _pending_entry()
    ChatService.cancel_tool_approvals()
    assert entry["approved"] is False
    assert entry["event"].is_set()
    # Stop already denied it; a racing resolve must not flip the outcome.
    assert ChatService.resolve_tool_approval("a1", "allow_once") is False
    assert entry["approved"] is False


def test_resolve_claims_the_entry_so_a_second_decision_is_a_noop():
    entry = _pending_entry()
    assert ChatService.resolve_tool_approval("a1", "allow_once") is True
    assert entry["approved"] is True
    assert ChatService.resolve_tool_approval("a1", "deny") is False
    assert entry["approved"] is True


def test_timeout_denies_even_when_a_resolve_lands_late(monkeypatch):
    # Force wait() to report a timeout, then let a resolve claim first: the
    # gate must still deny because the 300s elapsed without an earlier decision.
    monkeypatch.setattr(threading.Event, "wait", lambda self, timeout=None: False)
    events, result, thread = _run_gate(_change(), session_id="s1")
    assert events and events[0]["type"] == "approval"
    ChatService.resolve_tool_approval(events[0]["id"], "allow_once")
    thread.join(timeout=2)
    assert result["approved"] is False


def test_standing_session_decision_on_a_delete_entry_grants_no_write_trust():
    events, result, thread = _run_gate(_change("delete_file", "delete"), session_id="s1")
    assert ChatService.resolve_tool_approval(events[0]["id"], "always_session") is True
    thread.join(timeout=2)
    assert result["approved"] is True  # downgraded to allow_once: this delete proceeds
    assert "s1" not in ChatService._trusted_sessions


def test_standing_persistent_decision_on_a_delete_entry_grants_no_write_trust():
    events, result, thread = _run_gate(_change("delete_file", "delete"), session_id="s1")
    ChatService.resolve_tool_approval(events[0]["id"], "always_persistent")
    thread.join(timeout=2)
    assert result["approved"] is True
    assert tool_trust.persistent_enabled() is False


def test_delete_session_revokes_its_session_trust(monkeypatch):
    from api.routes import session as session_routes
    from api.services.session_service import SessionService

    ChatService._trusted_sessions.add("s1")
    monkeypatch.setattr(SessionService, "delete_session", lambda sid: True)
    session_routes.delete_session("s1")
    assert "s1" not in ChatService._trusted_sessions


def test_clear_all_sessions_revokes_every_session_trust(monkeypatch):
    from api.routes import session as session_routes
    from api.services.session_service import SessionService

    ChatService._trusted_sessions.update({"s1", "s2"})
    monkeypatch.setattr(SessionService, "delete_all", lambda: 2)
    session_routes.clear_all_sessions()
    assert ChatService._trusted_sessions == set()


def test_session_grant_auto_approves_the_next_same_session_gate():
    events, result, thread = _run_gate(_change(), session_id="s1")
    ChatService.resolve_tool_approval(events[0]["id"], "always_session")
    thread.join(timeout=2)
    assert result["approved"] is True

    events2, result2, thread2 = _run_gate(_change(), session_id="s1")
    thread2.join(timeout=2)
    assert result2["approved"] is True
    assert events2 and events2[0]["type"] == "notice"
    assert "(session)" in events2[0]["text"]


def test_out_of_root_write_still_gates_under_persistent_trust():
    tool_trust.set_persistent(True)
    change = FileChange("write_file", "../outside.txt", "create", "+x", 1, 0)
    events, result, thread = _run_gate(change, session_id="s1")
    assert events and events[0]["type"] == "approval"
    assert thread.is_alive()
    ChatService.resolve_tool_approval(events[0]["id"], "deny")
    thread.join(timeout=2)
    assert result["approved"] is False
