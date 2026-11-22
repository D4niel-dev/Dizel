"""Plan approvals block the tool thread and never consult write trust."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from fastapi import HTTPException

from api.routes.chat import PlanApprovalRequest, tools_plan_approve
from api.services.chat_service import ChatService
from dizel_core import tool_trust
from dizel_core.tools import FileChange, PlanChange


def _wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class PlanApprovalTests(unittest.TestCase):
    def setUp(self):
        ChatService.cancel_tool_approvals()
        self.addCleanup(ChatService.cancel_tool_approvals)

    def _start(self, change):
        events = []
        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                ChatService._request_plan_approval(change, events.append)
            ),
            daemon=True,
        )
        thread.start()
        self.assertTrue(_wait_for(lambda: events), "plan event was not emitted")
        return thread, events, results

    def test_event_arrives_then_approval_resolves_true(self):
        thread, events, results = self._start(PlanChange("Ship it", ("Tag", "Publish")))

        self.assertEqual(events[0]["type"], "plan")
        self.assertEqual(events[0]["task"], "Ship it")
        self.assertEqual(events[0]["steps"], ["Tag", "Publish"])
        self.assertTrue(ChatService.resolve_plan_approval(events[0]["id"], True))
        thread.join(2)
        self.assertEqual(results, [True])

    def test_discard_resolves_false(self):
        thread, events, results = self._start(PlanChange("Ship", ("Tag",)))

        self.assertTrue(ChatService.resolve_plan_approval(events[0]["id"], False))
        thread.join(2)
        self.assertEqual(results, [False])

    def test_full_trust_does_not_auto_approve_a_plan(self):
        with mock.patch.object(tool_trust, "persistent_enabled", return_value=True):
            thread, events, results = self._start(PlanChange("Ship", ("Tag",)))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "plan")
        ChatService.resolve_plan_approval(events[0]["id"], False)
        thread.join(2)
        self.assertEqual(results, [False])

    def test_cancel_denies_a_pending_plan(self):
        thread, events, results = self._start(PlanChange("Ship", ("Tag",)))

        ChatService.cancel_tool_approvals()
        thread.join(2)
        self.assertEqual(results, [False])

    def test_plan_and_write_approvals_cannot_cross_resolve(self):
        thread, events, results = self._start(PlanChange("Ship", ("Tag",)))
        self.assertFalse(
            ChatService.resolve_tool_approval(events[0]["id"], "allow_once")
        )
        self.assertTrue(ChatService.resolve_plan_approval(events[0]["id"], False))
        thread.join(2)

        change = FileChange("write_file", "notes.txt", "create", "+x", 1, 0)
        write_events = []
        write_results = []
        write_thread = threading.Thread(
            target=lambda: write_results.append(
                ChatService._request_tool_approval(change, write_events.append)
            ),
            daemon=True,
        )
        with mock.patch.object(tool_trust, "persistent_enabled", return_value=False):
            write_thread.start()
            self.assertTrue(_wait_for(lambda: write_events))
        self.assertFalse(
            ChatService.resolve_plan_approval(write_events[0]["id"], True)
        )
        self.assertTrue(
            ChatService.resolve_tool_approval(write_events[0]["id"], "deny")
        )
        write_thread.join(2)
        self.assertEqual(write_results, [False])

    def test_route_404_for_unknown_id(self):
        with self.assertRaises(HTTPException) as caught:
            tools_plan_approve(PlanApprovalRequest(approval_id="missing", approved=True))
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
