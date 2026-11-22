"""The apply-profile path forwards the Thinking effort end to end."""

from __future__ import annotations

import unittest
from unittest import mock

from api.routes.chat import ProfileRequest
from api.routes.chat import apply_profile as apply_profile_route
from api.services.chat_service import ChatService
from dizel_core.chat_manager import ChatManager


class ApplyProfileEffortTests(unittest.TestCase):
    def test_manager_stores_the_effort_profile(self):
        manager = ChatManager.__new__(ChatManager)
        manager.system_prompt = ""
        manager._active_model = "Dizel"
        manager._active_mode = "Fast"
        manager._active_thinking_effort = "medium"
        manager._active_profile = None

        manager.apply_profile("Dizel", "Thinking", "high")

        self.assertEqual(manager._active_thinking_effort, "high")
        self.assertEqual(manager._active_profile.sampling.temperature, 0.43)
        self.assertIn("Thinking · High", manager._active_profile.label)

    def test_route_forwards_the_effort(self):
        with mock.patch.object(
            ChatService, "apply_profile", return_value={"status": "ok"}
        ) as apply:
            apply_profile_route(
                ProfileRequest(
                    model_name="Dizel",
                    mode_name="Thinking",
                    thinking_effort="xhigh",
                )
            )
        apply.assert_called_once_with("Dizel", "Thinking", "xhigh", "fast")


    def test_route_forwards_the_speed(self):
        with mock.patch.object(
            ChatService, "apply_profile", return_value={"status": "ok"}
        ) as apply:
            apply_profile_route(
                ProfileRequest(
                    model_name="Dizel", mode_name="Fast", speed="turbo"
                )
            )
        apply.assert_called_once_with("Dizel", "Fast", "medium", "turbo")


if __name__ == "__main__":
    unittest.main()
