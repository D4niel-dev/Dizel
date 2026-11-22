"""The terminal gates a model plan behind a two-outcome checklist."""

from __future__ import annotations

import asyncio
import copy
import threading
import unittest
from unittest import mock

from textual.widgets import Button

from dizel_core.config_manager import ConfigManager
from dizel_core.tools import PlanChange
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.commands.builtins.plan import PlanCommand
from inference.tui.commands.parser import parse_command
from inference.tui.panels.modals.plan_approval import PlanApprovalModal


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class _FakeApp:
    generation_state = "IDLE"
    sent = None

    def __init__(self):
        self.mode = "Fast"

    def set_mode(self, mode, thinking_effort=None, announce=True):
        self.mode = mode

    def send_composed(self, display, content):
        self.sent = (display, content)
        return ""


class TuiPlanTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(DizelCMDApp, "_auto_load_model", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_update_check", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_mcp", lambda self: None),
            mock.patch.object(ConfigManager, "load", classmethod(lambda cls: _defaults())),
            mock.patch.object(ConfigManager, "save", classmethod(lambda cls, cfg: None)),
            mock.patch.object(UsageManager, "load_state", lambda self: None),
            mock.patch.object(UsageManager, "save_state", lambda self: None),
            mock.patch(
                "inference.tui.bridge.session_bridge.list_sessions", return_value=[]
            ),
            mock.patch(
                "inference.tui.bridge.session_bridge.save_session", return_value=""
            ),
            mock.patch(
                "inference.tui.bridge.session_bridge.load_session", return_value=None
            ),
        ]
        for patcher in self._patches:
            patcher.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for patcher in self._patches:
            patcher.stop()

    def _run(self, script, size=(120, 36)):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    def test_modal_lists_steps_and_escape_discards(self):
        async def script(app, pilot):
            results = []
            app.push_screen(
                PlanApprovalModal(PlanChange("Ship it", ("Tag", "Publish"))),
                results.append,
            )
            await pilot.pause()
            ids = [button.id for button in app.screen.query(Button)]
            steps = [step.render().plain for step in app.screen.query(".plan-step")]
            await pilot.press("escape")
            await pilot.pause()
            return ids, steps, results

        ids, steps, results = self._run(script)
        self.assertEqual(ids, ["plan-discard", "plan-execute"])
        self.assertEqual(steps, ["1. Tag", "2. Publish"])
        self.assertEqual(results, [False])

    def test_modal_y_executes(self):
        async def script(app, pilot):
            results = []
            app.push_screen(
                PlanApprovalModal(PlanChange("Ship", ("Tag",))), results.append
            )
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            return results

        self.assertEqual(self._run(script), [True])

    def test_bridge_plan_approval_uses_the_modal_even_with_full_trust(self):
        change = PlanChange("Ship it", ("Tag", "Publish"))

        async def script(app, pilot):
            results = []

            def worker():
                results.append(app.chat_bridge._approve_plan(change))

            with mock.patch(
                "dizel_core.tool_trust.persistent_enabled", return_value=True
            ):
                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                modal = None
                for _ in range(80):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, PlanApprovalModal):
                        modal = app.screen
                        break
                self.assertIsNotNone(modal)
                await pilot.press("y")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if results:
                        break
                return results

        self.assertEqual(self._run(script), [True])

    def test_plan_command_sends_the_drafting_prompt(self):
        app = _FakeApp()

        result = asyncio.run(
            PlanCommand().execute(app, parse_command("/plan fix the build"))
        )

        self.assertEqual(result, "")
        display, content = app.sent
        self.assertEqual(display, "Plan · fix the build")
        self.assertIn("Draft a short numbered plan", content)
        self.assertIn("Task: fix the build", content)
        self.assertEqual(app.mode, "Planning")

    def test_plan_command_needs_a_task_and_an_idle_app(self):
        app = _FakeApp()
        self.assertIn(
            "Usage: /plan <task>",
            asyncio.run(PlanCommand().execute(app, parse_command("/plan"))),
        )
        app.generation_state = "STREAMING"
        self.assertIn(
            "Still generating",
            asyncio.run(PlanCommand().execute(app, parse_command("/plan x"))),
        )
        self.assertIsNone(app.sent)


if __name__ == "__main__":
    unittest.main()
