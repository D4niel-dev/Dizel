"""TUI /edit replaces the last turn only after a two-step confirm."""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.commands.builtins.edit import EditCommand
from inference.tui.commands.parser import parse_command
from inference.tui.panels.multiline_input import MultilineInputModal
from inference.tui.rendering.history_message import HistoryMessage


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class _Bridge:
    def __init__(self, messages):
        self.current_messages = messages


class _CommandApp:
    generation_state = "IDLE"

    def __init__(self):
        self.session_bridge = _Bridge(
            [
                {"role": "user", "content": "first", "ts": 1.0},
                {"role": "assistant", "content": "answer", "ts": 2.0},
                {
                    "role": "user",
                    "content": "full expanded skill body\n\nsecond",
                    "display": {
                        "text": "second",
                        "skills": [{"name": "x", "title": "X"}],
                    },
                    "ts": 3.0,
                },
                {"role": "assistant", "content": "second answer", "ts": 4.0},
            ]
        )
        self.pushed = None
        self.edited = None

    def push_screen(self, modal, callback):
        self.pushed = (modal, callback)

    def edit_and_resend(self, index, text):
        self.edited = (index, text)
        return ""


class TuiEditTests(unittest.TestCase):
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

    def test_edit_command_prefills_the_visible_text(self):
        app = _CommandApp()

        result = asyncio.run(EditCommand().execute(app, parse_command("/edit")))

        self.assertEqual(result, "")
        modal, callback = app.pushed
        self.assertIsInstance(modal, MultilineInputModal)
        self.assertEqual(modal.initial_text, "second")
        self.assertTrue(modal.send_on_save)
        self.assertIn("replaces this turn", modal.notice_text)
        callback("edited prompt")
        self.assertEqual(app.edited, (2, "edited prompt"))

    def test_edit_command_guards_empty_sessions_and_streaming(self):
        app = _CommandApp()
        app.session_bridge.current_messages = []
        self.assertIn(
            "Nothing to edit yet",
            asyncio.run(EditCommand().execute(app, parse_command("/edit"))),
        )
        self.assertIsNone(app.pushed)

        app = _CommandApp()
        app.generation_state = "STREAMING"
        self.assertIn(
            "Still generating",
            asyncio.run(EditCommand().execute(app, parse_command("/edit"))),
        )
        self.assertIsNone(app.pushed)

    def test_modal_two_step_confirm(self):
        async def script(app, pilot):
            results = []
            app.push_screen(
                MultilineInputModal(
                    initial_text="hello",
                    send_on_save=True,
                    title="Edit message",
                    notice="Sending replaces this turn and every message after it.",
                ),
                results.append,
            )
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            armed = app.screen.notice_widget.render().plain
            still_open = isinstance(app.screen, MultilineInputModal)
            await pilot.press("ctrl+s")
            await pilot.pause()
            return armed, still_open, results

        armed, still_open, results = self._run(script)
        self.assertIn("Press ctrl+s again", armed)
        self.assertTrue(still_open)
        self.assertEqual(results, ["hello"])

    def test_edit_and_resend_truncates_and_resends(self):
        async def script(app, pilot):
            sent = []
            app.tool_bridge.run_tools_and_send = (
                lambda text, display=None: sent.append((text, display))
            )
            app.session_bridge.current_messages = [
                {"role": "user", "content": "first", "ts": 1.0},
                {"role": "assistant", "content": "first answer", "ts": 2.0},
                {"role": "user", "content": "second", "ts": 3.0},
                {"role": "assistant", "content": "second answer", "ts": 4.0},
            ]
            app._load_messages_into_app(app.session_bridge.current_messages)
            await pilot.pause()

            result = app.edit_and_resend(2, "edited prompt")
            await pilot.pause()

            return {
                "result": result,
                "contents": [m["content"] for m in app.session_bridge.current_messages],
                "roles": [m["role"] for m in app.session_bridge.current_messages],
                "history": [m["content"] for m in app.chat_bridge.manager.history],
                "sent": sent,
                "assistants": len(app.query(HistoryMessage)),
            }

        data = self._run(script)
        self.assertEqual(data["result"], "")
        self.assertEqual(
            data["contents"], ["first", "first answer", "edited prompt"]
        )
        self.assertEqual(data["roles"], ["user", "assistant", "user"])
        self.assertEqual(data["history"], ["first", "first answer"])
        self.assertEqual(data["sent"], [("edited prompt", {"text": "edited prompt", "skills": []})])
        self.assertEqual(data["assistants"], 1)


if __name__ == "__main__":
    unittest.main()
