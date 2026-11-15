"""The GUI Thought UI, ported to the terminal.

While reasoning streams: sparkles, a rotating phrase, a live clock, and a muted
railed body. Models without reasoning get the three-dot generating row and the
same phrase rotation until the first answer token. Reply statistics live in the
status bar, not under the message.
"""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.status_bar import StatusBar
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.rendering.assistant_message import THOUGHTS

THOUGHT = "Plan the answer line by line."
ANSWER = "Here is the answer."
RAW = f"<thinking>{THOUGHT}</thinking>\n{ANSWER}"


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


def _plain(widget) -> str:
    visual = widget.visual
    return getattr(visual, "plain", str(visual))


class ThinkingBlockTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(DizelCMDApp, "_auto_load_model", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_update_check", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_mcp", lambda self: None),
            mock.patch.object(ConfigManager, "load", classmethod(lambda cls: _defaults())),
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

    def _run(self, size, script):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    def test_live_reasoning_rotates_phrase_clock_and_folds(self):
        async def script(app, pilot):
            ws = app.query_one(WorkspacePanel)
            ws.clear_workspace(show_empty=False)
            ws.add_user_message("Write the answer.")
            await pilot.pause()
            ws.start_assistant_message()
            message = ws.current_message
            assert message is not None

            message.stream.append(f"<thinking>{THOUGHT}")
            message.update_stream()
            for _ in range(10):
                await pilot.pause(0.03)

            collapsible = message._think_coll
            active = {
                "display": collapsible.display,
                "title": collapsible.title,
                "marker": collapsible.marker,
                "thinking_class": collapsible.has_class("-thinking"),
                "stream_display": message._stream.display,
                "wait_display": message._wait.display,
            }
            earlier = collapsible.title
            for _ in range(12):
                await pilot.pause(0.1)
            later = collapsible.title

            message.stream.append(f"</thinking>\n{ANSWER}")
            message.update_stream()
            for _ in range(10):
                await pilot.pause(0.03)

            done_open = {
                "title": collapsible.title,
                "marker": collapsible.marker,
                "collapsed": collapsible.collapsed,
                "thinking_class": collapsible.has_class("-thinking"),
                "stream_display": message._stream.display,
                "wait_display": message._wait.display,
            }

            message.stream.finish()
            message.finalize(RAW)
            await pilot.pause(0.1)
            done_folded = {
                "title": collapsible.title,
                "marker": collapsible.marker,
                "collapsed": collapsible.collapsed,
            }
            return active, earlier, later, done_open, done_folded

        active, earlier, later, done_open, done_folded = self._run(
            (120, 36), script
        )

        self.assertTrue(active["display"])
        self.assertTrue(any(phrase in active["title"] for phrase in THOUGHTS))
        self.assertEqual(active["marker"], "✻")
        self.assertTrue(active["thinking_class"])
        self.assertFalse(active["stream_display"])
        self.assertFalse(active["wait_display"])
        self.assertNotEqual(earlier, later)

        self.assertTrue(done_open["title"].startswith("Thought for "))
        self.assertEqual(done_open["marker"], "-")
        self.assertFalse(done_open["collapsed"])
        self.assertFalse(done_open["thinking_class"])
        self.assertTrue(done_open["stream_display"])
        self.assertFalse(done_open["wait_display"])

        self.assertTrue(done_folded["title"].startswith("Thought for "))
        self.assertEqual(done_folded["marker"], "+")
        self.assertTrue(done_folded["collapsed"])

    def test_generating_row_covers_models_without_reasoning(self):
        async def script(app, pilot):
            ws = app.query_one(WorkspacePanel)
            ws.clear_workspace(show_empty=False)
            ws.add_user_message("Write the answer.")
            await pilot.pause()
            ws.start_assistant_message()
            message = ws.current_message
            assert message is not None

            await pilot.pause(0.25)
            before = {
                "wait_display": message._wait.display,
                "text": _plain(message._wait),
                "think_display": message._think_coll.display,
                "stream_display": message._stream.display,
            }
            frames = set()
            for _ in range(5):
                await pilot.pause(0.2)
                frames.add(_plain(message._wait)[:3])

            message.stream.append(ANSWER)
            message.update_stream()
            await pilot.pause(0.2)
            after = {
                "wait_display": message._wait.display,
                "stream_display": message._stream.display,
            }
            return before, frames, after

        before, frames, after = self._run((120, 36), script)

        self.assertTrue(before["wait_display"])
        self.assertTrue(any(phrase in before["text"] for phrase in THOUGHTS))
        self.assertFalse(before["think_display"])
        self.assertFalse(before["stream_display"])
        self.assertGreater(len(frames), 1)

        self.assertFalse(after["wait_display"])
        self.assertTrue(after["stream_display"])

    def test_reply_metrics_move_to_the_status_bar(self):
        async def script(app, pilot):
            ws = app.query_one(WorkspacePanel)
            ws.clear_workspace(show_empty=False)
            ws.add_user_message("Write the answer.")
            await pilot.pause()
            ws.start_assistant_message()
            message = ws.current_message
            assert message is not None
            await pilot.pause(0.7)

            message.stream.append(f"<thinking>{THOUGHT}</thinking>\n{ANSWER}")
            message.update_stream()
            message.stream.finish()
            message.finalize(RAW)
            await pilot.pause(0.2)

            status = app.query_one(StatusBar)
            return (
                app.last_reply,
                _plain(status.query_one("#status-left")),
                bool(app.query(".asm-meta")),
            )

        reply, status_left, meta = self._run((120, 36), script)

        self.assertIsNotNone(reply)
        self.assertEqual(reply[1], len(ANSWER))
        self.assertIn("chars", status_left)
        self.assertIn("tok/s", status_left)
        self.assertFalse(meta)


if __name__ == "__main__":
    unittest.main()
