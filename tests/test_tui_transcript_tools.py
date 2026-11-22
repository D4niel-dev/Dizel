"""Transcript search, reply length and clipboard, driven through the real app.

Same seams as test_tui_composer_placement: no settings writes, no sessions on
disk, no model load.
"""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.bridge.chat_bridge import ChatBridge
from inference.tui.commands.builtins import register_builtins
from inference.tui.commands.parser import parse_command
from inference.tui.commands.registry import registry
from inference.tui.panels.status_bar import StatusBar
from inference.tui.panels.workspace import WorkspacePanel


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class TranscriptToolsTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(DizelCMDApp, "_auto_load_model", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_update_check", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_mcp", lambda self: None),
            mock.patch.object(ConfigManager, "load", classmethod(lambda cls: _defaults())),
            mock.patch.object(ConfigManager, "save", classmethod(lambda cls, s: None)),
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
        register_builtins()

    def _stop_patches(self):
        for patcher in self._patches:
            patcher.stop()

    def _run(self, script, size=(140, 40)):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    async def _conversation(self, app, pilot):
        """Two finished turns: one with the needle, one without.

        Each turn is posted and paused on its own — a real generation always
        lands one at a time, and the workspace clears `current_message` when a
        turn completes.
        """

        async def turn(prompt, answer):
            workspace = app.query_one(WorkspacePanel)
            workspace.add_user_message(prompt)
            workspace.start_assistant_message()
            workspace.current_message.stream.append(answer)
            workspace.current_message.update_stream()
            workspace.post_message(ChatBridge.GenerationDone(answer))
            await pilot.pause()

        await turn("How do I run docker compose?", "Use docker compose up, then docker compose logs.")
        await turn("What about production?", "Ship the image, not the compose file.")
        return app.query_one(WorkspacePanel)

    # ── /find ──────────────────────────────────────────────────────────
    def test_find_lands_on_the_matching_turn_and_flags_it(self):
        async def script(app, pilot):
            workspace = await self._conversation(app, pilot)
            result = workspace.find("docker")
            await pilot.pause()
            flagged = [
                widget for widget in workspace._turns if "-find-hit" in widget[1].classes
            ]
            return result, len(workspace._turns), flagged

        result, turns, flagged = self._run(script)
        self.assertEqual(result["query"], "docker")
        # The prompt mentions docker too: both turns match, three times over.
        self.assertEqual(result["turns"], 2)
        self.assertEqual(result["matches"], 3)
        self.assertEqual(turns, 4)
        self.assertEqual(len(flagged), 1, "only the landed-on turn is tinted")

    def test_find_cycles_through_matching_turns(self):
        async def script(app, pilot):
            workspace = await self._conversation(app, pilot)
            first = workspace.find("docker")
            second = workspace.find("")
            third = workspace.find("")
            return first, second, third

        first, second, third = self._run(script)
        self.assertEqual(first["index"], 1)
        self.assertEqual(second["index"], 2)
        self.assertEqual(third["index"], 1, "wraps back to the first match")
        self.assertEqual(first["turns"], second["turns"])

    def test_find_reports_nothing_when_the_needle_is_absent(self):
        async def script(app, pilot):
            workspace = await self._conversation(app, pilot)
            return workspace.find("kubernetes")

        result = self._run(script)
        self.assertEqual(result["turns"], 0)
        self.assertEqual(result["matches"], 0)

    def test_find_through_the_command_names_the_turn(self):
        async def script(app, pilot):
            await self._conversation(app, pilot)
            return await registry.lookup("find").execute(
                app, parse_command("/find compose")
            )

        message = self._run(script)
        self.assertIn("turn 1 of", message)

    # ── Length ─────────────────────────────────────────────────────────
    def test_status_bar_shows_a_non_default_length(self):
        async def script(app, pilot):
            bar = app.query_one(StatusBar)
            before = bar.query_one("#status-left").render()
            app.active_verbosity = "detailed"
            await pilot.pause()
            after = bar.query_one("#status-left").render()
            return str(before), str(after)

        before, after = self._run(script)
        self.assertNotIn("length", before)
        self.assertIn("length detailed", after)

    def test_verbosity_command_updates_the_app(self):
        async def script(app, pilot):
            await registry.lookup("verbosity").execute(
                app, parse_command("/verbosity short")
            )
            return app.active_verbosity

        self.assertEqual(self._run(script), "short")

    # ── Clipboard ──────────────────────────────────────────────────────
    def test_copy_puts_the_last_reply_on_the_clipboard(self):
        async def script(app, pilot):
            workspace = await self._conversation(app, pilot)
            reply = await registry.lookup("copy").execute(app, parse_command("/copy"))
            return app._clipboard, reply

        clipboard, reply = self._run(script)
        self.assertEqual(clipboard, "Ship the image, not the compose file.")
        self.assertIn("OSC 52", reply)
