"""/debug diagnostics and render fixtures stay outside the saved transcript."""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.commands.parser import parse_command
from inference.tui.commands.registry import registry
from inference.tui.panels.modals.debug_panel import DebugPanelModal
from inference.tui.panels.status_bar import StatusBar
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.rendering.assistant_message import AssistantMessage
from inference.tui.rendering.tool_block import ToolLine


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


def _plain(widget) -> str:
    visual = widget.visual
    return getattr(visual, "plain", str(visual))


class DebugModeTests(unittest.TestCase):
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

    def _run(self, script):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=(160, 42)) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    async def _debug(self, app, pilot, text):
        command = registry.lookup("debug")
        self.assertIsNotNone(command)
        await command.execute(app, parse_command(text))
        await pilot.pause()

    def test_toggle_shows_the_debug_badge(self):
        # The final mode is off, so capture the badge before turning it off.
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=(160, 42)) as pilot:
                await pilot.pause()
                await pilot.pause()
                await self._debug(app, pilot, "/debug")
                status = app.query_one(StatusBar)
                badge = _plain(status.query_one("#status-right"))
                await self._debug(app, pilot, "/debug off")
                off_badge = _plain(status.query_one("#status-right"))
                return app.debug_mode, badge, off_badge

        enabled, badge, off_badge = asyncio.run(run())
        self.assertFalse(enabled)
        self.assertIn("DEBUG", badge)
        self.assertNotIn("DEBUG", off_badge)

    def test_help_status_and_usage_open_a_debug_panel(self):
        async def script(app, pilot):
            await self._debug(app, pilot, "/debug on")
            await self._debug(app, pilot, "/debug help")
            help_open = isinstance(app.screen, DebugPanelModal)
            help_body = app.screen._body
            await pilot.press("escape")
            await pilot.pause()

            await self._debug(app, pilot, "/debug status")
            status_open = isinstance(app.screen, DebugPanelModal)
            status_body = app.screen._body
            await pilot.press("escape")
            await pilot.pause()

            await self._debug(app, pilot, "/debug usage")
            usage_open = isinstance(app.screen, DebugPanelModal)
            return help_open, help_body, status_open, status_body, usage_open

        help_open, help_body, status_open, status_body, usage_open = self._run(
            script
        )
        self.assertTrue(help_open)
        self.assertIn("Render fixtures", help_body)
        self.assertTrue(status_open)
        self.assertIn("generation", status_body)
        self.assertTrue(usage_open)

    def test_markdown_and_thought_fixtures_append_assistant_turns(self):
        async def script(app, pilot):
            await self._debug(app, pilot, "/debug on")
            await self._debug(app, pilot, "/debug markdown")
            markdown_count = len(app.query(AssistantMessage))
            await self._debug(app, pilot, "/debug thought")
            thought_count = len(app.query(AssistantMessage))
            await self._debug(app, pilot, "/debug echo hello")
            echo_count = len(app.query(AssistantMessage))
            return markdown_count, thought_count, echo_count

        markdown_count, thought_count, echo_count = self._run(script)
        self.assertEqual(markdown_count, 1)
        self.assertEqual(thought_count, 2)
        self.assertEqual(echo_count, 3)

    def test_tools_fixture_appends_inspectable_rows(self):
        async def script(app, pilot):
            await self._debug(app, pilot, "/debug on")
            await self._debug(app, pilot, "/debug tools")
            rows = list(app.query(ToolLine))
            names = [row.tool_name for row in rows]
            detail = "\n".join(row.output_text() for row in rows)
            before = rows[0].expanded
            rows[0].action_toggle()
            await pilot.pause()
            expanded = rows[0].expanded
            return names, detail, before, expanded

        names, detail, before, expanded = self._run(script)
        self.assertEqual(
            names, ["grep", "read_file", "edit_file", "write_file", "web_search"]
        )
        self.assertIn("+  // reviewed by the tool fixture", detail)
        self.assertIn("Source: https://github.com/D4niel-dev/Dizel", detail)
        self.assertFalse(before)
        self.assertTrue(expanded)

    def test_staged_update_shows_a_persistent_badge(self):
        async def script(app, pilot):
            app.update_ready = "2.1.0"
            status = app.query_one(StatusBar)
            status.update_status()
            shown = _plain(status.query_one("#status-right"))
            app.update_ready = ""
            status.update_status()
            cleared = _plain(status.query_one("#status-right"))
            return shown, cleared

        shown, cleared = self._run(script)
        self.assertIn("UPDATE v2.1.0", shown)
        self.assertNotIn("UPDATE", cleared)

    def test_off_debug_rejects_inspection_commands(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            before = len(workspace.query(AssistantMessage))
            await self._debug(app, pilot, "/debug markdown")
            after = len(workspace.query(AssistantMessage))
            notice = _plain(app.query_one("NoticeLine"))
            return app.debug_mode, before, after, notice

        enabled, before, after, notice = self._run(script)
        self.assertFalse(enabled)
        self.assertEqual(before, after)
        self.assertIn("Debug mode is off", notice)


if __name__ == "__main__":
    unittest.main()
