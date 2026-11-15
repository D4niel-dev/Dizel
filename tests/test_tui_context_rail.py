"""The context rail is part of the chat surface, not an overlay."""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp, PanelOverlay
from inference.tui.panels.context_panel import ContextPanel
from inference.tui.panels.workspace import WorkspacePanel


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class _FakeMcpManager:
    def configured(self):
        return [
            {"id": "github", "name": "GitHub", "enabled": True},
            {"id": "linear", "name": "Linear", "enabled": True},
            {"id": "jira", "name": "Jira", "enabled": True},
            {"id": "notes", "name": "Notes", "enabled": False},
        ]

    def status(self):
        return [
            {"id": "github", "state": "ready", "error": "", "tools": [object()] * 3},
            {"id": "linear", "state": "error", "error": "auth failed", "tools": []},
            {"id": "jira", "state": "starting", "error": "", "tools": []},
        ]

    def shutdown(self):
        pass


class ContextRailTests(unittest.TestCase):
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

    def test_rail_is_hidden_on_the_welcome(self):
        async def script(app, pilot):
            panel = app.query_one(ContextPanel)
            self.assertFalse(panel.display)
            await pilot.press("ctrl+r")
            await pilot.pause(0.1)
            self.assertFalse(panel.display)
            self.assertNotIsInstance(app.screen, PanelOverlay)

        self._run((120, 36), script)

    def test_rail_appears_with_the_chat_and_cannot_close(self):
        async def script(app, pilot):
            panel = app.query_one(ContextPanel)
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            await pilot.pause(0.2)
            self.assertTrue(panel.display)
            self.assertEqual(panel.region.width, 32)

            await pilot.press("ctrl+r")
            await pilot.pause(0.1)
            self.assertTrue(panel.has_focus)
            self.assertNotIsInstance(app.screen, PanelOverlay)

            await pilot.press("escape")
            await pilot.pause(0.1)
            self.assertTrue(panel.display)
            self.assertNotIsInstance(app.screen, PanelOverlay)

            await pilot.press("ctrl+r")
            await pilot.pause(0.1)
            self.assertTrue(panel.display)

        self._run((120, 36), script)

    def test_rail_tracks_chat_session_state(self):
        async def script(app, pilot):
            panel = app.query_one(ContextPanel)
            self.assertFalse(panel.display)

            app.query_one(WorkspacePanel).add_user_message("Hello there")
            await pilot.pause(0.2)
            self.assertTrue(panel.display)

            app.action_new_session()
            await pilot.pause(0.2)
            self.assertFalse(panel.display)

            app.query_one(WorkspacePanel).add_user_message("New chat")
            await pilot.pause(0.2)
            self.assertTrue(panel.display)

            app.action_toggle_session()
            await pilot.pause(0.2)
            self.assertTrue(panel.display)

            app.query_one(WorkspacePanel).clear_workspace()
            await pilot.pause(0.2)
            self.assertFalse(panel.display)

        self._run((120, 36), script)

    def test_rail_fits_beside_the_chat_on_80_columns(self):
        async def script(app, pilot):
            panel = app.query_one(ContextPanel)
            composer = app.query_one("#composer-row")
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            await pilot.pause(0.2)
            self.assertTrue(panel.display)
            self.assertEqual(panel.region.width, 32)
            self.assertGreaterEqual(panel.region.x, composer.region.right)

        self._run((80, 24), script)

    def test_rail_lists_plugin_servers_and_status(self):
        async def script(app, pilot):
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            await pilot.pause(0.2)
            visual = app.query_one("#ctx-plugins").visual
            return getattr(visual, "plain", str(visual))

        with mock.patch(
            "dizel_core.mcp.manager.get_manager", return_value=_FakeMcpManager()
        ):
            text = self._run((120, 36), script)

        self.assertIn("GitHub", text)
        self.assertIn("3 tools", text)
        self.assertIn("Linear", text)
        self.assertIn("auth failed", text)
        self.assertIn("Jira", text)
        self.assertIn("connecting", text)
        self.assertIn("Notes", text)
        self.assertIn("off", text)
