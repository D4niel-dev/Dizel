"""Composer placement parity with the desktop.

Empty chat: the welcome and the composer centre as one column mid-screen.
Conversation: the composer docks to the bottom edge. Position only — the
composer itself is unchanged.

All disk and network seams are stubbed: no real settings, no session files,
no model load, no update check, no MCP spawn.
"""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.input_bar import PromptInput
from inference.tui.panels.slash_drawer import MAX_ROWS, SlashDrawer
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.rendering.empty_state import EmptyStateBlock

FAKE_SESSION = {
    "messages": [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]
}


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class ComposerPlacementTests(unittest.TestCase):
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

    def _run(self, size, script, **app_kwargs):
        async def run():
            app = DizelCMDApp(**app_kwargs)
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    @staticmethod
    def _geometry(app):
        container = app.query_one("#workspace-container")
        composer = app.query_one("#composer-row")
        workspace = app.query_one(WorkspacePanel)
        return {
            "empty": container.has_class("-empty"),
            "welcome": bool(workspace.query(EmptyStateBlock)),
            "composer_top": composer.region.y,
            "composer_bottom": composer.region.bottom,
            "container_top": container.region.y,
            "container_bottom": container.region.bottom,
            "scroll_max": workspace.max_scroll_y,
        }

    # ── Welcome: centred composition ───────────────────────────────────
    def test_welcome_centers_composer(self):
        async def script(app, pilot):
            return self._geometry(app)

        for size in ((150, 42), (120, 36), (100, 30)):
            with self.subTest(size=size):
                geo = self._run(size, script)
                self.assertTrue(geo["empty"])
                self.assertTrue(geo["welcome"])
                # Floating, not docked: at least one row of space below.
                self.assertLess(geo["composer_bottom"], geo["container_bottom"])
                # The welcome sits above the composer, not below the fold.
                self.assertGreater(geo["composer_top"], geo["container_top"] + 4)

    def test_small_welcome_fits_without_scroll(self):
        async def script(app, pilot):
            return self._geometry(app)

        geo = self._run((80, 24), script)
        self.assertTrue(geo["empty"])
        self.assertTrue(geo["welcome"])
        self.assertEqual(geo["scroll_max"], 0)
        self.assertLessEqual(geo["composer_bottom"], geo["container_bottom"])

    # ── Conversation: docked composer ──────────────────────────────────
    def test_first_message_docks_composer(self):
        async def script(app, pilot):
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            await pilot.pause()
            return self._geometry(app)

        geo = self._run((120, 36), script)
        self.assertFalse(geo["empty"])
        self.assertFalse(geo["welcome"])
        self.assertEqual(geo["composer_bottom"], geo["container_bottom"])

    def test_clear_returns_composer_to_center(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.add_user_message("Hello there")
            await pilot.pause()
            docked = self._geometry(app)
            app.action_clear_workspace()
            await pilot.pause()
            await pilot.pause()
            return docked, self._geometry(app)

        docked, cleared = self._run((120, 36), script)
        self.assertEqual(docked["composer_bottom"], docked["container_bottom"])
        self.assertTrue(cleared["empty"])
        self.assertTrue(cleared["welcome"])
        self.assertLess(cleared["composer_bottom"], cleared["container_bottom"])

    def test_new_session_returns_composer_to_center(self):
        async def script(app, pilot):
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            await pilot.pause()
            app.action_new_session()
            await pilot.pause()
            await pilot.pause()
            return self._geometry(app)

        geo = self._run((120, 36), script)
        self.assertTrue(geo["empty"])
        self.assertTrue(geo["welcome"])
        self.assertLess(geo["composer_bottom"], geo["container_bottom"])

    def test_slash_drawer_fuzzy_paging_and_tab(self):
        async def script(app, pilot):
            inp = app.query_one("#prompt-input", PromptInput)
            inp.value = "/smz"
            await pilot.pause()
            await pilot.pause()
            drawer = app.query_one(SlashDrawer)
            fuzzy = drawer.selected().name if drawer.is_open else None

            inp.value = "/"
            await pilot.pause()
            total = len(drawer._matches)
            await pilot.press("pagedown")
            await pilot.pause()
            index = drawer._index
            picked = drawer.selected().name
            await pilot.press("tab")
            await pilot.pause()
            return fuzzy, total, index, picked, inp.value

        fuzzy, total, index, picked, value = self._run((120, 36), script)
        self.assertEqual(fuzzy, "summarize")
        self.assertGreater(total, MAX_ROWS)
        self.assertEqual(index, MAX_ROWS)
        self.assertEqual(value, f"/{picked} ")

    def test_long_system_message_docks_composer(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.add_system_message("\n".join(f"line {index}" for index in range(80)))
            await pilot.pause()
            return self._geometry(app)

        geo = self._run((100, 30), script)
        self.assertFalse(geo["empty"])
        self.assertTrue(geo["welcome"])
        self.assertEqual(geo["composer_bottom"], geo["container_bottom"])
        self.assertLessEqual(geo["composer_bottom"], 30)
        self.assertGreater(geo["scroll_max"], 0)

    def test_resume_opens_docked(self):
        async def script(app, pilot):
            await pilot.pause()
            return self._geometry(app)

        with mock.patch(
            "inference.tui.bridge.session_bridge.load_session",
            return_value=FAKE_SESSION,
        ):
            geo = self._run((120, 36), script, resume_session="20261110_120000")
        self.assertFalse(geo["empty"])
        self.assertFalse(geo["welcome"])
        self.assertEqual(geo["composer_bottom"], geo["container_bottom"])

    def test_tab_switch_round_trip(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.add_user_message("Hello there")
            await pilot.pause()
            old_id = app.session_id
            app.action_new_session()
            await pilot.pause()
            await pilot.pause()
            centered = self._geometry(app)
            # The old tab's messages live on disk; the bridge reads them back.
            with mock.patch(
                "inference.tui.bridge.session_bridge.load_session",
                return_value=FAKE_SESSION,
            ):
                app.load_session_to_workspace(old_id)
            await pilot.pause()
            await pilot.pause()
            return centered, self._geometry(app)

        centered, docked = self._run((120, 36), script)
        self.assertTrue(centered["empty"])
        self.assertLess(centered["composer_bottom"], centered["container_bottom"])
        self.assertFalse(docked["empty"])
        self.assertEqual(docked["composer_bottom"], docked["container_bottom"])


if __name__ == "__main__":
    unittest.main()