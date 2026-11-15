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
from inference.tui.panels.context_panel import ContextPanel
from inference.tui.panels.input_bar import InputBar, PromptInput
from inference.tui.panels.notice_line import NoticeLine
from inference.tui.panels.slash_drawer import MAX_ROWS, SlashDrawer
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.rendering.empty_state import EmptyStateBlock, StarterShelf, SuggestionRow, Wordmark

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

    def test_welcome_is_the_simplified_composition(self):
        async def script(app, pilot):
            block = app.query_one(EmptyStateBlock)
            for selector in (".empty-kicker", ".empty-meta", ".empty-hint"):
                self.assertEqual(len(block.query(selector)), 0)
            self.assertEqual(len(block.query(".empty-logo")), 1)
            self.assertEqual(len(block.query(".empty-greeting")), 1)
            self.assertEqual(len(block.query(".empty-prompt")), 1)
            self.assertEqual(len(block.query(SuggestionRow)), 0)

            composer = app.query_one("#composer-row")
            logo = app.query_one("#empty-logo", Wordmark)
            greeting = block.query_one(".empty-greeting")
            prompt = block.query_one(".empty-prompt")
            self.assertLessEqual(logo.region.bottom, composer.region.y)
            self.assertEqual(prompt.region.y, greeting.region.bottom)
            composer_center = composer.region.x + composer.region.width // 2
            for widget in (logo, greeting, prompt):
                center = widget.region.x + widget.region.width // 2
                self.assertLessEqual(abs(center - composer_center), 1)

            shelf = app.query_one(StarterShelf)
            suggestions = list(shelf.query(SuggestionRow))
            self.assertEqual(len(suggestions), 3)
            self.assertTrue(all(row.display for row in suggestions))
            self.assertGreaterEqual(shelf.region.y, composer.region.bottom)

        self._run((120, 36), script)

    def test_short_welcome_uses_compact_wordmark_art(self):
        async def script(app, pilot):
            logo = app.query_one("#empty-logo", Wordmark)
            composer = app.query_one("#composer-row")
            container = app.query_one("#workspace-container")
            shelf = app.query_one(StarterShelf)

            self.assertGreaterEqual(logo.region.height, 3)
            self.assertLessEqual(logo.region.bottom, composer.region.y)
            self.assertGreaterEqual(shelf.region.y, composer.region.bottom)
            self.assertLessEqual(shelf.region.bottom, container.region.bottom)

        self._run((80, 24), script)

    def test_context_gauge_uses_spaced_segments(self):
        for percent, filled, empty in ((0, 0, 5), (50, 3, 2), (100, 5, 0)):
            _color, bar = InputBar._gauge(percent, 5)
            self.assertEqual(bar.count("▰"), filled)
            self.assertEqual(bar.count("▱"), empty)
            self.assertEqual(bar.count(" "), 4)

    def test_composer_rail_is_thick(self):
        async def script(app, pilot):
            return app.query_one(InputBar).styles.border_left

        border_type, _color = self._run((120, 36), script)
        self.assertEqual(str(border_type), "thick")

    def test_transient_notice_fades_inside_one_point_five_seconds(self):
        async def script(app, pilot):
            notice = app.query_one(NoticeLine)
            notice.show("temporary")
            await pilot.pause(0.1)
            shown = notice.has_class("-shown")
            await pilot.pause(1.5)
            return shown, notice.has_class("-shown")

        shown, hidden = self._run((120, 36), script)
        self.assertTrue(shown)
        self.assertFalse(hidden)

    def test_starter_shelf_leaves_with_the_empty_state(self):
        async def script(app, pilot):
            self.assertTrue(app.query(StarterShelf))
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            for _ in range(20):
                await pilot.pause(0.05)
                if not app.query(StarterShelf):
                    break
            self.assertFalse(app.query(StarterShelf))
            self.assertFalse(app.query_one("#workspace-container").has_class("-empty"))

            app.query_one(WorkspacePanel).clear_workspace()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.query(StarterShelf):
                    break
            self.assertTrue(app.query(StarterShelf))

        self._run((120, 36), script)

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

    def test_chat_uses_the_left_measure_and_the_welcome_stays_compact(self):
        async def script(app, pilot):
            empty_width = app.query_one(InputBar).region.width
            app.query_one(WorkspacePanel).add_user_message("Hello there")
            for _ in range(20):
                await pilot.pause(0.05)
                if app.query_one(ContextPanel).display:
                    break
            workspace = app.query_one(WorkspacePanel)
            composer = app.query_one(InputBar)
            rail = app.query_one(ContextPanel)
            return (
                empty_width,
                workspace.region,
                composer.region,
                rail.region,
            )

        empty_width, workspace, composer, rail = self._run((160, 42), script)
        self.assertEqual(empty_width, 96)
        self.assertEqual(workspace.x, 2)
        self.assertEqual(composer.x, 2)
        # The context rail is 32 cells and the conversation keeps two-cell
        # gutters on each side, so it fills everything else.
        self.assertEqual(workspace.width, 160 - 32 - 4)
        self.assertEqual(composer.width, workspace.width)
        self.assertEqual(rail.width, 32)
        self.assertLessEqual(workspace.right, rail.x)

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

    def test_slash_clear_keeps_the_welcome_centered(self):
        async def script(app, pilot):
            initial = self._geometry(app)
            workspace = app.query_one(WorkspacePanel)
            workspace.add_user_message("Hello there")
            await pilot.pause()

            inp = app.query_one("#prompt-input", PromptInput)
            inp.focus()
            inp.value = "/clear"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if app.query_one("#workspace-container").has_class("-empty"):
                    break
            notice = app.query_one(NoticeLine)
            composer = app.query_one(InputBar)
            return (
                initial,
                self._geometry(app),
                notice.has_class("-shown"),
                list(workspace.query(".system-line")),
                bool(app.query(StarterShelf)),
                (notice.region.x, notice.region.width),
                (composer.region.x, composer.region.width),
            )

        (
            initial,
            cleared,
            notice,
            lines,
            starters,
            notice_box,
            composer_box,
        ) = self._run((120, 36), script)
        self.assertTrue(cleared["empty"])
        self.assertTrue(cleared["welcome"])
        self.assertLess(cleared["composer_bottom"], cleared["container_bottom"])
        # The cleared column may gain the one-row notice, never a docked composer.
        self.assertLessEqual(abs(cleared["composer_top"] - initial["composer_top"]), 2)
        self.assertTrue(notice)
        self.assertEqual(lines, [])
        self.assertTrue(starters)
        # The notice is the composer's companion strip, not a full-width banner.
        self.assertEqual(notice_box, composer_box)

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