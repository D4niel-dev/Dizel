"""Mode pickers render colored glyphs and the effort picker dismisses a level."""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.modals.pickers import (
    ModePickerModal,
    ThinkingEffortPickerModal,
)


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class TuiModePickerTests(unittest.TestCase):
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

    def _run(self, script):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    def test_mode_picker_rows_are_colored_and_glyphed(self):
        async def script(app, pilot):
            app.push_screen(ModePickerModal(), lambda mode: None)
            await pilot.pause()
            return [
                (item.id, item.glyph, item.color) for item in app.screen._items
            ]

        rows = self._run(script)
        self.assertEqual(
            [row[0] for row in rows], ["Fast", "Planning", "Coding", "Thinking"]
        )
        self.assertTrue(all(row[1] for row in rows))
        self.assertEqual(rows[3][2], "#a78bfa")  # default medium effort

    def test_effort_picker_lists_five_levels_and_dismisses(self):
        async def script(app, pilot):
            results = []
            app.push_screen(ThinkingEffortPickerModal(), results.append)
            await pilot.pause()
            items = list(app.screen._items)
            self.assertEqual(
                [item.id for item in items],
                ["low", "medium", "high", "xhigh", "deep"],
            )
            self.assertEqual(len({item.color for item in items}), 5)
            app.screen.dismiss("high")
            await pilot.pause()
            return results

        self.assertEqual(self._run(script), ["high"])


if __name__ == "__main__":
    unittest.main()
