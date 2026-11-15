"""Session tab strip layout and scrolling.

Each tab owns its close control, create follows the newest tab, and the strip
turns into a one-row horizontal scroller above ten tabs. All disk and network
seams are stubbed.
"""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.session_tabs import SessionTabs, TabClose
from textual.widgets import Button, Tab


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class SessionTabTests(unittest.TestCase):
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

    def _run(self, count, script, size=(120, 36)):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                sessions = [f"20260101-0000{i:02d}" for i in range(count)]
                app.open_tabs = list(sessions)
                app.session_id = sessions[-1] if sessions else ""
                app.tab_title = lambda session_id: f"Chat {session_id[-2:]}"
                app.refresh_tabs()
                for _ in range(40):
                    await pilot.pause(0.05)
                    if len(app.query(TabClose)) == count:
                        break
                await pilot.pause(0.2)
                return await script(app, pilot)

        return asyncio.run(run())

    def test_close_and_create_follow_the_latest_tab(self):
        async def script(app, pilot):
            strip = app.query_one(SessionTabs)
            tabs = app.query_one("#session-tabs")
            children = list(tabs.query_one("#tabs-list").children)

            pattern = []
            for child in children:
                if isinstance(child, Tab):
                    pattern.append("tab")
                elif isinstance(child, TabClose):
                    pattern.append("close")
                elif child.id == "tab-new-inline":
                    pattern.append("new")
                else:
                    pattern.append("other")
            self.assertEqual(pattern, ["tab", "close", "tab", "close", "tab", "close", "new"])

            closes = list(app.query(TabClose))
            for close in closes:
                tab = app.query_one(f"#session-{close.session_id}", Tab)
                self.assertEqual(close.region.x, tab.region.right)
            newest = app.query_one("#tab-new-inline", Button)
            self.assertEqual(newest.region.x, closes[-1].region.right)
            self.assertEqual(
                [c.session_id for c in closes if c.has_class("-active")],
                [app.open_tabs[-1]],
            )
            self.assertFalse(strip.has_class("is-scrollable"))
            self.assertFalse(tabs.has_class("is-scrollable"))
            self.assertEqual(strip.region.height, 2)
            self.assertFalse(app.query("#tab-close"))
            self.assertFalse(app.query("#tab-new"))

        self._run(3, script)

    def test_ten_tabs_stay_on_one_row(self):
        async def script(app, pilot):
            strip = app.query_one(SessionTabs)
            tabs = app.query_one("#session-tabs")
            self.assertFalse(strip.has_class("is-scrollable"))
            self.assertEqual(tabs.region.height, 2)

        self._run(10, script)

    def test_eleven_tabs_scroll_to_the_newest(self):
        async def script(app, pilot):
            strip = app.query_one(SessionTabs)
            tabs = app.query_one("#session-tabs")
            scroller = tabs.query_one("#tabs-scroll")
            newest = app.query_one("#tab-new-inline", Button)

            self.assertTrue(strip.has_class("is-scrollable"))
            self.assertTrue(tabs.has_class("is-scrollable"))
            self.assertEqual(strip.region.height, 3)
            self.assertEqual(tabs.region.height, 3)
            self.assertEqual(str(scroller.styles.overflow_x), "auto")
            self.assertGreater(scroller.max_scroll_x, 0)
            self.assertEqual(scroller.scroll_x, scroller.max_scroll_x)
            self.assertLessEqual(newest.region.right, scroller.region.right)

        self._run(11, script)

    def test_create_button_opens_a_tab_after_the_newest(self):
        async def script(app, pilot):
            before = list(app.open_tabs)
            await pilot.click("#tab-new-inline")
            for _ in range(40):
                await pilot.pause(0.05)
                if (
                    len(app.open_tabs) == len(before) + 1
                    and len(app.query(TabClose)) == len(before) + 1
                ):
                    break
            self.assertEqual(len(app.open_tabs), len(before) + 1)
            self.assertNotIn(app.open_tabs[-1], before)
            self.assertEqual(
                [c.session_id for c in app.query(TabClose)], app.open_tabs
            )
            self.assertTrue(app.query_one("#tab-new-inline", Button).can_focus)

        self._run(3, script)

    def test_each_close_button_removes_its_own_tab(self):
        async def script(app, pilot):
            first = app.open_tabs[0]
            await pilot.click(f"#tab-x-{first}")
            for _ in range(30):
                await pilot.pause(0.05)
                if first not in app.open_tabs and len(app.query(TabClose)) == 2:
                    break
            self.assertNotIn(first, app.open_tabs)
            self.assertEqual(len(app.query(TabClose)), 2)
            self.assertEqual(
                [c.session_id for c in app.query(TabClose)], app.open_tabs
            )

        self._run(3, script)
