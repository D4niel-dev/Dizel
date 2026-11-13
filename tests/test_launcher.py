from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from textual.widgets import ListView

import inference.tui.launcher as launcher

SESSIONS = [
    {
        "id": "20261108_120000",
        "title": "Plan a small desktop app",
        "preview": "A workable order for this",
        "created": "2026-11-08T12:00:00",
    }
]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self._sessions = mock.patch.object(
            launcher.Launcher, "_sessions", staticmethod(lambda: SESSIONS)
        )
        self._sessions.start()

    def tearDown(self):
        self._sessions.stop()

    def _run(self, size, key=None):
        async def run():
            app = launcher.Launcher()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                if key is not None:
                    await pilot.press(key)
                    await pilot.pause()
                return {
                    "return_code": app.return_code,
                    "choices": len(app.choices),
                    "compact": app.query_one("#shell").has_class("-compact"),
                    "wide": app.query_one("#shell").has_class("-wide"),
                    "stacked": app.query_one("#deck").has_class("-stacked"),
                    "facts": app.query_one("#brand-facts").display,
                    "foot": app.query_one("#choice-foot").display,
                    "wordmark": app.query_one("#wordmark").display,
                    "list_height": app.query_one(ListView).size.height,
                }

        return asyncio.run(run())

    def test_exit_codes_preserved(self):
        self.assertEqual(self._run((80, 24), "1")["return_code"], launcher.TERMINAL)
        self.assertEqual(self._run((80, 24), "2")["return_code"], launcher.DESKTOP)
        self.assertEqual(self._run((80, 24), "3")["return_code"], launcher.RESUME)
        self.assertEqual(self._run((80, 24), "escape")["return_code"], launcher.CANCELLED)

    def test_compact_80x24_layout(self):
        result = self._run((80, 24))
        self.assertTrue(result["compact"])
        self.assertFalse(result["wide"])
        self.assertTrue(result["stacked"])
        self.assertFalse(result["facts"])
        self.assertFalse(result["foot"])
        self.assertTrue(result["wordmark"])
        self.assertEqual(result["list_height"], 9)

    def test_wide_layout(self):
        result = self._run((120, 32))
        self.assertFalse(result["compact"])
        self.assertTrue(result["wide"])
        self.assertFalse(result["stacked"])
        self.assertTrue(result["facts"])
        self.assertTrue(result["foot"])
        self.assertTrue(result["wordmark"])
        self.assertEqual(result["list_height"], 9)

    def test_resume_is_unavailable_without_history(self):
        with mock.patch.object(
            launcher.Launcher, "_sessions", staticmethod(lambda: [])
        ):
            result = self._run((80, 24), "3")
        self.assertIsNone(result["return_code"])
        self.assertEqual(result["choices"], 2)


if __name__ == "__main__":
    unittest.main()
