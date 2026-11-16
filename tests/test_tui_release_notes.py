"""The /release-notes command opens a modal showing the bundled CHANGELOG."""

from __future__ import annotations

import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.markdown import Markdown as RichMarkdown
from textual.widgets import Static

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.commands.builtins.release_notes import ReleaseNotesCommand
from inference.tui.commands.parser import parse_command
from inference.tui.panels.modals.release_notes import ReleaseNotesModal

CHANGELOG = "# Changelog\n\n## [9.9.9] - 2026-01-01\n\n### Added\n\n- A thing.\n"


class FakeApp:
    def __init__(self) -> None:
        self.screens: list = []

    def push_screen(self, screen, callback=None) -> None:
        self.screens.append(screen)


class ReleaseNotesCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_command_opens_the_modal_with_the_bundled_changelog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
            app = FakeApp()
            with mock.patch.dict(os.environ, {"DIZEL_ROOT": str(root)}):
                message = await ReleaseNotesCommand().execute(
                    app, parse_command("/release-notes")
                )

        self.assertEqual(message, "")
        self.assertEqual(len(app.screens), 1)
        modal = app.screens[0]
        self.assertIsInstance(modal, ReleaseNotesModal)
        self.assertIn("- A thing.", modal._text)

    async def test_command_without_a_changelog_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = FakeApp()
            with mock.patch.dict(os.environ, {"DIZEL_ROOT": tmp}):
                message = await ReleaseNotesCommand().execute(
                    app, parse_command("/release-notes")
                )

        self.assertIn("No release notes", message)
        self.assertEqual(app.screens, [])


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class ReleaseNotesModalTests(unittest.TestCase):
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

    def test_modal_renders_the_changelog_as_markdown(self):
        async def script(app, pilot):
            app.push_screen(ReleaseNotesModal(CHANGELOG, "9.9.9"))
            await pilot.pause()
            opened = isinstance(app.screen, ReleaseNotesModal)
            body = app.screen.query_one("#release-notes-text", Static)
            return opened, isinstance(body.visual._renderable, RichMarkdown)

        opened, is_markdown = self._run((120, 36), script)
        self.assertTrue(opened)
        self.assertTrue(is_markdown)

    def test_modal_closes_on_escape(self):
        async def script(app, pilot):
            results = []
            app.push_screen(ReleaseNotesModal(CHANGELOG, "9.9.9"), results.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            return results

        self.assertEqual(self._run((120, 36), script), [None])


if __name__ == "__main__":
    unittest.main()