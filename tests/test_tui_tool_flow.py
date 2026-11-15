"""Inline tool rows and the write-approval diff gate."""

from __future__ import annotations

import asyncio
import copy
import threading
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.tools import FileChange
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.bridge.tool_bridge import ToolBridge
from inference.tui.panels.modals.tool_approval import ToolApprovalModal
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.rendering.tool_block import ToolLine


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class ToolFlowTests(unittest.TestCase):
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

    def test_tool_rows_start_running_and_finish_in_place(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.clear_workspace(show_empty=False)
            workspace.add_user_message("Check the file.")
            await pilot.pause()

            workspace.post_message(
                ToolBridge.ToolStarted("grep", "pattern · src", key=11)
            )
            await pilot.pause()
            running = list(workspace.query(ToolLine))
            self.assertEqual(len(running), 1)
            running_row = running[0]
            before = (running_row.status, running_row.detail)

            workspace.post_message(
                ToolBridge.ToolCompleted("grep", "3 matches", key=11)
            )
            await pilot.pause()
            after = (running_row.status, running_row.detail)

            workspace.post_message(
                ToolBridge.ToolStarted("shell", "check", key=12)
            )
            await pilot.pause()
            workspace.post_message(
                ToolBridge.ToolFailed("shell", "exit 1", key=12)
            )
            await pilot.pause()
            failed = [
                row for row in workspace.query(ToolLine) if row.detail == "exit 1"
            ]
            return before, after, failed[0].status if failed else None

        before, after, failed = self._run((160, 42), script)
        self.assertEqual(before, ("RUNNING", "pattern · src"))
        self.assertEqual(after, ("DONE", "3 matches"))
        self.assertEqual(failed, "FAILED")

    def test_tool_completion_before_start_does_not_duplicate(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.clear_workspace(show_empty=False)
            workspace.add_user_message("Check the file.")
            await pilot.pause()

            workspace.post_message(
                ToolBridge.ToolCompleted("read_file", "README.md", key=21)
            )
            workspace.post_message(
                ToolBridge.ToolStarted("read_file", "README.md", key=21)
            )
            await pilot.pause()
            await pilot.pause()
            rows = list(workspace.query(ToolLine))
            return len(rows), rows[0].status if rows else None

        count, status = self._run((160, 42), script)
        self.assertEqual(count, 1)
        self.assertEqual(status, "DONE")

    def test_completed_tool_expands_its_drawer_in_place(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.clear_workspace(show_empty=False)
            workspace.post_message(
                ToolBridge.ToolStarted("edit_file", "src/App.tsx", key=31)
            )
            await pilot.pause()
            workspace.post_message(
                ToolBridge.ToolCompleted(
                    "edit_file",
                    "src/App.tsx",
                    key=31,
                    details={
                        "diff": (
                            "--- a/src/App.tsx\n"
                            "+++ b/src/App.tsx\n"
                            "@@ -1 +1 @@\n"
                            "-old line\n"
                            "+new line"
                        ),
                        "result": "Edited src/App.tsx (+1/-1)",
                    },
                )
            )
            await pilot.pause()

            row = list(workspace.query(ToolLine))[0]
            before = row.expanded
            preview = row.output_text()
            row.action_toggle()
            await pilot.pause()
            opened = row.expanded
            row.action_toggle()
            await pilot.pause()
            return before, preview, opened, row.expanded

        before, preview, opened, closed = self._run((160, 42), script)
        self.assertFalse(before)
        self.assertIn("+new line", preview)
        self.assertIn("-old line", preview)
        self.assertTrue(opened)
        self.assertFalse(closed)

    def test_write_approval_modal_shows_the_diff_and_approves(self):
        change = FileChange(
            "write_file",
            "notes.txt",
            "create",
            "--- a/notes.txt\n+++ b/notes.txt\n@@ -0,0 +1,2 @@\n+one\n+two",
            2,
            0,
        )

        async def script(app, pilot):
            results = []
            modal = ToolApprovalModal(change)
            self.assertTrue(modal._diff_text().plain.startswith("--- a/notes.txt"))
            self.assertIn("+one", modal._diff_text().plain)
            app.push_screen(modal, results.append)
            await pilot.pause()
            opened = isinstance(app.screen, ToolApprovalModal)
            await pilot.press("y")
            await pilot.pause()
            return opened, results

        opened, results = self._run((120, 36), script)
        self.assertTrue(opened)
        self.assertEqual(results, [True])

    def test_write_approval_modal_defaults_to_reject(self):
        change = FileChange("edit_file", "notes.txt", "edit", "", 0, 0)

        async def script(app, pilot):
            results = []
            app.push_screen(ToolApprovalModal(change), results.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            return results

        self.assertEqual(self._run((120, 36), script), [False])

    def test_chat_bridge_blocks_the_tool_thread_until_approval(self):
        change = FileChange(
            "write_file",
            "notes.txt",
            "create",
            "+one",
            1,
            0,
        )

        async def script(app, pilot):
            result = {}
            thread = threading.Thread(
                target=lambda: result.setdefault(
                    "approved", app.chat_bridge._approve_tool(change)
                ),
                daemon=True,
            )
            thread.start()
            for _ in range(30):
                await pilot.pause(0.05)
                if isinstance(app.screen, ToolApprovalModal):
                    break
            opened = isinstance(app.screen, ToolApprovalModal)
            await pilot.press("y")
            await pilot.pause()
            thread.join(timeout=2)
            return opened, result.get("approved"), thread.is_alive()

        opened, approved, alive = self._run((120, 36), script)
        self.assertTrue(opened)
        self.assertTrue(approved)
        self.assertFalse(alive)


if __name__ == "__main__":
    unittest.main()
