"""Reviewed write/edit tools stay inside the project and need approval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dizel_core import tools


class WriteToolTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name).resolve()
        self._patch = mock.patch.object(tools, "_project_root", return_value=self.root)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._temp.cleanup)

    def _run(self, name, args, *, approved=True):
        seen = []

        def approve(change):
            seen.append(change)
            return approved

        result = tools.execute({"name": name, "args": args}, approve=approve)
        return result, seen

    def test_write_creates_a_file_and_reports_a_reviewed_diff(self):
        result, changes = self._run(
            "write_file", {"path": "notes.txt", "content": "one\ntwo\n"}
        )

        self.assertIn("Created notes.txt", result)
        self.assertEqual((self.root / "notes.txt").read_text(encoding="utf-8"), "one\ntwo\n")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].action, "create")
        self.assertEqual(changes[0].additions, 2)
        self.assertIn("+one", changes[0].diff)

    def test_write_requires_an_approval_channel(self):
        result = tools.execute(
            {"name": "write_file", "args": {"path": "notes.txt", "content": "x"}}
        )
        self.assertIn("cannot approve", result)
        self.assertFalse((self.root / "notes.txt").exists())

    def test_declined_write_touches_nothing(self):
        result, changes = self._run(
            "write_file", {"path": "notes.txt", "content": "x"}, approved=False
        )
        self.assertIn("cancelled", result.lower())
        self.assertEqual(len(changes), 1)
        self.assertFalse((self.root / "notes.txt").exists())

    def test_write_cannot_escape_the_project_root(self):
        result = tools.execute(
            {"name": "write_file", "args": {"path": "../outside.txt", "content": "x"}},
            approve=lambda change: True,
        )
        self.assertIn("limited to the current project", result)
        self.assertFalse((self.root.parent / "outside.txt").exists())

    def test_edit_preserves_crlf_line_endings(self):
        target = self.root / "run.cmd"
        target.write_bytes(b"one\r\ntwo\r\n")

        result, changes = self._run(
            "edit_file", {"path": "run.cmd", "old": "two", "new": "three"}
        )

        self.assertIn("Edited run.cmd", result)
        self.assertEqual(target.read_bytes(), b"one\r\nthree\r\n")
        self.assertEqual(changes[0].deletions, 1)
        self.assertEqual(changes[0].additions, 1)

    def test_edit_requires_a_unique_snippet(self):
        target = self.root / "repeat.txt"
        target.write_text("same\nsame\n", encoding="utf-8")

        result, changes = self._run(
            "edit_file", {"path": "repeat.txt", "old": "same", "new": "once"}
        )

        self.assertIn("more than once", result)
        self.assertEqual(changes, [])
        self.assertEqual(target.read_text(encoding="utf-8"), "same\nsame\n")

    def test_write_payload_includes_written_content_and_diff(self):
        call = {
            "name": "write_file",
            "args": {"path": "notes.txt", "content": "one\ntwo\n"},
        }
        result = tools.execute(call, approve=lambda change: True)
        payload = tools.ui_payload(call, result)

        self.assertEqual(payload["content"], "one\ntwo\n")
        self.assertEqual(payload["action"], "create")
        self.assertEqual(payload["additions"], 2)
        self.assertIn("+one", payload["diff"])
        self.assertEqual(payload["path"], "notes.txt")

    def test_grep_payload_names_the_query_and_keeps_results(self):
        target = self.root / "notes.txt"
        target.write_text("first\nneedle here\n", encoding="utf-8")
        call = {
            "name": "grep",
            "args": {"pattern": "needle", "path": str(self.root), "glob": "*.txt"},
        }
        result = tools.execute(call)
        payload = tools.ui_payload(call, result)

        self.assertEqual(payload["query"], "needle")
        self.assertEqual(payload["path"], str(self.root))
        self.assertIn("needle here", payload["result"])

    def test_web_search_payload_has_a_search_url(self):
        call = {"name": "web_search", "args": {"query": "dizel local model"}}
        payload = tools.ui_payload(call, "Source: https://example.com")

        self.assertEqual(payload["query"], "dizel local model")
        self.assertIn("duckduckgo.com", payload["search_url"])

    def test_write_tools_only_appear_when_approval_is_available(self):
        self.assertNotIn("write_file", tools.catalog())
        self.assertIn("write_file", tools.catalog(include_writes=True))
        self.assertNotIn("write_file", tools.prompt_block(allow_writes=False))
        self.assertIn("write_file", tools.prompt_block(allow_writes=True))


if __name__ == "__main__":
    unittest.main()
