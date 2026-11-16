"""The delete tool is reviewed, root-bounded, recoverable, and never automatic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dizel_core import tools


class DeleteToolTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name).resolve()
        self._root_patch = mock.patch.object(
            tools, "_project_root", return_value=self.root
        )
        self._root_patch.start()
        self.addCleanup(self._root_patch.stop)
        self.addCleanup(self._temp.cleanup)
        # Never touch the real recycle bin in tests; record the call instead.
        self._trash_patch = mock.patch("send2trash.send2trash")
        self.trash = self._trash_patch.start()
        self.addCleanup(self._trash_patch.stop)

    def _file(self, name="notes.txt", text="one\ntwo\nthree\n"):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def _run(self, args, *, approved=True):
        seen = []

        def approve(change):
            seen.append(change)
            return approved

        result = tools.execute({"name": "delete_file", "args": args}, approve=approve)
        return result, seen

    def test_approved_delete_sends_the_file_to_the_recycle_bin(self):
        target = self._file()
        result, changes = self._run({"path": "notes.txt"})

        self.assertIn("Deleted notes.txt", result)
        self.trash.assert_called_once_with(str(target))
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].action, "delete")
        self.assertEqual(changes[0].additions, 0)
        self.assertEqual(changes[0].deletions, 3)
        self.assertIn("-one", changes[0].diff)

    def test_delete_requires_an_approval_channel(self):
        target = self._file()
        result = tools.execute({"name": "delete_file", "args": {"path": "notes.txt"}})

        self.assertIn("cannot approve", result)
        self.trash.assert_not_called()
        self.assertTrue(target.exists())

    def test_declined_delete_touches_nothing(self):
        target = self._file()
        result, changes = self._run({"path": "notes.txt"}, approved=False)

        self.assertIn("cancelled", result.lower())
        self.assertEqual(len(changes), 1)
        self.trash.assert_not_called()
        self.assertTrue(target.exists())

    def test_delete_cannot_escape_the_project_root(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("x", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))

        result = tools.execute(
            {"name": "delete_file", "args": {"path": "../outside.txt"}},
            approve=lambda change: True,
        )

        self.assertIn("limited to the current project", result)
        self.trash.assert_not_called()
        self.assertTrue(outside.exists())

    def test_delete_refuses_a_directory(self):
        folder = self.root / "subdir"
        folder.mkdir()

        result, changes = self._run({"path": "subdir"})

        self.assertIn("Not a regular file", result)
        self.assertEqual(changes, [])
        self.trash.assert_not_called()
        self.assertTrue(folder.is_dir())

    def test_delete_refuses_vcs_internals(self):
        config = self._file(".git/config", "[core]\n")

        result, changes = self._run({"path": ".git/config"})

        self.assertIn("Refusing", result)
        self.assertEqual(changes, [])
        self.trash.assert_not_called()
        self.assertTrue(config.exists())

    def test_delete_reports_a_missing_file(self):
        result, changes = self._run({"path": "ghost.txt"})

        self.assertIn("No such file", result)
        self.assertEqual(changes, [])
        self.trash.assert_not_called()

    def test_delete_allows_binary_files_unlike_writes(self):
        target = self.root / "image.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\n")

        result, _changes = self._run({"path": "image.png"})

        self.assertIn("Deleted image.png", result)
        self.trash.assert_called_once_with(str(target))

    def test_delete_failure_leaves_the_file_in_place(self):
        target = self._file()
        self.trash.side_effect = OSError("recycle bin unavailable")

        result, _changes = self._run({"path": "notes.txt"})

        self.assertIn("Failed:", result)
        self.assertTrue(target.exists())

    def test_delete_is_gated_like_other_reviewed_writes(self):
        self.assertNotIn("delete_file", tools.catalog())
        self.assertIn("delete_file", tools.catalog(include_writes=True))
        self.assertNotIn("delete_file", tools.prompt_block(allow_writes=False))
        self.assertIn("delete_file", tools.prompt_block(allow_writes=True))
        self.assertTrue(tools.catalog(include_writes=True)["delete_file"].requires_approval)

    def test_delete_payload_reports_the_removal(self):
        self._file()
        call = {"name": "delete_file", "args": {"path": "notes.txt"}}
        result = tools.execute(call, approve=lambda change: True)
        payload = tools.ui_payload(call, result)

        self.assertEqual(payload["path"], "notes.txt")
        self.assertEqual(payload["action"], "delete")
        self.assertEqual(payload["additions"], 0)
        self.assertEqual(payload["deletions"], 3)

    def test_declined_delete_is_reported_as_failed(self):
        self._file()
        outcome = tools.execute(
            {"name": "delete_file", "args": {"path": "notes.txt"}},
            approve=lambda change: False,
        )
        self.assertTrue(tools.outcome_failed("delete_file", outcome))


if __name__ == "__main__":
    unittest.main()