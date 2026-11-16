"""In-app release notes read the CHANGELOG bundled with the build."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dizel_core import release_notes


class ReadChangelogTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def test_reads_the_changelog_from_the_root(self):
        (self.root / "CHANGELOG.md").write_text(
            "# Changelog\n\n## [9.9.9] - 2026-01-01\n\n### Added\n\n- A thing.\n",
            encoding="utf-8",
        )

        text = release_notes.read_changelog(self.root)

        self.assertIn("## [9.9.9]", text)
        self.assertIn("- A thing.", text)

    def test_missing_changelog_is_empty(self):
        self.assertEqual(release_notes.read_changelog(self.root), "")


if __name__ == "__main__":
    unittest.main()