"""Release notes come from the matching CHANGELOG.md section."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_notes.py"


class ReleaseNotesTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        (self.root / "CHANGELOG.md").write_text(
            "# Changelog\n\n"
            "## [2.0.2] - 2026-12-01\n\n"
            "### Fixed\n\n- A later fix.\n\n"
            "## [2.0.1] - 2026-11-15\n\n"
            "### Added\n\n- The shipped feature.\n\n"
            "### Fixed\n\n- The shipped fix.\n",
            encoding="utf-8",
        )

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_extracts_only_the_requested_version(self):
        result = self._run("v2.0.1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("### Added", result.stdout)
        self.assertIn("- The shipped feature.", result.stdout)
        self.assertIn("- The shipped fix.", result.stdout)
        self.assertNotIn("A later fix", result.stdout)
        self.assertNotIn("## [2.0.1]", result.stdout)

    def test_missing_version_fails(self):
        result = self._run("9.9.9")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no section for 9.9.9", result.stderr)


if __name__ == "__main__":
    unittest.main()
