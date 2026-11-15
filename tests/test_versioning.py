"""The version command keeps every displayed app version in lockstep."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "set_version.py"


class VersionCommandTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self._seed("2.0.0")

    def _seed(self, version: str) -> None:
        version_file = self.root / "dizel_core" / "version.py"
        version_file.parent.mkdir(parents=True)
        version_file.write_text(
            f'APP_NAME = "Dizel"\n__version__ = "{version}"\nVERSION = __version__\n',
            encoding="utf-8",
        )

        gui = self.root / "inference" / "gui"
        gui.mkdir(parents=True)
        (gui / "package.json").write_text(
            json.dumps({"name": "dizel-gui", "version": version}, indent=2) + "\n",
            encoding="utf-8",
        )
        (gui / "package-lock.json").write_text(
            json.dumps(
                {
                    "name": "dizel-gui",
                    "version": version,
                    "packages": {"": {"name": "dizel-gui", "version": version}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        npm = self.root / "npm"
        npm.mkdir()
        (npm / "package.json").write_text(
            json.dumps({"name": "@d4niel-dev/dizel", "version": version}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _versions(self) -> tuple[str, str, str, str]:
        version = (self.root / "dizel_core" / "version.py").read_text(encoding="utf-8")
        gui = json.loads(
            (self.root / "inference" / "gui" / "package.json").read_text(encoding="utf-8")
        )
        lock = json.loads(
            (self.root / "inference" / "gui" / "package-lock.json").read_text(
                encoding="utf-8"
            )
        )
        npm = json.loads(
            (self.root / "npm" / "package.json").read_text(encoding="utf-8")
        )
        match = re.search(r'^__version__ = "([^"]+)"', version, re.MULTILINE)
        self.assertIsNotNone(match)
        return (
            match.group(1),
            gui["version"],
            lock["packages"][""]["version"],
            npm["version"],
        )

    def test_bump_patch_updates_the_canonical_and_manifest_versions(self):
        result = self._run("bump", "patch")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2.0.0 -> 2.0.1", result.stdout)
        self.assertEqual(self._versions(), ("2.0.1", "2.0.1", "2.0.1", "2.0.1"))
        self.assertEqual(self._run("check").returncode, 0)

    def test_bump_minor_and_major(self):
        self.assertEqual(self._run("bump", "minor").returncode, 0)
        self.assertEqual(self._versions()[0], "2.1.0")
        self.assertEqual(self._run("bump", "major").returncode, 0)
        self.assertEqual(self._versions()[0], "3.0.0")

    def test_legacy_set_form_still_stamps(self):
        result = self._run("2.1.0")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._versions(), ("2.1.0", "2.1.0", "2.1.0", "2.1.0"))

    def test_check_reports_a_manifest_drift(self):
        gui = self.root / "inference" / "gui" / "package.json"
        payload = json.loads(gui.read_text(encoding="utf-8"))
        payload["version"] = "1.9.9"
        gui.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        result = self._run("check")

        self.assertEqual(result.returncode, 1)
        self.assertIn("Version drift (expected 2.0.0)", result.stdout)
        self.assertIn("inference/gui/package.json: 1.9.9", result.stdout)

    def test_show_prints_the_current_version(self):
        result = self._run("show")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2.0.0")

    def test_startup_banners_read_the_stamped_sources(self):
        from dizel_core.version import __version__
        from inference.tui.launcher import _version as launcher_version
        from inference.tui.panels.status_bar import _version as status_version

        self.assertEqual(launcher_version(), __version__)
        self.assertEqual(status_version(), __version__)

        tui_main = (ROOT / "inference" / "tui" / "main.py").read_text(encoding="utf-8")
        self.assertIn("from dizel_core.version import __version__", tui_main)
        self.assertIn('f"Dizel {__version__}"', tui_main)

        # The desktop console banner is stamped through the GUI manifest.
        launcher = (ROOT / "dizel.cmd").read_text(encoding="utf-8")
        self.assertIn("require('./package.json').version", launcher)


if __name__ == "__main__":
    unittest.main()
