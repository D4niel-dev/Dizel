"""Electron window geometry persisting is exercised through its pure module."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "fixtures" / "window_state_check.cjs"


@unittest.skipIf(NODE is None, "node is not installed")
class WindowStateTests(unittest.TestCase):
    def test_sanitize_roundtrip_and_display_guard(self):
        result = subprocess.run(
            [NODE, str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)
