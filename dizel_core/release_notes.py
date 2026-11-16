"""The CHANGELOG bundled with the build, read in-app for release notes."""

from __future__ import annotations

import os
from typing import Optional

from dizel_core import updater


def read_changelog(root: Optional[os.PathLike | str] = None) -> str:
    """Return the bundled CHANGELOG.md text, or "" when it is not present."""
    path = updater.repo_root(root) / "CHANGELOG.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""