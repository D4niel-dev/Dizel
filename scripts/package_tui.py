"""Build the portable TUI archive attached to a Dizel release.

The archive contains application code only. User data (`.dizel`, `data`,
`logs`, `models`, `checkpoints`) is never included, so applying an update cannot
overwrite chats, settings, or model files.
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

APP_DIRS = ("api", "dizel_core", "model", "core", "utils", "inference/tui")
APP_FILES = (
    "config.py",
    "capabilities.json",
    "requirements.txt",
    "README.md",
    "dizel.cmd",
    "inference/__init__.py",
)
EXCLUDED_PARTS = {"__pycache__", ".git", "node_modules", ".dizel", "data", "logs"}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".safetensors",
    ".log",
}


def read_version(root: Path) -> str:
    text = (root / "dizel_core" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("Could not read __version__ from dizel_core/version.py")
    return match.group(1)


def iter_payload(root: Path):
    for folder in APP_DIRS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                yield path
    for name in APP_FILES:
        path = root / name
        if path.is_file():
            yield path


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def build(root: Path, output: Path, version: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"Dizel-TUI-{version}.zip"
    seen: set[Path] = set()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in iter_payload(root):
            relative = path.relative_to(root)
            if relative in seen or not included(path, root):
                continue
            seen.add(relative)
            bundle.write(path, relative.as_posix())
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--version", default="")
    parser.add_argument("--output", default=str(ROOT / "release-artifacts"))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    version = args.version.strip().lstrip("v") or read_version(root)
    archive = build(root, Path(args.output).resolve(), version)
    size = os.path.getsize(archive)
    with zipfile.ZipFile(archive) as bundle:
        count = len(bundle.infolist())
    print(f"Built {archive} · {count} files · {size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
