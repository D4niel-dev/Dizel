"""Stamp one release version into every place that reports it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


def stamp(version: str, root: Path = ROOT) -> None:
    version = version.strip().lstrip("v")
    if not SEMVER.match(version):
        raise SystemExit(f"Not a release version: {version}")

    version_file = root / "dizel_core" / "version.py"
    text = version_file.read_text(encoding="utf-8")
    text, count = re.subn(
        r'^__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Could not stamp dizel_core/version.py")
    version_file.write_text(text, encoding="utf-8")

    package_file = root / "inference" / "gui" / "package.json"
    package = json.loads(package_file.read_text(encoding="utf-8"))
    package["version"] = version
    package_file.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")

    lock_file = root / "inference" / "gui" / "package-lock.json"
    if lock_file.is_file():
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        lock["version"] = version
        packages = lock.get("packages")
        if isinstance(packages, dict) and isinstance(packages.get(""), dict):
            packages[""]["version"] = version
        lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    npm_package_file = root / "npm" / "package.json"
    if npm_package_file.is_file():
        npm_package = json.loads(npm_package_file.read_text(encoding="utf-8"))
        npm_package["version"] = version
        npm_package_file.write_text(
            json.dumps(npm_package, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Stamped {version}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    stamp(args.version, Path(args.root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
