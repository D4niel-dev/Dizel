"""Read, bump, stamp, and verify the one Dizel application version.

``dizel_core/version.py`` is the source used by the TUI, API, updater, and
release tools. The GUI package and npm launcher are ecosystem manifests that
must carry the same value, so this command writes all of them together.

    python scripts/set_version.py                  # show the current version
    python scripts/set_version.py bump patch       # 2.0.0 -> 2.0.1
    python scripts/set_version.py bump minor       # 2.0.1 -> 2.1.0
    python scripts/set_version.py set 2.1.0
    python scripts/set_version.py check            # fail on drift

The legacy ``python scripts/set_version.py 2.1.0`` form still works.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
VERSION_LINE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
BUMP_PARTS = ("major", "minor", "patch")


def _version_file(root: Path) -> Path:
    return root / "dizel_core" / "version.py"


def read_version(root: Path = ROOT) -> str:
    """Return the canonical application version."""
    match = VERSION_LINE.search(_version_file(root).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("Could not read __version__ from dizel_core/version.py")
    return match.group(1)


def _normalise(version: str) -> str:
    text = version.strip().lstrip("v")
    if not SEMVER.match(text):
        raise SystemExit(f"Not a release version: {version}")
    return text


def _write_if_changed(path: Path, text: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _json_version(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return str(version) if version is not None else None


def stamp(version: str, root: Path = ROOT) -> list[str]:
    """Write one version into every managed manifest; return changed paths."""
    version = _normalise(version)
    changed: list[str] = []

    version_path = _version_file(root)
    text = version_path.read_text(encoding="utf-8")
    next_text, count = re.subn(
        r'^__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Could not stamp dizel_core/version.py")
    if _write_if_changed(version_path, next_text):
        changed.append(str(version_path.relative_to(root)))

    gui_package = root / "inference" / "gui" / "package.json"
    package = json.loads(gui_package.read_text(encoding="utf-8"))
    if package.get("version") != version:
        package["version"] = version
        _write_if_changed(gui_package, json.dumps(package, indent=2) + "\n")
        changed.append(str(gui_package.relative_to(root)))

    gui_lock = root / "inference" / "gui" / "package-lock.json"
    if gui_lock.is_file():
        lock = json.loads(gui_lock.read_text(encoding="utf-8"))
        touched = False
        if lock.get("version") != version:
            lock["version"] = version
            touched = True
        packages = lock.get("packages")
        if isinstance(packages, dict) and isinstance(packages.get(""), dict):
            if packages[""].get("version") != version:
                packages[""]["version"] = version
                touched = True
        if touched:
            _write_if_changed(gui_lock, json.dumps(lock, indent=2) + "\n")
            changed.append(str(gui_lock.relative_to(root)))

    npm_package = root / "npm" / "package.json"
    if npm_package.is_file():
        package = json.loads(npm_package.read_text(encoding="utf-8"))
        if package.get("version") != version:
            package["version"] = version
            _write_if_changed(npm_package, json.dumps(package, indent=2) + "\n")
            changed.append(str(npm_package.relative_to(root)))

    return changed


def bump(part: str, root: Path = ROOT) -> tuple[str, str]:
    """Increment the canonical version and stamp every manifest."""
    part = part.strip().lower()
    if part not in BUMP_PARTS:
        raise SystemExit("Bump part must be major, minor, or patch.")
    current = read_version(root)
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", current)
    if not match:
        raise SystemExit(f"Not a bumpable version: {current}")
    major, minor, patch = (int(piece) for piece in match.groups())
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    version = f"{major}.{minor}.{patch}"
    stamp(version, root)
    return current, version


def check(root: Path = ROOT) -> list[str]:
    """Return human-readable mismatches; empty when every manifest agrees."""
    expected = read_version(root)
    mismatches: list[str] = []

    def compare(relative: str, actual: str | None) -> None:
        if actual != expected:
            mismatches.append(f"  {relative}: {actual or 'missing'}")

    compare(
        "inference/gui/package.json",
        _json_version(root / "inference" / "gui" / "package.json"),
    )
    lock_path = root / "inference" / "gui" / "package-lock.json"
    if lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mismatches.append("  inference/gui/package-lock.json: unreadable")
        else:
            compare("inference/gui/package-lock.json", str(lock.get("version") or ""))
            packages = lock.get("packages")
            if isinstance(packages, dict) and isinstance(packages.get(""), dict):
                compare(
                    "inference/gui/package-lock.json (packages[''])",
                    str(packages[""].get("version") or ""),
                )
    compare("npm/package.json", _json_version(root / "npm" / "package.json"))
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="show")
    parser.add_argument("value", nargs="?")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    command = (args.command or "show").strip()
    value = args.value.strip() if isinstance(args.value, str) else args.value

    # Keep the old `set_version.py 2.1.0` release-workflow invocation working.
    if value is None and SEMVER.match(command.lstrip("v")):
        command, value = "set", command
    command = command.lower()

    if command in ("show", "current", "status"):
        print(read_version(root))
        return 0

    if command in ("set", "stamp"):
        if not value:
            parser.error("set requires a version, for example: set 2.1.0")
        version = _normalise(value)
        changed = stamp(version, root)
        if changed:
            print(f"Stamped {version} in {len(changed)} file(s):")
            for path in changed:
                print(f"  {path}")
        else:
            print(f"{version} was already stamped everywhere.")
        return 0

    if command == "bump":
        old, new = bump(value or "patch", root)
        print(f"Bumped {old} -> {new}; stamped the GUI, lockfile, and npm launcher.")
        return 0

    if command in BUMP_PARTS and value is None:
        old, new = bump(command, root)
        print(f"Bumped {old} -> {new}; stamped the GUI, lockfile, and npm launcher.")
        return 0

    if command == "check":
        mismatches = check(root)
        if mismatches:
            print(f"Version drift (expected {read_version(root)}):")
            print("\n".join(mismatches))
            return 1
        print(f"All managed versions match {read_version(root)}.")
        return 0

    parser.error(
        "Unknown command. Use show, set <version>, bump <major|minor|patch>, or check."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
