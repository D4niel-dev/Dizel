"""Print the CHANGELOG.md section for one release version."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract(version: str, root: Path = ROOT) -> str:
    version = version.strip().lstrip("v")
    if not version:
        raise SystemExit("A release version is required.")
    changelog = root / "CHANGELOG.md"
    try:
        text = changelog.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Could not read {changelog}: {error}") from error

    section = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not section:
        raise SystemExit(f"CHANGELOG.md has no section for {version}.")
    body = section.group(1).strip()
    if not body:
        raise SystemExit(f"CHANGELOG.md section for {version} is empty.")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="release version, with or without a leading v")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)

    print(extract(args.version, Path(args.root).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
