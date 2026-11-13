"""Generate the release manifest the updater reads before downloading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def digest(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest(), os.path.getsize(path)


def asset(path: Path) -> dict:
    sha256, size = digest(path)
    return {"asset": path.name, "sha256": sha256, "size": size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--gui", required=True)
    parser.add_argument("--tui", required=True)
    parser.add_argument("--notes-file", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gui = Path(args.gui)
    tui = Path(args.tui)
    for path in (gui, tui):
        if not path.is_file():
            raise SystemExit(f"Missing release asset: {path}")

    notes = ""
    if args.notes_file:
        notes = Path(args.notes_file).read_text(encoding="utf-8").strip()
    else:
        notes = os.environ.get("RELEASE_NOTES", "").strip()

    version = args.version.strip().lstrip("v")
    payload = {
        "version": version,
        "tag": args.tag.strip() or f"v{version}",
        "released_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": notes[:20000],
        "gui": asset(gui),
        "tui": asset(tui),
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
