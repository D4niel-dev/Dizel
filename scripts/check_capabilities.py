"""Print the capability manifest, and optionally fail on entries nobody has.

    python scripts/check_capabilities.py
    python scripts/check_capabilities.py --strict

`--strict` exits non-zero if a capability is listed but exists in **neither**
surface — a manifest entry nobody can invoke is dead weight, and the only kind of
drift worth failing a build over.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dizel_core import capabilities  # noqa: E402


def main() -> int:
    rows = capabilities.load()
    if not rows:
        print("capabilities.json is missing or empty.")
        return 1

    print(capabilities.summary())
    print()
    print(capabilities.report())

    orphans = [c for c in rows if not c.tui and not c.gui]
    if orphans:
        print("\nListed but available nowhere:")
        for cap in orphans:
            print(f"  {cap.id} — {cap.name}")

    if "--strict" in sys.argv and orphans:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
