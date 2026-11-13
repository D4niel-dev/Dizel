"""The shared capability manifest.

One file, read by both surfaces, listing what each can do. A `null` is a recorded
gap rather than an undiscovered one — that is the whole point.

Nothing here enforces anything. It is a description, and a description that can
drift is worse than none, so `gaps()` exists to make the drift visible and
`scripts/check_capabilities.py` prints it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

MANIFEST = Path(__file__).resolve().parent.parent / "capabilities.json"


@dataclass
class Capability:
    id: str
    name: str
    tui: Optional[str]
    gui: Optional[str]

    def has(self, surface: str) -> bool:
        return bool(getattr(self, surface, None))


def load() -> List[Capability]:
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        Capability(
            id=item.get("id", ""),
            name=item.get("name", ""),
            tui=item.get("tui"),
            gui=item.get("gui"),
        )
        for item in data.get("capabilities", [])
    ]


def gaps(surface: str) -> List[Capability]:
    """Capabilities the named surface does not have."""
    return [cap for cap in load() if not cap.has(surface)]


def summary() -> str:
    """A two-column table, marking what is missing."""
    rows = load()
    if not rows:
        return "No capability manifest found at capabilities.json."

    width = max(len(cap.name) for cap in rows) + 2
    lines = [f"{'capability'.ljust(width)}TUI{'':<14}GUI", "-" * (width + 34)]
    for cap in rows:
        tui = cap.tui or "—"
        gui = cap.gui or "—"
        lines.append(f"{cap.name.ljust(width)}{tui:<17}{gui}")
    return "\n".join(lines)


def report() -> str:
    """What is missing where, for whoever is about to build the next feature."""
    rows = load()
    if not rows:
        return "No capability manifest found."
    tui_only = [c for c in rows if c.tui and not c.gui]
    gui_only = [c for c in rows if c.gui and not c.tui]
    both = [c for c in rows if c.tui and c.gui]

    lines = [f"{len(both)} of {len(rows)} capabilities exist in both surfaces."]
    if gui_only:
        lines.append("\nGUI only:")
        lines += [f"  {c.name}  ({c.gui})" for c in gui_only]
    if tui_only:
        lines.append("\nTUI only:")
        lines += [f"  {c.name}  ({c.tui})" for c in tui_only]
    return "\n".join(lines)
