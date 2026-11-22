"""The shared capability manifest.

One file, read by both surfaces, listing what each can do. A `null` is a recorded
gap rather than an undiscovered one — that is the whole point.

A surface entry carries a display label and, where possible, a machine-readable
`check` naming how it is verified: terminal slash commands and key bindings by
`tests/test_capabilities.py`, desktop palette entries and DOM controls by the
GUI smoke harness. Entries without a check are verified by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Union

MANIFEST = Path(__file__).resolve().parent.parent / "capabilities.json"
VERSION = 2

KINDS: Dict[str, FrozenSet[str]] = {
    "tui": frozenset({"command", "binding"}),
    "gui": frozenset({"palette", "control"}),
}
SURFACES = tuple(KINDS)


class ManifestError(ValueError):
    """The manifest is missing, malformed, or lists an impossible check."""


@dataclass(frozen=True)
class Check:
    kind: str
    value: str


@dataclass(frozen=True)
class SurfaceEntry:
    label: str
    check: Optional[Check] = None


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    tui: Optional[SurfaceEntry]
    gui: Optional[SurfaceEntry]

    def entry(self, surface: str) -> Optional[SurfaceEntry]:
        return getattr(self, surface, None)

    def has(self, surface: str) -> bool:
        return self.entry(surface) is not None


PathLike = Union[str, Path, None]


def _surface_entry(surface: str, raw) -> Optional[SurfaceEntry]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestError(f"{surface} entry must be an object or null")
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ManifestError(f"{surface} entry needs a label")
    raw_check = raw.get("check")
    if raw_check is None:
        return SurfaceEntry(label=label)
    if not isinstance(raw_check, dict):
        raise ManifestError(f"{surface} check must be an object")
    kind = raw_check.get("kind")
    value = raw_check.get("value")
    if kind not in KINDS[surface]:
        raise ManifestError(
            f"{surface} check kind {kind!r} is not one of {sorted(KINDS[surface])}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{surface} check needs a value")
    return SurfaceEntry(label=label, check=Check(kind=kind, value=value))


def load(path: PathLike = None) -> List[Capability]:
    source = Path(path) if path else MANIFEST
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ManifestError(f"cannot read {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(data, dict):
        raise ManifestError("manifest must be an object")
    if data.get("version") != VERSION:
        raise ManifestError(f"manifest version must be {VERSION}")
    raw_rows = data.get("capabilities")
    if not isinstance(raw_rows, list):
        raise ManifestError("manifest needs a capabilities list")

    rows: List[Capability] = []
    seen = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ManifestError("each capability must be an object")
        cap_id = raw.get("id")
        name = raw.get("name")
        if not isinstance(cap_id, str) or not cap_id.strip():
            raise ManifestError("capability needs an id")
        if cap_id in seen:
            raise ManifestError(f"duplicate capability id {cap_id!r}")
        seen.add(cap_id)
        if not isinstance(name, str) or not name.strip():
            raise ManifestError(f"{cap_id} needs a name")
        rows.append(
            Capability(
                id=cap_id,
                name=name,
                tui=_surface_entry("tui", raw.get("tui")),
                gui=_surface_entry("gui", raw.get("gui")),
            )
        )
    return rows


def gaps(surface: str) -> List[Capability]:
    """Capabilities the named surface does not have."""
    return [row for row in load() if not row.has(surface)]


def _label(row: Capability, surface: str) -> str:
    entry = row.entry(surface)
    return entry.label if entry else "—"


def summary() -> str:
    """A two-column table, marking what is missing."""
    rows = load()
    if not rows:
        return "No capability manifest found at capabilities.json."

    name_width = max(len(row.name) for row in rows) + 2
    tui_width = max(len(_label(row, "tui")) for row in rows) + 2
    lines = [
        f"{'capability'.ljust(name_width)}{'TUI'.ljust(tui_width)}GUI",
        "-" * (name_width + tui_width + 24),
    ]
    for row in rows:
        lines.append(
            f"{row.name.ljust(name_width)}"
            f"{_label(row, 'tui').ljust(tui_width)}{_label(row, 'gui')}"
        )
    return "\n".join(lines)


def report() -> str:
    """What is missing where, for whoever is about to build the next feature."""
    rows = load()
    if not rows:
        return "No capability manifest found."
    both = [row for row in rows if row.tui and row.gui]
    tui_only = [row for row in rows if row.tui and not row.gui]
    gui_only = [row for row in rows if row.gui and not row.tui]

    lines = [f"{len(both)} of {len(rows)} capabilities exist in both surfaces."]
    if gui_only:
        lines.append("\nGUI only:")
        lines += [f"  {row.name}  ({row.gui.label})" for row in gui_only]
    if tui_only:
        lines.append("\nTUI only:")
        lines += [f"  {row.name}  ({row.tui.label})" for row in tui_only]
    return "\n".join(lines)
