"""The desktop half of the capability manifest, checked against its sources.

The terminal's claims are verified by driving the real command registry and
binding table. The desktop cannot be booted in a unit test, but every checked
control is marked in the markup with `data-capability`, so the claim can be
verified from the source tree instead of a screenshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dizel_core import capabilities

GUI_SRC = Path(__file__).resolve().parents[1] / "inference" / "gui" / "src"


def _gui_sources() -> str:
    if not GUI_SRC.is_dir():
        pytest.skip("GUI sources are not present in this checkout.")
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in GUI_SRC.rglob("*")
        if path.suffix in {".ts", ".tsx"} and path.is_file()
    )


def _commands_source() -> str:
    path = GUI_SRC / "services" / "commands.ts"
    if not path.is_file():
        pytest.skip("GUI command table is not present in this checkout.")
    return path.read_text(encoding="utf-8")


def test_every_checked_desktop_control_is_marked_in_the_markup():
    source = _gui_sources()
    missing = []
    for row in capabilities.load():
        entry = row.gui
        if not entry or not entry.check or entry.check.kind != "control":
            continue
        value = entry.check.value
        # Rendered as an attribute, or handed to Picker, which renders it.
        if (
            f'data-capability="{value}"' not in source
            and f'triggerDataCapability="{value}"' not in source
        ):
            missing.append((row.id, value))
    assert missing == []


def test_every_checked_desktop_palette_entry_exists():
    source = _commands_source()
    missing = []
    for row in capabilities.load():
        entry = row.gui
        if not entry or not entry.check or entry.check.kind != "palette":
            continue
        value = entry.check.value
        if f'name: "{value}"' not in source and f'debug("{value}"' not in source:
            missing.append((row.id, value))
    assert missing == []


def test_desktop_gaps_are_recorded_as_nulls_not_omissions():
    """A surface entry with neither a label nor a check is a lie by omission."""
    for row in capabilities.load():
        for entry in (row.tui, row.gui):
            if entry is None:
                continue
            assert entry.label.strip()
