"""The capability manifest must describe real surfaces, not aspirational ones."""

from __future__ import annotations

import json

import pytest

from dizel_core import capabilities


def write_manifest(tmp_path, data):
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_manifest_loads_with_surface_entries():
    rows = capabilities.load()
    assert len(rows) >= 20
    by_id = {row.id: row for row in rows}
    entry = by_id["sessions.open"].tui
    assert entry is not None
    assert entry.label == "Ctrl+T"
    assert entry.check is not None and entry.check.kind == "binding"


def test_every_capability_is_reachable_somewhere():
    orphans = [row.id for row in capabilities.load() if row.tui is None and row.gui is None]
    assert orphans == []


def test_surface_entries_are_valid():
    rows = capabilities.load()
    ids = [row.id for row in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row.name.strip()
        for surface in ("tui", "gui"):
            entry = row.entry(surface)
            if entry is None:
                continue
            assert entry.label.strip()
            if entry.check:
                assert entry.check.kind in capabilities.KINDS[surface]
                assert entry.check.value.strip()


def test_tui_commands_resolve_in_the_real_registry():
    from inference.tui.commands.builtins import register_builtins
    from inference.tui.commands.registry import registry

    register_builtins()
    missing = []
    for row in capabilities.load():
        entry = row.tui
        if entry and entry.check and entry.check.kind == "command":
            if registry.lookup(entry.check.value) is None:
                missing.append((row.id, entry.check.value))
    assert missing == []


def test_tui_bindings_resolve_in_the_app():
    from inference.tui.app import DizelCMDApp

    keys = set()
    for binding in DizelCMDApp.BINDINGS:
        key = getattr(binding, "key", None)
        if key is None and isinstance(binding, tuple):
            key = binding[0]
        keys.add(key)

    missing = []
    for row in capabilities.load():
        entry = row.tui
        if entry and entry.check and entry.check.kind == "binding":
            if entry.check.value not in keys:
                missing.append((row.id, entry.check.value))
    assert missing == []


def test_usage_window_points_at_a_real_terminal_command():
    rows = {row.id: row for row in capabilities.load()}
    usage = rows["usage.window"].tui
    assert usage is not None
    assert usage.check is not None
    assert usage.check.kind == "command"
    assert usage.check.value == "debug"


def test_copy_reply_reaches_the_clipboard_on_both_surfaces():
    """Was a terminal gap. /copy writes OSC 52, so the gap is now closed."""
    rows = {row.id: row for row in capabilities.load()}
    tui = rows["chat.copy"].tui
    assert tui is not None
    assert tui.check is not None and tui.check.kind == "command"
    assert tui.check.value == "copy"
    assert rows["chat.copy"].gui is not None


def test_malformed_manifest_raises(tmp_path):
    path = write_manifest(tmp_path, {"version": 1, "capabilities": []})
    with pytest.raises(capabilities.ManifestError):
        capabilities.load(path)


def test_unknown_check_kind_raises(tmp_path):
    path = write_manifest(
        tmp_path,
        {
            "version": 2,
            "capabilities": [
                {
                    "id": "x.y",
                    "name": "X",
                    "tui": {"label": "x", "check": {"kind": "vibes", "value": "x"}},
                    "gui": None,
                }
            ],
        },
    )
    with pytest.raises(capabilities.ManifestError):
        capabilities.load(path)


def test_duplicate_ids_raise(tmp_path):
    entry = {"id": "x.y", "name": "X", "tui": {"label": "x"}, "gui": None}
    path = write_manifest(tmp_path, {"version": 2, "capabilities": [entry, entry]})
    with pytest.raises(capabilities.ManifestError):
        capabilities.load(path)


def test_summary_marks_gaps():
    text = capabilities.summary()
    assert "Open the session list" in text
    assert "Ctrl+T" in text
    assert "—" in text


def test_report_counts_capabilities():
    text = capabilities.report()
    assert "capabilities exist in both surfaces" in text
