"""Project memory routes and the terminal command share one store."""

from __future__ import annotations

import asyncio

import pytest

from api.routes import project as project_routes
from dizel_core import memory_store, tools
from inference.tui.commands.builtins.memory import MemoryCommand
from inference.tui.commands.parser import parse_command


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "DIZEL.md").write_text("Always use tabs.", encoding="utf-8")
    monkeypatch.setattr(memory_store, "MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setattr(tools, "_project_root", lambda: root)
    return root


def test_routes_add_list_remove_and_clear(_isolated):
    added = project_routes.add_memory(project_routes.MemoryText(text="Prefers black."))
    assert added["text"] == "Prefers black."

    listed = project_routes.project_memory()
    assert listed["entries"][0]["text"] == "Prefers black."
    assert "DIZEL.md" in listed["rules_summary"]

    removed = project_routes.remove_memory(
        project_routes.MemoryId(id=listed["entries"][0]["id"])
    )
    assert removed["removed"] is True
    assert project_routes.project_memory()["entries"] == []

    project_routes.add_memory(project_routes.MemoryText(text="one"))
    project_routes.add_memory(project_routes.MemoryText(text="two"))
    assert project_routes.clear_memory()["removed"] == 2


class FakeApp:
    shown = None

    def show_debug_output(self, title, body):
        self.shown = (title, body)


def test_memory_command_lists_entries_and_rules(_isolated):
    memory_store.add("Prefers black.", project_root=_isolated)
    app = FakeApp()

    result = asyncio.run(MemoryCommand().execute(app, parse_command("/memory")))

    assert result == ""
    assert app.shown[0] == "Project memory"
    assert "Prefers black." in app.shown[1]
    assert "DIZEL.md" in app.shown[1]


def test_memory_command_add_remove_and_clear(_isolated):
    app = FakeApp()
    added = asyncio.run(
        MemoryCommand().execute(app, parse_command("/memory add Uses CRLF for .cmd."))
    )
    assert "saved" in added.lower()
    assert memory_store.entries(_isolated)[0]["text"] == "Uses CRLF for .cmd."

    removed = asyncio.run(
        MemoryCommand().execute(app, parse_command("/memory remove 1"))
    )
    assert "removed" in removed.lower()
    assert memory_store.entries(_isolated) == []

    memory_store.add("one", project_root=_isolated)
    cleared = asyncio.run(MemoryCommand().execute(app, parse_command("/memory clear")))
    assert "1" in cleared
    assert memory_store.entries(_isolated) == []


def test_memory_command_rules_summary(_isolated):
    app = FakeApp()

    result = asyncio.run(MemoryCommand().execute(app, parse_command("/memory rules")))

    assert result == ""
    assert app.shown[0] == "Project rules"
    assert "DIZEL.md" in app.shown[1]
