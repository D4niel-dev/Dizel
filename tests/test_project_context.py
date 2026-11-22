"""Project rules, memory composition, and the approval-gated remember tool."""

from __future__ import annotations

import pytest

from dizel_core import memory_store, project_context


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_store, "MEMORY_DIR", str(tmp_path / "memory"))


def make_project(base, files=None):
    root = base / "project"
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (files or {}).items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_memory_round_trip(tmp_path):
    root = make_project(tmp_path)

    entry = memory_store.add("Uses CRLF for .cmd files.", project_root=root)

    assert entry["source"] == "user"
    assert memory_store.entries(root)[0]["text"] == "Uses CRLF for .cmd files."
    assert memory_store.remove(entry["id"], root) is True
    assert memory_store.entries(root) == []


def test_projects_do_not_share_memory(tmp_path):
    first = make_project(tmp_path / "a")
    second = make_project(tmp_path / "b")

    memory_store.add("belongs to the first project", project_root=first)

    assert memory_store.entries(first)
    assert memory_store.entries(second) == []


def test_memory_rejects_empty_text(tmp_path):
    with pytest.raises(ValueError):
        memory_store.add("   ", project_root=make_project(tmp_path))


def test_memory_caps_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_store, "MAX_ENTRIES", 2)
    root = make_project(tmp_path)
    memory_store.add("one", project_root=root)
    memory_store.add("two", project_root=root)

    with pytest.raises(ValueError):
        memory_store.add("three", project_root=root)


def test_memory_caps_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_store, "MAX_CHARS", 20)
    root = make_project(tmp_path)
    memory_store.add("a" * 15, project_root=root)

    with pytest.raises(ValueError):
        memory_store.add("b" * 10, project_root=root)


def test_clear_returns_removed_count(tmp_path):
    root = make_project(tmp_path)
    memory_store.add("one", project_root=root)
    memory_store.add("two", project_root=root)

    assert memory_store.clear(root) == 2
    assert memory_store.entries(root) == []


def test_rules_are_read_in_order_and_labeled(tmp_path):
    root = make_project(
        tmp_path,
        {"DIZEL.md": "Always use tabs.", "AGENTS.md": "Run pytest before commits."},
    )

    context = project_context.load(root)
    blocks = project_context.system_blocks(context)

    assert [source["name"] for source in context.sources] == ["DIZEL.md", "AGENTS.md"]
    assert "Always use tabs." in blocks
    assert "Run pytest before commits." in blocks
    assert "Project rules" in blocks


def test_rules_are_truncated_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(project_context, "MAX_CHARS", 40)
    root = make_project(tmp_path, {"DIZEL.md": "x" * 200})

    context = project_context.load(root)

    assert len(context.rules) < 200
    assert "truncated" in context.rules


def test_missing_rules_are_fine(tmp_path):
    context = project_context.load(make_project(tmp_path))

    assert context.sources == []
    assert project_context.system_blocks(context) == ""


def test_memory_is_part_of_the_system_blocks(tmp_path):
    root = make_project(tmp_path)
    memory_store.add("Prefers black formatting.", project_root=root)

    blocks = project_context.system_blocks(project_context.load(root))

    assert "Prefers black formatting." in blocks
    assert "Project memory" in blocks


def test_system_content_joins_prompt_and_context():
    from dizel_core.chat_manager import ChatManager

    manager = ChatManager()
    manager.system_prompt = "BASE"
    manager.project_context = "PROJECT"

    assert manager.system_content() == "BASE\n\nPROJECT"

    manager.project_context = ""
    assert manager.system_content() == "BASE"


def test_refresh_reads_the_project(tmp_path, monkeypatch):
    from dizel_core import tools

    root = make_project(tmp_path, {"DIZEL.md": "Prefer small commits."})
    monkeypatch.setattr(tools, "_project_root", lambda: root)
    from dizel_core.chat_manager import ChatManager

    manager = ChatManager()
    manager.refresh_project_context()

    assert "Prefer small commits." in manager.project_context


def test_remember_requires_approval_and_stores(tmp_path, monkeypatch):
    from dizel_core import tools

    root = make_project(tmp_path)
    monkeypatch.setattr(tools, "_project_root", lambda: root)

    approved = tools.execute(
        {"name": "remember", "args": {"text": "Uses CRLF for .cmd files."}},
        approve=lambda change: True,
    )
    assert "Saved" in approved
    assert memory_store.entries(root)[0]["text"] == "Uses CRLF for .cmd files."

    denied = tools.execute(
        {"name": "remember", "args": {"text": "Never happens."}},
        approve=lambda change: False,
    )
    assert "cancel" in denied.lower()
    assert len(memory_store.entries(root)) == 1


def test_remember_without_an_approver_is_refused(tmp_path, monkeypatch):
    from dizel_core import tools

    root = make_project(tmp_path)
    monkeypatch.setattr(tools, "_project_root", lambda: root)

    outcome = tools.execute({"name": "remember", "args": {"text": "No."}})

    assert "refused" in outcome.lower()
    assert memory_store.entries(root) == []
