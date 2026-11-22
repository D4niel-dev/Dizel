"""The commands and helpers added for clipboard, search, length and handoff."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from inference.tui.commands.builtins.__init__ import register_builtins
from inference.tui.commands.builtins.copy import CopyCommand
from inference.tui.commands.builtins.find import FindCommand
from inference.tui.commands.builtins.open_gui import OpenInGuiCommand
from inference.tui.commands.builtins.verbosity import VerbosityCommand
from inference.tui.commands.parser import parse_command
from inference.tui.commands.registry import registry


def run(command, app, text):
    """These commands are async by interface only; drive one to completion."""
    return asyncio.run(command.execute(app, parse_command(text)))


@pytest.fixture(autouse=True)
def _builtins():
    register_builtins()


def test_new_commands_are_registered():
    for name in ("copy", "find", "verbosity", "open"):
        assert registry.lookup(name) is not None, name
    # /find is reachable as /search too.
    assert registry.lookup("search") is registry.lookup("find")


class FakeApp:
    """Only what the commands under test actually touch."""

    def __init__(self, messages=None, session_id="20260101_120000"):
        self.session_bridge = SimpleNamespace(current_messages=messages or [])
        self.session_id = session_id
        self.clipboard = None
        self.panel = SimpleNamespace(calls=[], find=None)
        self.active_verbosity = "normal"

    def copy_to_clipboard(self, text):
        self.clipboard = text

    def query_one(self, selector):
        return self.panel


def test_copy_takes_the_last_reply_and_explains_osc52():
    app = FakeApp(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "two"},
        ]
    )
    out = run(CopyCommand(), app, "/copy")
    assert app.clipboard == "two"
    assert "OSC 52" in out


def test_copy_counts_backwards_and_refuses_out_of_range():
    app = FakeApp(
        [
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
    )
    run(CopyCommand(), app, "/copy 2")
    assert app.clipboard == "one"

    out = run(CopyCommand(), app, "/copy 9")
    assert "No reply 9" in out
    assert app.clipboard == "one"


def test_copy_all_includes_both_roles():
    app = FakeApp(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    run(CopyCommand(), app, "/copy all")
    assert "You: hi" in app.clipboard and "Dizel: hello" in app.clipboard


def test_copy_says_so_when_there_is_nothing_yet():
    app = FakeApp([{"role": "user", "content": "hi"}])
    out = run(CopyCommand(), app, "/copy")
    assert "Nothing to copy" in out
    assert app.clipboard is None


def test_verbosity_sets_the_value_and_saves_the_shared_key(monkeypatch):
    saved = {}

    class FakeConfig:
        @staticmethod
        def load():
            return {"token_budget": {"verbosity": "normal", "chat_budget": 150}}

        @staticmethod
        def save(settings):
            saved.update(settings)

    monkeypatch.setattr("dizel_core.config_manager.ConfigManager", FakeConfig)

    app = FakeApp()
    out = run(VerbosityCommand(), app, "/verbosity detailed")
    assert app.active_verbosity == "detailed"
    assert saved["token_budget"]["verbosity"] == "detailed"
    # The rest of the section must survive the write.
    assert saved["token_budget"]["chat_budget"] == 150
    assert "detailed" in out


def test_verbosity_rejects_unknown_values():
    app = FakeApp()
    out = run(VerbosityCommand(), app, "/verbosity verbose")
    assert "Unknown length" in out
    assert app.active_verbosity == "normal"


def test_find_reports_the_first_match_and_repeats():
    app = FakeApp()

    def find(query):
        app.panel.calls.append(query)
        return {"query": "docker", "index": 1, "turns": 2, "matches": 3}

    app.panel.find = find
    out = run(FindCommand(), app, "/find docker")
    assert "turn 1 of 2" in out
    assert "3 matches" in out

    # A bare /find repeats the previous query rather than clearing it.
    run(FindCommand(), app, "/find")
    assert app.panel.calls == ["docker", ""]


def test_find_reports_no_match_plainly():
    app = FakeApp()
    app.panel.find = lambda query: {"query": "zzz", "index": 0, "turns": 0, "matches": 0}
    out = run(FindCommand(), app, "/find zzz")
    assert "No match" in out


def test_open_refuses_a_chat_that_was_never_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.tui.commands.builtins.open_gui._repo_root", lambda: tmp_path
    )
    app = FakeApp(session_id="")
    out = run(OpenInGuiCommand(), app, "/open")
    assert "no saved id" in out


def test_session_tags_round_trip(tmp_path, monkeypatch):
    from dizel_core import history_manager

    monkeypatch.setattr(history_manager, "HISTORY_DIR", str(tmp_path))
    session_id = history_manager.save_session(
        [{"role": "user", "content": "hi"}], title="Tag me"
    )

    stored = history_manager.set_session_tags(session_id, ["#Design", "design", " ", "gui"])
    assert stored == ["design", "gui"]

    path = next(tmp_path.glob(f"{session_id}*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tags"] == ["design", "gui"]

    assert history_manager.set_session_tags("missing", ["x"]) is None
