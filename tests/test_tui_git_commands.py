"""The terminal's review/commit commands hand the diff to one model turn."""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from dizel_core import tools
from inference.tui.commands.builtins.git_commands import (
    DraftCommitCommand,
    ReviewCommand,
)
from inference.tui.commands.parser import parse_command


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(root, "add", "app.py")
    git(root, "commit", "-q", "-m", "initial")
    monkeypatch.setattr(tools, "_project_root", lambda: root)
    return root


class FakeApp:
    sent = None

    def send_composed(self, display, content):
        self.sent = (display, content)
        return ""


def test_review_sends_the_working_diff(repo):
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    app = FakeApp()

    result = asyncio.run(ReviewCommand().execute(app, parse_command("/review")))

    assert result == ""
    assert app.sent is not None
    display, content = app.sent
    assert "Review the working-tree diff" in display
    assert "print('changed')" in content


def test_commit_uses_the_staged_diff(repo):
    (repo / "app.py").write_text("print('staged')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    (repo / "app.py").write_text("print('staged')\nprint('unstaged')\n", encoding="utf-8")
    app = FakeApp()

    result = asyncio.run(DraftCommitCommand().execute(app, parse_command("/commit")))

    assert result == ""
    display, content = app.sent
    assert "Draft a commit message" in display
    assert "commit message" in content.lower()
    assert "unstaged" not in content


def test_review_on_a_clean_tree_does_not_send(repo):
    app = FakeApp()

    result = asyncio.run(ReviewCommand().execute(app, parse_command("/review")))

    assert "no working-tree changes" in result.lower()
    assert app.sent is None


def test_commit_with_nothing_staged_does_not_send(repo):
    (repo / "app.py").write_text("print('unstaged')\n", encoding="utf-8")
    app = FakeApp()

    result = asyncio.run(DraftCommitCommand().execute(app, parse_command("/commit")))

    assert "stage" in result.lower()
    assert app.sent is None
