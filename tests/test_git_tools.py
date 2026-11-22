"""Read-only git helpers for /review and /commit."""

from __future__ import annotations

import subprocess

import pytest

from dizel_core import git_tools, tools


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


def test_working_diff_includes_unstaged_changes(repo):
    (repo / "app.py").write_text("print('hello')\nprint('more')\n", encoding="utf-8")

    change = git_tools.working_diff()

    assert change["available"] is True
    assert "print('more')" in change["diff"]
    assert any(row["path"] == "app.py" for row in change["files"])


def test_staged_diff_excludes_unstaged_changes(repo):
    (repo / "app.py").write_text("print('staged')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    (repo / "app.py").write_text("print('staged')\nprint('unstaged')\n", encoding="utf-8")

    change = git_tools.staged_diff()

    assert "print('staged')" in change["diff"]
    assert "unstaged" not in change["diff"]


def test_commit_prompt_returns_a_message_request(repo):
    (repo / "app.py").write_text("print('staged')\n", encoding="utf-8")
    git(repo, "add", "app.py")

    prompt = git_tools.commit_prompt(git_tools.staged_diff())

    assert "commit message" in prompt.lower()
    assert "staged" in prompt.lower()


def test_review_prompt_is_about_the_diff(repo):
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

    prompt = git_tools.review_prompt(git_tools.working_diff())

    assert "review" in prompt.lower()
    assert "print('changed')" in prompt


def test_nothing_staged_is_explained(repo):
    change = git_tools.staged_diff()

    assert change["available"] is True
    assert "stage" in change["error"].lower()


def test_clean_tree_review_is_explained(repo):
    change = git_tools.working_diff()

    assert change["available"] is True
    assert "no working-tree changes" in change["error"].lower()


def test_diff_is_capped(repo, monkeypatch):
    monkeypatch.setattr(git_tools, "MAX_DIFF_CHARS", 200)
    (repo / "app.py").write_text("print('x')\n" * 100, encoding="utf-8")

    change = git_tools.working_diff()

    assert change["truncated"] is True
    assert "truncated" in change["diff"].lower()
    assert len(change["diff"]) < 400


def test_non_repo_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "plain"
    root.mkdir()
    monkeypatch.setattr(tools, "_project_root", lambda: root)

    change = git_tools.working_diff()

    assert change["available"] is False
    assert "not a git repository" in change["error"].lower()
