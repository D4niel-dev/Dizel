"""Shared write-trust policy: tiers, root bounds, and the never-auto-delete rule."""

from __future__ import annotations

import pytest

from dizel_core import config_manager, tool_trust
from dizel_core.config_manager import ConfigManager
from dizel_core.tools import FileChange


@pytest.fixture()
def tmp_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    return tmp_path


def _write(path: str = "notes.txt") -> FileChange:
    return FileChange("write_file", path, "create", "+one", 1, 0)


def _edit(path: str = "notes.txt") -> FileChange:
    return FileChange("edit_file", path, "edit", "-old\n+new", 1, 1)


def _delete(path: str = "notes.txt") -> FileChange:
    return FileChange("delete_file", path, "delete", "-one", 0, 1)


def test_default_is_manual(tmp_path):
    assert tool_trust.should_auto_approve(
        _write(), "s1", session_grants=set(), persistent=False, root=tmp_path
    ) is False


def test_persistent_auto_approves_writes_and_edits(tmp_path):
    for change in (_write(), _edit()):
        assert tool_trust.should_auto_approve(
            change, "s1", session_grants=set(), persistent=True, root=tmp_path
        ) is True


def test_session_grant_is_per_session(tmp_path):
    grants = {"s1"}
    assert tool_trust.should_auto_approve(
        _write(), "s1", session_grants=grants, persistent=False, root=tmp_path
    ) is True
    assert tool_trust.should_auto_approve(
        _write(), "s2", session_grants=grants, persistent=False, root=tmp_path
    ) is False
    assert tool_trust.should_auto_approve(
        _write(), "", session_grants=grants, persistent=False, root=tmp_path
    ) is False


def test_delete_and_read_tools_never_auto_approve(tmp_path):
    # The user mandate: no setting, no trust tier may auto-approve a delete.
    for change in (_delete(), FileChange("read_file", "notes.txt", "read", "", 0, 0)):
        assert tool_trust.should_auto_approve(
            change, "s1", session_grants={"s1"}, persistent=True, root=tmp_path
        ) is False


def test_out_of_root_never_auto_approves(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    for path in ("../outside.txt", str(outside)):
        assert tool_trust.should_auto_approve(
            _write(path), "s1", session_grants={"s1"}, persistent=True, root=tmp_path
        ) is False
    assert tool_trust.should_auto_approve(
        _write(str(tmp_path / "notes.txt")), "s1",
        session_grants=set(), persistent=True, root=tmp_path,
    ) is True


def test_approved_for():
    assert tool_trust.approved_for(tool_trust.ALLOW_ONCE) is True
    assert tool_trust.approved_for(tool_trust.ALWAYS_SESSION) is True
    assert tool_trust.approved_for(tool_trust.ALWAYS_PERSISTENT) is True
    assert tool_trust.approved_for(tool_trust.DENY) is False
    assert tool_trust.approved_for("bogus") is False


def test_record_and_revoke_session_grant():
    grants: set[str] = set()
    tool_trust.record_session_grant(grants, "s1")
    tool_trust.record_session_grant(grants, "")  # empty id is never recorded
    assert grants == {"s1"}
    tool_trust.revoke_session_grant(grants, "s1")
    tool_trust.revoke_session_grant(grants, "missing")  # no-op
    assert grants == set()


def test_persistent_round_trip(tmp_config):
    assert tool_trust.persistent_enabled() is False
    tool_trust.set_persistent(True)
    assert tool_trust.persistent_enabled() is True
    cfg = ConfigManager.load()
    assert cfg["tools"]["always_allow_writes"] is True
    tool_trust.set_persistent(False)
    assert tool_trust.persistent_enabled() is False


def test_persistent_read_fails_closed(tmp_config, monkeypatch):
    def boom(cls):
        raise OSError("disk gone")

    monkeypatch.setattr(ConfigManager, "load", classmethod(boom))
    assert tool_trust.persistent_enabled() is False


def test_default_config_has_tools_key():
    assert ConfigManager._DEFAULTS["tools"]["always_allow_writes"] is False


def test_load_without_file_does_not_share_nested_defaults(tmp_config):
    # Regression: the no-file load path returned a shallow copy, so mutating
    # a nested section (as set_persistent does) polluted the class defaults.
    cfg = ConfigManager.load()
    cfg.setdefault("tools", {})["always_allow_writes"] = True
    assert ConfigManager._DEFAULTS["tools"]["always_allow_writes"] is False


def test_set_persistent_normalizes_a_non_dict_tools_section(tmp_config):
    (tmp_config / "settings.json").write_text('{"tools": "yes"}', encoding="utf-8")
    tool_trust.set_persistent(True)  # must not raise on a malformed section
    assert tool_trust.persistent_enabled() is True


def test_corrupt_settings_file_fails_closed(tmp_config):
    (tmp_config / "settings.json").write_text("{not valid json", encoding="utf-8")
    assert tool_trust.persistent_enabled() is False
