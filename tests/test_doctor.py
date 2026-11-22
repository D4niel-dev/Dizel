"""Runtime doctor: fast, offline checks that never print secrets."""

from __future__ import annotations

import json

import pytest

from dizel_core import config_manager, doctor, history_manager


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(history_manager, "HISTORY_DIR", str(tmp_path / "history"))


def row_for(rows, check_id):
    return next(row for row in rows if row.id == check_id)


def write_settings(tmp_path, monkeypatch, data):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(settings))
    return settings


def test_python_and_core_dependencies_are_ok():
    rows = doctor.run_checks()
    assert row_for(rows, "python").status == "ok"
    assert row_for(rows, "core_deps").status == "ok"


def test_missing_core_dependency_fails(monkeypatch):
    monkeypatch.setattr(doctor, "CORE_MODULES", ("definitely_missing_dizel_module",))
    row = row_for(doctor.run_checks(), "core_deps")
    assert row.status == "fail"
    assert "definitely_missing_dizel_module" in row.detail


def test_malformed_settings_fail_without_printing_content(tmp_path, monkeypatch):
    secret = "sk-super-secret-value"
    settings = tmp_path / "settings.json"
    settings.write_text(f'{{"api_router": {{"api_key": "{secret}"', encoding="utf-8")
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(settings))

    rows = doctor.run_checks()
    row = row_for(rows, "settings")
    assert row.status == "fail"
    assert secret not in doctor.summary_text(rows)
    assert secret not in json.dumps(doctor.as_dict(rows))


def test_provider_key_presence_is_reported_not_printed(tmp_path, monkeypatch):
    secret = "sk-super-secret-value"
    write_settings(
        tmp_path,
        monkeypatch,
        {"api_router": {"provider": "openai", "api_key": secret}},
    )

    rows = doctor.run_checks()
    row = row_for(rows, "provider")
    assert row.status == "ok"
    assert "configured" in row.detail.lower()
    assert secret not in doctor.summary_text(rows)
    assert secret not in json.dumps(doctor.as_dict(rows))


def test_missing_provider_key_warns(tmp_path, monkeypatch):
    write_settings(
        tmp_path,
        monkeypatch,
        {"api_router": {"provider": "anthropic", "api_key": ""}},
    )
    row = row_for(doctor.run_checks(), "provider")
    assert row.status == "warn"
    assert "anthropic" in row.detail.lower()


def test_local_provider_is_ok(tmp_path, monkeypatch):
    write_settings(
        tmp_path,
        monkeypatch,
        {"api_router": {"provider": "local", "api_key": ""}},
    )
    row = row_for(doctor.run_checks(), "provider")
    assert row.status == "ok"


def test_checkpoint_history_reports_missing_files(tmp_path, monkeypatch):
    write_settings(
        tmp_path,
        monkeypatch,
        {"checkpoint_history": [str(tmp_path / "gone.gguf")]},
    )
    row = row_for(doctor.run_checks(), "checkpoints")
    assert row.status == "warn"
    assert "0 of 1" in row.detail


def test_missing_settings_are_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    row = row_for(doctor.run_checks(), "settings")
    assert row.status == "ok"


def test_missing_usage_state_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    row = row_for(doctor.run_checks(), "usage_state")
    assert row.status == "ok"


def test_config_dir_absent_is_reported_as_deferred(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_manager, "SETTINGS_FILE", str(tmp_path / "nested" / "settings.json")
    )
    row = row_for(doctor.run_checks(), "config_dir")
    assert row.status == "ok"
    assert "first save" in row.detail


def test_history_counts_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(history_manager, "HISTORY_DIR", str(tmp_path))
    (tmp_path / "s1_chat.json").write_text(
        json.dumps({"id": "s1", "messages": [{"role": "user", "content": "hi"}]}),
        encoding="utf-8",
    )
    row = row_for(doctor.run_checks(), "history")
    assert row.status == "ok"
    assert "1" in row.detail


def test_summary_and_dict_carry_statuses():
    rows = doctor.run_checks()
    text = doctor.summary_text(rows)
    assert "[ok]" in text
    payload = doctor.as_dict(rows)
    assert len(payload["checks"]) == len(rows)
    assert payload["counts"]["ok"] + payload["counts"]["warn"] + payload["counts"]["fail"] == len(rows)
    assert payload["status"] in {"ok", "warn", "fail"}


# ── The terminal surface ───────────────────────────────────────────────────


def test_tui_doctor_command_shows_the_report():
    import asyncio

    from inference.tui.commands.builtins.doctor import DoctorCommand
    from inference.tui.commands.parser import parse_command

    class App:
        shown = None

        def show_debug_output(self, title, body):
            self.shown = (title, body)

    app = App()
    result = asyncio.run(DoctorCommand().execute(app, parse_command("/doctor")))
    assert app.shown is not None
    assert app.shown[0] == "Runtime doctor"
    assert "Python" in app.shown[1]
    assert "[ok]" in app.shown[1]
    assert result == ""


def test_diagnostic_bundle_includes_the_doctor_report():
    from inference.tui.commands.builtins.debug import _bundle_text

    text = _bundle_text(FakeApp())
    assert "Runtime doctor" in text
    assert "Core packages" in text


class _Manager:
    is_ready = True
    effective_ctx_len = 2048
    model_info = {"device": "cpu"}


class _Bridge:
    manager = _Manager()


class _Sessions:
    current_messages = []


class _Usage:
    max_capacity = 2048
    usage = 120.0
    compaction_count = 0
    MAX_COMPACTIONS = 5
    is_exhausted = False

    def time_until_reset(self):
        return "0m"

    @property
    def percentage(self):
        return 6


class FakeApp:
    debug_mode = True
    generation_state = "IDLE"
    active_model = "Dizel 1.5"
    active_provider = "local"
    active_mode = "Fast"
    context_tokens = 120
    budget_tokens = 2048
    session_id = "s1"
    open_tabs = []
    update_ready = ""
    tools_revision = 0
    trust_revision = 0
    _tool_states = {}
    chat_bridge = _Bridge()
    session_bridge = _Sessions()

    def __init__(self):
        self.usage_manager = _Usage()
