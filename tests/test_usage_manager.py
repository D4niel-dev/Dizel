"""Per-provider usage windows: independent counters, resets, and migration."""

from __future__ import annotations

import json
import time

import pytest

from dizel_core.usage_manager import UsageManager


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    import dizel_core.config_manager as config_manager

    monkeypatch.setattr(config_manager, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    return tmp_path


def _state_file(tmp_path):
    return tmp_path / "usage_state.json"


def _charge_words(usage: UsageManager, words: int) -> None:
    usage.charge_text(" ".join(["word"] * words))


def test_providers_keep_independent_counters(tmp_path):
    usage = UsageManager()
    usage.activate("openai", 10_000)
    _charge_words(usage, 100)
    assert usage.usage == pytest.approx(10.0)

    usage.activate("local", 1_024)
    assert usage.usage == 0.0
    _charge_words(usage, 50)
    assert usage.usage == pytest.approx(5.0)

    usage.activate("openai", 10_000)
    assert usage.usage == pytest.approx(10.0)


def test_counters_survive_a_reload(tmp_path):
    usage = UsageManager()
    usage.activate("anthropic", 200_000)
    _charge_words(usage, 100)
    usage.save_state()

    again = UsageManager()
    assert again.active_provider == "anthropic"
    assert again.usage == pytest.approx(10.0)


def test_expired_window_resets_only_the_active_provider(tmp_path):
    usage = UsageManager()
    usage.activate("openai", 1_000)
    _charge_words(usage, 100)
    usage.activate("local", 1_000)
    _charge_words(usage, 50)

    usage.activate("openai", 1_000)
    usage.expire_window()
    usage.check_reset()
    assert usage.usage == 0.0

    usage.activate("local", 1_000)
    assert usage.usage == pytest.approx(5.0)


def test_compactions_are_per_provider(tmp_path):
    usage = UsageManager()
    usage.activate("openai", 1_000)
    usage.run_compaction()
    assert usage.compaction_count == 1

    usage.activate("local", 1_000)
    assert usage.compaction_count == 0

    usage.activate("openai", 1_000)
    assert usage.compaction_count == 1


def test_set_capacity_applies_to_the_active_provider(tmp_path):
    usage = UsageManager()
    usage.activate("openai", 1_000)
    usage.set_capacity(5_000)
    assert usage.max_capacity == 5_000

    usage.activate("local", 2_000)
    assert usage.max_capacity == 2_000

    usage.activate("openai")
    assert usage.max_capacity == 5_000


def test_flat_state_migrates_into_the_local_bucket(tmp_path):
    legacy = {"usage": 42.0, "last_reset": time.time(), "compaction_count": 2}
    _state_file(tmp_path).write_text(json.dumps(legacy), encoding="utf-8")

    usage = UsageManager()
    assert usage.active_provider == "local"
    assert usage.usage == pytest.approx(42.0)
    assert usage.compaction_count == 2

    usage.save_state()
    saved = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
    assert saved["active"] == "local"
    assert saved["providers"]["local"]["usage"] == pytest.approx(42.0)
