"""Usage and cost: provider windows with clearly-labelled estimates."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from dizel_core import config_manager, provider_pricing, usage_manager, usage_report
from inference.tui.commands.builtins.cost import CostCommand
from inference.tui.commands.parser import parse_command


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, "SETTINGS_FILE", str(tmp_path / "settings.json"))


def test_pricing_math_and_unknown_providers():
    assert provider_pricing.estimate_cost("openai", 1_000_000) == pytest.approx(6.25)
    assert provider_pricing.estimate_cost("local", 1_000_000) == 0.0
    assert provider_pricing.estimate_cost("ollama", 1_000_000) == 0.0
    assert provider_pricing.estimate_cost("some-new-provider", 1_000_000) is None
    assert "estimate" in provider_pricing.rate_label("openai").lower()
    assert provider_pricing.rate_label("local").lower() == "free"


def test_snapshot_has_every_provider_bucket():
    usage = usage_manager.UsageManager()
    usage.activate("openai", 128000)
    usage.charge_text(" ".join(["word"] * 100))
    usage.activate("anthropic", 200000)

    report = usage_report.snapshot(usage)

    rows = {row["provider"]: row for row in report["providers"]}
    assert set(rows) >= {"local", "openai", "anthropic"}
    assert rows["openai"]["used"] > 0
    assert rows["openai"]["estimated_cost"] > 0
    assert rows["anthropic"]["active"] is True
    assert rows["local"]["estimated_cost"] == 0.0
    assert report["window_hours"] == 12


def test_report_text_labels_estimates_and_local_free():
    usage = usage_manager.UsageManager()
    usage.activate("openai", 128000)
    usage.charge_text(" ".join(["word"] * 100))

    text = usage_report.report_text(usage_report.snapshot(usage))

    assert "estimate" in text.lower()
    assert "openai" in text.lower()
    assert "free" in text.lower()


class FakeApp:
    shown = None

    def __init__(self):
        self.usage_manager = usage_manager.UsageManager()

    def show_debug_output(self, title, body):
        self.shown = (title, body)


def test_tui_cost_command_renders_the_report():
    app = FakeApp()
    result = asyncio.run(CostCommand().execute(app, parse_command("/cost")))

    assert result == ""
    assert app.shown[0] == "Usage & cost"
    assert "estimate" in app.shown[1].lower()


def test_usage_route_returns_the_report():
    from api.routes.usage import usage_report_route

    usage = usage_manager.UsageManager()
    usage.activate("local", 2048)
    usage.charge_text("hello there")

    with mock.patch(
        "api.services.chat_service.get_usage_manager", return_value=usage
    ):
        payload = usage_report_route()

    assert payload["providers"]
    assert payload["window_hours"] == 12
