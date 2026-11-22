"""The terminal shows stored scorecards and points at the ways to run one."""

from __future__ import annotations

import asyncio
from unittest import mock

from dizel_core import scorecards
from inference.tui.commands.builtins.scorecard import ScorecardCommand
from inference.tui.commands.parser import parse_command


class _Manager:
    checkpoint_path = "C:/models/card.gguf"


class _Bridge:
    manager = _Manager()


class App:
    chat_bridge = _Bridge()
    shown = None

    def show_debug_output(self, title, body):
        self.shown = (title, body)


def card():
    return {
        "version": 1,
        "suite": "core",
        "suite_version": 1,
        "name": "card.gguf",
        "checkpoint": "card.gguf",
        "evaluated_at": "2026-11-18T12:00:00",
        "total": 15,
        "passed": 12,
        "pass_rate": 0.8,
        "avg_score": 0.83,
        "tok_per_s": 41.2,
        "by_category": {"reasoning": 0.75, "coding": 1.0},
        "cases": [],
    }


def test_shows_the_stored_card():
    app = App()
    with mock.patch.object(scorecards, "load", return_value=card()):
        result = asyncio.run(ScorecardCommand().execute(app, parse_command("/scorecard")))

    assert result == ""
    assert app.shown[0] == "Scorecard"
    assert "12 / 15" in app.shown[1]
    assert "reasoning" in app.shown[1].lower()


def test_without_a_checkpoint_explains():
    class Manager:
        checkpoint_path = ""

    class Bridge:
        manager = Manager()

    class Bare:
        chat_bridge = Bridge()

    result = asyncio.run(ScorecardCommand().execute(Bare(), parse_command("/scorecard")))

    assert "load one" in result.lower() or "load a" in result.lower()


def test_without_a_card_points_at_the_runners():
    class Bare(App):
        pass

    app = Bare()
    with mock.patch.object(scorecards, "load", return_value=None):
        result = asyncio.run(ScorecardCommand().execute(app, parse_command("/scorecard")))

    assert "scripts/run_scorecard.py" in result
    assert "desktop" in result.lower()


def test_run_points_at_desktop_or_cli():
    app = App()
    result = asyncio.run(ScorecardCommand().execute(app, parse_command("/scorecard run")))

    assert "scripts/run_scorecard.py" in result
    assert "desktop" in result.lower()
