"""Scorecards: local eval results that travel with a checkpoint, not a promise."""

from __future__ import annotations

import json
import os

import pytest

from dizel_core import scorecards
from evaluation import suite


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecards, "SCORECARDS_DIR", str(tmp_path / "scorecards"))


def make_checkpoint(tmp_path, name="Dizel-1.5-SFT.gguf", payload=b"weights"):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def fake_card(**overrides):
    card = {
        "version": 1,
        "suite": "core",
        "suite_version": 1,
        "checkpoint": "Dizel-1.5-SFT.gguf",
        "evaluated_at": "2026-11-18T12:00:00",
        "total": 15,
        "passed": 12,
        "pass_rate": 0.8,
        "avg_score": 0.83,
        "by_category": {"reasoning": 0.75, "coding": 1.0},
        "tok_per_s": 41.2,
        "cases": [
            {"id": "g1", "category": "reasoning", "passed": True, "score": 1.0, "duration_ms": 900}
        ],
    }
    card.update(overrides)
    return card


def test_save_and_load_round_trip(tmp_path):
    path = make_checkpoint(tmp_path)

    saved = scorecards.save(path, fake_card())
    loaded = scorecards.load(path)

    assert loaded is not None
    assert loaded["passed"] == 12 and loaded["total"] == 15
    assert loaded["path"] == os.path.abspath(path)
    assert loaded["size"] == len(b"weights")
    assert saved["fingerprint"] == loaded["fingerprint"]


def test_a_changed_checkpoint_invalidates_the_card(tmp_path):
    path = make_checkpoint(tmp_path)
    scorecards.save(path, fake_card())

    with open(path, "ab") as handle:
        handle.write(b"more")

    assert scorecards.load(path) is None


def test_missing_checkpoint_has_no_card(tmp_path):
    assert scorecards.load(str(tmp_path / "nope.gguf")) is None


def test_summary_card_has_the_badge_fields(tmp_path):
    path = make_checkpoint(tmp_path)
    saved = scorecards.save(path, fake_card())

    summary = scorecards.summary(saved)

    assert summary["passed"] == 12
    assert summary["total"] == 15
    assert summary["pass_rate"] == 0.8
    assert summary["tok_per_s"] == 41.2
    assert summary["evaluated_at"] == "2026-11-18T12:00:00"


def test_report_names_counts_categories_and_speed(tmp_path):
    path = make_checkpoint(tmp_path)
    saved = scorecards.save(path, fake_card())

    text = scorecards.report(saved)

    assert "12 / 15" in text
    assert "reasoning" in text.lower()
    assert "41.2" in text


def test_run_core_suite_scores_a_correct_handler(tmp_path):
    loader = suite.EvalRunner()
    cases = {case.prompt: case.expected for case in loader.load_cases()}

    card = suite.run_core_suite(lambda prompt: cases[prompt], checkpoint="fake.gguf")

    assert card["total"] == 15
    assert card["passed"] == 15
    assert card["pass_rate"] == 1.0
    assert card["suite"] == "core"
    assert len(card["cases"]) == 15


def test_run_core_suite_scores_a_wrong_handler(tmp_path):
    card = suite.run_core_suite(lambda prompt: "definitely not the answer")

    assert card["total"] == 15
    assert card["pass_rate"] < 0.5


def test_core_suite_records_timing_and_speed():
    loader = suite.EvalRunner()
    cases = {case.prompt: case.expected for case in loader.load_cases()}

    card = suite.run_core_suite(lambda prompt: cases[prompt])

    assert card["tok_per_s"] > 0
    assert all(case["duration_ms"] >= 0 for case in card["cases"])
