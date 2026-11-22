"""Recently loaded checkpoints endpoint: newest first, with existence flags."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import dizel_core.config_manager as cm
from dizel_core import checkpoint_history


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_recent_models_lists_newest_first_with_existence(client, tmp_path):
    first = tmp_path / "dizel-1.5-sft-best.pt"
    first.write_bytes(b"1234")
    gone = tmp_path / "deleted-model.gguf"

    checkpoint_history.record(str(first))
    checkpoint_history.record(str(gone))

    response = client.get("/api/models/recent")
    assert response.status_code == 200, response.text
    items = response.json()

    assert [item["path"] for item in items] == [str(gone.resolve()), str(first.resolve())]
    assert items[0]["exists"] is False
    assert items[0]["kind"] == "gguf"
    assert items[1]["exists"] is True
    assert items[1]["size"] == 4
    assert items[1]["display_name"]


def test_recent_models_is_empty_without_history(client):
    response = client.get("/api/models/recent")
    assert response.status_code == 200
    assert response.json() == []


def _card():
    return {
        "version": 1,
        "suite": "core",
        "suite_version": 1,
        "total": 15,
        "passed": 12,
        "pass_rate": 0.8,
        "avg_score": 0.83,
        "tok_per_s": 41.2,
        "evaluated_at": "2026-11-18T12:00:00",
        "by_category": {"coding": 0.9},
    }


def test_recent_models_carry_their_scorecard(client, tmp_path, monkeypatch):
    from dizel_core import scorecards

    monkeypatch.setattr(scorecards, "SCORECARDS_DIR", str(tmp_path / "scorecards"))
    path = tmp_path / "Dizel-1.5-SFT.gguf"
    path.write_bytes(b"weights")
    checkpoint_history.record(str(path))
    scorecards.save(str(path), _card())

    items = client.get("/api/models/recent").json()

    assert items[0]["scorecard"]["passed"] == 12
    assert items[0]["scorecard"]["tok_per_s"] == 41.2


def test_scorecard_routes_read_stored_cards(client, tmp_path, monkeypatch):
    from dizel_core import scorecards

    monkeypatch.setattr(scorecards, "SCORECARDS_DIR", str(tmp_path / "scorecards"))
    path = tmp_path / "Dizel-1.5-SFT.gguf"
    path.write_bytes(b"weights")
    scorecards.save(str(path), _card())

    ok = client.get("/api/models/scorecard", params={"path": str(path)})
    assert ok.status_code == 200
    assert ok.json()["passed"] == 12

    missing = client.get("/api/models/scorecard", params={"path": str(tmp_path / "x.gguf")})
    assert missing.status_code == 404
