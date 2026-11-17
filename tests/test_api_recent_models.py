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
