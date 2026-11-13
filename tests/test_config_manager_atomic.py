"""Settings writes swap a temp file into place instead of truncating in situ."""

import json

import dizel_core.config_manager as cm


def test_save_load_roundtrip_and_no_temp_file_remains(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "SETTINGS_FILE", str(tmp_path / "settings.json"))

    cm.ConfigManager.save({"checkpoint": "atomic.pt", "mcp": {"servers": []}})

    loaded = cm.ConfigManager.load()
    assert loaded["checkpoint"] == "atomic.pt"
    assert loaded["mcp"]["servers"] == []

    assert not (tmp_path / "settings.json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []
    # The final file is complete JSON, not a partial write.
    on_disk = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["checkpoint"] == "atomic.pt"
