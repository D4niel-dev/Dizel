import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NPM = ROOT / "npm"
BIN = NPM / "bin" / "dizel.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_node(*args, env=None):
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [NODE, str(BIN), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=merged,
    )


def test_package_manifest_is_publishable():
    package = json.loads((NPM / "package.json").read_text(encoding="utf-8"))
    assert package["name"] == "dizel"
    assert package["bin"] == {"dizel": "bin/dizel.js"}
    assert package["license"] == "MIT"
    assert package["engines"]["node"] == ">=18"
    assert package["publishConfig"]["access"] == "public"
    assert "dependencies" not in package


def test_help_and_version():
    help_result = run_node("--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "dizel install" in help_result.stdout
    assert "dizel tui" in help_result.stdout
    assert "dizel doctor" in help_result.stdout

    version_result = run_node("version")
    assert version_result.returncode == 0, version_result.stderr
    package = json.loads((NPM / "package.json").read_text(encoding="utf-8"))
    assert version_result.stdout.strip() == f"dizel {package['version']}"


def test_doctor_json_is_offline_and_structured(tmp_path):
    home = tmp_path / "home"
    result = run_node("doctor", "--json", env={"DIZEL_HOME": str(home)})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["launcher"]
    assert report["node"]
    assert report["platform"]
    assert "python" in report
    assert "tui" in report
    assert "gui" in report
    assert "models" in report


def test_dry_run_prints_verified_release_urls(tmp_path):
    manifest = {
        "version": "9.9.9",
        "tag": "v9.9.9",
        "tui": {
            "asset": "Dizel-TUI-9.9.9.zip",
            "sha256": "a" * 64,
            "size": 123,
        },
        "gui": {
            "asset": "Dizel-Setup-9.9.9.exe",
            "sha256": "b" * 64,
            "size": 456,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    home = tmp_path / "home"

    result = run_node(
        "install",
        "--tui",
        "--dry-run",
        "--json",
        env={
            "DIZEL_HOME": str(home),
            "DIZEL_RELEASE_MANIFEST": str(manifest_path),
        },
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["version"] == "9.9.9"
    assert payload["tag"] == "v9.9.9"
    assert payload["tui"]["asset"] == "Dizel-TUI-9.9.9.zip"
    assert payload["tui"]["url"] == (
        "https://github.com/D4niel-dev/Dizel/releases/download/"
        "v9.9.9/Dizel-TUI-9.9.9.zip"
    )
    assert payload["gui"]["url"].endswith("/Dizel-Setup-9.9.9.exe")
    assert not (home / "downloads").exists()
