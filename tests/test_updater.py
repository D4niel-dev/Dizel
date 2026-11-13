from __future__ import annotations

import hashlib
import http.server
import tempfile
import threading
import time
import unittest
import zipfile
from functools import partial
from pathlib import Path
from unittest import mock

from dizel_core import updater


class VersionTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(updater.normalize_version("v2.01.0"), "2.1.0")
        self.assertEqual(updater.normalize_version("2.0.0-rc.1"), "2.0.0-rc.1")

    def test_newer(self):
        self.assertTrue(updater.is_newer("2.1.0", "2.0.0"))
        self.assertFalse(updater.is_newer("2.0.0", "2.0.0"))
        self.assertFalse(updater.is_newer("2.0.0-rc.1", "2.0.0"))
        self.assertTrue(updater.is_newer("2.0.1", "2.0.0-rc.1"))
        self.assertTrue(updater.is_newer("2.0.0-rc.10", "2.0.0-rc.2"))


class ManifestTests(unittest.TestCase):
    def test_manifest_wins_over_release_asset_digest(self):
        release = {
            "tag_name": "v2.1.0",
            "name": "Dizel v2.1.0",
            "body": "Release notes",
            "html_url": "https://github.com/D4niel-dev/Dizel/releases/tag/v2.1.0",
            "published_at": "2026-11-07T00:00:00Z",
            "prerelease": False,
            "assets": [
                {
                    "name": "Dizel-Setup-2.1.0.exe",
                    "browser_download_url": "https://github.com/D4niel-dev/Dizel/releases/download/v2.1.0/Dizel-Setup-2.1.0.exe",
                    "size": 1000,
                    "digest": "sha256:release-side",
                },
                {
                    "name": "Dizel-TUI-2.1.0.zip",
                    "browser_download_url": "https://github.com/D4niel-dev/Dizel/releases/download/v2.1.0/Dizel-TUI-2.1.0.zip",
                    "size": 500,
                    "digest": "sha256:release-tui",
                },
                {
                    "name": "dizel-update.json",
                    "browser_download_url": "https://github.com/D4niel-dev/Dizel/releases/download/v2.1.0/dizel-update.json",
                    "size": 100,
                },
            ],
        }
        manifest = {
            "version": "2.1.0",
            "gui": {
                "asset": "Dizel-Setup-2.1.0.exe",
                "sha256": "a" * 64,
                "size": 1000,
            },
            "tui": {
                "asset": "Dizel-TUI-2.1.0.zip",
                "sha256": "b" * 64,
                "size": 500,
            },
        }

        def fake_get(url, timeout=15.0):
            return release if "api.github.com" in url else manifest

        with mock.patch.object(updater, "_get_json", side_effect=fake_get):
            info = updater.fetch_latest_release()

        self.assertIsNotNone(info)
        self.assertEqual(info.version, "2.1.0")
        self.assertEqual(info.gui.sha256, "a" * 64)
        self.assertEqual(info.tui.sha256, "b" * 64)
        self.assertEqual(info.tui.size, 500)


class ArchiveTests(unittest.TestCase):
    def _zip(self, path: Path, files: dict[str, str]) -> None:
        with zipfile.ZipFile(path, "w") as bundle:
            for name, text in files.items():
                bundle.writestr(name, text)

    def test_stage_and_apply(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            (root / "dizel_core").mkdir(parents=True)
            (root / "inference" / "tui").mkdir(parents=True)
            (root / "data" / "history").mkdir(parents=True)
            (root / "dizel_core" / "updater.py").write_text("old updater", encoding="utf-8")
            (root / "inference" / "tui" / "main.py").write_text("old main", encoding="utf-8")
            (root / "dizel.cmd").write_text("old launcher", encoding="utf-8")
            (root / "data" / "history" / "keep.json").write_text("keep me", encoding="utf-8")

            archive = Path(temp) / "Dizel-TUI-9.9.9.zip"
            self._zip(
                archive,
                {
                    "dizel_core/updater.py": "new updater",
                    "inference/tui/main.py": "new main",
                    "inference/__init__.py": "",
                    "config.py": "new config",
                    "dizel.cmd": "new launcher",
                    "data/history/keep.json": "overwrite attempt",
                },
            )
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            info = updater.UpdateInfo(
                version="9.9.9",
                tag="v9.9.9",
                name="Dizel 9.9.9",
                notes="",
                url="https://github.com/D4niel-dev/Dizel/releases/tag/v9.9.9",
                tui=updater.UpdateAsset(
                    name=archive.name,
                    url="https://github.com/D4niel-dev/Dizel/releases/download/v9.9.9/x.zip",
                    sha256=sha,
                ),
            )

            pending = updater.stage_tui_update(archive, info, root=root)
            self.assertTrue(pending.is_file())
            result = updater.apply_pending_update(root=root)

            self.assertEqual(result["applied"], "9.9.9")
            self.assertEqual((root / "dizel_core" / "updater.py").read_text(), "new updater")
            self.assertEqual((root / "inference" / "tui" / "main.py").read_text(), "new main")
            self.assertEqual((root / "config.py").read_text(), "new config")
            self.assertEqual((root / "dizel.cmd").read_text(), "old launcher")
            self.assertEqual(
                (root / "data" / "history" / "keep.json").read_text(), "keep me"
            )
            self.assertFalse(updater._pending_path(root).exists())

    def test_rejects_zip_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "evil.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "nope")
            with self.assertRaises(updater.UpdateError):
                updater._extract_zip(archive, Path(temp) / "out")
            self.assertFalse((Path(temp) / "escape.txt").exists())


class StateTests(unittest.TestCase):
    def test_auto_check_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertTrue(updater.auto_check_enabled(root))
            self.assertTrue(updater.should_auto_check(root, interval=0))
            updater.set_auto_check(False, root=root)
            self.assertFalse(updater.auto_check_enabled(root))
            self.assertFalse(updater.should_auto_check(root, interval=0))
            updater.set_auto_check(True, root=root)
            updater.write_state(root, last_check=time.time())
            self.assertFalse(updater.should_auto_check(root, interval=3600))
            updater.write_state(root, last_check=0.0)
            self.assertTrue(updater.should_auto_check(root, interval=3600))

    def test_stage_rejects_unsafe_version(self):
        info = updater.UpdateInfo(
            version="../../escape",
            tag="v9.9.9",
            name="Dizel 9.9.9",
            notes="",
            url="https://github.com/D4niel-dev/Dizel/releases/tag/v9.9.9",
            tui=updater.UpdateAsset(
                name="Dizel-TUI-9.9.9.zip",
                url="https://github.com/D4niel-dev/Dizel/releases/download/v9.9.9/Dizel-TUI-9.9.9.zip",
                sha256="a" * 64,
            ),
        )
        with self.assertRaises(updater.UpdateError):
            updater.stage_tui_update("missing.zip", info)

    def test_tui_download_requires_digest(self):
        info = updater.UpdateInfo(
            version="9.9.9",
            tag="v9.9.9",
            name="Dizel 9.9.9",
            notes="",
            url="https://github.com/D4niel-dev/Dizel/releases/tag/v9.9.9",
            tui=updater.UpdateAsset(
                name="Dizel-TUI-9.9.9.zip",
                url="https://github.com/D4niel-dev/Dizel/releases/download/v9.9.9/Dizel-TUI-9.9.9.zip",
            ),
        )
        with self.assertRaises(updater.UpdateError):
            updater.download_tui_update(info)

    def test_tui_download_rejects_unsafe_asset_name(self):
        info = updater.UpdateInfo(
            version="9.9.9",
            tag="v9.9.9",
            name="Dizel 9.9.9",
            notes="",
            url="https://github.com/D4niel-dev/Dizel/releases/tag/v9.9.9",
            tui=updater.UpdateAsset(
                name="../../escape.zip",
                url="https://github.com/D4niel-dev/Dizel/releases/download/v9.9.9/escape.zip",
                sha256="a" * 64,
            ),
        )
        with self.assertRaises(updater.UpdateError):
            updater.download_tui_update(info)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.payload = b"hello update"
        (self.folder / "file.bin").write_bytes(self.payload)
        handler = partial(QuietHandler, directory=self.folder)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/file.bin"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def test_download_and_checksum(self):
        expected = hashlib.sha256(self.payload).hexdigest()
        progress = []
        with mock.patch.object(updater, "_validate_url", side_effect=lambda url: url):
            target = updater.download_asset(
                updater.UpdateAsset(
                    name="file.bin",
                    url=self.url,
                    size=len(self.payload),
                    sha256=expected,
                ),
                self.folder / "download.bin",
                progress=lambda received, total: progress.append((received, total)),
            )
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(progress[-1][0], len(self.payload))

    def test_bad_checksum_discards_file(self):
        with mock.patch.object(updater, "_validate_url", side_effect=lambda url: url):
            with self.assertRaises(updater.UpdateError):
                updater.download_asset(
                    updater.UpdateAsset(name="file.bin", url=self.url, sha256="0" * 64),
                    self.folder / "download.bin",
                )
        self.assertFalse((self.folder / "download.bin").exists())


if __name__ == "__main__":
    unittest.main()
