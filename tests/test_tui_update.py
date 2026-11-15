from __future__ import annotations

import unittest
from unittest import mock

from inference.tui import main as tui_main
from inference.tui.commands.builtins.update_cmd import CheckUpdateCommand, UpdateCommand
from inference.tui.commands.parser import parse_command


class FakeApp:
    def __init__(self) -> None:
        self.restart_requested = False
        self.update_ready = ""
        self.timers = []
        self.notices = []

    def call_from_thread(self, callback, *args) -> None:
        callback(*args)

    def _log_system(self, message: str, level: str = "info") -> None:
        self.notices.append((message, level))

    def request_update_restart(self, delay: float = 0.6) -> None:
        self.restart_requested = True
        self.timers.append(delay)

    def exit(self) -> None:
        pass


class UpdateCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_update_reports_current_release(self):
        from dizel_core import updater

        app = FakeApp()
        result = updater.CheckResult(current="2.0.0", available=False, info=None)
        with mock.patch.object(updater, "check_for_update", return_value=result):
            message = await CheckUpdateCommand().execute(app, parse_command("/check-update"))
        self.assertIn("up to date", message)

    async def test_check_update_points_at_install(self):
        from dizel_core import updater

        app = FakeApp()
        info = updater.UpdateInfo(
            version="2.1.0",
            tag="v2.1.0",
            name="Dizel 2.1.0",
            notes="",
            url="https://github.com/D4niel-dev/Dizel/releases/tag/v2.1.0",
            tui=updater.UpdateAsset(
                name="Dizel-TUI-2.1.0.zip",
                url="https://github.com/D4niel-dev/Dizel/releases/download/v2.1.0/x.zip",
                sha256="b" * 64,
            ),
        )
        result = updater.CheckResult(current="2.0.0", available=True, info=info)
        with mock.patch.object(updater, "check_for_update", return_value=result):
            message = await CheckUpdateCommand().execute(app, parse_command("/check-update"))
        self.assertIn("/update install", message)
        self.assertEqual(app.update_ready, "2.1.0")

    async def test_install_restarts_with_a_staged_update(self):
        from dizel_core import updater

        app = FakeApp()
        with mock.patch.object(updater, "pending_update", return_value={"version": "2.1.0"}):
            message = await UpdateCommand().execute(app, parse_command("/update install"))

        self.assertTrue(app.restart_requested)
        self.assertEqual(len(app.timers), 1)
        self.assertIn("2.1.0", message)
        self.assertTrue(any("Restarting Dizel" in note for note, _ in app.notices))

    async def test_install_downloads_then_restarts(self):
        from dizel_core import updater

        app = FakeApp()
        with (
            mock.patch.object(
                updater, "pending_update", side_effect=[None, {"version": "2.1.0"}]
            ),
            mock.patch(
                "inference.tui.commands.builtins.update_cmd._check",
                new=mock.AsyncMock(return_value="downloaded"),
            ),
        ):
            message = await UpdateCommand().execute(app, parse_command("/update install"))

        self.assertTrue(app.restart_requested)
        self.assertIn("2.1.0", message)


class UpdateBootstrapTests(unittest.TestCase):
    def test_apply_staged_update_noop_without_pending(self):
        with mock.patch("dizel_core.updater.pending_update", return_value=None):
            self.assertIsNone(tui_main._apply_staged_update("C:/dizel"))

    def test_apply_staged_update_returns_result(self):
        result = {"applied": "2.1.0", "previous": "2.0.0"}
        with (
            mock.patch("dizel_core.updater.pending_update", return_value={"version": "2.1.0"}),
            mock.patch("dizel_core.updater.auto_apply_allowed", return_value=True),
            mock.patch("dizel_core.updater.apply_pending_update", return_value=result),
            mock.patch.object(tui_main, "_say"),
        ):
            self.assertEqual(tui_main._apply_staged_update("C:/dizel"), result)

    def test_apply_staged_update_skips_a_dirty_checkout(self):
        with (
            mock.patch("dizel_core.updater.pending_update", return_value={"version": "2.1.0"}),
            mock.patch("dizel_core.updater.auto_apply_allowed", return_value=False),
            mock.patch("dizel_core.updater.apply_pending_update") as apply,
            mock.patch.object(tui_main, "_say"),
        ):
            self.assertIsNone(tui_main._apply_staged_update("C:/dizel"))
        apply.assert_not_called()

    def test_forced_apply_ignores_a_dirty_checkout(self):
        result = {"applied": "2.1.0", "previous": "2.0.0"}
        with (
            mock.patch("dizel_core.updater.pending_update", return_value={"version": "2.1.0"}),
            mock.patch("dizel_core.updater.auto_apply_allowed", return_value=False),
            mock.patch("dizel_core.updater.apply_pending_update", return_value=result),
            mock.patch.object(tui_main, "_say"),
        ):
            self.assertEqual(
                tui_main._apply_staged_update("C:/dizel", force=True), result
            )

    def test_relaunch_uses_the_same_python_and_arguments(self):
        process = mock.Mock()
        process.wait.return_value = 7
        with (
            mock.patch.object(tui_main.sys, "executable", "C:/Python/python.exe"),
            mock.patch.object(tui_main.subprocess, "Popen", return_value=process) as popen,
            mock.patch.object(tui_main, "_say"),
        ):
            code = tui_main._relaunch_tui(["--resume"])

        self.assertEqual(code, 7)
        self.assertEqual(
            popen.call_args.args[0],
            ["C:/Python/python.exe", "-m", "inference.tui.main", "--resume"],
        )

    def test_relaunch_survives_ctrl_c_in_the_supervisor(self):
        process = mock.Mock()
        process.wait.side_effect = [KeyboardInterrupt, 7]
        with (
            mock.patch.object(tui_main.sys, "executable", "C:/Python/python.exe"),
            mock.patch.object(tui_main.subprocess, "Popen", return_value=process),
            mock.patch.object(tui_main, "_say"),
        ):
            self.assertEqual(tui_main._relaunch_tui([]), 7)


if __name__ == "__main__":
    unittest.main()
