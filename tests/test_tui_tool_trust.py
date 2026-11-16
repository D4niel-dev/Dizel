"""TUI write trust: four-decision gate, bridge auto-approval, app grant state."""

from __future__ import annotations

import asyncio
import copy
import threading
import unittest
from unittest import mock

from textual.widgets import Button, Checkbox

from dizel_core import tool_trust
from dizel_core.config_manager import ConfigManager
from dizel_core.tools import FileChange
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.modals.settings_modal import SettingsModal
from inference.tui.panels.modals.tool_approval import ToolApprovalModal


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


def _write_change():
    return FileChange("write_file", "notes.txt", "create", "+one\n+two", 2, 0)


def _delete_change():
    return FileChange("delete_file", "notes.txt", "delete", "-one", 0, 1)


class TuiTrustTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(DizelCMDApp, "_auto_load_model", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_update_check", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_mcp", lambda self: None),
            mock.patch.object(ConfigManager, "load", classmethod(lambda cls: _defaults())),
            mock.patch.object(ConfigManager, "save", classmethod(lambda cls, cfg: None)),
            mock.patch.object(UsageManager, "load_state", lambda self: None),
            mock.patch.object(UsageManager, "save_state", lambda self: None),
            mock.patch(
                "inference.tui.bridge.session_bridge.list_sessions", return_value=[]
            ),
            mock.patch(
                "inference.tui.bridge.session_bridge.save_session", return_value=""
            ),
            mock.patch(
                "inference.tui.bridge.session_bridge.load_session", return_value=None
            ),
        ]
        for patcher in self._patches:
            patcher.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for patcher in self._patches:
            patcher.stop()

    def _run(self, size, script):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    def test_write_gate_offers_four_decisions(self):
        async def script(app, pilot):
            results = []
            app.push_screen(ToolApprovalModal(_write_change()), results.append)
            await pilot.pause()
            buttons = [b.id for b in app.screen.query(Button)]
            hint = app.screen.query_one(".approval-trust-hint").render().plain
            await pilot.press("s")
            await pilot.pause()
            return buttons, hint, results

        buttons, hint, results = self._run((120, 36), script)
        self.assertEqual(
            buttons,
            ["approval-reject", "approval-once", "approval-session", "approval-persistent"],
        )
        self.assertIn("revoke", hint.lower())
        self.assertEqual(results, ["always_session"])

    def test_write_gate_persistent_key(self):
        async def script(app, pilot):
            results = []
            app.push_screen(ToolApprovalModal(_write_change()), results.append)
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            return results

        self.assertEqual(self._run((120, 36), script), ["always_persistent"])

    def test_delete_gate_ignores_trust_keys(self):
        async def script(app, pilot):
            results = []
            app.push_screen(ToolApprovalModal(_delete_change()), results.append)
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            still_open_s = isinstance(app.screen, ToolApprovalModal)
            await pilot.press("a")
            await pilot.pause()
            still_open_a = isinstance(app.screen, ToolApprovalModal)
            buttons = [b.id for b in app.screen.query(Button)]
            await pilot.press("y")
            await pilot.pause()
            return still_open_s, still_open_a, buttons, results

        open_s, open_a, buttons, results = self._run((120, 36), script)
        self.assertTrue(open_s)
        self.assertTrue(open_a)
        self.assertEqual(buttons, ["approval-reject", "approval-ok"])
        self.assertEqual(results, ["allow_once"])

    def test_bridge_auto_approves_persistent_writes_with_an_audit_note(self):
        async def script(app, pilot):
            with mock.patch.object(tool_trust, "persistent_enabled", return_value=True):
                with mock.patch.object(app, "_log_system", wraps=app._log_system) as log:
                    approved = app.chat_bridge._approve_tool(_write_change())
                    await pilot.pause()
                    await pilot.pause()
            gated = isinstance(app.screen, ToolApprovalModal)
            notes = [str(c.args[0]) for c in log.call_args_list]
            return approved, gated, notes

        approved, gated, notes = self._run((120, 36), script)
        self.assertTrue(approved)
        self.assertFalse(gated)
        self.assertTrue(any("Auto-approved write_file notes.txt (persistent)" in n for n in notes))

    def test_bridge_auto_approves_only_the_granted_session(self):
        async def script(app, pilot):
            app.chat_bridge._session_id = "s1"
            app._trusted_sessions.add("s1")
            approved = app.chat_bridge._approve_tool(_write_change())
            await pilot.pause()
            return approved

        self.assertTrue(self._run((120, 36), script))

    def test_bridge_delete_still_gates_with_every_trust_tier(self):
        async def script(app, pilot):
            app.chat_bridge._session_id = "s1"
            app._trusted_sessions.add("s1")
            result = {}
            with mock.patch.object(tool_trust, "persistent_enabled", return_value=True):
                thread = threading.Thread(
                    target=lambda: result.setdefault(
                        "approved", app.chat_bridge._approve_tool(_delete_change())
                    ),
                    daemon=True,
                )
                thread.start()
                for _ in range(30):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, ToolApprovalModal):
                        break
                gated = isinstance(app.screen, ToolApprovalModal)
                await pilot.press("y")
                await pilot.pause()
                thread.join(timeout=2)
            return gated, result.get("approved")

        gated, approved = self._run((120, 36), script)
        self.assertTrue(gated)
        self.assertTrue(approved)

    def test_bridge_session_decision_records_the_grant(self):
        async def script(app, pilot):
            app.chat_bridge._session_id = "s-tab"
            before = app.trust_revision
            result = {}
            thread = threading.Thread(
                target=lambda: result.setdefault(
                    "approved", app.chat_bridge._approve_tool(_write_change())
                ),
                daemon=True,
            )
            thread.start()
            for _ in range(30):
                await pilot.pause(0.05)
                if isinstance(app.screen, ToolApprovalModal):
                    break
            await pilot.press("s")
            await pilot.pause()
            thread.join(timeout=2)
            return (
                result.get("approved"),
                "s-tab" in app._trusted_sessions,
                app.trust_revision > before,
            )

        approved, granted, bumped = self._run((120, 36), script)
        self.assertTrue(approved)
        self.assertTrue(granted)
        self.assertTrue(bumped)

    def test_forget_tab_revokes_session_trust(self):
        async def script(app, pilot):
            app._trusted_sessions.add("tab-x")
            app.forget_tab("tab-x")
            await pilot.pause()
            return "tab-x" in app._trusted_sessions

        self.assertFalse(self._run((120, 36), script))

    def test_settings_saves_always_allow_writes(self):
        saved = {}

        async def script(app, pilot):
            with mock.patch.object(
                ConfigManager, "save", classmethod(lambda cls, cfg: saved.update(cfg))
            ):
                modal = SettingsModal()
                app.push_screen(modal)
                await pilot.pause()
                modal._show_section("tools")
                await pilot.pause()
                box = modal.query_one("#settings-tools-always-allow-writes", Checkbox)
                box.value = True
                await pilot.press("ctrl+s")
                await pilot.pause()
            return saved.get("tools", {}).get("always_allow_writes")

        self.assertIs(self._run((140, 44), script), True)

    def test_settings_revoke_clears_the_session_grant(self):
        async def script(app, pilot):
            app.session_id = "s1"
            app._trusted_sessions.add("s1")
            modal = SettingsModal()
            app.push_screen(modal)
            await pilot.pause()
            modal._show_section("tools")
            await pilot.pause()
            box = modal.query_one("#settings-trust-revoke-session", Checkbox)
            box.value = True
            await pilot.press("ctrl+s")
            await pilot.pause()
            return "s1" in app._trusted_sessions

        self.assertFalse(self._run((140, 44), script))

    def test_status_bar_shows_writes_auto_marker(self):
        async def script(app, pilot):
            with mock.patch.object(tool_trust, "persistent_enabled", return_value=True):
                app.trust_revision += 1
                await pilot.pause()
                on = app.query_one("#status-right").render().plain
            app.trust_revision += 1
            await pilot.pause()
            off = app.query_one("#status-right").render().plain
            return on, off

        on, off = self._run((140, 40), script)
        self.assertIn("writes:auto", on)
        self.assertNotIn("writes:auto", off)

    def test_status_bar_marker_for_session_grant(self):
        async def script(app, pilot):
            app.session_id = "s1"
            app._trusted_sessions.add("s1")
            app.trust_revision += 1
            await pilot.pause()
            return app.query_one("#status-right").render().plain

        self.assertIn("writes:auto", self._run((140, 40), script))

    def test_status_bar_repaints_do_not_read_config_from_disk(self):
        async def script(app, pilot):
            from inference.tui.panels.status_bar import StatusBar

            bar = app.query_one(StatusBar)
            calls = {"n": 0}

            def counting_load(cls):
                calls["n"] += 1
                return _defaults()

            with mock.patch.object(ConfigManager, "load", classmethod(counting_load)):
                bar.update_status()
                base = calls["n"]
                for _ in range(20):
                    bar.update_status()
                repaint_reads = calls["n"] - base
                for _ in range(16):
                    bar._tick()
                tick_reads = calls["n"] - base - repaint_reads
                app.trust_revision += 1
                await pilot.pause()
                revision_reads = calls["n"] - base - repaint_reads - tick_reads
            return repaint_reads, tick_reads, revision_reads

        repaint_reads, tick_reads, revision_reads = self._run((140, 40), script)
        self.assertEqual(repaint_reads, 0)
        self.assertGreaterEqual(tick_reads, 1)
        self.assertGreaterEqual(revision_reads, 1)


    def test_settings_save_survives_a_non_dict_tools_section(self):
        bad = _defaults()
        bad["tools"] = "yes"

        async def script(app, pilot):
            with mock.patch.object(
                ConfigManager, "load", classmethod(lambda cls: copy.deepcopy(bad))
            ):
                modal = SettingsModal()
                app.push_screen(modal)
                await pilot.pause()
                modal._show_section("tools")
                await pilot.pause()
                error = None
                try:
                    modal._save()
                except Exception as exc:  # noqa: BLE001 - regression probe
                    error = exc
                await pilot.pause()
                return error

        self.assertIsNone(self._run((140, 44), script))

    def test_bridge_persistent_decision_persists_and_bumps_revision(self):
        saved = {}

        async def script(app, pilot):
            app.chat_bridge._session_id = "s1"
            before = app.trust_revision
            result = {}
            with mock.patch.object(
                ConfigManager, "save", classmethod(lambda cls, cfg: saved.update(cfg))
            ):
                thread = threading.Thread(
                    target=lambda: result.setdefault(
                        "approved", app.chat_bridge._approve_tool(_write_change())
                    ),
                    daemon=True,
                )
                thread.start()
                for _ in range(30):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, ToolApprovalModal):
                        break
                await pilot.press("a")
                await pilot.pause()
                thread.join(timeout=2)
            return (
                result.get("approved"),
                saved.get("tools", {}).get("always_allow_writes"),
                app.trust_revision > before,
            )

        approved, persisted, bumped = self._run((120, 36), script)
        self.assertTrue(approved)
        self.assertIs(persisted, True)
        self.assertTrue(bumped)

    def test_bridge_session_auto_approval_notes_the_session_tier(self):
        async def script(app, pilot):
            app.chat_bridge._session_id = "s1"
            app._trusted_sessions.add("s1")
            with mock.patch.object(app, "_log_system", wraps=app._log_system) as log:
                approved = app.chat_bridge._approve_tool(_write_change())
                await pilot.pause()
                await pilot.pause()
            notes = [str(c.args[0]) for c in log.call_args_list]
            return approved, notes

        approved, notes = self._run((120, 36), script)
        self.assertTrue(approved)
        self.assertTrue(any("(session)" in n for n in notes))

    def test_settings_revoke_only_save_persists_no_trust_key(self):
        saved = {}

        async def script(app, pilot):
            app.session_id = "s1"
            app._trusted_sessions.add("s1")
            before = app.trust_revision
            with mock.patch.object(
                ConfigManager, "save", classmethod(lambda cls, cfg: saved.update(cfg))
            ):
                modal = SettingsModal()
                app.push_screen(modal)
                await pilot.pause()
                modal._show_section("tools")
                await pilot.pause()
                box = modal.query_one("#settings-trust-revoke-session", Checkbox)
                box.value = True
                await pilot.press("ctrl+s")
                await pilot.pause()
            return (
                any(k.startswith("@trust") for k in saved),
                saved.get("tools", {}).get("always_allow_writes"),
                app.trust_revision > before,
                "s1" in app._trusted_sessions,
            )

        persisted_trust, flag, bumped, still_granted = self._run((140, 44), script)
        self.assertFalse(persisted_trust)
        self.assertIs(flag, False)
        self.assertTrue(bumped)
        self.assertFalse(still_granted)


if __name__ == "__main__":
    unittest.main()
