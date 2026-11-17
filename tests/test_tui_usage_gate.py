"""TUI usage policy: provider buckets, blocking, and compaction."""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.modals.settings_modal import SettingsModal


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class TuiUsageGateTests(unittest.TestCase):
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

    def test_enforce_usage_activates_the_provider_bucket(self):
        async def script(app, pilot):
            manager = app.chat_bridge.manager
            manager._provider_slug = "openai"
            manager._api_provider = None
            manager._api_model = ""
            manager._backend = None
            manager._cfg = {"api_router": {}}
            app.active_provider = "openai"
            app.chat_bridge.enforce_usage()
            return (
                app.usage_manager.active_provider,
                int(app.usage_manager.max_capacity),
            )

        provider, capacity = self._run((120, 36), script)
        self.assertEqual(provider, "openai")
        self.assertEqual(capacity, 131072)

    def test_enforce_usage_blocks_a_spent_window(self):
        async def script(app, pilot):
            usage = app.usage_manager
            cap = int(app.chat_bridge.manager.effective_ctx_len)
            usage.activate("local", cap)
            usage.force_percent(100)
            message = app.chat_bridge.enforce_usage()
            await pilot.pause()
            return message

        message = self._run((120, 36), script)
        self.assertIsNotNone(message)
        self.assertIn("spent", message.lower())
        self.assertIn("local", message.lower())

    def test_enforce_usage_compacts_at_the_threshold(self):
        async def script(app, pilot):
            app.chat_bridge.manager._provider_slug = "local"
            app.active_provider = "local"
            usage = app.usage_manager
            cap = int(app.chat_bridge.manager.effective_ctx_len)
            usage.activate("local", cap)
            usage.force_percent(84)
            called = {}
            with mock.patch.object(
                app.chat_bridge.manager,
                "compact_context",
                side_effect=lambda: called.setdefault("compacted", True),
            ):
                message = app.chat_bridge.enforce_usage()
            await pilot.pause()
            return message, called.get("compacted"), usage.usage, usage.compaction_count

        message, compacted, remaining, compactions = self._run((120, 36), script)
        self.assertIsNone(message)
        self.assertTrue(compacted)
        self.assertEqual(remaining, 0.0)
        self.assertEqual(compactions, 1)

    def test_enforce_usage_refuses_when_compactions_are_spent(self):
        async def script(app, pilot):
            usage = app.usage_manager
            cap = int(app.chat_bridge.manager.effective_ctx_len)
            usage.activate("local", cap)
            for _ in range(usage.MAX_COMPACTIONS):
                usage.run_compaction()
            usage.force_percent(90)
            message = app.chat_bridge.enforce_usage()
            await pilot.pause()
            return message

        message = self._run((120, 36), script)
        self.assertIsNotNone(message)
        self.assertIn("context limit", message.lower())

    def test_tool_bridge_stops_before_starting_a_message(self):
        async def script(app, pilot):
            started = {}
            sent = {}
            with mock.patch.object(
                app.chat_bridge,
                "enforce_usage",
                return_value="Usage window spent for local",
            ), mock.patch.object(
                app,
                "_log_system",
                wraps=app._log_system,
            ) as log, mock.patch.object(
                app.tool_bridge,
                "_start_assistant",
                side_effect=lambda: started.setdefault("started", True),
            ), mock.patch.object(
                app.chat_bridge.manager,
                "send_with_tools",
                side_effect=lambda **kwargs: sent.setdefault("sent", True),
            ):
                app.tool_bridge._run("hello")
                await pilot.pause()
                notes = [str(c.args[0]) for c in log.call_args_list]
            return started.get("started", False), sent.get("sent", False), notes

        started, sent, notes = self._run((120, 36), script)
        self.assertFalse(started)
        self.assertFalse(sent)
        self.assertTrue(any("spent" in note.lower() for note in notes))

    def test_sync_usage_provider_follows_the_manager(self):
        async def script(app, pilot):
            manager = app.chat_bridge.manager
            manager._provider_slug = "anthropic"
            manager._api_provider = None
            manager._api_model = ""
            manager._backend = None
            manager._cfg = {"api_router": {}}
            app.active_provider = "anthropic"
            app.sync_usage_provider()
            return (
                app.usage_manager.active_provider,
                int(app.usage_manager.max_capacity),
            )

        provider, capacity = self._run((120, 36), script)
        self.assertEqual(provider, "anthropic")
        self.assertEqual(capacity, 200000)


    def test_settings_saves_the_active_provider_usage_limit(self):
        saved = {}
        defaults = _defaults()
        defaults["api_router"] = {**defaults["api_router"], "provider": "ollama"}

        async def script(app, pilot):
            with mock.patch.object(
                ConfigManager, "load", classmethod(lambda cls: copy.deepcopy(defaults))
            ), mock.patch.object(
                ConfigManager, "save", classmethod(lambda cls, cfg: saved.update(cfg))
            ):
                modal = SettingsModal()
                app.push_screen(modal)
                await pilot.pause()
                modal._show_section("api")
                await pilot.pause()
                field = modal.query_one("#settings-api-router-provider-limits")
                field.value = "64000"
                await pilot.press("ctrl+s")
                await pilot.pause()
            return saved.get("api_router", {}).get("provider_limits")

        self.assertEqual(self._run((140, 44), script), {"ollama": 64000})


if __name__ == "__main__":
    unittest.main()
