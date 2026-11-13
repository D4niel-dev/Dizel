"""Chat and behaviour preferences shared with the desktop.

The desktop's Chat and Shortcuts tabs now drive the terminal too. These tests
pin the shared defaults, the transcript options they control, the composer
keys, the transcript width, and the settings-modal round trip.

All disk and network seams are stubbed: no real settings, no session files,
no model load, no update check, no MCP spawn.
"""

from __future__ import annotations

import asyncio
import copy
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from rich.console import Console
from textual.widgets import Input

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.input_bar import InputBar, PromptInput
from inference.tui.panels.modals.settings_modal import SettingsModal
from inference.tui.panels.multiline_input import MultilineInputModal
from inference.tui.panels.status_bar import StatusBar
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.rendering.assistant_message import _safe_reveal
from inference.tui.rendering.message_block import (
    MessageBlock,
    format_clock,
    message_style,
)
from inference.tui.settings_apply import CHAT_DEFAULTS


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


def _config(**appearance):
    cfg = _defaults()
    cfg.setdefault("appearance", {}).update(appearance)
    return cfg


class PreferenceLayerTests(unittest.TestCase):
    """The pure layer: defaults, clock formats, and render options."""

    def test_defaults_match_desktop(self):
        self.assertEqual(CHAT_DEFAULTS["typing_animation"], "smooth")
        self.assertEqual(CHAT_DEFAULTS["message_width"], "comfortable")
        self.assertFalse(CHAT_DEFAULTS["show_model_tag"])
        self.assertFalse(CHAT_DEFAULTS["escape_clears_draft"])
        self.assertTrue(CHAT_DEFAULTS["enter_to_send"])
        self.assertTrue(CHAT_DEFAULTS["auto_focus_composer"])
        self.assertTrue(CHAT_DEFAULTS["arrow_up_edits_last"])

    def test_format_clock(self):
        stamp = time.mktime((2026, 11, 10, 14, 5, 0, 0, 0, -1))
        self.assertEqual(format_clock(stamp, True), "14:05")
        self.assertEqual(format_clock(stamp, False), "2:05 PM")

    def test_message_style_reads_prefs(self):
        prefs = dict(CHAT_DEFAULTS)
        prefs.update(
            show_timestamps=False,
            separators=True,
            message_spacing="compact",
            syntax_highlighting=False,
        )
        app = SimpleNamespace(_chat_prefs=prefs)
        style = message_style(app, timestamp=time.time())
        self.assertIsNone(style["timestamp"])
        self.assertTrue(style["separators"])
        self.assertEqual(style["spacing"], "compact")
        self.assertFalse(style["syntax"])

        prefs["show_timestamps"] = True
        style = message_style(app, timestamp=time.time())
        self.assertIsNotNone(style["timestamp"])

    def test_message_block_renders_timestamp_and_rule(self):
        console = Console(
            record=True, width=70, force_terminal=False, color_system=None
        )
        block = MessageBlock(
            "ASSISTANT",
            "Hello there",
            timestamp="14:05",
            separators=True,
            syntax=False,
        )
        console.print(block)
        out = console.export_text()
        self.assertIn("14:05", out)
        self.assertIn("─", out)

    def test_safe_reveal_holds_back_cut_inside_think_tag(self):
        text = "abc<thinking>inner</thinking>"
        self.assertEqual(_safe_reveal(text, 6), 3)
        self.assertEqual(_safe_reveal(text, 3), 3)
        self.assertEqual(_safe_reveal(text, 2), 2)


class ChatPreferenceRuntimeTests(unittest.TestCase):
    """Each pref, proved against the running app."""

    def setUp(self):
        self._patches = [
            mock.patch.object(DizelCMDApp, "_auto_load_model", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_update_check", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_mcp", lambda self: None),
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

    def _run(self, size, script, cfg=None):
        def load(cls):
            return copy.deepcopy(cfg) if cfg is not None else _defaults()

        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        with mock.patch.object(ConfigManager, "load", classmethod(load)):
            return asyncio.run(run())

    def test_enter_to_send_off_opens_editor_and_ctrl_s_sends(self):
        async def script(app, pilot):
            app._chat_prefs["enter_to_send"] = False
            app.tool_bridge.run_tools_and_send = lambda text: None
            inp = app.query_one("#prompt-input", PromptInput)
            inp.focus()
            inp.value = "hello"
            await pilot.pause()
            inp.post_message(Input.Submitted(inp, "hello"))
            await pilot.pause()
            await pilot.pause()
            opened = isinstance(app.screen, MultilineInputModal)
            idle = app.generation_state
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.pause()
            return (
                opened,
                idle,
                isinstance(app.screen, MultilineInputModal),
                app.generation_state,
                [m["content"] for m in app.session_bridge.current_messages],
            )

        opened, idle, still_open, state, messages = self._run((120, 36), script)
        self.assertTrue(opened)
        self.assertEqual(idle, "IDLE")
        self.assertFalse(still_open)
        self.assertEqual(state, "THINKING")
        self.assertEqual(messages[-1], "hello")

    def test_escape_clears_draft(self):
        async def script(app, pilot):
            inp = app.query_one("#prompt-input", PromptInput)
            inp.focus()
            inp.value = "draft"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            kept = inp.value
            app._chat_prefs["escape_clears_draft"] = True
            await pilot.press("escape")
            await pilot.pause()
            return kept, inp.value

        kept, cleared = self._run((120, 36), script)
        self.assertEqual(kept, "draft")
        self.assertEqual(cleared, "")

    def test_arrow_up_recall_and_draft_protection(self):
        async def script(app, pilot):
            bar = app.query_one(InputBar)
            inp = app.query_one("#prompt-input", PromptInput)
            inp.focus()
            bar.history = ["first prompt"]
            bar.history_idx = 1
            await pilot.press("up")
            await pilot.pause()
            recalled = inp.value

            inp.value = "typed draft"
            bar.history_idx = 1
            await pilot.press("up")
            await pilot.pause()
            protected = inp.value

            inp.value = ""
            bar.history_idx = 1
            app._chat_prefs["arrow_up_edits_last"] = False
            await pilot.press("up")
            await pilot.pause()
            return recalled, protected, inp.value

        recalled, protected, disabled = self._run((120, 36), script)
        self.assertEqual(recalled, "first prompt")
        self.assertEqual(protected, "typed draft")
        self.assertEqual(disabled, "")

    def test_transcript_width_classes(self):
        async def script(app, pilot):
            container = app.query_one("#workspace-container")
            workspace = app.query_one(WorkspacePanel)
            row = app.query_one("#transcript-row")
            return (
                container.has_class("-width-wide"),
                container.has_class("-width-full"),
                workspace.region.width,
                row.region.width,
            )

        wide, full, width, row_width = self._run(
            (150, 42), script, cfg=_config(message_width="wide")
        )
        self.assertTrue(wide)
        self.assertFalse(full)
        self.assertEqual(width, min(120, row_width))

        wide, full, width, row_width = self._run(
            (150, 42), script, cfg=_config(message_width="full")
        )
        self.assertFalse(wide)
        self.assertTrue(full)
        self.assertEqual(width, row_width)

    def test_starter_suggestions_switch(self):
        async def script(app, pilot):
            return [bool(w.display) for w in app.query(".empty-suggestion")]

        visible = self._run((120, 36), script)
        hidden = self._run(
            (120, 36), script, cfg=_config(show_suggestions=False)
        )
        self.assertTrue(visible)
        self.assertTrue(all(visible))
        self.assertTrue(hidden)
        self.assertFalse(any(hidden))

    def test_auto_focus_switch(self):
        async def script(app, pilot):
            return isinstance(app.focused, PromptInput)

        self.assertTrue(self._run((120, 36), script))
        self.assertFalse(
            self._run((120, 36), script, cfg=_config(auto_focus_composer=False))
        )

    def test_animations_gate_status_spinner(self):
        async def script(app, pilot):
            status = app.query_one(StatusBar)
            app.generation_state = "STREAMING"
            app._chat_prefs["animations"] = False
            status._glyph = "●"
            status._tick()
            frozen = status._glyph
            app._chat_prefs["animations"] = True
            status._tick()
            return frozen, status._glyph

        frozen, moving = self._run((120, 36), script)
        self.assertEqual(frozen, "●")
        self.assertNotEqual(moving, "●")

    def test_new_messages_record_a_timestamp(self):
        async def script(app, pilot):
            app.session_bridge.add_message("user", "hello")
            return app.session_bridge.current_messages[-1].get("ts")

        stamp = self._run((120, 36), script)
        self.assertIsInstance(stamp, float)

    def test_settings_modal_round_trip(self):
        async def script(app, pilot):
            modal = SettingsModal()
            app.push_screen(modal)
            await pilot.pause()
            await pilot.pause()
            order = list(modal._section_order)
            defaults = {
                key: modal._fields[f"appearance.{key}"].control.value
                for key in (
                    "show_timestamps",
                    "clock_24h",
                    "typing_animation",
                    "escape_clears_draft",
                    "message_width",
                    "enter_to_send",
                )
            }
            modal._fields["appearance.show_timestamps"].control.value = False
            modal._fields["appearance.typing_animation"].control.value = "cinematic"
            modal._fields["appearance.message_width"].control.value = "wide"
            await pilot.pause()
            saved = {}
            with mock.patch.object(
                ConfigManager,
                "save",
                classmethod(lambda cls, cfg: saved.update(copy.deepcopy(cfg))),
            ), mock.patch.object(
                app.chat_bridge.manager, "reload_sampling", lambda: None
            ), mock.patch.object(
                app.chat_bridge.manager, "reload_provider", lambda: None
            ):
                modal._save()
                await pilot.pause()
            width_applied = app.query_one("#workspace-container").has_class(
                "-width-wide"
            )
            return (
                order,
                defaults,
                saved.get("appearance", {}),
                dict(app._chat_prefs),
                width_applied,
            )

        order, defaults, saved, live, width_applied = self._run((150, 42), script)
        self.assertIn("chat", order)
        self.assertIn("behaviour", order)
        self.assertLess(order.index("chat"), order.index("tools"))
        self.assertTrue(defaults["show_timestamps"])
        self.assertTrue(defaults["clock_24h"])
        self.assertEqual(defaults["typing_animation"], "smooth")
        self.assertFalse(defaults["escape_clears_draft"])
        self.assertEqual(defaults["message_width"], "comfortable")
        self.assertTrue(defaults["enter_to_send"])
        self.assertFalse(saved["show_timestamps"])
        self.assertEqual(saved["typing_animation"], "cinematic")
        self.assertEqual(saved["message_width"], "wide")
        self.assertFalse(live["show_timestamps"])
        self.assertEqual(live["typing_animation"], "cinematic")
        self.assertTrue(width_applied)


if __name__ == "__main__":
    unittest.main()
