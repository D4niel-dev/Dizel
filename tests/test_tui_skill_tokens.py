"""Attached skills: /skill attaches a chip; backspace removes; send expands."""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

from dizel_core.config_manager import ConfigManager
from dizel_core.usage_manager import UsageManager
from inference.tui.app import DizelCMDApp
from inference.tui.panels.workspace import WorkspacePanel


def _defaults():
    return copy.deepcopy(ConfigManager._DEFAULTS)


class SkillTokenTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(DizelCMDApp, "_auto_load_model", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_update_check", lambda self: None),
            mock.patch.object(DizelCMDApp, "_start_mcp", lambda self: None),
            mock.patch.object(ConfigManager, "load", classmethod(lambda cls: _defaults())),
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

    def _run(self, script):
        async def run():
            app = DizelCMDApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.pause()
                return await script(app, pilot)

        return asyncio.run(run())

    def test_skill_command_attaches_instead_of_inserting(self):
        async def script(app, pilot):
            from inference.tui.commands.builtins import register_builtins
            from inference.tui.commands.parser import parse_command
            from inference.tui.commands.registry import registry

            register_builtins()
            command = registry.lookup("email")
            assert command is not None, "builtin /email missing"
            result = await command.execute(app, parse_command("/email hello"))
            inp = app.query_one("#prompt-input")
            row = app.query_one("#input-skills")
            return app.attached_skills, inp.value, result, row.display, str(row.render())

        attached, value, result, visible, row_text = self._run(script)
        self.assertEqual([s["name"] for s in attached], ["email"])
        self.assertIn("hello", attached[0]["prompt"])
        self.assertEqual(value, "")
        self.assertIn("Attached", result)
        self.assertTrue(visible)
        self.assertIn(attached[0]["title"], row_text)
        self.assertIn("backspace", row_text.lower())

    def test_backspace_on_empty_prompt_removes_the_last_skill(self):
        async def script(app, pilot):
            app.attach_skill({"name": "a", "title": "A", "prompt": "pa"})
            app.attach_skill({"name": "b", "title": "B", "prompt": "pb"})
            inp = app.query_one("#prompt-input")
            inp.focus()
            await pilot.press("backspace")
            await pilot.pause()
            after_empty = [s["name"] for s in app.attached_skills]

            inp.value = "hi"
            inp.focus()
            await pilot.press("backspace")
            await pilot.pause()
            return after_empty, [s["name"] for s in app.attached_skills], inp.value

        after_empty, after_text, value = self._run(script)
        self.assertEqual(after_empty, ["a"])
        self.assertEqual(after_text, ["a"])
        self.assertEqual(value, "h")

    def test_submit_composes_skills_and_stores_display(self):
        async def script(app, pilot):
            app.attach_skill({"name": "email", "title": "Email", "prompt": "PROMPT"})
            captured = {}
            app.tool_bridge.run_tools_and_send = (
                lambda text, display=None: captured.update(text=text, display=display)
            )
            inp = app.query_one("#prompt-input")
            inp.value = "do it"
            inp.focus()
            await pilot.press("enter")
            await pilot.pause()
            return (
                captured,
                app.session_bridge.current_messages[-1],
                app.attached_skills,
            )

        captured, stored, attached = self._run(script)
        self.assertEqual(captured["text"], "PROMPT\n\ndo it")
        self.assertEqual(
            captured["display"],
            {"text": "do it", "skills": [{"name": "email", "title": "Email"}]},
        )
        self.assertEqual(stored["content"], "PROMPT\n\ndo it")
        self.assertEqual(stored["display"], captured["display"])
        self.assertEqual(attached, [])

    def test_skills_only_submission_sends_the_expanded_prompt(self):
        async def script(app, pilot):
            app.attach_skill({"name": "email", "title": "Email", "prompt": "PROMPT"})
            captured = {}
            app.tool_bridge.run_tools_and_send = (
                lambda text, display=None: captured.update(text=text, display=display)
            )
            inp = app.query_one("#prompt-input")
            inp.focus()
            await pilot.press("enter")
            await pilot.pause()
            return captured, app.session_bridge.current_messages[-1]

        captured, stored = self._run(script)
        self.assertEqual(captured["text"], "PROMPT")
        self.assertEqual(
            captured["display"],
            {"text": "", "skills": [{"name": "email", "title": "Email"}]},
        )
        self.assertEqual(stored["content"], "PROMPT")

    def test_transcript_shows_skill_chips_above_the_typed_text(self):
        async def script(app, pilot):
            workspace = app.query_one(WorkspacePanel)
            workspace.add_user_message(
                "do it",
                {"text": "do it", "skills": [{"name": "email", "title": "Email"}]},
            )
            await pilot.pause()
            chips = list(workspace.query(".skill-chips"))
            rendered = " ".join(str(widget.render()) for widget in chips)
            return len(chips), rendered

        count, rendered = self._run(script)
        self.assertEqual(count, 1)
        self.assertIn("Email", rendered)

    def test_resumed_session_renders_skill_chips(self):
        async def script(app, pilot):
            data = {
                "id": "skill-session",
                "title": "Skill session",
                "messages": [
                    {
                        "role": "user",
                        "content": "PROMPT BODY\n\nhello",
                        "display": {
                            "text": "hello",
                            "skills": [{"name": "email", "title": "Email"}],
                        },
                        "ts": 0,
                    }
                ],
            }
            with mock.patch(
                "inference.tui.bridge.session_bridge.load_session", return_value=data
            ):
                app.load_session_to_workspace("skill-session")
            await pilot.pause()
            workspace = app.query_one(WorkspacePanel)
            chips = list(workspace.query(".skill-chips"))
            rendered = " ".join(str(widget.render()) for widget in chips)
            return len(chips), rendered

        count, rendered = self._run(script)
        self.assertEqual(count, 1)
        self.assertIn("Email", rendered)


if __name__ == "__main__":
    unittest.main()
