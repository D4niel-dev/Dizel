from __future__ import annotations

import asyncio
import unittest

from dizel_core.token_budget import _PROFILES, get_model_profile, model_family
from inference.tui.commands.builtins.mode import ModeCommand
from inference.tui.commands.builtins.tier import TierCommand
from inference.tui.commands.parser import parse_command
from inference.tui.constants import MODE_DETAILS, MODES


class ModeProfileTests(unittest.TestCase):
    def test_every_family_has_the_full_mode_set(self):
        expected = {"Fast", "Planning", "Coding", "Thinking", "Deep Think"}
        for family in ("Dizel", "Mila", "Generic", "Flash", "Pro"):
            self.assertEqual(set(_PROFILES[family]), expected)

    def test_deep_think_is_deeper_than_thinking(self):
        thinking = get_model_profile("Dizel", "Thinking")
        deep = get_model_profile("Dizel", "Deep Think")
        self.assertGreater(deep.budget_multiplier, thinking.budget_multiplier)
        self.assertIn("<think>", deep.system_prompt)
        self.assertIn("Deep Think", deep.label)

    def test_non_dizel_profiles_keep_their_identity(self):
        mila = get_model_profile("Mila", "Deep Think")
        generic = get_model_profile("Generic", "Deep Think")
        self.assertIn("You are Mila", mila.system_prompt)
        self.assertNotIn("You are Dizel", mila.system_prompt)
        self.assertNotIn("You are Dizel", generic.system_prompt)


class TuiModeTests(unittest.TestCase):
    class FakeApp:
        is_dizel_model = False

        def __init__(self):
            self.active_mode = "Fast"
            self.screens = []

        def set_mode(self, mode):
            self.active_mode = mode

        def push_screen(self, screen, callback=None):
            self.screens.append((screen, callback))

    def test_modes_include_deep_think(self):
        self.assertIn("Deep Think", MODES)
        self.assertIn("Deep Think", MODE_DETAILS)

    def test_mode_command_accepts_multiword_deep_think(self):
        async def run():
            app = self.FakeApp()
            command = ModeCommand()
            await command.execute(app, parse_command("/mode deep think"))
            self.assertEqual(app.active_mode, "Deep Think")
            await command.execute(app, parse_command("/mode deep-think"))
            self.assertEqual(app.active_mode, "Deep Think")
            await command.execute(app, parse_command("/mode deepthink"))
            self.assertEqual(app.active_mode, "Deep Think")

        asyncio.run(run())

    def test_tier_command_rejects_non_dizel_models(self):
        async def run():
            app = self.FakeApp()
            result = await TierCommand().execute(app, parse_command("/tier"))
            self.assertIn("Dizel models only", result)
            self.assertEqual(app.screens, [])

        asyncio.run(run())


class ProfileFamilyTests(unittest.TestCase):
    def test_family_detection(self):
        self.assertEqual(model_family("Dizel 1.5 Distill"), "Dizel")
        self.assertEqual(model_family("Dizel 1.6 Flash Preview"), "Dizel")
        self.assertEqual(model_family("Mila Preview"), "Mila")
        self.assertEqual(model_family("Milan 7B"), "Generic")
        self.assertEqual(model_family("Qwen2.5 7B Instruct"), "Generic")
        self.assertEqual(model_family(""), "Generic")

if __name__ == "__main__":
    unittest.main()
