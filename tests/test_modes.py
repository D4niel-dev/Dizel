from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from dizel_core.token_budget import (
    THINKING_EFFORTS,
    _PROFILES,
    get_model_profile,
    model_family,
    normalize_thinking_effort,
)
from inference.tui.app import DizelCMDApp
from inference.tui.commands.builtins import register_builtins
from inference.tui.commands.builtins.mode import ModeCommand
from inference.tui.commands.parser import parse_command
from inference.tui.commands.registry import registry
from inference.tui.constants import MODE_DETAILS, MODES


class ModeProfileTests(unittest.TestCase):
    def test_every_family_has_four_modes(self):
        expected = {"Fast", "Planning", "Coding", "Thinking"}
        for family in ("Dizel", "Mila", "Generic", "Flash", "Pro"):
            self.assertEqual(set(_PROFILES[family]), expected)

    def test_effort_changes_sampling_and_budget(self):
        low = get_model_profile("Dizel", "Thinking", "low")
        medium = get_model_profile("Dizel", "Thinking", "medium")
        high = get_model_profile("Dizel", "Thinking", "high")
        deep = get_model_profile("Dizel", "Thinking", "deep")
        self.assertEqual(medium.sampling.temperature, 0.40)
        self.assertEqual(medium.budget_multiplier, 1.35)
        self.assertEqual(low.sampling.temperature, 0.35)
        self.assertEqual(low.budget_multiplier, 1.01)
        self.assertEqual(high.sampling.temperature, 0.43)
        self.assertEqual(high.budget_multiplier, 1.55)
        self.assertEqual(deep.sampling.temperature, 0.50)
        self.assertEqual(deep.budget_multiplier, 1.89)
        self.assertGreater(deep.budget_multiplier, medium.budget_multiplier)
        self.assertIn("<think>", deep.system_prompt)
        self.assertIn("Max", deep.label)

    def test_effort_defaults_and_normalization(self):
        self.assertEqual(
            get_model_profile("Dizel", "Thinking").system_prompt,
            get_model_profile("Dizel", "Thinking", "medium").system_prompt,
        )
        self.assertEqual(normalize_thinking_effort("X-High"), "xhigh")
        self.assertEqual(normalize_thinking_effort("deep think"), "deep")
        self.assertEqual(normalize_thinking_effort("nonsense"), "medium")
        self.assertEqual(
            set(THINKING_EFFORTS), {"low", "medium", "high", "xhigh", "deep"}
        )

    def test_non_dizel_profiles_keep_their_identity(self):
        mila = get_model_profile("Mila", "Thinking", "deep")
        generic = get_model_profile("Generic", "Thinking", "deep")
        self.assertIn("You are Mila", mila.system_prompt)
        self.assertNotIn("You are Dizel", mila.system_prompt)
        self.assertNotIn("You are Dizel", generic.system_prompt)


class TuiModeTests(unittest.TestCase):
    class FakeApp:
        def __init__(self):
            self.active_mode = "Fast"
            self.screens = []
            self.mode_calls = []

        def set_mode(self, mode, thinking_effort=None, announce=True):
            self.active_mode = mode
            self.mode_calls.append((mode, thinking_effort))

        def set_thinking_effort(self, effort, announce=True):
            self.mode_calls.append(("effort", effort))

        def push_screen(self, screen, callback=None):
            self.screens.append((screen, callback))

    def test_modes_exclude_deep_think(self):
        self.assertIn("Thinking", MODES)
        self.assertNotIn("Deep Think", MODES)
        self.assertNotIn("Deep Think", MODE_DETAILS)

    def test_mode_command_accepts_efforts_and_aliases(self):
        async def run():
            app = self.FakeApp()
            command = ModeCommand()
            await command.execute(app, parse_command("/mode coding"))
            self.assertEqual(app.active_mode, "Coding")
            await command.execute(app, parse_command("/mode thinking high"))
            self.assertEqual(app.active_mode, "Thinking")
            await command.execute(app, parse_command("/mode deep"))
            self.assertIn(("Thinking", "deep"), app.mode_calls)
            await command.execute(app, parse_command("/mode deep think"))
            self.assertIn(("Thinking", "deep"), app.mode_calls)
            await command.execute(app, parse_command("/mode thinking x-high"))
            self.assertIn(("Thinking", "xhigh"), app.mode_calls)

        asyncio.run(run())

    def test_tier_command_is_gone(self):
        register_builtins()
        self.assertIsNone(registry.lookup("tier"))

    def test_response_label_and_color_track_the_effort(self):
        stub = SimpleNamespace(
            active_mode="Thinking", active_thinking_effort="high", active_tier="Flash"
        )
        self.assertEqual(DizelCMDApp.response_label.fget(stub), "Thinking · High")
        self.assertEqual(DizelCMDApp.response_color.fget(stub), "#f472b6")

    def test_dizel_profiles_resolve_without_a_tier(self):
        stub = SimpleNamespace(model_family="Dizel", active_tier="Pro")
        self.assertEqual(DizelCMDApp._profile_family(stub), "Dizel")


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
