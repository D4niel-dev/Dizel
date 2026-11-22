"""Response length must affect both the budget and the prompt.

The GUI sends short/normal/detailed while the budget table used low/normal/high,
so short and detailed used to fall through to 1.0 and did nothing at all.
"""

from __future__ import annotations

import unittest

from dizel_core.chat_manager import ChatManager
from dizel_core.model_backend import _V15_SYSTEM_PROMPT_MAX_CHARS, _v15_system_prompt
from dizel_core.token_budget import (
    VERBOSITY_CLAUSES,
    VERBOSITY_MULTIPLIERS,
    allocate_token_budget,
    verbosity_clause,
)


class VerbosityBudgetTests(unittest.TestCase):
    def test_all_gui_spellings_have_distinct_multipliers(self):
        short = VERBOSITY_MULTIPLIERS["short"]
        normal = VERBOSITY_MULTIPLIERS["normal"]
        detailed = VERBOSITY_MULTIPLIERS["detailed"]
        self.assertLess(short, normal)
        self.assertLess(normal, detailed)

    def test_allocate_budget_honours_gui_spellings(self):
        from dizel_core.token_budget import TaskType

        budgets = {
            value: allocate_token_budget(TaskType.CHAT, 100, 100, 2048, verbosity=value)
            for value in ("short", "normal", "detailed")
        }
        self.assertEqual(len(set(budgets.values())), 3)
        self.assertLess(budgets["short"], budgets["normal"])
        self.assertLess(budgets["normal"], budgets["detailed"])

    def test_tui_spellings_share_the_clauses(self):
        self.assertEqual(verbosity_clause("low"), VERBOSITY_CLAUSES["short"])
        self.assertEqual(verbosity_clause("high"), VERBOSITY_CLAUSES["detailed"])
        self.assertEqual(verbosity_clause("nonsense"), "")


class VerbosityPromptTests(unittest.TestCase):
    def _manager(self, verbosity="normal"):
        manager = ChatManager.__new__(ChatManager)
        manager.system_prompt = "Base rules."
        manager.mode_prompt = "Mode: Fast — be direct."
        manager.project_context = "Project rules."
        manager.turn_verbosity = verbosity
        return manager

    def test_normal_adds_nothing(self):
        manager = self._manager("normal")
        self.assertEqual(
            manager.system_content(), "Base rules.\n\nMode: Fast — be direct.\n\nProject rules."
        )

    def test_clause_sits_between_mode_and_project(self):
        manager = self._manager("detailed")
        content = manager.system_content()
        self.assertLess(content.index("Mode: Fast"), content.index("Response length"))
        self.assertLess(content.index("Response length"), content.index("Project rules"))

    def test_short_has_a_clause(self):
        content = self._manager("short").system_content()
        self.assertIn("Response length: Short", content)

    def test_collapsed_prompt_fits_the_cap_for_every_mode_and_length(self):
        from dizel_core.token_budget import get_model_profile

        # Use real profile prompts: the collapse path recognises the shipped
        # prefixes, and a synthetic string would not exercise it.
        for family in ("Dizel", "Mila", "Generic"):
            for mode in ("Fast", "Thinking", "Coding", "Planning"):
                for verbosity in ("short", "normal", "detailed"):
                    profile = get_model_profile(family, mode)
                    manager = self._manager(verbosity)
                    manager.system_prompt = profile.system_prompt
                    manager.mode_prompt = ""
                    manager.project_context = ""
                    self.assertLessEqual(
                        len(_v15_system_prompt(manager.system_content())),
                        _V15_SYSTEM_PROMPT_MAX_CHARS,
                        f"{family}/{mode}/{verbosity}",
                    )

    def test_turn_verbosity_can_override_config(self):
        manager = self._manager("detailed")
        manager.turn_verbosity = "short"
        self.assertIn("Response length: Short", manager.system_content())
        self.assertNotIn("Detailed", manager.system_content())


if __name__ == "__main__":
    unittest.main()
