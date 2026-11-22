"""Every Dizel Thinking effort prompt must survive both tiers."""

from __future__ import annotations

import unittest

from dizel_core.model_backend import _V15_SYSTEM_PROMPT_MAX_CHARS, _v15_system_prompt
from dizel_core.token_budget import THINKING_EFFORTS, get_model_profile


class DizelModePromptTests(unittest.TestCase):
    def test_every_thinking_effort_fits_the_15_system_cap(self):
        for effort in THINKING_EFFORTS:
            prompt = get_model_profile("Dizel", "Thinking", effort).system_prompt
            collapsed = " ".join(prompt.split())
            self.assertLessEqual(
                len(collapsed), _V15_SYSTEM_PROMPT_MAX_CHARS, f"{effort}: {len(collapsed)}"
            )

    def test_every_prompt_names_dizel_and_survives_compaction(self):
        for effort in THINKING_EFFORTS:
            prompt = get_model_profile("Dizel", "Thinking", effort).system_prompt
            collapsed = " ".join(prompt.split())
            self.assertIn("dizel", collapsed.lower())
            self.assertEqual(_v15_system_prompt(prompt), collapsed, effort)

    def test_flash_and_pro_inherit_every_effort(self):
        for family in ("Flash", "Pro"):
            for effort in THINKING_EFFORTS:
                self.assertEqual(
                    get_model_profile(family, "Thinking", effort).system_prompt,
                    get_model_profile("Dizel", "Thinking", effort).system_prompt,
                )

    def test_efforts_keep_distinct_instructions(self):
        prompts = {
            effort: get_model_profile("Dizel", "Thinking", effort).system_prompt
            for effort in THINKING_EFFORTS
        }
        self.assertEqual(len(set(prompts.values())), len(THINKING_EFFORTS))
        self.assertIn("<think>", prompts["deep"])
        self.assertTrue(any("(low)" in p for p in prompts.values()))


if __name__ == "__main__":
    unittest.main()
