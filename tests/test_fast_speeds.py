"""Fast has three reply speeds, mirroring Thinking effort.

Higher speed spends a larger budget and carries a longer mode clause. Temperature
and top-p must stay put: they do not control wall-clock time.
"""

from __future__ import annotations

import unittest

from dizel_core.model_backend import _V15_SYSTEM_PROMPT_MAX_CHARS, _v15_system_prompt
from dizel_core.token_budget import (
    _FAST_SPEED_CLAUSES,
    FAST_SPEEDS,
    _PROFILES,
    get_model_profile,
    normalize_fast_speed,
)


class FastSpeedTests(unittest.TestCase):
    def test_default_reproduces_todays_fast(self):
        before = get_model_profile("Dizel", "Fast")
        after = get_model_profile("Dizel", "Fast", "medium", "fast")
        self.assertEqual(before.system_prompt, after.system_prompt)
        self.assertEqual(before.budget_multiplier, after.budget_multiplier)

    def test_budget_increases_with_speed(self):
        budgets = [
            get_model_profile("Dizel", "Fast", "medium", speed).budget_multiplier
            for speed in FAST_SPEEDS
        ]
        self.assertEqual(budgets, sorted(budgets))
        self.assertEqual(len(set(budgets)), 3)

    def test_clauses_grow_with_speed(self):
        lengths = [len(_FAST_SPEED_CLAUSES[speed]) for speed in FAST_SPEEDS]
        self.assertEqual(lengths, sorted(lengths))

    def test_sampling_is_unchanged(self):
        base = _PROFILES["Dizel"]["Fast"].sampling
        for speed in FAST_SPEEDS:
            profile = get_model_profile("Dizel", "Fast", "medium", speed)
            self.assertEqual(profile.sampling.temperature, base.temperature)
            self.assertEqual(profile.sampling.top_k, base.top_k)
            self.assertEqual(profile.sampling.top_p, base.top_p)
            self.assertEqual(profile.sampling.repetition_penalty, base.repetition_penalty)

    def test_every_family_fits_the_system_prompt_cap(self):
        for family in ("Dizel", "Mila", "Generic", "Flash", "Pro"):
            for speed in FAST_SPEEDS:
                profile = get_model_profile(family, "Fast", "medium", speed)
                collapsed = _v15_system_prompt(profile.system_prompt)
                self.assertLessEqual(
                    len(collapsed),
                    _V15_SYSTEM_PROMPT_MAX_CHARS,
                    f"{family}/{speed}",
                )

    def test_speed_is_ignored_outside_fast(self):
        for mode in ("Thinking", "Planning", "Coding"):
            plain = get_model_profile("Dizel", mode)
            fast = get_model_profile("Dizel", mode, "medium", "ultrafast")
            self.assertEqual(plain.system_prompt, fast.system_prompt)
            self.assertEqual(plain.budget_multiplier, fast.budget_multiplier)


class NormalizeSpeedTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_fast_speed("normal"), "fast")
        self.assertEqual(normalize_fast_speed("default"), "fast")
        self.assertEqual(normalize_fast_speed("ultra"), "ultrafast")
        self.assertEqual(normalize_fast_speed(" ULTRA-FAST "), "ultrafast")

    def test_unknown_falls_back_to_fast(self):
        for value in ("", "nonsense", None, 42):
            self.assertEqual(normalize_fast_speed(value), "fast")


if __name__ == "__main__":
    unittest.main()
