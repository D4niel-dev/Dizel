"""Plan approval lifts only the current turn's tool-round ceiling."""

from __future__ import annotations

import unittest

from dizel_core.chat_manager import PLAN_ROUND_BUDGET, plan_approved, round_limit


class PlanBudgetTests(unittest.TestCase):
    def test_default_budget_is_unchanged(self):
        self.assertEqual(round_limit(3, False), 3)
        self.assertEqual(round_limit(5, False), 5)

    def test_approval_raises_the_budget(self):
        self.assertEqual(round_limit(3, True), PLAN_ROUND_BUDGET)
        self.assertEqual(round_limit(24, True), 24)

    def test_approved_plan_outcome_is_detected(self):
        calls = [{"name": "read_file"}, {"name": "plan"}]
        outcomes = ["[read_file] ok", "[plan] Approved. Execute the plan now."]

        self.assertTrue(plan_approved(calls, outcomes))

    def test_denied_or_unrelated_outcomes_do_not_count(self):
        calls = [{"name": "plan"}, {"name": "plan"}]
        outcomes = [
            "[plan] Plan discarded. Do not execute it.",
            "[plan] Plan refused: no approver.",
        ]

        self.assertFalse(plan_approved(calls, outcomes))
        self.assertFalse(plan_approved([{"name": "read_file"}], ["[plan] Approved. x"]))


if __name__ == "__main__":
    unittest.main()
