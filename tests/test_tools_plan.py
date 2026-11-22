"""A plan is a reviewed, multi-step proposal — never an auto-approved action."""

from __future__ import annotations

import unittest
from pathlib import Path

from dizel_core import tool_trust, tools


class PlanToolTests(unittest.TestCase):
    def _plan(self, args, approved=True):
        seen = []

        def approve(change):
            seen.append(change)
            return approved

        result = tools.execute({"name": "plan", "args": args}, approve=approve)
        return result, seen

    def test_plan_normalizes_numbered_steps_and_returns_them(self):
        result, changes = self._plan(
            {
                "task": "Fix the launcher",
                "steps": "1. Read the logs\n2) Patch the script\n- Run the smoke test",
            }
        )

        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertIsInstance(change, tools.PlanChange)
        self.assertEqual(change.task, "Fix the launcher")
        self.assertEqual(
            change.steps, ("Read the logs", "Patch the script", "Run the smoke test")
        )
        self.assertTrue(result.startswith("[plan] Approved."))
        self.assertIn("2. Patch the script", result)

    def test_plan_caps_steps_at_eight(self):
        steps = "\n".join(f"{i}. step {i}" for i in range(1, 12))

        _, changes = self._plan({"steps": steps})

        self.assertEqual(len(changes[0].steps), 8)
        self.assertEqual(changes[0].steps[-1], "step 8")

    def test_plan_needs_at_least_one_step(self):
        result = tools.execute(
            {"name": "plan", "args": {"steps": "  \n1.  \n"}},
            approve=lambda change: True,
        )
        self.assertIn("at least one step", result)

    def test_declined_plan_says_it_was_discarded(self):
        result, changes = self._plan({"steps": "1. Do it"}, approved=False)

        self.assertEqual(len(changes), 1)
        self.assertTrue(result.startswith("[plan] Plan discarded."))
        self.assertIn("Do not execute", result)

    def test_plan_requires_an_approval_channel(self):
        result = tools.execute({"name": "plan", "args": {"steps": "1. Do it"}})
        self.assertIn("cannot approve", result)

    def test_plan_never_auto_approves_under_full_trust(self):
        change = tools.PlanChange("Ship", ("Tag",))

        self.assertFalse(
            tool_trust.should_auto_approve(
                change,
                "session-1",
                session_grants={"session-1"},
                persistent=True,
                root=Path(".").resolve(),
            )
        )

    def test_plan_display_and_payload(self):
        call = {"name": "plan", "args": {"task": "Ship", "steps": "1. Tag\n2. Publish"}}

        self.assertEqual(tools.display_detail(call), "Plan · 2 steps")
        result = tools.execute(call, approve=lambda change: True)
        payload = tools.ui_payload(call, result)
        self.assertEqual(payload["steps"], ["Tag", "Publish"])
        self.assertEqual(payload["task"], "Ship")

    def test_plan_is_listed_only_when_approval_is_available(self):
        self.assertNotIn("plan", tools.catalog())
        self.assertIn("plan", tools.catalog(include_writes=True))


if __name__ == "__main__":
    unittest.main()
