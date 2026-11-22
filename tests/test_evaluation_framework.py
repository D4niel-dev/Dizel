"""Regression guards for the modernized evaluation framework.

The scorecard dict is consumed by the sidecar and both clients, so its key set is
part of the contract — this test asserts it rather than assuming it.
"""

from __future__ import annotations

import unittest

from evaluation import Scorer, ScoringMethod, suite
from evaluation.schema import EvalCase, EvalCategory, EvalResult, EvalRun


def _case(scoring=ScoringMethod.CONTAINS, expected="42", weight=1.0, **metadata):
    return EvalCase(
        prompt="p",
        expected=expected,
        category=EvalCategory.REASONING,
        scoring=scoring,
        weight=weight,
        metadata=metadata,
    )


class ScoringTests(unittest.TestCase):
    def test_existing_methods_keep_their_values(self):
        scorer = Scorer()
        self.assertEqual(scorer.score(_case(ScoringMethod.EXACT_MATCH, "42"), "42").score, 1.0)
        self.assertEqual(scorer.score(_case(ScoringMethod.CONTAINS, "42"), "is 42").score, 1.0)
        self.assertEqual(scorer.score(_case(ScoringMethod.SIMILARITY, "abc"), "abc").score, 1.0)
        self.assertAlmostEqual(scorer.score(_case(ScoringMethod.NUMERIC, "42"), "41.9").score, 0.9976, places=4)

    def test_registered_custom_scorer_is_used(self):
        Scorer.register_custom("double_check", lambda case, actual: 1.0 if "ok" in actual else 0.0)
        result = Scorer().score(
            _case(ScoringMethod.CUSTOM, expected="", custom_scorer="double_check"),
            "all ok",
        )
        self.assertEqual(result.score, 1.0)
        self.assertIsNone(result.error)

    def test_unregistered_custom_scorer_is_an_error_not_a_zero(self):
        result = Scorer().score(
            _case(ScoringMethod.CUSTOM, expected="", custom_scorer="missing"), "x"
        )
        self.assertEqual(result.score, 0.0)
        self.assertIn("unregistered", result.error or "")

    def test_manual_no_longer_counts_as_half_a_pass(self):
        result = Scorer().score(_case(ScoringMethod.MANUAL), "whatever")
        self.assertEqual(result.score, 0.0)
        self.assertIn("review", result.error or "")

    def test_case_weight_is_carried_to_the_result(self):
        result = Scorer().score(_case(weight=2.5), "42")
        self.assertEqual(result.weight, 2.5)


class AggregationTests(unittest.TestCase):
    def test_avg_score_is_weighted_but_pass_rate_is_not(self):
        run = EvalRun(
            results=[
                EvalResult("a", True, 1.0, "", "", "dizel", EvalCategory.REASONING, weight=3.0),
                EvalResult("b", False, 0.0, "", "", "dizel", EvalCategory.REASONING, weight=1.0),
            ]
        )
        self.assertAlmostEqual(run.avg_score, 0.75)
        self.assertEqual(run.pass_rate, 0.5)

    def test_timestamps_are_timezone_aware(self):
        result = Scorer().score(_case(), "42")
        run = EvalRun(results=[result])
        self.assertIsNotNone(result.timestamp.tzinfo)
        self.assertIsNotNone(run.started_at.tzinfo)


class ContractTests(unittest.TestCase):
    def test_core_suite_dict_keys_are_unchanged(self):
        card = suite.run_core_suite(lambda prompt: "answer 42", checkpoint="test")
        self.assertEqual(
            set(card),
            {
                "version", "suite", "suite_version", "checkpoint", "evaluated_at",
                "total", "passed", "pass_rate", "avg_score", "by_category",
                "tok_per_s", "cases",
            },
        )
        self.assertEqual(card["total"], 15)

    def test_runner_no_longer_has_the_second_store(self):
        from evaluation.runner import EvalRunner

        self.assertFalse(hasattr(EvalRunner, "save_run"))


if __name__ == "__main__":
    unittest.main()
