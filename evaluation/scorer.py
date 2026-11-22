"""
evaluation/scorer.py — Automated scoring for eval cases.

Supports exact match, contains, fuzzy similarity, numeric and named custom scoring.
"""

import re
from difflib import SequenceMatcher
from typing import Callable, Dict, Optional

from .schema import EvalCase, EvalResult, EvalCategory, ScoringMethod

# Programmatic scorers for cases with ScoringMethod.CUSTOM. JSON cases select one
# with metadata["custom_scorer"] = name; callers register the callable at runtime.
_CUSTOM_SCORERS: Dict[str, Callable[[EvalCase, str], float]] = {}


class Scorer:
    """Scores evaluation results using the method specified in each case."""

    @staticmethod
    def register_custom(name: str, fn: Callable[[EvalCase, str], float]) -> None:
        _CUSTOM_SCORERS[str(name)] = fn

    def score(self, case: EvalCase, actual: str,
              duration_ms: float = 0.0) -> EvalResult:
        """Score a single case against the actual output."""
        try:
            score_val = self._compute_score(case, actual)
            passed = score_val >= case.tolerance
        except Exception as e:
            return EvalResult(
                case_id=case.id,
                passed=False,
                score=0.0,
                actual_output=actual,
                expected_output=case.expected,
                agent=case.agent,
                category=case.category,
                duration_ms=duration_ms,
                error=f"Scoring error: {e}",
            )

        return EvalResult(
            case_id=case.id,
            passed=passed,
            score=round(score_val, 4),
            actual_output=actual,
            expected_output=case.expected,
            agent=case.agent,
            category=case.category,
            duration_ms=duration_ms,
            weight=case.weight,
        )

    def _compute_score(self, case: EvalCase, actual: str) -> float:
        method = case.scoring
        expected = case.expected

        if method == ScoringMethod.EXACT_MATCH:
            return 1.0 if actual.strip() == expected.strip() else 0.0

        elif method == ScoringMethod.CONTAINS:
            return 1.0 if expected.lower() in actual.lower() else 0.0

        elif method == ScoringMethod.SIMILARITY:
            return self._similarity(expected, actual)

        elif method == ScoringMethod.NUMERIC:
            return self._numeric_match(expected, actual, case.tolerance)

        elif method == ScoringMethod.CUSTOM:
            name = str((case.metadata or {}).get("custom_scorer", ""))
            fn = _CUSTOM_SCORERS.get(name)
            if fn is None:
                raise ValueError(f"unregistered custom scorer: {name or '(none)'}")
            return max(0.0, min(1.0, float(fn(case, actual))))

        elif method == ScoringMethod.MANUAL:
            # A silent 0.5 counted as a half-pass and inflated pass rates.
            raise ValueError("manual scoring requires review")

        return 0.0

    @staticmethod
    def _similarity(expected: str, actual: str) -> float:
        """Fuzzy string similarity using SequenceMatcher."""
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()

    @staticmethod
    def _numeric_match(expected: str, actual: str, tolerance: float) -> float:
        """Check if numeric values match within tolerance."""
        try:
            exp_nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", expected)]
            act_nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", actual)]
            if not exp_nums or not act_nums:
                return 0.0
            # Compare first number found
            diff = abs(exp_nums[0] - act_nums[0])
            max_val = max(abs(exp_nums[0]), 1.0)
            relative_error = diff / max_val
            return max(0.0, 1.0 - relative_error)
        except (ValueError, IndexError):
            return 0.0
