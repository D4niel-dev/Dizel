"""
evaluation/regression.py — Detect regressions across versions.

Compares EvalRun results to find performance drops
between checkpoints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schema import EvalRun


@dataclass
class RegressionAlert:
    """A detected performance regression."""
    category: str
    old_score: float
    new_score: float
    delta: float
    severity: str       # "warning" | "critical"

    def __repr__(self) -> str:
        return (
            f"Regression({self.category}: {self.old_score:.2f}→{self.new_score:.2f} "
            f"Δ{self.delta:+.2f} [{self.severity}])"
        )


class RegressionDetector:
    """Compares eval runs to detect score regressions."""

    def __init__(self, warn_threshold: float = 0.05, critical_threshold: float = 0.15):
        self._warn = warn_threshold
        self._critical = critical_threshold

    def compare(self, baseline: EvalRun, current: EvalRun) -> List[RegressionAlert]:
        """Compare two runs and return regression alerts."""
        alerts = []

        old_scores = baseline.scores_by_category()
        new_scores = current.scores_by_category()

        all_categories = set(old_scores) | set(new_scores)

        for cat in sorted(all_categories):
            old = old_scores.get(cat, 0.0)
            new = new_scores.get(cat, 0.0)
            delta = new - old

            if delta < -self._critical:
                alerts.append(RegressionAlert(cat, old, new, delta, "critical"))
            elif delta < -self._warn:
                alerts.append(RegressionAlert(cat, old, new, delta, "warning"))

        # Overall pass rate comparison
        old_rate = baseline.pass_rate
        new_rate = current.pass_rate
        rate_delta = new_rate - old_rate

        if rate_delta < -self._critical:
            alerts.append(RegressionAlert(
                "overall_pass_rate", old_rate, new_rate, rate_delta, "critical"
            ))
        elif rate_delta < -self._warn:
            alerts.append(RegressionAlert(
                "overall_pass_rate", old_rate, new_rate, rate_delta, "warning"
            ))

        return alerts

    def has_regressions(self, baseline: EvalRun, current: EvalRun) -> bool:
        """Quick check: any regressions at all?"""
        return len(self.compare(baseline, current)) > 0

    def has_critical(self, baseline: EvalRun, current: EvalRun) -> bool:
        """Quick check: any critical regressions?"""
        alerts = self.compare(baseline, current)
        return any(a.severity == "critical" for a in alerts)

    def summary(self, baseline: EvalRun, current: EvalRun) -> str:
        """Human-readable regression summary."""
        alerts = self.compare(baseline, current)
        if not alerts:
            return f"✅ No regressions. Score: {current.avg_score:.2f} (was {baseline.avg_score:.2f})"

        lines = [f"⚠ {len(alerts)} regression(s) detected:"]
        for a in alerts:
            icon = "🔴" if a.severity == "critical" else "🟡"
            lines.append(f"  {icon} {a.category}: {a.old_score:.2f} → {a.new_score:.2f} ({a.delta:+.2f})")
        return "\n".join(lines)
