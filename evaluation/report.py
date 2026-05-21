"""
evaluation/report.py — Generate comparison reports from eval runs.

Produces human-readable and machine-readable evaluation reports.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from .schema import EvalRun


class ReportGenerator:
    """Generates evaluation reports in text and JSON formats."""

    def text_report(self, run: EvalRun) -> str:
        """Generate a human-readable text report."""
        lines = [
            "═" * 60,
            f"  DIZEL EVALUATION REPORT",
            f"  Checkpoint: {run.checkpoint or 'N/A'}",
            f"  Run ID: {run.id}",
            f"  Date: {run.started_at.strftime('%Y-%m-%d %H:%M')}",
            "═" * 60,
            "",
            f"  Total Cases:  {run.total}",
            f"  Passed:       {run.passed}",
            f"  Failed:       {run.failed}",
            f"  Pass Rate:    {run.pass_rate:.1%}",
            f"  Avg Score:    {run.avg_score:.4f}",
            "",
            "─" * 60,
            "  SCORES BY CATEGORY",
            "─" * 60,
        ]

        by_cat = run.scores_by_category()
        for cat, score in sorted(by_cat.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {cat:<22} {bar} {score:.2f}")

        lines.append("")
        lines.append("─" * 60)
        lines.append("  FAILED CASES")
        lines.append("─" * 60)

        failed = [r for r in run.results if not r.passed]
        if not failed:
            lines.append("  None — all cases passed! ✓")
        else:
            for r in failed[:20]:  # show first 20 failures
                lines.append(f"  [{r.category.value}] {r.case_id}")
                lines.append(f"    Expected: {r.expected_output[:80]}...")
                lines.append(f"    Got:      {r.actual_output[:80]}...")
                lines.append(f"    Score:    {r.score:.4f}")
                if r.error:
                    lines.append(f"    Error:    {r.error}")
                lines.append("")

        lines.append("═" * 60)
        return "\n".join(lines)

    def comparison_report(self, baseline: EvalRun, current: EvalRun) -> str:
        """Generate a side-by-side comparison of two runs."""
        lines = [
            "═" * 60,
            "  COMPARISON REPORT",
            f"  Baseline: {baseline.checkpoint or baseline.id}",
            f"  Current:  {current.checkpoint or current.id}",
            "═" * 60,
            "",
            f"  {'Metric':<22} {'Baseline':>10} {'Current':>10} {'Delta':>10}",
            "  " + "─" * 54,
            f"  {'Pass Rate':<22} {baseline.pass_rate:>10.1%} {current.pass_rate:>10.1%} {current.pass_rate - baseline.pass_rate:>+10.1%}",
            f"  {'Avg Score':<22} {baseline.avg_score:>10.4f} {current.avg_score:>10.4f} {current.avg_score - baseline.avg_score:>+10.4f}",
            "",
            "  " + "─" * 54,
            f"  {'Category':<22} {'Baseline':>10} {'Current':>10} {'Delta':>10}",
            "  " + "─" * 54,
        ]

        old_cats = baseline.scores_by_category()
        new_cats = current.scores_by_category()
        all_cats = sorted(set(old_cats) | set(new_cats))

        for cat in all_cats:
            old = old_cats.get(cat, 0.0)
            new = new_cats.get(cat, 0.0)
            delta = new - old
            indicator = "🔴" if delta < -0.1 else ("🟢" if delta > 0.05 else "⚪")
            lines.append(f"  {indicator} {cat:<20} {old:>10.2f} {new:>10.2f} {delta:>+10.2f}")

        lines.append("")
        lines.append("═" * 60)
        return "\n".join(lines)

    def save_json(self, run: EvalRun, output_dir: str) -> str:
        """Save a run as a JSON report."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"report_{run.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)
        return path
