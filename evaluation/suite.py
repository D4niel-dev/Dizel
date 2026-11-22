"""Run the fixed core benchmark suite into a stored scorecard record."""

from __future__ import annotations

from datetime import datetime

from .runner import EvalRunner

CORE_SUITE = "core"
CORE_SUITE_VERSION = 1


def run_core_suite(handler, checkpoint: str = "") -> dict:
    """Score `handler(prompt) -> text` over the 15-case core suite."""
    runner = EvalRunner()
    run = runner.run({"dizel": handler}, checkpoint=checkpoint)
    tokens = sum(max(1, len(result.actual_output or "") // 4) for result in run.results)
    seconds = sum(result.duration_ms for result in run.results) / 1000.0
    return {
        "version": 1,
        "suite": CORE_SUITE,
        "suite_version": CORE_SUITE_VERSION,
        "checkpoint": checkpoint,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "total": run.total,
        "passed": run.passed,
        "pass_rate": round(run.pass_rate, 4),
        "avg_score": round(run.avg_score, 4),
        "by_category": {
            name: round(score, 4) for name, score in run.scores_by_category().items()
        },
        "tok_per_s": round(tokens / seconds, 1) if seconds else 0.0,
        "cases": [
            {
                "id": result.case_id,
                "category": result.category.value,
                "passed": result.passed,
                "score": result.score,
                "duration_ms": round(result.duration_ms),
            }
            for result in run.results
        ],
    }
