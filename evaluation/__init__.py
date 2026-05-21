"""
evaluation — Dizel Evaluation Infrastructure.

Benchmark testing, automated scoring, regression detection,
and report generation for model checkpoints.

Usage:
    from evaluation import EvalRunner, ReportGenerator

    runner = EvalRunner()
    run = runner.run({"dizel": my_handler}, checkpoint="v1.2")
    print(ReportGenerator().text_report(run))
"""

from .schema import EvalCase, EvalResult, EvalRun, EvalCategory, ScoringMethod
from .scorer import Scorer
from .runner import EvalRunner
from .regression import RegressionDetector, RegressionAlert
from .report import ReportGenerator


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "EvalCategory",
    "ScoringMethod",
    "Scorer",
    "EvalRunner",
    "RegressionDetector",
    "RegressionAlert",
    "ReportGenerator",
]
