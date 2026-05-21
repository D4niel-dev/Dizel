"""
evaluation/runner.py — Run benchmarks against checkpoints.

Loads eval cases, executes them against agent handlers,
and produces scored EvalRun results.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .schema import EvalCase, EvalCategory, EvalRun
from .scorer import Scorer


# Agent handler: takes prompt string, returns response string
EvalHandler = Callable[[str], str]


class EvalRunner:
    """Loads and runs evaluation benchmarks."""

    def __init__(self, benchmarks_dir: Optional[str] = None):
        self._benchmarks_dir = benchmarks_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "benchmarks"
        )
        self._scorer = Scorer()

    def load_cases(self, category: Optional[EvalCategory] = None) -> List[EvalCase]:
        """Load eval cases from the benchmarks directory."""
        cases = []
        if not os.path.isdir(self._benchmarks_dir):
            return cases

        for fname in os.listdir(self._benchmarks_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._benchmarks_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    case = EvalCase.from_dict(item)
                    if category is None or case.category == category:
                        cases.append(case)
            except (json.JSONDecodeError, KeyError):
                continue
        return cases

    def run(self, handlers: Dict[str, EvalHandler],
            checkpoint: str = "",
            category: Optional[EvalCategory] = None) -> EvalRun:
        """
        Execute all matching eval cases and return a scored EvalRun.
        handlers: dict mapping agent name → callable(prompt) → response
        """
        cases = self.load_cases(category)
        run = EvalRun(checkpoint=checkpoint)

        for case in cases:
            handler = handlers.get(case.agent)
            if not handler:
                continue

            start = time.perf_counter()
            try:
                actual = handler(case.prompt)
            except Exception as e:
                actual = f"ERROR: {e}"
            elapsed_ms = (time.perf_counter() - start) * 1000

            result = self._scorer.score(case, actual, duration_ms=elapsed_ms)
            run.results.append(result)

        run.completed_at = datetime.utcnow()
        return run

    def run_single(self, case: EvalCase, handler: EvalHandler) -> dict:
        """Run a single case for quick testing."""
        start = time.perf_counter()
        try:
            actual = handler(case.prompt)
        except Exception as e:
            actual = f"ERROR: {e}"
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = self._scorer.score(case, actual, duration_ms=elapsed_ms)
        return result.to_dict()

    def save_run(self, run: EvalRun, output_dir: Optional[str] = None):
        """Persist an EvalRun to disk as JSON."""
        out_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results"
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"eval_{run.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)
        return path
