# Evaluation framework modernization — design spec

**Status:** Draft (2026-11-22). Direction chosen: modernize in place. Awaiting spec review.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** `evaluation/` package, benchmark tracking, scoring gaps, deprecations and docs. No change to the live scorecard contract.

---

## 1. Summary

`evaluation/` works — `api/services/scorecard_service.py` and `scripts/run_scorecard.py` both call `run_core_suite()` — but it has been untouched since the deleted PySide6 GUI era. The concrete rot:

- `evaluation/suite.py` is **untracked** and `evaluation/benchmarks/` is **gitignored**, so a fresh checkout has no cases.
- `evaluation/__init__.py:11` still uses `checkpoint="v1.2"`.
- `ScoringMethod.CUSTOM` falls through to `0.0`; `MANUAL` silently returns `0.5`, which inflates pass rates.
- `EvalCase.weight` is declared but never used.
- `datetime.utcnow()` is deprecated.
- `EvalRunner.save_run()` writes to a second, unused store while the live path already persists through `dizel_core.scorecards` under `data/scorecards/`.

This modernizes the implementation without changing the scorecard dict consumed by the sidecar.

---

## 2. Goals and non-goals

### Goals

1. Benchmark cases and the suite are versioned; a fresh checkout can run them.
2. The package documents the shipped model, not v1.2.
3. Every `ScoringMethod` has honest behaviour: CUSTOM is implementable, MANUAL cannot silently look like a half-pass.
4. Case `weight` affects the aggregate score.
5. No deprecated datetime calls.
6. One persistence path — the existing `scorecards` store.
7. Tests cover scoring and aggregation; README documents the run command.

### Non-goals

- No new benchmark cases and no change to the 15 toy prompts. RFC-0008 still forbids treating them as a release gate.
- No release-gate machinery, no regression alerting.
- No change to `run_core_suite()`'s returned dict keys — the sidecar and both clients depend on them.
- No change to `scripts/run_scorecard.py`'s CLI.
- No deletion of `regression.py` / `report.py`; they stay as clearly-labelled experimental scaffolding.

---

## 3. Design

### 3.1 Versioning and tracking

- `.gitignore`: remove `evaluation/benchmarks/` **and** add an explicit `!evaluation/benchmarks/core_benchmarks.json` exception after the global `*.json` rule — removing the directory rule alone left the file ignored. No `evaluation/results/` path remains (see 3.4).
- `evaluation/__init__.py`: docstring example uses `checkpoint="Dizel-1.5-Distill"` and states the shipped suite is the 15-case core suite.
- `evaluation/suite.py` is already correct but untracked; with the ignore removed, both it and the benchmark JSON are pickable up by the next commit. No staging or commit in this change.

### 3.2 Scoring

In `scorer.py`:

- `Scorer.register_custom(name, fn)` adds a named callable to a module registry. A case opts in with `metadata["custom_scorer"] = name`. Missing or bad registration produces a scoring error through the existing `EvalResult.error` path — never a silent `0.0`.
- `MANUAL` produces `score=0.0` with `error="manual scoring requires review"`. A half-credit placeholder that counts as a pass is worse than an explicit error.
- Keep EXACT_MATCH / CONTAINS / SIMILARITY / NUMERIC behaviour identical.

In `schema.py`:

- `EvalResult` gains `weight: float = 1.0`, populated by `Scorer.score()` from the case.
- `EvalRun.avg_score` becomes the weighted mean. `pass_rate` stays a plain count ratio — mixing weights into both would make the two numbers disagree in a confusing way.

### 3.3 Datetime

Replace `datetime.utcnow()` with `datetime.now(timezone.utc)` in `schema.py` and `runner.py`. ISO strings gain `+00:00`; all current readers parse or display them as strings, so no consumer change is expected. Tests will confirm.

### 3.4 Persistence

Delete `EvalRunner.save_run()` and its unused `results/` path. `scripts/run_scorecard.py` already calls `scorecards.save()`, which fingerprints the checkpoint and stores under `data/scorecards/`. Keeping a second store invites drift.

### 3.5 Docs

README: add a short "Evaluation" subsection with the run command, where the scorecard is stored, and the RFC-0008 warning that the 15 cases are not a release gate.

---

## 4. Files

| File | Change |
|---|---|
| `.gitignore` | Un-ignore `evaluation/benchmarks/`. |
| `evaluation/__init__.py` | Current model in the example and usage docstring. |
| `evaluation/scorer.py` | Custom scorer registry; honest MANUAL; set result weight. |
| `evaluation/schema.py` | `EvalResult.weight`, weighted `avg_score`, timezone-aware datetimes. |
| `evaluation/runner.py` | Timezone-aware completion timestamp; remove `save_run()`. |
| `tests/test_evaluation_framework.py` (new) | Scoring methods, weights, deprecation-free timestamps. |
| `README.md` | Evaluation run instructions and RFC warning. |

---

## 5. Data flow

```
benchmarks/core_benchmarks.json  (now versioned)
   → EvalRunner.load_cases()
   → Scorer.score(case, actual)   ← custom registry / honest manual
   → EvalResult(weight=case.weight)
   → EvalRun.avg_score (weighted)
   → run_core_suite() dict        ← unchanged shape
   → scorecards.save()            ← existing single store
```

---

## 6. Error handling

- An unregistered custom scorer name surfaces through `EvalResult.error`; the run continues.
- A handler exception is still captured as `ERROR: …` text and scored normally.
- Missing benchmark cases still raise, so a bad checkout fails loudly — with the folder versioned this becomes a real check instead of a fresh-install surprise.

---

## 7. Testing

**`tests/test_evaluation_framework.py`:**

1. EXACT_MATCH / CONTAINS / SIMILARITY / NUMERIC produce today's values (regression guard).
2. A registered custom scorer is used; an unregistered name produces an error result.
3. MANUAL produces `score=0.0` and a review error, not 0.5.
4. Weighted `avg_score` differs from the unweighted mean when weights differ; `pass_rate` stays unweighted.
5. Timestamps are timezone-aware.
6. `run_core_suite()` still returns the exact key set the sidecar reads.

Also run `tests/test_scorecards.py`, `tests/test_scorecard_service.py`, and `tests/test_tui_scorecard.py`.

---

## 8. Decisions

- **No new cases.** The user asked to modernize scripts, not to redesign the benchmark.
- **Scorecard contract frozen.** The sidecar, TUI and GUI all read the same dict.
- **One store.** Delete `save_run()` rather than maintain two.
- **MANUAL fails loudly.** Silent half-credit is a false signal.
- **Keep dead modules for now.** Removing `regression.py`/`report.py` is a separate, reversible cleanup; labelling is enough here.
