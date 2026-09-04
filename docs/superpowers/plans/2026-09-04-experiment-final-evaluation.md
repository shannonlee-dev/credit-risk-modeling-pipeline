# Experiment / Final Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate Train-only experiments from selection-driven untouched-Holdout evaluation while preserving existing entry points.

**Architecture:** Core builders and pure evaluation remain independent of artifacts. Experiment modules accept only Train data and profiles; final evaluators accept resolved selection plus Train/Holdout data. Workflow and reporting own stage orchestration and serialization.

**Tech Stack:** Python 3.10, pandas, NumPy, scikit-learn, matplotlib, pytest.

**Spec:** `docs/experiment_final_evaluation_architecture.md`

## Global Constraints

- Preserve the generator, 10,000-row dataset, `RANDOM_STATE=42`, 80:20 splits, five folds, Pipeline preprocessing, metrics, rule baseline, and batch benchmark meaning.
- Experiments must not receive Holdout data; final evaluation must not run CV, GridSearchCV, OOF prediction, threshold sweeps, or sensitivity analyses.
- Only reporting writes artifacts. No external dependencies, generic ML framework, nested CV, or automatic threshold policy.
- Preserve `python train.py`, `run_analysis`, `run_classification`, and `run_regression` as delegating compatibility paths.

---

### Task 1: Pure evaluation and result contracts

**Files:** Create `credit_risk/evaluation.py`, `credit_risk/results.py`; modify `tests/test_train.py`.

**Produces:** `apply_threshold`, threshold table, metric helpers, experiment/final selection dataclasses.

- [ ] Write tests that a literal score vector produces threshold labels and TP/FP/FN/precision/recall/F1 without presentation columns, and that incomplete final selections reject.
- [ ] Run `uv run pytest tests/test_train.py -k 'threshold or selection' -v`; confirm missing-module failure.
- [ ] Implement the pure helpers and dataclass validation/conversion.
- [ ] Re-run the focused tests; confirm pass.

### Task 2: Core builders and Train-only experiment functions

**Files:** Modify `credit_risk/classification.py`, `credit_risk/regression.py`; create `credit_risk/experiments/{__init__.py,config.py,thresholds.py,classification.py,regression.py}`; modify tests.

**Produces:** Full/smoke profiles and experiment functions accepting only `x_train, y_train, config`.

- [ ] Write tests for profile candidate differences and experiment result fields without Holdout metrics/predictions.
- [ ] Run focused tests; confirm import/behavior failure.
- [ ] Extract builders and move current Train CV/OOF/sensitivity/coefficient calculations into experiment modules with unchanged candidate values.
- [ ] Re-run focused tests; confirm pass.

### Task 3: Selection-driven final evaluators

**Files:** Modify `credit_risk/classification.py`, `credit_risk/regression.py`, `credit_risk/evaluation.py`; modify tests.

**Produces:** Final classification/regression evaluation from a validated selection.

- [ ] Write tests that final results use literal selected thresholds/alphas and monkeypatch experiment primitives to raise without affecting final evaluation.
- [ ] Run focused tests; confirm failure.
- [ ] Implement full-Train fit and one Holdout prediction/evaluation with no search/OOF calls.
- [ ] Re-run focused tests; confirm pass.

### Task 4: Provenance, reporting, and two-stage workflow

**Files:** Modify `credit_risk/data.py`, `credit_risk/workflow.py`, `credit_risk/reporting.py`; modify tests.

**Produces:** dataset/protocol IDs, selection-template validation, separated experiment/final artifacts and `metrics.json`.

- [ ] Write tests for fingerprint mismatch rejection and stage-specific output paths.
- [ ] Run focused tests; confirm failure.
- [ ] Implement provenance, JSON serialization, and workflow entry points.
- [ ] Re-run focused tests; confirm pass.

### Task 5: Compatibility facade and CLI

**Files:** Modify `credit_risk/workflow.py`, `train.py`, `README.md`, `docs/model_selection.md`; modify tests.

**Produces:** `experiment`, `final`, `all`, bare-CLI alias, and legacy facades delegating to new paths.

- [ ] Write subprocess/API tests for bare CLI and stage commands using a small generated dataset.
- [ ] Run focused tests; confirm failure.
- [ ] Implement CLI parsing and compatibility conversions without a legacy computation path; update docs and artifact references.
- [ ] Run focused tests and `uv run pytest -v`; confirm pass.

### Task 6: Final review

**Files:** All changed files.

- [ ] Run `uv run pytest -v`, `uv run python train.py --help`, and `git diff --check`.
- [ ] Inspect the diff for accidental artifact writes, duplicate logic, or changed statistical constants.
