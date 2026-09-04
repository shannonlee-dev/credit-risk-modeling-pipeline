# Experiment / Final Evaluation Architecture Refactor

## 1. Pre-refactor architecture summary

현재 파이프라인은 재현 가능하고 전처리 누수 방지라는 중요한 성질을 갖지만, Train-only model selection과 untouched holdout 평가가 한 번의 `train_classification()`/`train_regression()` 호출에 결합되어 있다. `workflow.run_analysis()`는 이 혼합 결과를 단일 `artifacts/`에 저장한다. 따라서 통계적 순서는 현재도 대체로 올바르지만, 함수 signature·artifact·CLI만으로는 두 lifecycle을 구분할 수 없다.

`data_gen.py`는 저장소에 없다. 실제 데이터 생성 진입점은 `scripts/generate_data.py`이며 `N_SAMPLES = 10_000`, `RANDOM_STATE = 42`다.

## 2. Pre-refactor call/data flow

```text
python train.py
  -> workflow.run_analysis(data_path, output_dir, fast=False)
     -> data.load_and_validate_data()
     -> data.split_classification_data()   # 80:20, stratified, random_state=42
     -> data.split_regression_data()       # 80:20, random_state=42
     -> workflow.run_classification(..., x_test, y_test)
        -> classification.train_classification(..., x_test, y_test)
           -> Train CV / GridSearch / Train OOF threshold sweep
           -> full-Train fit -> Holdout scores, labels, metrics
        -> reporting.save_classification_artifacts(result, y_test, output_dir)
     -> workflow.run_regression(..., x_test, y_test)
        -> regression.train_regression(..., x_test, y_test)
           -> Train CV alpha scan -> full-Train fit -> Holdout metrics
        -> reporting.save_regression_artifacts(result, y_test, output_dir)
     -> legacy root metrics serialization (removed)
```

- `classification.train_classification()` constructs Logistic and RF `Pipeline`s, fits Logistic/RF on Train before and after selection as needed, runs Logistic CV and RF `GridSearchCV` on Train, generates OOF probabilities with `cross_val_predict`, then predicts the Holdout. Rule predictions are created directly from `x_test`.
- `_threshold_sweep()` consumes `y_train` and Train OOF scores, but adds `is_baseline` and `is_selected` presentation flags. Chosen thresholds (`0.45`, `0.33`) are module constants and are applied to Holdout in the same function.
- `_random_forest_saturation_analysis()` creates Train-CV sensitivity rows, additionally fitting candidates on full Train to time a Train feature batch. `_feature_importance()` reads the tuned RF fitted on full Train.
- `regression.train_regression()` loops alpha candidates; each candidate gets Train CV RMSE and a full-Train fit for coefficient paths. It then fits the two selected models on full Train and predicts Holdout.
- Only reporting writes files today. This is a property to retain.

## 3. Problems / mixed responsibilities

`classification.py` currently combines core model construction, CV selection, OOF analysis, human decision constants, final Holdout evaluation, latency experiments, feature interpretation, DataFrame assembly, and a large nested return contract. It receives Holdout data even though its CV/OOF work must not use it; a future edit can accidentally blur that boundary.

`regression.py` similarly interleaves pipeline construction, alpha search, coefficient sensitivity, final fit, Holdout metrics, and prediction tables. `reporting.py` is I/O-only but its `save_classification_artifacts()` consumes both experiment and final fields, forcing one result object and one output directory. `metrics.json` serializes model-selection evidence, Holdout metrics, and benchmark data under a name suggesting final metrics only.

## 4. Statistical invariants

The refactor preserves exactly:

- generator formula, 10,000 rows, and `random_state=42`;
- classification 80:20 stratified split and regression 80:20 split;
- preprocessing inside sklearn `Pipeline`, so imputation/scaling fit only on Train or CV Train folds;
- 5-fold Train-only Logistic C selection, RF `GridSearchCV`, RF sensitivity, OOF threshold evidence, and Ridge/Lasso alpha selection;
- current metrics, rule baseline, and batch benchmark semantics;
- one Holdout evaluation after all selection; Holdout labels never select a model, hyperparameter, alpha, or threshold.

The existing limitation remains explicit: selected hyperparameters are chosen with all-Train CV, then OOF scores for that selected setting are regenerated on the same Train. This is not nested OOF, but it is not Holdout leakage. Nested CV is out of scope.

## 5. Documentation drift

Code and tests are authoritative. The current committed artifacts match current filenames and current default result schema; prediction CSVs are ignored by `.gitignore`. README accurately describes `fast=True` as a reduced candidate set while retaining five folds.

One substantive drift exists: `docs/model_selection.md` says the RF search fixes `n_estimators=100` and describes the depth selection as an initial `max_depth=8` choice, while current code searches `max_depth in [None, 8, 16]` together with `min_samples_split`. The documentation must be corrected during migration. `README.md` also intentionally describes the old, mixed artifact layout and must be updated for the new workflow.

## 6. Three refactoring options

| Option | Design | Advantages | Costs / risk |
| --- | --- | --- | --- |
| A. Minimal extraction | Keep `classification.py`/`regression.py`; move a few CV helpers to one `experiments.py`. | Lowest file churn. | Core modules still know candidate grids and selection; final boundary remains weak; reporting still needs mixed dictionaries. |
| B. Thin core + `experiments/` package | Keep model builders/rule baseline in core; add `evaluation.py`, `results.py`, and focused experiment modules. | Makes Train-only signatures and final-only evaluators obvious; enough structure for two stages without framework machinery. | Moderate migration and result-contract test changes. |
| C. Explicit models/selection/evaluation layers | Introduce separate model, selection, policy, artifact service layers. | Strong formal separation. | Too many wrappers/files for this portfolio project; high overengineering and compatibility cost. |

## 7. Recommended option

Choose **Option B**. It makes the statistical boundary structural: experiment functions cannot receive Holdout arguments; final evaluators receive a resolved selection and no experiment config. It leaves classifiers and regressors as recognizably sklearn-focused modules and avoids generic base classes, strategy objects, or dependency injection.

## 8. Proposed file tree

```text
credit_risk/
  constants.py                 # invariants, names, default paths
  data.py                      # load/validate/split/fingerprint
  preprocessing.py
  classification.py            # builders, rule baseline, compatibility facade
  regression.py                # builders, compatibility facade
  evaluation.py                # pure scoring and threshold application
  results.py                   # small dataclass contracts and JSON conversion
  experiments/
    __init__.py
    config.py                  # full/smoke candidate profiles
    thresholds.py              # pure Train-OOF threshold table
    classification.py          # Train-only CV, OOF, RF analysis, benchmark
    regression.py              # Train-only alpha/coefficients
  workflow.py                  # stage orchestration and compatibility facade
  reporting.py                 # only JSON/CSV/PNG serialization
train.py                       # all / experiment / final CLI
```

## 9. Responsibility of every affected module

`classification.py` supplies `build_logistic_classifier()`, `build_random_forest_classifier()`, `build_decision_tree_classifier()` if retained, and `rule_based_predict()`. `regression.py` supplies Ridge/Lasso builders. Neither imports experiment configuration, reporting, or artifact paths.

`evaluation.py` owns `apply_threshold`, classification metrics, and regression metrics for already-created predictions. `experiments/*` imports core builders, preprocessing through builders, and evaluation threshold helpers; it has no reporting imports. `results.py` is dependency-light contracts/validation. `workflow.py` loads/splits, calls stage computations, resolves selections, and asks reporting to persist their results. `reporting.py` imports results/constants only; it never trains a model.

## 10. Dependency diagram

```text
constants, data, preprocessing
          |             |
          v             v
classification, regression ---> evaluation <--- experiments.config
          ^                         ^                 |
          |                         +--- experiments -+
          +--------------------------------------------+
                              |
                           workflow
                              |
                           reporting
```

No edge points from core or experiments to reporting, nor from final evaluation to experiment configuration. This avoids circular imports and prevents `workflow.py` from implementing CV, grid search, or threshold logic.

## 11. Experiment / human selection / final evaluation data flow

```text
Experiment
load + deterministic split
 -> classification experiment(x_class_train, y_class_train, profile)
 -> regression experiment(x_reg_train, y_reg_train, profile)
 -> experiment artifacts + selection.template.json
 -> exit (no Holdout metric, prediction, ROC, or confusion matrix)

Human
review CV / OOF / sensitivity / coefficients
 -> copy template to selection.json
 -> choose classifier, threshold, and RF tree count where applicable

Final
load + same deterministic split
 -> validate data + protocol provenance against selection
 -> selected full-Train fit
 -> one untouched Holdout prediction/evaluation
 -> final artifacts
```

`run_analysis()` remains a convenience facade: run experiment, resolve the repository-approved deterministic default selection, then call the same final evaluator. It never duplicates old logic.

## 12. Function-by-function migration map

| Current | Proposed | Reason |
| --- | --- | --- |
| `rule_based_predict` | `classification.py` | Core baseline. |
| `evaluate_classifier` | `evaluation.evaluate_classification` | Pure scoring after predictions exist. |
| `_average_latency_ms` | `experiments.classification.average_latency_ms` | Comparative experiment benchmark. |
| `_feature_importance` | `experiments.classification.feature_importance` | RF interpretation evidence. |
| `_threshold_sweep` | `evaluation.evaluate_thresholds` | Pure table; remove annotation flags. |
| `_random_forest_saturation_analysis` | `experiments.classification.analyze_rf_tree_counts` | Train-only sensitivity. |
| `_analyze_logistic_c_values` | `experiments.classification.analyze_logistic_c_values` | Train-only selection. |
| `_selected_threshold_metrics` | experiment-result summary helper | Derives evidence, not final decision. |
| `train_classification` | compatibility facade over new experiment + final functions | Preserve semipublic imports without duplicate implementation. |
| `_regularized_pipeline` | `regression.build_regularized_regressor` | Core construction. |
| `train_regression` | compatibility facade | Preserve current call pattern. |
| `run_classification`, `run_regression` | workflow compatibility facades | Retain public usage while delegating. |
| `run_analysis` | `run_experiment` + default-selection resolution + `run_final_evaluation` | Existing no-argument behavior becomes `all`. |
| `DEFAULT_CLASSIFICATION_THRESHOLD` | `constants.py` baseline annotation only | Not a selected operating decision. |
| selected thresholds / model | `FinalSelection` / `selection.json` | Human operating decisions. |
| Logistic/RF/alpha candidates, fast lists | `experiments.config.ExperimentProfile` | Candidate scope belongs to experiments. |
| `REGRESSION_ALPHAS` | full profile (or invariant default consumed by it) | Candidate list, not final selection. |
| `fast=True` | `SMOKE_EXPERIMENT` mapping at facade boundary | Prevent boolean propagation through new APIs. |

## 13. Proposed internal/public APIs

Pseudo-signatures intentionally show data access boundaries:

```python
def build_logistic_classifier(c: float) -> Pipeline: ...
def build_random_forest_classifier(
    n_estimators: int, max_depth: int | None, min_samples_split: int
) -> Pipeline: ...
def build_regularized_regressor(model_name: str, alpha: float) -> Pipeline: ...

def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray: ...
def evaluate_thresholds(
    y_true: pd.Series, scores: np.ndarray, thresholds: Sequence[float]
) -> pd.DataFrame: ...
def evaluate_classification(y_true, predictions, scores) -> ClassificationMetrics: ...
def evaluate_regression(y_true, raw_predictions) -> RegressionMetrics: ...

def run_classification_experiment(
    x_train: pd.DataFrame, y_train: pd.Series,
    config: ClassificationExperimentConfig,
) -> ClassificationExperimentResult: ...
def run_regression_experiment(
    x_train: pd.DataFrame, y_train: pd.Series,
    config: RegressionExperimentConfig,
) -> RegressionExperimentResult: ...

def evaluate_final_classification(
    x_train, y_train, x_holdout, y_holdout, selection: FinalSelection
) -> FinalClassificationResult: ...
def evaluate_final_regression(
    x_train, y_train, x_holdout, y_holdout, selection: FinalSelection
) -> FinalRegressionResult: ...

def run_experiment(data_path, output_dir, profile: ExperimentProfile) -> ExperimentResult: ...
def run_final_evaluation(data_path, selection_path, output_dir) -> FinalEvaluationResult: ...
def run_analysis(data_path=..., output_dir=..., fast=False) -> dict: ...
```

Experiment APIs have no `x_holdout`/`y_holdout`; final APIs receive no candidate profile. Final classification may fit the selected Logistic and selected tuned RF only if the existing comparison contract requires both; `selection.classification_model` determines the primary reported selection, while no unselected hyperparameter search is performed.

## 14. Config/profile strategy

Use three frozen, small configs: `ClassificationExperimentConfig`, `RegressionExperimentConfig`, and `ExperimentProfile(name, classification, regression)`. Full preserves all existing lists; smoke preserves the existing reduced C, `min_samples_split`, and tree-count candidates. The profile also includes the threshold candidate grid. Constants retain split/CV/random-state/schema values.

## 15. Result dataclass strategy

Use only five result contracts: `ClassificationExperimentResult`, `RegressionExperimentResult`, `ExperimentResult`, `FinalSelection`, and `FinalEvaluationResult`. Their nested metrics may remain dictionaries/DataFrames where that is clearer. Dataclasses should provide explicit `to_dict()`/`from_dict()` conversions; DataFrames are written by reporting, not JSON-encoded as nested records by default.

## 16. Selection contract / JSON strategy

`selection.template.json` is generated from the experiment result. CV-selected C, RF `max_depth`, RF `min_samples_split`, Ridge alpha, and Lasso alpha are filled. Human choices remain `null`: `selected_model`, the selected-model threshold, and RF `n_estimators` when RF is selected. Allow optional logistic/RF thresholds to record considered choices, but validation requires exactly the selected model's threshold.

```json
{
  "schema_version": 1,
  "experiment_id": "sha256:...",
  "dataset_fingerprint": "sha256:...",
  "protocol_fingerprint": "sha256:...",
  "classification": {
    "selected_model": null,
    "logistic_regression": {"C": 0.01, "threshold": null},
    "random_forest": {
      "n_estimators": null, "max_depth": 8,
      "min_samples_split": 40, "threshold": null
    }
  },
  "regression": {"ridge_alpha": 1.0, "lasso_alpha": 0.1}
}
```

The repository-approved policy used by `all` lives in `credit_risk.selection.PROJECT_DEFAULT_SELECTION`. Its model, thresholds, and RF tree count are combined explicitly with experiment-derived hyperparameters and provenance, then persisted as `experiment/selection.json`. The compatibility path then passes that same validated selection to the final evaluator without reloading it from disk.

## 17. Dataset fingerprint / experiment provenance

`data.dataset_fingerprint(path)` computes SHA-256 over raw CSV bytes. `protocol_fingerprint(profile)` computes SHA-256 over canonical JSON containing schema version, `RANDOM_STATE`, split size/stratification, `CV_FOLDS`, and all profile candidate lists. `experiment_id` is SHA-256 over the dataset fingerprint plus protocol fingerprint.

`experiment.json` and its template carry all three. Final loading recomputes the current dataset and protocol fingerprints and rejects a selection mismatch before fitting. This detects changed data, a changed profile, and changed split/CV protocol without MLflow/DVC. A legacy `all` call constructs its selection from the matching just-created experiment result, so it cannot mismatch.

## 18. Artifact directory strategy

```text
artifacts/
  experiment/
    experiment.json
    selection.template.json
    logistic_threshold_sweep.csv / .png
    random_forest_threshold_sweep.csv / .png
    random_forest_grid_search.csv
    random_forest_n_estimators_curve.png
    feature_importance.png
    regularization_coefficients.png
    regression_alpha_cv.csv
  final/
    metrics.json
    confusion_matrix.png
    roc_curve.png
    classification_predictions.csv   # ignored
    credit_score_predictions.csv     # ignored
```

## 19. `experiment.json` strategy

`experiment.json` contains provenance, profile name, selected CV hyperparameters, concise CV summaries, OOF operating-point summaries, and benchmark metadata. Detailed grid/sweep/alpha rows belong in CSV. It includes no Holdout values.

## 20. Slim `final/metrics.json` strategy

`final/metrics.json` contains dataset summary, validated selection, primary final classification metrics (including FP/FN), all retained final regression metrics/alphas, and no CV/grid/OOF/coefficient/benchmark detail. The batch benchmark is an experiment artifact because it is comparative evidence, not serving latency. RF importance likewise remains experiment evidence even when Logistic is final.

Reporting adds baseline/selected markers to a copy of pure threshold tables for CSV/plot rendering. Computation does not know `is_baseline`, `is_selected`, paths, JSON, CSV, or PNG.

## 21. Backward compatibility

Keep `python train.py`, `run_analysis`, `run_classification`, `run_regression`, and direct current imports during the first migration. Existing tests directly import only `rule_based_predict`, `train_classification`, `train_regression`, and workflow functions; none establish `train.py` re-exports. Do not invent re-exports.

## 22. CLI UX

CLI parsing should support:

```text
python train.py                 # alias: all
python train.py all [--profile full|smoke]
python train.py experiment [--profile full|smoke]
python train.py final --selection artifacts/experiment/selection.json
```

`experiment` writes only experiment artifacts. `final` requires a complete, matching selection and writes only final artifacts. `all` uses the approved default selection after its experiment. README should label `experiment -> human review -> final` as recommended and `all` as deterministic convenience reproduction.

## 23. Testing strategy

Retain generator reproducibility, schema/split, preprocessor, rule baseline, no-write computation, matplotlib import, and legacy facade tests. Split today's broad `test_classification_compares_models_and_saves_artifacts`, regression, and `run_analysis` tests by contract:

- pure tests: threshold table counts/metrics without annotations, threshold application, classification metrics, regression metrics and clipping;
- experiment boundary tests: experiment signatures/results have no Holdout input; monkeypatch `GridSearchCV`, `cross_validate`, and `cross_val_predict` to prove final evaluation never invokes them; pipelines retain preprocessing;
- final tests: selected parameters/threshold are used for full-Train fit and Holdout prediction, with final schema only;
- selection/provenance tests: required human fields, bad model/threshold/params, dataset mismatch, and protocol mismatch reject before fitting;
- I/O tests: computation writes nothing, only reporting writes, experiment creates no final ROC/confusion/prediction files, and final creates no experiment sweep/grid files;
- compatibility/reproducibility tests: no-command CLI, `run_analysis(..., fast=True)` maps to smoke, full/smoke candidates, old wrappers, ignored prediction CSVs, and final metrics agreement with its returned Holdout metrics.

## 24. Incremental migration plan

| Step | Change | Compatibility/statistical guard | Required test / artifact effect |
| --- | --- | --- | --- |
| 1 | Extract `evaluation.py` and pure threshold calculation. | No behavior change; retain temporary wrappers. | Existing classification tests plus new pure tests; no artifact change. |
| 2 | Add profile/config and map `fast=True` at facades. | Candidate values exact. | Full/smoke candidate tests. |
| 3 | Add Train-only classification experiment result/function. | No Holdout argument. | CV/OOF tests; no files. |
| 4 | Add final classification evaluator. | No CV/search/sweep call. | Monkeypatch boundary and Holdout metric tests. |
| 5 | Split regression experiment/final evaluator. | Alpha CV stays Train-only. | CV and final regression tests. |
| 6 | Introduce result contracts and selection validation. | Preserve legacy dictionary conversion. | Schema/validation tests. |
| 7 | Add provenance/fingerprints. | Reject before fitting. | mismatch tests. |
| 8 | Split reporting and artifact paths. | Preserve ignored prediction policy. | stage-specific artifact tests. |
| 9 | Add CLI subcommands and make facades delegate. | Bare CLI remains `all`. | subprocess/legacy tests. |
| 10 | Update README/docs/artifacts and remove temporary internal compatibility only after callers migrate. | Correct RF depth documentation. | full suite and artifact-contract check. |

## 25. Risks / trade-offs

The largest compatibility risk is callers/tests relying on the current large dictionaries and root artifact paths. Address it with facade return conversion in the first release, deprecation notes in README, and a deliberate later removal rather than two computation paths. The largest statistical risk is accidentally treating OOF evidence as a final metric; separate types, signatures, and directories prevent that.

## 26. Things deliberately not abstracted

Do not add an abstract experiment base class, model strategy/factory hierarchy, repository/service/DI layer, generic metric registry, plugin system, MLflow/DVC, serving/model registry, auto-threshold policy, or nested-CV framework. These add indirection without improving this project's concrete Train-selection/Holdout boundary.

## 27. Before / after workflow comparison

| Current | Target |
| --- | --- |
| `run_analysis` calls one classifier function that both selects and evaluates. | `run_experiment` performs Train-only evidence; `run_final_evaluation` consumes a validated selection. |
| Threshold constants are embedded in classification computation. | Human decision lives in `selection.json`; pure threshold calculation has no decision annotation. |
| One `metrics.json` mixes CV, OOF, Holdout, and timing. | `experiment/experiment.json` summarizes evidence; `final/metrics.json` contains Holdout results only. |
| Root artifact directory mixes selection plots and final plots. | `experiment/` and `final/` communicate lifecycle. |
| Bare CLI executes the only path. | Bare CLI aliases deterministic `all`; recommended flow explicitly separates experiment, review, and final. |

## 28. Implementation approval checklist

- [ ] Preserve every invariant in section 4 and the documented non-nested-OOF limitation.
- [ ] Make Holdout data impossible to pass to experiment APIs and experiment configuration unavailable to final evaluators.
- [ ] Keep calculations file-free; reporting is the only writer.
- [ ] Preserve legacy entry points as delegating facades before changing public result shapes.
- [ ] Require fingerprint/protocol validation for external `selection.json`.
- [ ] Confirm experiment produces no final artifacts and final runs no selection routines.
- [ ] Update README, `model_selection.md`, artifact references, and tests together.
