# Credit Risk Modeling Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 규칙 기반·분류·회귀 모델을 누수 없이 학습하고 실제 성능 비교표와 시각화를 생성하는 재현 가능한 신용 위험 분석 시스템을 구축한다.

**Architecture:** `data_gen.py`가 제공된 규칙으로 원본 CSV를 생성하고, `train.py`의 역할별 함수들이 검증·분할·전처리·학습·평가·산출물 저장을 수행한다. 공통 `ColumnTransformer`는 모든 모델 Pipeline 내부에서 Train Set에만 fit하며, 모델 선택은 Train 내부 CV로 끝낸 후 Test Set은 최종 평가에만 사용한다.

**Tech Stack:** Python 3.10+, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pytest

**Spec:** `docs/superpowers/specs/2026-08-30-credit-risk-pipeline-design.md`

## Global Constraints

- `docs/private/mission.md`와 `docs/private/rubric.md`의 필수 항목만 구현하고 보너스 항목은 구현하지 않는다.
- 데이터 생성은 제공된 `N_SAMPLES=10000`, `RANDOM_STATE=42`와 수식을 유지한다.
- 모델 입력은 `age`, `annual_income`, `spending_score`, `debt_ratio`, `credit_card_count`, `overdue_count_6m`만 사용한다.
- 분류 분할은 `test_size=0.2`, `stratify=y`, `random_state=42`를 사용하고 모든 CV는 Train Set 내부 5-fold로 수행한다.
- 전처리, 스케일링, 인코딩, 클래스 가중치 학습은 Test Set에 fit하지 않는다.
- Ridge/Lasso Alpha는 `0.01`, `0.1`, `1`, `10`, `100`만 사용한다.
- GridSearchCV는 총 100개 이하 하이퍼파라미터 조합만 탐색한다.
- 외부 데이터와 딥러닝 프레임워크는 사용하지 않는다.

---

### Task 1: 재현 가능한 데이터 생성과 프로젝트 환경

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `data_gen.py`
- Create: `tests/test_data_gen.py`

**Interfaces:**
- Consumes: 출력 경로 `str | pathlib.Path`, 기본값 `finance_data.csv`
- Produces: `generate_finance_data(output_path: str | Path = "finance_data.csv") -> pandas.DataFrame`

- [ ] **Step 1: 의존성과 제외 파일 설정 작성**

```text
# requirements.txt
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
pytest==8.3.3
```

`.gitignore`에는 `*.csv`, `.venv/`, `__pycache__/`, `.pytest_cache/`를 한 줄씩 추가한다. 이어서 `python3 -m venv .venv`와 `.venv/bin/pip install -r requirements.txt`로 격리 환경을 준비한다.

- [ ] **Step 2: 생성 데이터 계약을 나타내는 실패 테스트 작성**

```python
from pandas.testing import assert_frame_equal

from data_gen import EXPECTED_COLUMNS, generate_finance_data


def test_generate_finance_data_is_reproducible(tmp_path):
    first = generate_finance_data(tmp_path / "first.csv")
    second = generate_finance_data(tmp_path / "second.csv")

    assert first.shape == (10_000, 8)
    assert list(first.columns) == EXPECTED_COLUMNS
    assert 0.10 <= first["is_overdue"].mean() <= 0.15
    assert first["credit_score"].between(0, 1000).all()
    assert_frame_equal(first, second)
```

- [ ] **Step 3: 테스트가 모듈 부재로 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/test_data_gen.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'data_gen'`.

- [ ] **Step 4: 제공된 생성 수식을 함수와 CLI로 구현**

```python
N_SAMPLES = 10_000
RANDOM_STATE = 42
EXPECTED_COLUMNS = [
    "age", "annual_income", "spending_score", "debt_ratio",
    "credit_card_count", "overdue_count_6m", "credit_score", "is_overdue",
]


def generate_finance_data(output_path="finance_data.csv"):
    np.random.seed(RANDOM_STATE)
    data = {
        "age": np.random.randint(20, 70, N_SAMPLES),
        "annual_income": np.random.normal(5000, 2000, N_SAMPLES).round(0),
        "spending_score": np.random.randint(1, 100, N_SAMPLES),
        "debt_ratio": np.random.uniform(0, 1, N_SAMPLES).round(2),
        "credit_card_count": np.random.randint(1, 10, N_SAMPLES),
        "overdue_count_6m": np.random.poisson(0.5, N_SAMPLES),
    }
    df = pd.DataFrame(data)
    df["annual_income"] = df["annual_income"].apply(lambda value: max(value, 1500))
    df["credit_score"] = (
        300 + (df["annual_income"] / 100) * 3
        - df["overdue_count_6m"] * 50 - df["debt_ratio"] * 100
        + np.random.normal(0, 30, N_SAMPLES)
    ).clip(0, 1000).round(0)
    threshold = df["credit_score"].quantile(0.15)
    df["is_overdue"] = np.where(
        (df["credit_score"] < threshold) & (np.random.rand(N_SAMPLES) > 0.2), 1, 0
    )
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df
```

- [ ] **Step 5: 데이터 테스트와 실제 생성 실행 확인**

Run: `.venv/bin/python -m pytest tests/test_data_gen.py -v && .venv/bin/python data_gen.py`

Expected: PASS, `finance_data.csv` 생성, 10,000건과 약 10~15% 연체율 출력, CSV는 `git status`에 나타나지 않음.

- [ ] **Step 6: 데이터 생성 기능 커밋**

```bash
git add .gitignore requirements.txt data_gen.py tests/test_data_gen.py
git -c user.name=shannonlee-dev commit -m "feat(data): generate reproducible finance dataset"
```

---

### Task 2: 공통 전처리와 규칙 기반 분류

**Files:**
- Create: `train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `pandas.DataFrame` with eight required columns
- Produces: `load_and_validate_data(path)`, `split_classification_data(df)`, `split_regression_data(df)`, `class_distribution(y)`, `build_preprocessor()`, `rule_based_predict(row)`, `evaluate_classifier(y_true, predictions, scores, prediction_latency_ms) -> dict[str, float]`

- [ ] **Step 1: 입력 검증·분할·전처리·규칙 계약의 실패 테스트 작성**

```python
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from data_gen import generate_finance_data
from train import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
    build_preprocessor, class_distribution, load_and_validate_data,
    rule_based_predict, split_classification_data,
)


def test_pipeline_contracts_prevent_target_leakage(tmp_path):
    df = generate_finance_data(tmp_path / "finance.csv")
    x_train, x_test, y_train, y_test = split_classification_data(df)

    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "credit_score" not in x_train.columns
    assert "is_overdue" not in x_train.columns
    assert len(x_train) == 8_000 and len(x_test) == 2_000
    assert abs(y_train.mean() - y_test.mean()) < 0.001
    assert isinstance(build_preprocessor(), ColumnTransformer)


@pytest.mark.parametrize("updates", [
    {"overdue_count_6m": 3},
    {"debt_ratio": 0.9, "annual_income": 4000},
    {"annual_income": 2000},
    {"spending_score": 95, "debt_ratio": 0.75},
    {"credit_card_count": 8, "debt_ratio": 0.7},
    {"age": 22, "debt_ratio": 0.8},
])
def test_rule_model_has_six_explicit_risk_conditions(updates):
    safe = pd.Series({
        "age": 40, "annual_income": 7000, "spending_score": 50,
        "debt_ratio": 0.2, "credit_card_count": 2, "overdue_count_6m": 0,
    })
    assert rule_based_predict(safe) == 0
    risky = safe.copy()
    for key, value in updates.items():
        risky[key] = value
    assert rule_based_predict(risky) == 1


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"age": [30]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="누락된 필수 열"):
        load_and_validate_data(path)
```

- [ ] **Step 2: 테스트가 공개 인터페이스 부재로 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`

Expected: FAIL because `train.py` or imported names do not exist.

- [ ] **Step 3: 최소 공통 파이프라인과 여섯 규칙 구현**

```python
FEATURE_COLUMNS = ["age", "annual_income", "spending_score", "debt_ratio",
                   "credit_card_count", "overdue_count_6m"]
NUMERIC_FEATURES = ["age", "annual_income", "spending_score", "debt_ratio",
                    "overdue_count_6m"]
CATEGORICAL_FEATURES = ["credit_card_count"]


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def rule_based_predict(row):
    if row["overdue_count_6m"] >= 2: return 1
    if row["debt_ratio"] > 0.80 and row["annual_income"] < 4500: return 1
    if row["annual_income"] < 2500: return 1
    if row["spending_score"] > 90 and row["debt_ratio"] > 0.70: return 1
    if row["credit_card_count"] >= 8 and row["debt_ratio"] > 0.65: return 1
    if row["age"] < 25 and row["debt_ratio"] > 0.75: return 1
    return 0
```

분할 함수는 분류에 stratify를 적용하고, 분포 함수는 `count_0`, `count_1`, `positive_rate`를 반환한다. `evaluate_classifier`는 Accuracy, Precision, Recall, F1, ROC-AUC, prediction latency를 float로 반환한다.

- [ ] **Step 4: 공통 파이프라인 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`

Expected: PASS for validation, leakage, split, preprocessing, and rule behavior.

- [ ] **Step 5: 누수 방지 전처리와 규칙 베이스라인 커밋**

```bash
git add train.py tests/test_train.py
git -c user.name=shannonlee-dev commit -m "feat(pipeline): add leakage-safe preprocessing"
```

---

### Task 3: 불균형 분류 모델·튜닝·시각화

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`
- Create at runtime: `artifacts/confusion_matrix.png`
- Create at runtime: `artifacts/roc_curve.png`
- Create at runtime: `artifacts/feature_importance.png`

**Interfaces:**
- Consumes: classification Train/Test tuples and output directory
- Produces: `run_classification(x_train, x_test, y_train, y_test, output_dir, grid=None, fast=False) -> dict`

- [ ] **Step 1: 분류 결과 계약의 실패 테스트 작성**

```python
def test_classification_returns_all_models_and_artifacts(tmp_path):
    df = generate_finance_data(tmp_path / "finance.csv").head(1200)
    x_train, x_test, y_train, y_test = split_classification_data(df)
    result = run_classification(
        x_train, x_test, y_train, y_test, tmp_path / "artifacts",
        grid={"model__n_estimators": [20], "model__max_depth": [8],
              "model__min_samples_split": [2]},
    )

    assert set(result["metrics"]) == {
        "Rule Baseline", "Logistic Regression", "Random Forest",
        "Random Forest (Tuned)",
    }
    for metrics in result["metrics"].values():
        assert {"accuracy", "precision", "recall", "f1", "roc_auc",
                "prediction_latency_ms"} <= metrics.keys()
    assert result["best_params"]
    for name in ["confusion_matrix.png", "roc_curve.png", "feature_importance.png"]:
        assert (tmp_path / "artifacts" / name).is_file()
```

- [ ] **Step 2: 새 분류 함수가 없어 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_classification_returns_all_models_and_artifacts -v`

Expected: FAIL because `run_classification` is not defined.

- [ ] **Step 3: Train 내부 CV와 네 모델 비교 구현**

```python
logistic = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                 random_state=RANDOM_STATE)),
])
forest = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)),
])
grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 8, 16],
    "model__min_samples_split": [2, 5],
}
search = GridSearchCV(
    clone(forest), grid, scoring="f1", cv=StratifiedKFold(5, shuffle=True,
    random_state=RANDOM_STATE), n_jobs=-1, refit=True,
)
```

Logistic Regression의 Train 5-fold F1/AUC, 기본 Random Forest, GridSearchCV 최적 모델을 fit하고 Test Set은 각 최종 모델 평가에만 사용한다. 규칙 기반은 0/1 점수를 사용한다. `perf_counter`로 5회 `predict_proba` 평균 지연시간을 기록한다.

- [ ] **Step 4: Confusion Matrix·ROC·Feature Importance 저장 구현**

`ConfusionMatrixDisplay`, `RocCurveDisplay`, `get_feature_names_out()`, 튜닝된 Forest의 `feature_importances_`를 사용한다. 모든 Figure는 `tight_layout()`, `figure.savefig(output_dir / "artifact_name.png", dpi=150)`, `plt.close(figure)` 순서로 저장한다.

- [ ] **Step 5: 분류 테스트와 전체 관련 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`

Expected: PASS and three non-empty PNG files in the temporary artifacts directory.

- [ ] **Step 6: 분류 기능 커밋**

```bash
git add train.py tests/test_train.py
git -c user.name=shannonlee-dev commit -m "feat(classification): compare weighted risk models"
```

---

### Task 4: Ridge/Lasso 선택·평가·계수 변화

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`
- Create at runtime: `artifacts/regularization_coefficients.png`

**Interfaces:**
- Consumes: regression Train/Test tuples and output directory
- Produces: `run_regression(x_train, x_test, y_train, y_test, output_dir, fast=False) -> dict`

- [ ] **Step 1: 회귀 모델 선택과 출력 범위 계약의 실패 테스트 작성**

```python
def test_regression_selects_alpha_on_train_and_clips_only_output(tmp_path):
    df = generate_finance_data(tmp_path / "finance.csv").head(1200)
    x_train, x_test, y_train, y_test = split_regression_data(df)
    result = run_regression(x_train, x_test, y_train, y_test,
                            tmp_path / "artifacts")

    assert set(result["test_metrics"]) == {"Ridge", "Lasso"}
    assert set(result["selected_alpha"]) == {"Ridge", "Lasso"}
    assert set(result["cv_rmse"]["Ridge"]) == {"0.01", "0.1", "1", "10", "100"}
    for metrics in result["test_metrics"].values():
        assert {"rmse", "mae", "r2"} <= metrics.keys()
    assert result["predictions"]["Ridge"].between(0, 1000).all()
    assert result["predictions"]["Lasso"].between(0, 1000).all()
    assert (tmp_path / "artifacts" / "regularization_coefficients.png").is_file()
```

- [ ] **Step 2: 회귀 함수 부재로 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_regression_selects_alpha_on_train_and_clips_only_output -v`

Expected: FAIL because `run_regression` is not defined.

- [ ] **Step 3: Train 내부 5-fold Alpha 선택 구현**

```python
ALPHAS = [0.01, 0.1, 1, 10, 100]
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for model_name, estimator_type in {"Ridge": Ridge, "Lasso": Lasso}.items():
    for alpha in ALPHAS:
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator_type(alpha=alpha, max_iter=20_000)),
        ])
        scores = cross_val_score(
            pipeline, x_train, y_train, scoring="neg_root_mean_squared_error",
            cv=cv, n_jobs=-1,
        )
        cv_rmse[model_name][str(alpha)] = float(-scores.mean())
```

모델별 최소 CV RMSE Alpha를 고르고 전체 Train Set에 refit한다. Test 원시 예측으로 RMSE, MAE, R²를 딱 한 번 계산하고, 저장용 예측 Series만 `np.clip(raw_predictions, 0, 1000)` 처리한다.

- [ ] **Step 4: Train 기반 계수 변화 시각화 구현**

각 Alpha Pipeline을 전체 Train Set에 fit하고 `preprocessor.get_feature_names_out()`와 `model.coef_`를 저장한다. Ridge/Lasso 두 subplot에 log Alpha 축과 전처리 후 특성별 선을 그리고 `regularization_coefficients.png`로 저장한다.

- [ ] **Step 5: 회귀 테스트와 전체 테스트 통과 확인**

Run: `.venv/bin/python -m pytest -v`

Expected: all tests PASS; metrics use raw predictions while returned predictions remain within 0~1000.

- [ ] **Step 6: 회귀 기능 커밋**

```bash
git add train.py tests/test_train.py
git -c user.name=shannonlee-dev commit -m "feat(regression): compare regularized credit models"
```

---

### Task 5: 전체 실행·결과 저장·README

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`
- Create: `README.md`
- Create at runtime and commit: `artifacts/metrics.json`
- Create at runtime and commit: `artifacts/confusion_matrix.png`
- Create at runtime and commit: `artifacts/roc_curve.png`
- Create at runtime and commit: `artifacts/feature_importance.png`
- Create at runtime and commit: `artifacts/regularization_coefficients.png`

**Interfaces:**
- Consumes: `--data finance_data.csv`, `--output-dir artifacts`
- Produces: `run_analysis(data_path="finance_data.csv", output_dir="artifacts", fast=False) -> dict` and reproducible CLI artifacts

- [ ] **Step 1: 전체 실행 산출물 계약의 실패 테스트 작성**

```python
def test_run_analysis_saves_metrics_and_predictions(tmp_path):
    data_path = tmp_path / "finance.csv"
    generate_finance_data(data_path)
    output_dir = tmp_path / "artifacts"
    result = run_analysis(data_path, output_dir, fast=True)

    assert {"data_distribution", "classification", "regression"} <= result.keys()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "classification_predictions.csv").is_file()
    assert (output_dir / "credit_score_predictions.csv").is_file()
```

- [ ] **Step 2: 오케스트레이션 부재로 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_run_analysis_saves_metrics_and_predictions -v`

Expected: FAIL because `run_analysis` is not defined.

- [ ] **Step 3: 전체 흐름과 CLI 구현**

```python
def run_analysis(data_path="finance_data.csv", output_dir="artifacts", fast=False):
    df = load_and_validate_data(data_path)
    class_split = split_classification_data(df)
    regression_split = split_regression_data(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "data_distribution": {
            "all": class_distribution(df["is_overdue"]),
            "train": class_distribution(class_split[2]),
            "test": class_distribution(class_split[3]),
        },
        "classification": run_classification(*class_split, output_dir, fast=fast),
        "regression": run_regression(*regression_split, output_dir, fast=fast),
    }
    report = {
        "data_distribution": result["data_distribution"],
        "classification": {
            key: value for key, value in result["classification"].items()
            if key != "predictions"
        },
        "regression": {
            key: value for key, value in result["regression"].items()
            if key != "predictions"
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
```

CLI는 입력 파일이 없으면 `python3 data_gen.py` 실행 안내를 포함한 오류를 출력한다. `fast=True`는 테스트에서 Grid 후보와 CV 비용만 축소하며 기본 CLI는 설계의 전체 5-fold와 12개 Grid 조합을 사용한다.

- [ ] **Step 4: 전체 실행 흐름 커밋**

```bash
git add train.py tests/test_train.py
git -c user.name=shannonlee-dev commit -m "feat(pipeline): orchestrate analysis artifacts"
```

- [ ] **Step 5: 전체 테스트 통과 후 실제 데이터로 최종 분석 실행**

Run: `.venv/bin/python -m pytest -v && .venv/bin/python data_gen.py && .venv/bin/python train.py`

Expected: all tests PASS; `artifacts/metrics.json`, 두 예측 CSV, 네 PNG 생성; 데이터 분포와 모델 성능이 콘솔에 출력됨.

- [ ] **Step 6: 실제 결과를 README에 기록**

README 순서는 제목·한 문장 설명·핵심 기능·빠른 시작·실제 결과·설계 및 해석·프로젝트 구조로 구성한다. `metrics.json`의 실제 값을 소수점 넷째 자리까지 반올림해 분류·회귀 표에 기록한다. 다음 키 매핑을 사용해 문서와 산출물의 값이 일치하도록 한다.

```markdown
| README 행 | metrics.json 경로 |
|---|---|
| 규칙 기반 | `classification.metrics.Rule Baseline` |
| Logistic Regression | `classification.metrics.Logistic Regression` |
| Random Forest | `classification.metrics.Random Forest` |
| Random Forest (Tuned) | `classification.metrics.Random Forest (Tuned)` |
| Ridge | `regression.selected_alpha.Ridge`, `regression.test_metrics.Ridge` |
| Lasso | `regression.selected_alpha.Lasso`, `regression.test_metrics.Lasso` |
```

README에는 데이터 분포, Accuracy/F1/AUC 선정 근거, 누수 방지, 표준화 공간의 L1/L2 계수 해석, 기본/튜닝 Forest 차이와 편향-분산, SMOTE/class weight 비교, Test 분포 보존, 규칙 기반 AUC 한계, 오탐·미탐 비용과 임계값, 지연시간·복잡도를 고려한 최종 운영 모델 선택을 실제 수치와 연결해 작성한다. Feature Importance 상위 특성의 방향과 한계를 최소 3줄로 해석한다.

- [ ] **Step 7: 결과와 문서 검증**

Run: `.venv/bin/python -m pytest -v && .venv/bin/python -m compileall -q data_gen.py train.py && git diff --check`

Expected: all commands exit 0; README 표의 수치가 `artifacts/metrics.json`과 일치함.

- [ ] **Step 8: 결과와 문서 커밋**

```bash
git add README.md artifacts/metrics.json artifacts/*.png
git -c user.name=shannonlee-dev commit -m "docs(readme): report credit risk experiment results"
```

---

### Task 6: 최종 요구사항 대조와 저장소 검증

**Files:**
- Verify: all tracked project files
- Preserve untracked: `docs/private/`

**Interfaces:**
- Consumes: completed repository and generated `finance_data.csv`
- Produces: passing validation and feature-scoped commit history

- [ ] **Step 1: 미션·루브릭 16개 항목과 구현 대조**

Run: `rg -n "Accuracy|F1|ROC-AUC|RMSE|MAE|R²|class_weight|SMOTE|편향|분산|임계값|지연시간|데이터 누수" README.md`

Expected: 각 평가 근거가 README에서 검색되고 관련 PNG가 링크됨.

- [ ] **Step 2: 전체 검증 재실행**

Run: `.venv/bin/python -m pytest -v && .venv/bin/python data_gen.py && .venv/bin/python train.py`

Expected: tests PASS and full pipeline completes without warnings or errors.

- [ ] **Step 3: Git 상태와 기능 단위 이력 확인**

Run: `git status --short --branch && git log --oneline --decorate -10 && git check-ignore -v finance_data.csv`

Expected: `finance_data.csv` is ignored, `docs/private/`만 의도적으로 untracked, 데이터·분류·회귀·문서 기능 커밋이 분리됨.
