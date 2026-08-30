"""Train and evaluate credit-risk classification and regression models."""

from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "overdue_count_6m",
    "credit_card_count",
]
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "overdue_count_6m",
]
CATEGORICAL_FEATURES = ["credit_card_count"]
TARGET_COLUMNS = ["credit_score", "is_overdue"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS
ALPHAS = [0.01, 0.1, 1, 10, 100]


def load_and_validate_data(path: str | Path) -> pd.DataFrame:
    """Load a generated CSV and reject incomplete schemas."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {data_path}. "
            "먼저 `python3 data_gen.py`를 실행하세요."
        )

    df = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"누락된 필수 열: {', '.join(missing)}")
    return df


def split_classification_data(df: pd.DataFrame):
    """Create a reproducible stratified classification split."""
    return train_test_split(
        df[FEATURE_COLUMNS],
        df["is_overdue"],
        test_size=0.2,
        stratify=df["is_overdue"],
        random_state=RANDOM_STATE,
    )


def split_regression_data(df: pd.DataFrame):
    """Create a reproducible regression split."""
    return train_test_split(
        df[FEATURE_COLUMNS],
        df["credit_score"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )


def class_distribution(target: pd.Series) -> dict[str, int | float]:
    """Summarize binary class counts and positive rate."""
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    return {
        "count_0": int(counts.loc[0]),
        "count_1": int(counts.loc[1]),
        "positive_rate": float(target.mean()),
    }


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-safe numeric and categorical preprocessing."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def rule_based_predict(row: pd.Series) -> int:
    """Classify overdue risk with six explicit business rules."""
    if row["overdue_count_6m"] >= 2:
        return 1
    if row["debt_ratio"] > 0.80 and row["annual_income"] < 4500:
        return 1
    if row["annual_income"] < 2500:
        return 1
    if row["spending_score"] > 90 and row["debt_ratio"] > 0.70:
        return 1
    if row["credit_card_count"] >= 8 and row["debt_ratio"] > 0.65:
        return 1
    if row["age"] < 25 and row["debt_ratio"] > 0.75:
        return 1
    return 0


def evaluate_classifier(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    prediction_latency_ms: float,
) -> dict[str, float]:
    """Calculate classification metrics on an untouched test target."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "prediction_latency_ms": float(prediction_latency_ms),
    }


def _average_latency_ms(predictor, repeats: int = 5) -> float:
    predictor()
    started = perf_counter()
    for _ in range(repeats):
        predictor()
    return (perf_counter() - started) * 1000 / repeats


def _save_confusion_matrices(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (name, values) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            values,
            display_labels=["Normal", "Overdue"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_roc_curves(
    y_test: pd.Series,
    scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, values in scores.items():
        RocCurveDisplay.from_predictions(y_test, values, name=name, ax=axis)
    axis.plot([0, 1], [0, 1], "k--", label="Random")
    axis.set_title("Overdue Risk ROC Curves")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_feature_importance(model: Pipeline, output_path: Path) -> dict[str, float]:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    importances = model.named_steps["model"].feature_importances_
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance, x="importance", y="feature", ax=axis)
    axis.set_title("Tuned Random Forest Feature Importance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return {
        row.feature: float(row.importance)
        for row in importance.itertuples(index=False)
    }


def run_classification(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
    grid: dict | None = None,
) -> dict:
    """Compare rule, linear, and ensemble overdue-risk classifiers."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    logistic = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_validate(
        logistic,
        x_train,
        y_train,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        cv=cv,
        n_jobs=-1,
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    parameter_grid = grid or {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_split": [2, 5],
    }
    search = GridSearchCV(
        clone(forest),
        parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    tuned_forest = search.best_estimator_

    rule_predictions = x_test.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_test)[:, 1]
    forest_scores = forest.predict_proba(x_test)[:, 1]
    tuned_scores = tuned_forest.predict_proba(x_test)[:, 1]
    logistic_predictions = (logistic_scores >= 0.5).astype(int)
    forest_predictions = (forest_scores >= 0.5).astype(int)
    tuned_predictions = (tuned_scores >= 0.5).astype(int)

    latencies = {
        "Rule Baseline": _average_latency_ms(
            lambda: x_test.apply(rule_based_predict, axis=1).to_numpy()
        ),
        "Logistic Regression": _average_latency_ms(
            lambda: logistic.predict_proba(x_test)
        ),
        "Random Forest": _average_latency_ms(
            lambda: forest.predict_proba(x_test)
        ),
        "Random Forest (Tuned)": _average_latency_ms(
            lambda: tuned_forest.predict_proba(x_test)
        ),
    }
    predictions = {
        "Rule Baseline": rule_predictions,
        "Logistic Regression": logistic_predictions,
        "Random Forest": forest_predictions,
        "Random Forest (Tuned)": tuned_predictions,
    }
    scores = {
        "Rule Baseline": rule_predictions.astype(float),
        "Logistic Regression": logistic_scores,
        "Random Forest": forest_scores,
        "Random Forest (Tuned)": tuned_scores,
    }
    metrics = {
        name: evaluate_classifier(
            y_test,
            predictions[name],
            scores[name],
            latencies[name],
        )
        for name in predictions
    }

    _save_confusion_matrices(
        y_test,
        {
            "Logistic Regression": logistic_predictions,
            "Random Forest (Tuned)": tuned_predictions,
        },
        destination / "confusion_matrix.png",
    )
    _save_roc_curves(y_test, scores, destination / "roc_curve.png")
    feature_importance = _save_feature_importance(
        tuned_forest,
        destination / "feature_importance.png",
    )

    prediction_table = pd.DataFrame(
        {
            "actual_is_overdue": y_test.to_numpy(),
            "rule_prediction": rule_predictions,
            "logistic_probability": logistic_scores,
            "random_forest_probability": forest_scores,
            "overdue_probability": tuned_scores,
        }
    )
    prediction_table.to_csv(
        destination / "classification_predictions.csv",
        index=False,
    )

    return {
        "metrics": metrics,
        "logistic_cv": {
            "f1_mean": float(cv_scores["test_f1"].mean()),
            "roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        },
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "feature_importance": feature_importance,
        "predictions": prediction_table,
    }


def _regularized_pipeline(model_name: str, alpha: float) -> Pipeline:
    if model_name == "Ridge":
        estimator = Ridge(alpha=alpha)
    else:
        estimator = Lasso(
            alpha=alpha,
            max_iter=20_000,
            random_state=RANDOM_STATE,
        )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _save_coefficient_paths(
    coefficients: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, model_name in zip(axes, ["Ridge", "Lasso"]):
        first_alpha = str(ALPHAS[0])
        feature_names = coefficients[model_name][first_alpha].keys()
        for feature_name in feature_names:
            values = [
                coefficients[model_name][str(alpha)][feature_name]
                for alpha in ALPHAS
            ]
            axis.plot(ALPHAS, values, marker="o", label=feature_name)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(f"{model_name} Coefficient Paths")
        axis.set_xlabel("Alpha")
        axis.set_ylabel("Coefficient in standardized feature space")
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
) -> dict:
    """Select Ridge/Lasso alpha on Train CV and evaluate once on Test."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rmse: dict[str, dict[str, float]] = {"Ridge": {}, "Lasso": {}}
    coefficients: dict[str, dict[str, dict[str, float]]] = {
        "Ridge": {},
        "Lasso": {},
    }

    for model_name in ["Ridge", "Lasso"]:
        for alpha in ALPHAS:
            pipeline = _regularized_pipeline(model_name, alpha)
            scores = cross_val_score(
                pipeline,
                x_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            )
            alpha_key = str(alpha)
            cv_rmse[model_name][alpha_key] = float(-scores.mean())

            pipeline.fit(x_train, y_train)
            feature_names = pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()
            model_coefficients = pipeline.named_steps["model"].coef_
            coefficients[model_name][alpha_key] = {
                feature.replace("numeric__", "").replace(
                    "categorical__",
                    "",
                ): float(coefficient)
                for feature, coefficient in zip(
                    feature_names,
                    model_coefficients,
                )
            }

    selected_alpha = {
        model_name: float(
            min(cv_rmse[model_name], key=cv_rmse[model_name].get)
        )
        for model_name in ["Ridge", "Lasso"]
    }
    test_metrics: dict[str, dict[str, float]] = {}
    clipped_predictions: dict[str, pd.Series] = {}

    for model_name in ["Ridge", "Lasso"]:
        final_model = _regularized_pipeline(
            model_name,
            selected_alpha[model_name],
        )
        final_model.fit(x_train, y_train)
        raw_predictions = final_model.predict(x_test)
        test_metrics[model_name] = {
            "rmse": float(root_mean_squared_error(y_test, raw_predictions)),
            "mae": float(mean_absolute_error(y_test, raw_predictions)),
            "r2": float(r2_score(y_test, raw_predictions)),
        }
        clipped_predictions[model_name] = pd.Series(
            np.clip(raw_predictions, 0, 1000),
            name=model_name,
        )

    _save_coefficient_paths(
        coefficients,
        destination / "regularization_coefficients.png",
    )
    prediction_table = pd.DataFrame(
        {
            "actual_credit_score": y_test.to_numpy(),
            "ridge_prediction": clipped_predictions["Ridge"].to_numpy(),
            "lasso_prediction": clipped_predictions["Lasso"].to_numpy(),
        }
    )
    prediction_table.to_csv(
        destination / "credit_score_predictions.csv",
        index=False,
    )

    return {
        "cv_rmse": cv_rmse,
        "selected_alpha": selected_alpha,
        "test_metrics": test_metrics,
        "coefficients": coefficients,
        "predictions": clipped_predictions,
    }
