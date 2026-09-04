"""Shared constants for the credit-risk pipeline."""

from pathlib import Path


# Reproducibility and evaluation
RANDOM_STATE = 42
CV_FOLDS = 5

# Project paths
DEFAULT_DATA_PATH = Path("data/generated/finance_data.csv")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Dataset schema
REGRESSION_TARGET = "credit_score"
CLASSIFICATION_TARGET = "is_overdue"
FEATURE_COLUMNS = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "overdue_count_6m",
    "credit_card_count",
]

# Shared model constraints and identifiers
CREDIT_SCORE_MIN = 0
CREDIT_SCORE_MAX = 1000
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
RULE_BASELINE_MODEL = "Rule Baseline"
LOGISTIC_REGRESSION_MODEL = "Logistic Regression"
DECISION_TREE_MODEL = "Decision Tree"
RANDOM_FOREST_MODEL = "Random Forest"
TUNED_RANDOM_FOREST_MODEL = "Random Forest (Tuned)"
REGRESSION_MODELS = ("Ridge", "Lasso")
REGRESSION_ALPHAS = [0.01, 0.1, 1, 10, 100]
