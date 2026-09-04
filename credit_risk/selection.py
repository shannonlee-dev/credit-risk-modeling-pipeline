"""Repository-approved policy for convenience final evaluations."""

from credit_risk.constants import LOGISTIC_REGRESSION_MODEL


PROJECT_SELECTION_POLICY = {
    "selected_model": LOGISTIC_REGRESSION_MODEL,
    "logistic_threshold": 0.45,
    "random_forest_n_estimators": 100,
    "random_forest_threshold": 0.33,
}
