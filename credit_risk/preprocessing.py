"""Shared leakage-safe feature preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.constants import FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Build the shared numeric preprocessing path."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("numeric", numeric, FEATURE_COLUMNS)])
