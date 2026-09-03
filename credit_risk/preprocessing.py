"""Shared leakage-safe feature preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build numeric and categorical preprocessing paths."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES)])
