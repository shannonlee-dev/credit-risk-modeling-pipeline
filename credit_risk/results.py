"""Small result and selection contracts for two-stage execution."""

from dataclasses import asdict, dataclass

from credit_risk.constants import LOGISTIC_REGRESSION_MODEL, TUNED_RANDOM_FOREST_MODEL


@dataclass(frozen=True)
class FinalSelection:
    """Resolved choices allowed to reach final Holdout evaluation."""

    selected_model: str
    logistic_c: float
    logistic_threshold: float | None
    random_forest_n_estimators: int | None
    random_forest_max_depth: int | None
    random_forest_min_samples_split: int
    random_forest_threshold: float | None
    ridge_alpha: float
    lasso_alpha: float
    dataset_fingerprint: str | None = None
    protocol_fingerprint: str | None = None
    experiment_id: str | None = None

    def __post_init__(self) -> None:
        if self.selected_model not in {
            LOGISTIC_REGRESSION_MODEL,
            TUNED_RANDOM_FOREST_MODEL,
        }:
            raise ValueError("selected_model must be Logistic Regression or Random Forest (Tuned)")
        if self.selected_model == LOGISTIC_REGRESSION_MODEL and self.logistic_threshold is None:
            raise ValueError("selected Logistic Regression requires logistic threshold")
        if self.selected_model == TUNED_RANDOM_FOREST_MODEL and (
            self.random_forest_threshold is None
            or self.random_forest_n_estimators is None
        ):
            raise ValueError("selected Random Forest requires n_estimators and threshold")

    @classmethod
    def from_dict(cls, value: dict) -> "FinalSelection":
        classification = value["classification"]
        logistic = classification["logistic_regression"]
        forest = classification["random_forest"]
        regression = value["regression"]
        return cls(
            selected_model=classification["selected_model"],
            logistic_c=float(logistic["C"]),
            logistic_threshold=logistic["threshold"],
            random_forest_n_estimators=forest["n_estimators"],
            random_forest_max_depth=forest["max_depth"],
            random_forest_min_samples_split=int(forest["min_samples_split"]),
            random_forest_threshold=forest["threshold"],
            ridge_alpha=float(regression["ridge_alpha"]),
            lasso_alpha=float(regression["lasso_alpha"]),
            dataset_fingerprint=value.get("dataset_fingerprint"),
            protocol_fingerprint=value.get("protocol_fingerprint"),
            experiment_id=value.get("experiment_id"),
        )

    def to_dict(self) -> dict:
        return asdict(self)
