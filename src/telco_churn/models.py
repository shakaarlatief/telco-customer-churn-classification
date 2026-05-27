"""Model factories for the Telco Customer Churn project.

The functions in this module return unfitted estimators. Keeping model
construction in one place makes notebooks cleaner and makes later experiments
more reproducible.

Section 05 introduces linear classifiers:

- RidgeClassifier as a regularized least-squares classifier;
- LogisticRegression with L2 regularization;
- LogisticRegression with L1 regularization;
- class-weighted logistic regression as a simple imbalance-aware variant.

The functions intentionally return scikit-learn estimators rather than fitting
them. Fitting happens inside cross-validation pipelines in the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline

from telco_churn.config import RANDOM_STATE


@dataclass
class EDAInspiredRuleClassifier(BaseEstimator, ClassifierMixin):
    """Simple transparent rule-based churn classifier.

    The classifier assigns one risk point for each manually selected high-risk
    condition and predicts churn when the total risk score is at least
    ``risk_threshold``.

    This estimator is intentionally simple and deterministic. It is useful as a
    bridge between exploratory analysis and learned models.
    """

    risk_threshold: int = 2

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        self.classes_ = np.array([0, 1])
        return self

    def _risk_score(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X), dtype=float)

        score += (X["Contract"] == "Month-to-month").astype(float)
        score += (X["PaymentMethod"] == "Electronic check").astype(float)
        score += (X["InternetService"] == "Fiber optic").astype(float)
        score += (X["OnlineSecurity"] == "No").astype(float)
        score += (X["TechSupport"] == "No").astype(float)

        return score

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        score = self._risk_score(X)
        return (score >= self.risk_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self._risk_score(X)
        max_score = 5.0
        probability = np.clip(score / max_score, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def make_eda_inspired_rule_classifier(
    *,
    risk_threshold: int = 2,
) -> EDAInspiredRuleClassifier:
    """Create the EDA-inspired rule classifier."""
    return EDAInspiredRuleClassifier(risk_threshold=risk_threshold)


def make_most_frequent_dummy_classifier() -> DummyClassifier:
    """Create a majority-class dummy classifier."""
    return DummyClassifier(strategy="most_frequent")


def make_prior_probability_dummy_classifier() -> DummyClassifier:
    """Create a prior-probability dummy classifier.

    The hard predictions are the majority class, but predicted probabilities
    equal the empirical class distribution in each training fold.
    """
    return DummyClassifier(strategy="prior")


def make_stratified_dummy_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a stratified random dummy classifier."""
    return DummyClassifier(strategy="stratified", random_state=random_state)


def make_uniform_dummy_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a uniform random dummy classifier."""
    return DummyClassifier(strategy="uniform", random_state=random_state)


def make_ridge_classifier(
    *,
    alpha: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = RANDOM_STATE,
) -> RidgeClassifier:
    """Create a regularized least-squares linear classifier.

    ``RidgeClassifier`` fits a linear classifier using squared-error logic with
    an L2 penalty. It is used here as the practical version of least-squares
    classification because unregularized least-squares classification can be
    numerically unstable after one-hot encoding.
    """
    return RidgeClassifier(
        alpha=alpha,
        class_weight=class_weight,
        random_state=random_state,
    )


def _sklearn_version_at_least(major: int, minor: int) -> bool:
    """Return whether the installed scikit-learn version is at least major.minor.

    This small helper avoids adding a new dependency only to compare versions.
    It is used because newer scikit-learn versions deprecate the old
    LogisticRegression ``penalty`` argument in favour of ``l1_ratio``.
    """
    import sklearn

    version_parts = sklearn.__version__.split(".")[:2]

    try:
        installed_major = int(version_parts[0])
        installed_minor = int(version_parts[1])
    except (IndexError, ValueError):
        return False

    return (installed_major, installed_minor) >= (major, minor)


def make_logistic_regression_classifier(
    *,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    solver: str | None = None,
    l1_ratio: float | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Create a logistic regression classifier.

    The function supports both older and newer scikit-learn APIs.

    In older scikit-learn versions, regularization is controlled by the
    ``penalty`` argument. In newer versions, the same idea is represented by
    ``l1_ratio``:

    - ``l1_ratio=0`` corresponds to L2 regularization;
    - ``l1_ratio=1`` corresponds to L1 regularization;
    - intermediate values correspond to elastic-net regularization.

    Scikit-learn's ``C`` is inverse regularization strength. Smaller values
    imply stronger regularization.
    """
    if solver is None:
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "elasticnet":
            solver = "saga"
        else:
            solver = "lbfgs"

    kwargs = {
        "C": C,
        "class_weight": class_weight,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
    }

    uses_l1_ratio_api = _sklearn_version_at_least(1, 8)

    if uses_l1_ratio_api:
        if penalty == "l2":
            kwargs["l1_ratio"] = 0.0
        elif penalty == "l1":
            kwargs["l1_ratio"] = 1.0
        elif penalty == "elasticnet":
            kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio
        elif penalty in {None, "none"}:
            kwargs["C"] = np.inf
            kwargs["l1_ratio"] = 0.0
        else:
            raise ValueError(f"Unsupported penalty: {penalty!r}")
    else:
        kwargs["penalty"] = penalty
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio

    return LogisticRegression(**kwargs)


def make_l2_logistic_regression_classifier(
    *,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Create L2-regularized logistic regression."""
    return make_logistic_regression_classifier(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )


def make_l1_logistic_regression_classifier(
    *,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Create L1-regularized logistic regression."""
    return make_logistic_regression_classifier(
        penalty="l1",
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )


def make_classifier_pipeline(
    *,
    preprocessor,
    classifier,
) -> Pipeline:
    """Create a preprocessing-plus-classifier pipeline.

    The preprocessor is fitted inside each cross-validation training fold when
    the pipeline is passed to cross-validation utilities. This prevents leakage
    from validation folds into scaling, encoding, or imputation steps.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
