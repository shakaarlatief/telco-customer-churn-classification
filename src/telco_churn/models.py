"""Model constructors used across churn classification notebooks.

This module contains lightweight model factory functions and small reusable
estimator definitions. It should not contain full experiments, hyperparameter
searches, result interpretation, or report logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from telco_churn.config import RANDOM_STATE


def make_most_frequent_dummy_classifier() -> DummyClassifier:
    """Create a majority-class baseline classifier."""
    return DummyClassifier(strategy="most_frequent")


def make_prior_probability_dummy_classifier() -> DummyClassifier:
    """Create a prior-probability dummy classifier."""
    return DummyClassifier(strategy="prior")


def make_stratified_dummy_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a random baseline that samples from the training class prior."""
    return DummyClassifier(strategy="stratified", random_state=random_state)


def make_uniform_dummy_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a random baseline that samples uniformly across classes."""
    return DummyClassifier(strategy="uniform", random_state=random_state)


class EDAInspiredChurnRuleClassifier(BaseEstimator, ClassifierMixin):
    """Transparent rule-based churn-risk baseline inspired by EDA.

    The rule assigns one risk point for each observed high-risk condition:

    - month-to-month contract;
    - electronic check payment;
    - fiber optic internet service;
    - no online security;
    - no tech support.

    The predicted class is churn when the number of risk points is at least
    ``risk_threshold``. The score returned by ``predict_proba`` is the normalized
    risk count. It is not a calibrated probability.
    """

    def __init__(self, risk_threshold: int = 2):
        self.risk_threshold = risk_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """Fit the rule classifier."""
        self.classes_ = np.array([0, 1])
        return self

    def _risk_count(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the number of high-risk rule conditions per row."""
        required_columns = [
            "Contract",
            "PaymentMethod",
            "InternetService",
            "OnlineSecurity",
            "TechSupport",
        ]
        missing_columns = [column for column in required_columns if column not in X]

        if missing_columns:
            raise KeyError(
                "The EDA-inspired rule classifier is missing required columns: "
                f"{missing_columns}"
            )

        risk_count = np.zeros(len(X), dtype=float)
        risk_count += (X["Contract"] == "Month-to-month").to_numpy(dtype=float)
        risk_count += (X["PaymentMethod"] == "Electronic check").to_numpy(dtype=float)
        risk_count += (X["InternetService"] == "Fiber optic").to_numpy(dtype=float)
        risk_count += (X["OnlineSecurity"] == "No").to_numpy(dtype=float)
        risk_count += (X["TechSupport"] == "No").to_numpy(dtype=float)

        return risk_count

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn when the risk count reaches the threshold."""
        risk_count = self._risk_count(X)
        return (risk_count >= self.risk_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return normalized rule scores in probability-array shape."""
        risk_count = self._risk_count(X)
        positive_score = risk_count / 5.0
        negative_score = 1.0 - positive_score
        return np.column_stack([negative_score, positive_score])


def make_eda_inspired_rule_classifier(
    *,
    risk_threshold: int = 2,
) -> EDAInspiredChurnRuleClassifier:
    """Create the EDA-inspired rule baseline."""
    return EDAInspiredChurnRuleClassifier(risk_threshold=risk_threshold)
