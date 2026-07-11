"""Frozen final-procedure estimators for development refit artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class FrozenProbabilityVotingEnsemble:
    """A fitted soft-voting ensemble over binary probability estimators."""

    member_ids: tuple[str, ...]
    member_display_names: tuple[str, ...]
    member_weights: tuple[float, ...]
    estimators: tuple[Any, ...]
    decision_threshold: float
    calibration_method: str = "none"
    calibration_status: str = "deferred_fast_completion"

    def __post_init__(self) -> None:
        """Validate fitted ensemble metadata."""
        if not self.member_ids:
            raise ValueError("member_ids must not be empty.")
        if len(self.member_ids) != len(self.member_display_names):
            raise ValueError("member display-name count must match member_ids.")
        if len(self.member_ids) != len(self.member_weights):
            raise ValueError("member weight count must match member_ids.")
        if len(self.member_ids) != len(self.estimators):
            raise ValueError("estimator count must match member_ids.")
        weights = np.asarray(self.member_weights, dtype=float)
        if not np.all(np.isfinite(weights)):
            raise ValueError("member weights must be finite.")
        if not np.isclose(float(weights.sum()), 1.0, atol=1e-12):
            raise ValueError("member weights must sum to one.")
        if not np.isfinite(float(self.decision_threshold)):
            raise ValueError("decision threshold must be finite.")
        if not 0.0 <= float(self.decision_threshold) <= 1.0:
            raise ValueError("decision threshold must be between zero and one.")
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.member_ids_ = tuple(self.member_ids)
        self.member_display_names_ = tuple(self.member_display_names)
        self.member_weights_ = tuple(float(weight) for weight in self.member_weights)
        self.decision_threshold_ = float(self.decision_threshold)

    def fit(self, X: Any, y: Any = None) -> "FrozenProbabilityVotingEnsemble":
        """Return self because member estimators are already fitted."""
        return self

    def _positive_class_probability(self, estimator: Any, X: Any) -> np.ndarray:
        """Extract class-one probabilities from one fitted binary estimator."""
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("member predict_proba must return an n-by-2 matrix.")
        classes = np.asarray(getattr(estimator, "classes_", self.classes_))
        matches = np.flatnonzero(classes == 1)
        if matches.size != 1:
            raise ValueError("member estimator must expose class label 1.")
        scores = probabilities[:, int(matches[0])]
        if not np.all(np.isfinite(scores)):
            raise ValueError("member probabilities must be finite.")
        return scores

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return two-column averaged binary probabilities."""
        weighted_scores: list[np.ndarray] = []
        for estimator, weight in zip(self.estimators, self.member_weights_):
            weighted_scores.append(
                float(weight) * self._positive_class_probability(estimator, X)
            )
        positive = np.sum(np.vstack(weighted_scores), axis=0)
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        """Return class predictions using the frozen OOF-selected threshold."""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.decision_threshold_).astype(int)


def validate_equal_weights(weights: Sequence[float], *, expected_count: int) -> tuple[float, ...]:
    """Validate exactly equal ensemble weights and return them as floats."""
    if len(weights) != expected_count:
        raise ValueError(f"Expected {expected_count} weights, got {len(weights)}.")
    numeric = tuple(float(weight) for weight in weights)
    if len(set(numeric)) != 1:
        raise ValueError("Ensemble weights must be exactly equal.")
    if not np.isclose(sum(numeric), 1.0, atol=1e-12):
        raise ValueError("Ensemble weights must sum to one.")
    return numeric
