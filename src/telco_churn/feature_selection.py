"""Fold-safe feature-selection policies for final Telco model comparison.

Feature selection is treated as a fitted part of a complete candidate procedure.  A
selector therefore sits after the relevant feature-policy transformer and numerical /
categorical representation preprocessor, but before the final classifier.  Its learned
variance masks, mutual-information scores, and sparse logistic coefficients are fitted
only on the active training partition of an inner or outer cross-validation split.

The implemented policies are deliberately bounded:

``S0_NONE``
    Preserve the full represented feature matrix.

``S1_VARIANCE_MUTUAL_INFO``
    Remove zero-variance represented columns, then keep the highest mutual-information
    features.  The selector knows which leading preprocessed columns are continuous
    numeric features and which remaining one-hot columns are discrete indicators.

``S2_L1_LOGISTIC_SELECT_FROM_MODEL``
    Fit an L1-regularized logistic-regression selector and use scikit-learn's
    ``SelectFromModel`` thresholding rule.  A deterministic one-feature fallback
    protects downstream estimators when an extremely strong penalty shrinks every
    coefficient below the chosen threshold.

The module owns generic selector mechanics only.  Candidate-specific compatibility
rules belong in ``telco_churn.candidates`` so this module remains independent of model
registry identifiers.
"""

from __future__ import annotations

from functools import partial
from typing import Final, Literal, Mapping

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.utils.validation import check_is_fitted

from telco_churn.models import make_logistic_regression_classifier


FeatureSelectionPolicyId = Literal[
    "S0_NONE",
    "S1_VARIANCE_MUTUAL_INFO",
    "S2_L1_LOGISTIC_SELECT_FROM_MODEL",
]

FEATURE_SELECTION_NONE: Final[FeatureSelectionPolicyId] = "S0_NONE"
FEATURE_SELECTION_VARIANCE_MUTUAL_INFO: Final[FeatureSelectionPolicyId] = (
    "S1_VARIANCE_MUTUAL_INFO"
)
FEATURE_SELECTION_L1_LOGISTIC: Final[FeatureSelectionPolicyId] = (
    "S2_L1_LOGISTIC_SELECT_FROM_MODEL"
)

FEATURE_SELECTION_POLICY_IDS: Final[tuple[FeatureSelectionPolicyId, ...]] = (
    FEATURE_SELECTION_NONE,
    FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
    FEATURE_SELECTION_L1_LOGISTIC,
)


class FeatureSelectionPolicyError(ValueError):
    """Raised when a feature-selection policy or selector parameter is invalid."""


def validate_feature_selection_policy_id(policy_id: str) -> FeatureSelectionPolicyId:
    """Validate and return one declared feature-selection policy identifier."""
    if policy_id not in FEATURE_SELECTION_POLICY_IDS:
        raise FeatureSelectionPolicyError(
            f"Unknown feature-selection policy {policy_id!r}. Expected one of "
            f"{list(FEATURE_SELECTION_POLICY_IDS)!r}."
        )
    return policy_id  # type: ignore[return-value]


def _to_dense_array(X) -> np.ndarray:
    """Return a two-dimensional dense array without changing feature order."""
    if sparse.issparse(X):
        X = X.toarray()
    result = np.asarray(X)
    if result.ndim != 2:
        raise FeatureSelectionPolicyError("Feature selectors require a two-dimensional matrix.")
    return result


def _mixed_mutual_information(
    X: np.ndarray,
    y,
    *,
    discrete_features: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Score continuous numeric and one-hot indicator columns with mutual information."""
    return mutual_info_classif(
        X,
        y,
        discrete_features=discrete_features,
        random_state=int(random_state),
    )


class VarianceThresholdMutualInfoSelector(BaseEstimator, TransformerMixin):
    """Apply zero-variance filtering followed by mixed-type mutual-information selection.

    The incoming feature matrix is assumed to be ordered exactly as the project's
    feature-policy one-hot preprocessors produce it: all numeric columns first, followed
    by one-hot categorical indicators.  The selector first removes zero-variance columns,
    then passes a post-filter discrete-feature mask to ``mutual_info_classif``.  This is
    important because the leading scaled numeric columns are continuous whereas the
    one-hot indicators are discrete.

    ``requested_k`` is interpreted as an upper bound.  A small training fold may contain
    fewer nonconstant represented columns than the requested value, for example because
    a rare category is absent.  In that situation the selector safely uses every retained
    column rather than failing a long nested-CV experiment for a harmless schema-width
    difference.
    """

    def __init__(
        self,
        *,
        requested_k: int,
        n_numeric_features: int,
        random_state: int,
    ) -> None:
        self.requested_k = requested_k
        self.n_numeric_features = n_numeric_features
        self.random_state = random_state

    def fit(self, X, y):
        """Fit variance and mutual-information selectors on the active training matrix."""
        if int(self.requested_k) < 1:
            raise FeatureSelectionPolicyError("requested_k must be at least one.")
        if int(self.n_numeric_features) < 0:
            raise FeatureSelectionPolicyError("n_numeric_features must be non-negative.")

        self.variance_selector_ = VarianceThreshold(threshold=0.0)
        X_variance = self.variance_selector_.fit_transform(X, y)
        variance_support = np.asarray(self.variance_selector_.get_support(), dtype=bool)
        retained_positions = np.flatnonzero(variance_support)
        n_retained = int(retained_positions.size)
        if n_retained == 0:
            raise FeatureSelectionPolicyError(
                "Variance filtering removed every represented feature column."
            )

        effective_k = min(int(self.requested_k), n_retained)
        discrete_features = retained_positions >= int(self.n_numeric_features)
        X_dense = _to_dense_array(X_variance)
        score_func = partial(
            _mixed_mutual_information,
            discrete_features=np.asarray(discrete_features, dtype=bool),
            random_state=int(self.random_state),
        )
        self.mutual_information_selector_ = SelectKBest(
            score_func=score_func,
            k=effective_k,
        )
        self.mutual_information_selector_.fit(X_dense, y)

        mutual_information_support = np.asarray(
            self.mutual_information_selector_.get_support(), dtype=bool
        )
        selected_support = np.zeros_like(variance_support, dtype=bool)
        selected_support[retained_positions[mutual_information_support]] = True

        self.n_features_in_ = int(variance_support.size)
        self.variance_support_ = variance_support
        self.mutual_information_support_ = mutual_information_support
        self.selected_support_ = selected_support
        self.effective_k_ = int(effective_k)
        self.n_features_after_variance_ = n_retained
        self.n_selected_features_ = int(selected_support.sum())
        return self

    def transform(self, X):
        """Apply the fitted variance and mutual-information masks in training order."""
        check_is_fitted(
            self,
            attributes=(
                "variance_selector_",
                "mutual_information_selector_",
                "selected_support_",
            ),
        )
        X_variance = self.variance_selector_.transform(X)
        X_dense = _to_dense_array(X_variance)
        return self.mutual_information_selector_.transform(X_dense)

    def get_support(self, indices: bool = False):
        """Return the composed support mask in the original represented-column space."""
        check_is_fitted(self, "selected_support_")
        if indices:
            return np.flatnonzero(self.selected_support_)
        return self.selected_support_.copy()


class L1LogisticSelectFromModel(BaseEstimator, TransformerMixin):
    """Use an L1 logistic model as a fold-safe ``SelectFromModel`` feature selector.

    The wrapped logistic estimator uses a balanced loss so the selector is not dominated
    by the majority class during the binary churn task.  ``C`` controls the L1 penalty
    and the threshold follows scikit-learn's standard ``SelectFromModel`` syntax, such as
    ``"mean"`` or ``"median"``.  When no coefficient meets the threshold, the largest
    absolute coefficient is retained deterministically so downstream classifiers always
    receive at least one column.
    """

    def __init__(
        self,
        *,
        C: float,
        threshold: str,
        max_iter: int,
        random_state: int,
    ) -> None:
        self.C = C
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the L1 selector and construct a stable selected-column mask."""
        if float(self.C) <= 0:
            raise FeatureSelectionPolicyError("L1 selector C must be strictly positive.")
        if int(self.max_iter) < 1:
            raise FeatureSelectionPolicyError("L1 selector max_iter must be positive.")
        if self.threshold not in {"mean", "median"}:
            raise FeatureSelectionPolicyError(
                "L1 selector threshold must be either 'mean' or 'median'."
            )

        estimator = make_logistic_regression_classifier(
            penalty="l1",
            C=float(self.C),
            class_weight="balanced",
            solver="saga",
            max_iter=int(self.max_iter),
            random_state=int(self.random_state),
        )
        self.selector_ = SelectFromModel(
            estimator=estimator,
            threshold=self.threshold,
        )
        self.selector_.fit(X, y)

        support = np.asarray(self.selector_.get_support(), dtype=bool)
        if support.ndim != 1:
            raise FeatureSelectionPolicyError("L1 selector returned an invalid support shape.")
        if not support.any():
            coefficients = np.asarray(self.selector_.estimator_.coef_, dtype=float)
            importances = np.max(np.abs(coefficients), axis=0)
            if importances.ndim != 1 or importances.size != support.size:
                raise FeatureSelectionPolicyError(
                    "L1 selector produced coefficient importances with an invalid shape."
                )
            if not np.isfinite(importances).all():
                raise FeatureSelectionPolicyError(
                    "L1 selector produced non-finite coefficient importances."
                )
            support[int(np.argmax(importances))] = True

        self.n_features_in_ = int(support.size)
        self.support_mask_ = support
        self.n_selected_features_ = int(support.sum())
        return self

    def transform(self, X):
        """Return only columns retained by the fitted L1 selection rule."""
        check_is_fitted(self, "support_mask_")
        if sparse.issparse(X):
            return X[:, self.support_mask_]
        array = np.asarray(X)
        if array.ndim != 2:
            raise FeatureSelectionPolicyError("Feature selectors require a two-dimensional matrix.")
        return array[:, self.support_mask_]

    def get_support(self, indices: bool = False):
        """Return the fitted selected-column mask."""
        check_is_fitted(self, "support_mask_")
        if indices:
            return np.flatnonzero(self.support_mask_)
        return self.support_mask_.copy()


def make_feature_selector(
    policy_id: str,
    *,
    n_numeric_features: int,
    parameters: Mapping[str, object] | None,
    random_state: int,
):
    """Build one unfitted selector or ``'passthrough'`` for the no-selection policy."""
    policy_id = validate_feature_selection_policy_id(policy_id)
    parameters = dict(parameters or {})

    if policy_id == FEATURE_SELECTION_NONE:
        if parameters:
            raise FeatureSelectionPolicyError(
                "S0_NONE must not receive selector-specific hyperparameters."
            )
        return "passthrough"

    if policy_id == FEATURE_SELECTION_VARIANCE_MUTUAL_INFO:
        allowed = {"selection_k"}
        unexpected = sorted(set(parameters).difference(allowed))
        if unexpected:
            raise FeatureSelectionPolicyError(
                f"S1_VARIANCE_MUTUAL_INFO received unsupported parameters: {unexpected!r}."
            )
        if "selection_k" not in parameters:
            raise FeatureSelectionPolicyError(
                "S1_VARIANCE_MUTUAL_INFO requires 'selection_k'."
            )
        return VarianceThresholdMutualInfoSelector(
            requested_k=int(parameters["selection_k"]),
            n_numeric_features=int(n_numeric_features),
            random_state=int(random_state),
        )

    if policy_id == FEATURE_SELECTION_L1_LOGISTIC:
        allowed = {"selection_l1_C", "selection_l1_threshold"}
        unexpected = sorted(set(parameters).difference(allowed))
        if unexpected:
            raise FeatureSelectionPolicyError(
                "S2_L1_LOGISTIC_SELECT_FROM_MODEL received unsupported parameters: "
                f"{unexpected!r}."
            )
        missing = sorted(allowed.difference(parameters))
        if missing:
            raise FeatureSelectionPolicyError(
                "S2_L1_LOGISTIC_SELECT_FROM_MODEL is missing parameters: "
                f"{missing!r}."
            )
        return L1LogisticSelectFromModel(
            C=float(parameters["selection_l1_C"]),
            threshold=str(parameters["selection_l1_threshold"]),
            max_iter=8_000,
            random_state=int(random_state),
        )

    raise RuntimeError(f"Unexpected validated feature-selection policy {policy_id!r}.")
