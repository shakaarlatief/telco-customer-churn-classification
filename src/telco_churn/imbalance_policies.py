"""Fold-safe imbalance-treatment primitives for final Telco model comparison.

The final comparison treats imbalance handling as a complete candidate-procedure
component rather than a correction applied after cross-validation. This module defines
the initial bounded policy family and the reusable samplers that later candidate
pipelines will place strictly inside their fitted training path.

``I0_NONE``
    Keep the observed training-fold class distribution.

``I1_CLASS_WEIGHT_BALANCED``
    Derive balanced binary-class weights from the active training target. The concrete
    estimator adapter is candidate-specific because libraries expose weighting through
    different parameter names.

``I2_RANDOM_OVERSAMPLING``
    Duplicate minority-class training rows after a compatible numeric representation
    has been fitted from the active training fold.

``I3_RANDOM_UNDERSAMPLING``
    Randomly discard majority-class training rows after a compatible numeric
    representation has been fitted from the active training fold.

``I4_SMOTENC``
    Create synthetic minority examples from a mixed numeric/categorical feature-policy
    table before one-hot encoding. It is intentionally separated from ordinary SMOTE:
    interpolating one-hot indicators would create invalid fractional categories.

The module deliberately owns generic policy mechanics only. Candidate-specific routing,
classifier weight injection, and pipeline placement are added after the sampler contracts
have passed their independent training-only smoke test.
"""

from __future__ import annotations

from collections import Counter
from typing import Final, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from telco_churn.feature_policies import (
    FeaturePolicyId,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
    validate_feature_policy_id,
)


ImbalancePolicyId = Literal[
    "I0_NONE",
    "I1_CLASS_WEIGHT_BALANCED",
    "I2_RANDOM_OVERSAMPLING",
    "I3_RANDOM_UNDERSAMPLING",
    "I4_SMOTENC",
]

IMBALANCE_NONE: Final[ImbalancePolicyId] = "I0_NONE"
IMBALANCE_CLASS_WEIGHT_BALANCED: Final[ImbalancePolicyId] = "I1_CLASS_WEIGHT_BALANCED"
IMBALANCE_RANDOM_OVERSAMPLING: Final[ImbalancePolicyId] = "I2_RANDOM_OVERSAMPLING"
IMBALANCE_RANDOM_UNDERSAMPLING: Final[ImbalancePolicyId] = "I3_RANDOM_UNDERSAMPLING"
IMBALANCE_SMOTENC: Final[ImbalancePolicyId] = "I4_SMOTENC"

IMBALANCE_POLICY_IDS: Final[tuple[ImbalancePolicyId, ...]] = (
    IMBALANCE_NONE,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
)


class ImbalancePolicyError(ValueError):
    """Raised when an imbalance policy or sampler configuration is invalid."""


def validate_imbalance_policy_id(policy_id: str) -> ImbalancePolicyId:
    """Validate and return one declared imbalance-treatment identifier."""
    if policy_id not in IMBALANCE_POLICY_IDS:
        raise ImbalancePolicyError(
            f"Unknown imbalance policy {policy_id!r}. Expected one of "
            f"{list(IMBALANCE_POLICY_IDS)!r}."
        )
    return policy_id  # type: ignore[return-value]


def is_resampling_policy(policy_id: str) -> bool:
    """Return whether a policy changes the number of fitted training rows."""
    policy_id = validate_imbalance_policy_id(policy_id)
    return policy_id in {
        IMBALANCE_RANDOM_OVERSAMPLING,
        IMBALANCE_RANDOM_UNDERSAMPLING,
        IMBALANCE_SMOTENC,
    }


def normalize_imbalance_parameters(
    policy_id: str,
    parameters: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate and normalize conditional parameters for one imbalance policy.

    Persistent HPO stores a candidate configuration as one flat JSON-compatible
    dictionary. This helper makes every imbalance-policy branch explicit and rejects
    parameters that do not belong to the selected branch. That prevents an apparently
    valid resume from silently ignoring, for example, an SMOTENC neighbour count on a
    random-under-sampling configuration.
    """
    policy_id = validate_imbalance_policy_id(policy_id)
    result = dict(parameters or {})

    if policy_id in {IMBALANCE_NONE, IMBALANCE_CLASS_WEIGHT_BALANCED}:
        if result:
            raise ImbalancePolicyError(
                f"{policy_id} must not receive imbalance-specific parameters."
            )
        return {}

    if policy_id in {IMBALANCE_RANDOM_OVERSAMPLING, IMBALANCE_RANDOM_UNDERSAMPLING}:
        required = {"imbalance_sampling_strategy"}
    elif policy_id == IMBALANCE_SMOTENC:
        required = {
            "imbalance_sampling_strategy",
            "imbalance_smotenc_k_neighbors",
        }
    else:
        raise RuntimeError(f"Unexpected validated imbalance policy {policy_id!r}.")

    unexpected = sorted(set(result).difference(required))
    missing = sorted(required.difference(result))
    if unexpected or missing:
        raise ImbalancePolicyError(
            f"{policy_id} has unexpected parameters {unexpected!r} or missing "
            f"parameters {missing!r}."
        )
    return result


def _binary_class_counts(y: Sequence[int] | pd.Series | np.ndarray) -> dict[int, int]:
    """Validate a binary target and return deterministic class counts."""
    values = np.asarray(y)
    if values.ndim != 1:
        raise ImbalancePolicyError("Imbalance policies require a one-dimensional target.")
    if values.size == 0:
        raise ImbalancePolicyError("Imbalance policies require at least one training row.")

    counts = Counter(int(value) for value in values)
    observed_classes = set(counts)
    if observed_classes != {0, 1}:
        raise ImbalancePolicyError(
            "Binary churn imbalance policies require both class labels 0 and 1; "
            f"observed {sorted(observed_classes)!r}."
        )
    return {0: int(counts[0]), 1: int(counts[1])}


def class_count_summary(y: Sequence[int] | pd.Series | np.ndarray) -> dict[int, int]:
    """Return validated binary counts for a training-fold target vector."""
    return _binary_class_counts(y)


def balanced_class_weight_mapping(
    y: Sequence[int] | pd.Series | np.ndarray,
) -> dict[int, float]:
    """Compute scikit-learn-style balanced binary class weights from training labels.

    For a training fold with :math:`n` rows and :math:`n_c` observations in class
    :math:`c`, the returned weight is :math:`n / (2n_c)`. Each class therefore has the
    same total weighted mass. The calculation must occur from the active training target,
    never from a validation or held-out test partition.
    """
    counts = _binary_class_counts(y)
    total = float(counts[0] + counts[1])
    return {
        0: total / (2.0 * float(counts[0])),
        1: total / (2.0 * float(counts[1])),
    }


class BalancedSampleWeightClassifier(ClassifierMixin, BaseEstimator):
    """Apply fold-local balanced class weighting through ``sample_weight``.

    Estimator libraries use different constructor names for class weighting and some
    estimators expose weighting only through ``fit(sample_weight=...)``. This adapter
    implements one consistent I1 mechanism: it derives
    :math:`n / (2 n_c)` from the active fitting target, clones the wrapped classifier,
    and passes the resulting per-row weights into fitting. No validation or held-out rows
    participate in this calculation.

    Optional score methods are delegated only after fitting. Consequently, a wrapped
    margin-only estimator remains margin-only, while a probabilistic estimator continues
    to expose ``predict_proba``.
    """

    def __init__(self, *, estimator) -> None:
        self.estimator = estimator

    def fit(self, X, y):
        """Fit a cloned classifier with balanced weights from this target vector."""
        weights_by_class = balanced_class_weight_mapping(y)
        target = np.asarray(y)
        sample_weight = np.asarray(
            [weights_by_class[int(value)] for value in target],
            dtype=float,
        )
        fitted_estimator = clone(self.estimator)
        try:
            fitted_estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            raise ImbalancePolicyError(
                "I1_CLASS_WEIGHT_BALANCED requires a classifier whose fit method "
                "accepts sample_weight."
            ) from exc

        classes = getattr(fitted_estimator, "classes_", None)
        if classes is None:
            raise ImbalancePolicyError(
                "A class-weighted binary classifier must expose classes_ after fitting."
            )
        self.estimator_ = fitted_estimator
        self.classes_ = np.asarray(classes)
        self.class_counts_ = _binary_class_counts(y)
        self.class_weight_mapping_ = dict(weights_by_class)
        self.sample_weight_ = sample_weight
        return self

    def predict(self, X):
        """Predict labels using the fitted wrapped classifier."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def __getattr__(self, name: str):
        """Delegate optional fitted-estimator methods without changing score semantics."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            fitted_estimator = object.__getattribute__(self, "estimator_")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(fitted_estimator, name)


def _validate_sampling_strategy(value: object, *, policy_id: str) -> float:
    """Validate a binary resampling ratio in the imbalanced-learn float convention."""
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ImbalancePolicyError(
            f"{policy_id} sampling_strategy must be a numeric value in (0, 1]."
        ) from exc
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ImbalancePolicyError(
            f"{policy_id} sampling_strategy must lie in (0, 1], received {ratio!r}."
        )
    return ratio


def _require_imblearn():
    """Import imbalanced-learn lazily with an actionable environment error."""
    try:
        import imblearn
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for final-comparison resampling policies. "
            "Install the repository requirements before using I2, I3, or I4."
        ) from exc
    return imblearn


def make_random_resampler(
    policy_id: str,
    *,
    sampling_strategy: object,
    random_state: int,
):
    """Build a deterministic random over- or under-sampler for one fitted pipeline.

    Random resampling is representation-agnostic and will later be inserted after the
    representation preprocessor but before feature selection and the classifier. It is
    not valid for I0, I1, or I4 because those policies respectively do no resampling,
    alter a loss, or require mixed-feature SMOTENC placement.
    """
    policy_id = validate_imbalance_policy_id(policy_id)
    ratio = _validate_sampling_strategy(sampling_strategy, policy_id=policy_id)
    _require_imblearn()

    if policy_id == IMBALANCE_RANDOM_OVERSAMPLING:
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(
            sampling_strategy=ratio,
            random_state=int(random_state),
        )
    if policy_id == IMBALANCE_RANDOM_UNDERSAMPLING:
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(
            sampling_strategy=ratio,
            random_state=int(random_state),
        )
    raise ImbalancePolicyError(
        "make_random_resampler supports only I2_RANDOM_OVERSAMPLING and "
        "I3_RANDOM_UNDERSAMPLING."
    )


def make_smotenc_resampler(
    *,
    n_numeric_features: int,
    n_categorical_features: int,
    sampling_strategy: object,
    k_neighbors: object,
    random_state: int,
):
    """Build a deterministic SMOTENC sampler for a mixed feature-policy table.

    The sampler receives a boolean Python list instead of a NumPy boolean array for the
    categorical mask. This is accepted by imbalanced-learn across supported releases and
    avoids ambiguous scalar-comparison behaviour in some dependency combinations.
    """
    if int(n_numeric_features) < 1:
        raise ImbalancePolicyError("SMOTENC requires at least one numeric feature.")
    if int(n_categorical_features) < 1:
        raise ImbalancePolicyError("SMOTENC requires at least one categorical feature.")
    ratio = _validate_sampling_strategy(
        sampling_strategy,
        policy_id=IMBALANCE_SMOTENC,
    )
    try:
        neighbors = int(k_neighbors)
    except (TypeError, ValueError) as exc:
        raise ImbalancePolicyError("SMOTENC k_neighbors must be a positive integer.") from exc
    if neighbors < 1:
        raise ImbalancePolicyError("SMOTENC k_neighbors must be at least one.")

    _require_imblearn()
    from imblearn.over_sampling import SMOTENC

    categorical_mask = [False] * int(n_numeric_features) + [True] * int(
        n_categorical_features
    )
    return SMOTENC(
        categorical_features=categorical_mask,
        sampling_strategy=ratio,
        k_neighbors=neighbors,
        random_state=int(random_state),
    )


class FeaturePolicySamplerImputer(BaseEstimator, TransformerMixin):
    """Impute one policy-expanded mixed table while preserving a pandas schema.

    SMOTENC must receive non-missing numerical and categorical inputs before the later
    one-hot or native-categorical representation is fitted. This transformer learns only
    training-partition medians and deterministic categorical modes. It returns numeric
    columns followed by categorical string columns, matching the declared feature-policy
    order used to build the SMOTENC boolean categorical mask.
    """

    def __init__(self, *, policy_id: FeaturePolicyId):
        self.policy_id = policy_id

    def fit(self, X: pd.DataFrame, y=None):
        """Estimate fold-local fallback values for one feature-policy table."""
        policy_id = validate_feature_policy_id(self.policy_id)
        numeric_features = feature_policy_numeric_features(policy_id)
        categorical_features = feature_policy_categorical_features(policy_id)
        X_frame = self._validate_policy_frame(
            X,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        numeric_fill_values: dict[str, float] = {}
        for column in numeric_features:
            values = pd.to_numeric(X_frame[column], errors="coerce")
            median = float(values.median(skipna=True))
            numeric_fill_values[column] = median if np.isfinite(median) else 0.0

        categorical_fill_values: dict[str, str] = {}
        for column in categorical_features:
            values = X_frame[column].astype("string").dropna().astype(str)
            if values.empty:
                categorical_fill_values[column] = "__MISSING__"
            else:
                categorical_fill_values[column] = str(sorted(values.mode().tolist())[0])

        self.policy_id_ = policy_id
        self.numeric_features_ = tuple(numeric_features)
        self.categorical_features_ = tuple(categorical_features)
        self.numeric_fill_values_ = numeric_fill_values
        self.categorical_fill_values_ = categorical_fill_values
        self.feature_names_in_ = np.asarray(
            [*numeric_features, *categorical_features],
            dtype=object,
        )
        self.feature_names_out_ = self.feature_names_in_.copy()
        self.n_features_in_ = int(self.feature_names_in_.size)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a complete mixed-type DataFrame in the declared policy order."""
        check_is_fitted(
            self,
            attributes=(
                "policy_id_",
                "numeric_features_",
                "categorical_features_",
                "numeric_fill_values_",
                "categorical_fill_values_",
            ),
        )
        X_frame = self._validate_policy_frame(
            X,
            numeric_features=list(self.numeric_features_),
            categorical_features=list(self.categorical_features_),
        )

        numeric = pd.DataFrame(index=X_frame.index)
        for column in self.numeric_features_:
            numeric[column] = (
                pd.to_numeric(X_frame[column], errors="coerce")
                .fillna(self.numeric_fill_values_[column])
                .astype(float)
            )

        categorical = pd.DataFrame(index=X_frame.index)
        for column in self.categorical_features_:
            categorical[column] = (
                X_frame[column]
                .astype("string")
                .fillna(self.categorical_fill_values_[column])
                .astype(str)
            )

        return pd.concat([numeric, categorical], axis=1).loc[
            :, self.feature_names_out_.tolist()
        ]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the fixed sampler input schema after fitting."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    @staticmethod
    def _validate_policy_frame(
        X: pd.DataFrame,
        *,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> pd.DataFrame:
        """Validate and order the deterministic post-policy feature schema."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeaturePolicySamplerImputer expects a pandas DataFrame produced by "
                "FeaturePolicyTransformer."
            )
        expected = [*numeric_features, *categorical_features]
        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise ImbalancePolicyError(
                "Feature-policy output is missing columns required for mixed resampling: "
                f"{missing!r}."
            )
        return X.loc[:, expected].copy()
