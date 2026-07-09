"""Training-only smoke test for the C24 TabNet candidate.

The test validates C24 routing and mechanics on development training data only. It
does not estimate model quality and never loads the held-out test set.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_TABNET,
    CandidateRegistryError,
    build_candidate_pipeline,
    suggest_candidate_parameters,
    supported_feature_policies,
    supported_feature_selection_policies,
    supported_imbalance_policies,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_LINEAR_EXPANDED,
    FEATURE_POLICY_RAW,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
)
from telco_churn.feature_policy_pipelines import (  # noqa: E402
    CloneSafeFeaturePolicyTabNetClassifier,
    FeaturePolicyNativeCategoricalPreprocessor,
)
from telco_churn.feature_selection import (  # noqa: E402
    FEATURE_SELECTION_NONE,
    FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
)
from telco_churn.hpo import extract_continuous_scores  # noqa: E402
from telco_churn.imbalance_policies import (  # noqa: E402
    BalancedSampleWeightClassifier,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_NONE,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
)


SAMPLE_SIZE = 300
KNOWN_NOISY_TABNET_WARNING_SNIPPETS = (
    "Best weights from best epoch are automatically used!",
    "Please import `spmatrix` from the `scipy.sparse` namespace",
)


class DeterministicSmokeTrial:
    """Minimal Optuna-compatible trial selecting the first declared smoke value."""

    def suggest_categorical(self, name: str, choices):
        if not choices:
            raise AssertionError(f"Empty categorical choice set for {name!r}.")
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, **kwargs):
        if low > high or not np.isfinite(low) or not np.isfinite(high):
            raise AssertionError(f"Invalid float range for {name!r}.")
        return float(low)

    def suggest_int(self, name: str, low: int, high: int, **kwargs):
        if low > high:
            raise AssertionError(f"Invalid integer range for {name!r}.")
        return int(low)


def make_partition() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return one deterministic development-only train/validation partition."""
    train_df = load_train_data()
    X, y = split_features_target(train_df)
    if len(y) > SAMPLE_SIZE:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=SAMPLE_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    return train_test_split(
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE + 24,
    )


def parameters_for_route(
    *,
    feature_policy: str,
    imbalance_policy: str,
) -> dict[str, object]:
    """Build a low-cost forced C24 S0 route."""
    parameters = suggest_candidate_parameters(
        DeterministicSmokeTrial(),
        candidate_id=CANDIDATE_TABNET,
        profile="smoke",
    )
    parameters["feature_policy"] = feature_policy
    parameters["feature_selection_policy"] = FEATURE_SELECTION_NONE
    parameters["imbalance_policy"] = imbalance_policy
    parameters.pop("imbalance_sampling_strategy", None)
    parameters.pop("imbalance_smotenc_k_neighbors", None)
    return parameters


def assert_routes() -> None:
    """Verify C24's feature, selector, and weighted-only imbalance contracts."""
    if supported_feature_policies(CANDIDATE_TABNET) != (
        FEATURE_POLICY_RAW,
        FEATURE_POLICY_DOMAIN,
    ):
        raise AssertionError("C24 must support exactly F0 and F1.")
    for feature_policy in (FEATURE_POLICY_RAW, FEATURE_POLICY_DOMAIN):
        if supported_feature_selection_policies(
            CANDIDATE_TABNET,
            feature_policy,
        ) != (FEATURE_SELECTION_NONE,):
            raise AssertionError("C24 must remain S0-only.")
        if supported_imbalance_policies(
            CANDIDATE_TABNET,
            feature_policy,
            FEATURE_SELECTION_NONE,
        ) != (IMBALANCE_NONE, IMBALANCE_CLASS_WEIGHT_BALANCED):
            raise AssertionError("C24 must expose only I0 and I1.")


def assert_probability_output(fitted_pipeline, X_validation: pd.DataFrame) -> None:
    """Validate binary predictions and finite probability scores."""
    predictions = np.asarray(fitted_pipeline.predict(X_validation))
    if predictions.shape != (len(X_validation),) or not np.isin(predictions, [0, 1]).all():
        raise AssertionError("C24 returned invalid binary labels.")

    scores, kind = extract_continuous_scores(fitted_pipeline, X_validation)
    scores = np.asarray(scores, dtype=float)
    if kind != "probability":
        raise AssertionError(f"C24 must expose probabilities, not {kind!r}.")
    if scores.shape != (len(X_validation),) or not np.isfinite(scores).all():
        raise AssertionError("C24 returned invalid continuous scores.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise AssertionError("C24 returned values outside [0, 1].")


def _unwrap_tabnet_classifier(fitted_pipeline) -> CloneSafeFeaturePolicyTabNetClassifier:
    classifier = fitted_pipeline.named_steps["classifier"]
    if isinstance(classifier, BalancedSampleWeightClassifier):
        classifier = classifier.estimator_
    if not isinstance(classifier, CloneSafeFeaturePolicyTabNetClassifier):
        raise AssertionError("C24 did not build a TabNet classifier wrapper.")
    return classifier


def assert_native_string_preprocessor(fitted_pipeline, feature_policy: str) -> None:
    """Verify C24 uses the existing native categorical string preprocessor."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    if not isinstance(preprocessor, FeaturePolicyNativeCategoricalPreprocessor):
        raise AssertionError("C24 must use the native categorical preprocessor.")
    if preprocessor.categorical_dtype:
        raise AssertionError("C24 must receive categorical columns as strings.")
    expected_categoricals = feature_policy_categorical_features(feature_policy)
    if tuple(preprocessor.categorical_features) != tuple(expected_categoricals):
        raise AssertionError("C24 native preprocessor has the wrong categorical schema.")


def assert_tabnet_mechanics(
    fitted_pipeline,
    *,
    feature_policy: str,
    X_validation: pd.DataFrame,
) -> None:
    """Verify TabNet category mappings, embedding metadata, and CPU-safe settings."""
    wrapper = _unwrap_tabnet_classifier(fitted_pipeline)
    numeric_features = tuple(feature_policy_numeric_features(feature_policy))
    categorical_features = tuple(feature_policy_categorical_features(feature_policy))

    if wrapper.device_name != "cpu" or wrapper.num_workers != 0:
        raise AssertionError("C24 must use CPU-safe TabNet settings.")
    if tuple(wrapper.numeric_features) != numeric_features:
        raise AssertionError("C24 TabNet wrapper has the wrong numeric schema.")
    if tuple(wrapper.categorical_features) != categorical_features:
        raise AssertionError("C24 TabNet wrapper has the wrong categorical schema.")

    expected_cat_idxs = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
    )
    if wrapper.cat_idxs_ != expected_cat_idxs:
        raise AssertionError("C24 TabNet cat_idxs do not match the fitted schema.")
    expected_cat_dims = [
        len(wrapper.category_mappings_[column]) + 1
        for column in categorical_features
    ]
    if wrapper.cat_dims_ != expected_cat_dims:
        raise AssertionError("C24 TabNet cat_dims do not match the fitted mappings.")

    unknown_frame = X_validation.iloc[:3].copy()
    first_categorical = categorical_features[0]
    unknown_frame[first_categorical] = unknown_frame[first_categorical].astype(str)
    unknown_frame.loc[:, first_categorical] = "__C24_UNKNOWN_CATEGORY__"
    policy_frame = fitted_pipeline.named_steps["feature_policy"].transform(unknown_frame)
    native_frame = fitted_pipeline.named_steps["preprocessor"].transform(policy_frame)
    encoded = wrapper._transform_with_mappings(native_frame)
    unknown_index = wrapper.cat_idxs_[0]
    if not np.all(encoded[:, unknown_index] == 0):
        raise AssertionError("C24 must encode unknown validation categories as zero.")


def assert_no_known_noisy_warnings(caught_warnings: list[warnings.WarningMessage]) -> None:
    """Ensure the wrapper suppresses only the known harmless TabNet warning noise."""
    observed = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught_warnings
    ]
    noisy = [
        message
        for message in observed
        if any(snippet in message for snippet in KNOWN_NOISY_TABNET_WARNING_SNIPPETS)
    ]
    if noisy:
        raise AssertionError(
            "C24 TabNet smoke still recorded known noisy warnings: "
            f"{noisy!r}."
        )


def fit_and_check(
    *,
    feature_policy: str,
    imbalance_policy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    seed: int,
) -> None:
    """Fit one declared route and validate the C24 contract."""
    pipeline = build_candidate_pipeline(
        CANDIDATE_TABNET,
        parameters_for_route(
            feature_policy=feature_policy,
            imbalance_policy=imbalance_policy,
        ),
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fitted_pipeline = clone(pipeline).fit(X_train, y_train)
    assert_no_known_noisy_warnings(caught_warnings)
    if imbalance_policy == IMBALANCE_CLASS_WEIGHT_BALANCED and not isinstance(
        fitted_pipeline.named_steps["classifier"],
        BalancedSampleWeightClassifier,
    ):
        raise AssertionError("C24 I1 route must use the balanced sample-weight adapter.")
    assert_probability_output(fitted_pipeline, X_validation)
    assert_native_string_preprocessor(fitted_pipeline, feature_policy)
    assert_tabnet_mechanics(
        fitted_pipeline,
        feature_policy=feature_policy,
        X_validation=X_validation,
    )


def assert_rejections() -> None:
    """Ensure excluded C24 routes fail before fitting."""
    invalid_f2 = parameters_for_route(
        feature_policy=FEATURE_POLICY_RAW,
        imbalance_policy=IMBALANCE_NONE,
    )
    invalid_f2["feature_policy"] = FEATURE_POLICY_LINEAR_EXPANDED
    try:
        build_candidate_pipeline(CANDIDATE_TABNET, invalid_f2, random_state=RANDOM_STATE)
    except CandidateRegistryError:
        pass
    else:
        raise AssertionError("C24 must reject F2.")

    invalid_selection = parameters_for_route(
        feature_policy=FEATURE_POLICY_RAW,
        imbalance_policy=IMBALANCE_NONE,
    )
    invalid_selection["feature_selection_policy"] = FEATURE_SELECTION_VARIANCE_MUTUAL_INFO
    try:
        build_candidate_pipeline(
            CANDIDATE_TABNET,
            invalid_selection,
            random_state=RANDOM_STATE,
        )
    except CandidateRegistryError:
        pass
    else:
        raise AssertionError("C24 must reject non-S0 feature selection.")

    for imbalance_policy in (
        IMBALANCE_RANDOM_OVERSAMPLING,
        IMBALANCE_RANDOM_UNDERSAMPLING,
        IMBALANCE_SMOTENC,
    ):
        invalid_imbalance = parameters_for_route(
            feature_policy=FEATURE_POLICY_RAW,
            imbalance_policy=IMBALANCE_NONE,
        )
        invalid_imbalance["imbalance_policy"] = imbalance_policy
        if imbalance_policy in {
            IMBALANCE_RANDOM_OVERSAMPLING,
            IMBALANCE_RANDOM_UNDERSAMPLING,
            IMBALANCE_SMOTENC,
        }:
            invalid_imbalance["imbalance_sampling_strategy"] = 0.75
        if imbalance_policy == IMBALANCE_SMOTENC:
            invalid_imbalance["imbalance_smotenc_k_neighbors"] = 3
        try:
            build_candidate_pipeline(
                CANDIDATE_TABNET,
                invalid_imbalance,
                random_state=RANDOM_STATE,
            )
        except (CandidateRegistryError, ValueError):
            pass
        else:
            raise AssertionError(f"C24 must reject {imbalance_policy}.")


def main() -> None:
    """Run C24's focused training-only smoke test."""
    validate_candidate_registry()
    assert_routes()
    X_train, X_validation, y_train, _ = make_partition()
    fit_and_check(
        feature_policy=FEATURE_POLICY_RAW,
        imbalance_policy=IMBALANCE_NONE,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        seed=RANDOM_STATE + 241,
    )
    fit_and_check(
        feature_policy=FEATURE_POLICY_DOMAIN,
        imbalance_policy=IMBALANCE_CLASS_WEIGHT_BALANCED,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        seed=RANDOM_STATE + 242,
    )
    assert_rejections()
    print("TabNet final-comparison smoke test passed.")


if __name__ == "__main__":
    main()
