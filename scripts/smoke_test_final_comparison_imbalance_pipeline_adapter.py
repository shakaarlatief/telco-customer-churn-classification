"""Training-only integration smoke test for the imbalance-pipeline adapter.

This test does not yet expose imbalance policies in the candidate registry. It verifies
the pipeline topologies that the later registry-routing phase will reuse: no treatment,
fold-local balanced sample weighting, random over/undersampling after preprocessing, and
raw-only SMOTENC before one-hot encoding. The held-out test set is never loaded.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_RAW,
    feature_policy_categorical_features,
)
from telco_churn.feature_policy_pipelines import (  # noqa: E402
    CloneSafeFeaturePolicyCatBoostClassifier,
    FeaturePolicyPipelineError,
    REPRESENTATION_NATIVE_CATEGORICAL_STRING,
    REPRESENTATION_SPARSE_SCALED,
    apply_imbalance_policy_to_pipeline,
    make_feature_policy_classifier_pipeline,
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
from telco_churn.models import make_logistic_regression_classifier  # noqa: E402


SAMPLE_SIZE = 480


def make_development_partition() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a deterministic, development-only train/validation split."""
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
        random_state=RANDOM_STATE + 1,
    )


def make_logistic_base_pipeline(*, policy_id: str):
    """Construct one low-cost raw classifier pipeline with no estimator-side weighting."""
    classifier = make_logistic_regression_classifier(
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=4_000,
        random_state=RANDOM_STATE,
    )
    return make_feature_policy_classifier_pipeline(
        policy_id=policy_id,
        representation=REPRESENTATION_SPARSE_SCALED,
        classifier=classifier,
        random_state=RANDOM_STATE,
    )


def assert_valid_probability_scores(pipeline, X_validation: pd.DataFrame) -> None:
    """Verify valid class-one probabilities from a fitted integration pipeline."""
    scores, score_kind = extract_continuous_scores(pipeline, X_validation)
    scores = np.asarray(scores, dtype=float)
    if score_kind != "probability":
        raise AssertionError("The logistic integration route must expose probabilities.")
    if scores.shape != (len(X_validation),) or not np.isfinite(scores).all():
        raise AssertionError("The integration route returned invalid continuous scores.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise AssertionError("The integration route returned values outside [0, 1].")


def run_adapter_smoke(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Fit every general imbalance topology on development data only."""
    cases = (
        (IMBALANCE_NONE, {}, ["feature_policy", "preprocessor", "feature_selection", "classifier"]),
        (
            IMBALANCE_CLASS_WEIGHT_BALANCED,
            {},
            ["feature_policy", "preprocessor", "feature_selection", "classifier"],
        ),
        (
            IMBALANCE_RANDOM_OVERSAMPLING,
            {"imbalance_sampling_strategy": 0.75},
            ["feature_policy", "preprocessor", "sampler", "feature_selection", "classifier"],
        ),
        (
            IMBALANCE_RANDOM_UNDERSAMPLING,
            {"imbalance_sampling_strategy": 0.75},
            ["feature_policy", "preprocessor", "sampler", "feature_selection", "classifier"],
        ),
        (
            IMBALANCE_SMOTENC,
            {
                "imbalance_sampling_strategy": 0.75,
                "imbalance_smotenc_k_neighbors": 3,
            },
            [
                "feature_policy",
                "sampler_imputer",
                "sampler",
                "preprocessor",
                "feature_selection",
                "classifier",
            ],
        ),
    )

    for index, (policy_id, parameters, expected_steps) in enumerate(cases, start=1):
        print(f"  [{index}/{len(cases)}] Checking {policy_id}...", flush=True)
        base = make_logistic_base_pipeline(policy_id=FEATURE_POLICY_RAW)
        routed = apply_imbalance_policy_to_pipeline(
            base,
            imbalance_policy=policy_id,
            imbalance_parameters=parameters,
            random_state=RANDOM_STATE + index,
        )
        fitted = clone(routed).fit(X_train, y_train)
        if list(fitted.named_steps) != expected_steps:
            raise AssertionError(
                f"{policy_id} created {list(fitted.named_steps)!r}, expected {expected_steps!r}."
            )
        if policy_id == IMBALANCE_CLASS_WEIGHT_BALANCED:
            weighted = fitted.named_steps["classifier"]
            if not isinstance(weighted, BalancedSampleWeightClassifier):
                raise AssertionError("I1 must wrap the final classifier.")
            mass_zero = weighted.class_weight_mapping_[0] * weighted.class_counts_[0]
            mass_one = weighted.class_weight_mapping_[1] * weighted.class_counts_[1]
            if not np.isclose(mass_zero, mass_one):
                raise AssertionError("I1 did not equalize total fitted class mass.")
        assert_valid_probability_scores(fitted, X_validation)

    domain_base = make_logistic_base_pipeline(policy_id=FEATURE_POLICY_DOMAIN)
    try:
        apply_imbalance_policy_to_pipeline(
            domain_base,
            imbalance_policy=IMBALANCE_SMOTENC,
            imbalance_parameters={
                "imbalance_sampling_strategy": 0.75,
                "imbalance_smotenc_k_neighbors": 3,
            },
            random_state=RANDOM_STATE,
        )
    except FeaturePolicyPipelineError:
        pass
    else:
        raise AssertionError("F1 must reject SMOTENC before any estimator fit.")


def run_catboost_weight_smoke(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Verify that the policy-aware CatBoost wrapper accepts I1 sample weights."""
    classifier = CloneSafeFeaturePolicyCatBoostClassifier(
        iterations=50,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        categorical_features=(
            "SeniorCitizen",
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ),
        random_state=RANDOM_STATE,
        thread_count=1,
    )
    base = make_feature_policy_classifier_pipeline(
        policy_id=FEATURE_POLICY_RAW,
        representation=REPRESENTATION_NATIVE_CATEGORICAL_STRING,
        classifier=classifier,
        random_state=RANDOM_STATE,
    )
    fitted = clone(
        apply_imbalance_policy_to_pipeline(
            base,
            imbalance_policy=IMBALANCE_CLASS_WEIGHT_BALANCED,
            random_state=RANDOM_STATE,
        )
    ).fit(X_train, y_train)
    assert_valid_probability_scores(fitted, X_validation)


def main() -> None:
    X_train, X_validation, y_train, _ = make_development_partition()
    print("Checking imbalance pipeline-adapter topologies on development data only...", flush=True)
    run_adapter_smoke(X_train, X_validation, y_train)
    print("Checking CatBoost weighted adapter compatibility...", flush=True)
    run_catboost_weight_smoke(X_train, X_validation, y_train)
    print("Final-comparison imbalance pipeline-adapter smoke test passed.")


if __name__ == "__main__":
    main()
