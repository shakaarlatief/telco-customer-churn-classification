"""Training-only routing smoke test for candidate-specific feature policies.

The script validates every declared candidate-policy combination without estimating
model quality.  It makes a small stratified sample from ``train.csv``, fits each
complete pipeline on a training partition, and verifies that the selected policy is
the first fold-local pipeline step, exposes its declared deterministic schema, and
produces valid continuous scores on a validation partition.

The test is intentionally separate from the persistent Optuna smoke test.  Optuna
already validates resumable HPO mechanics.  This script validates the larger matrix of
feature-policy compatibility before those policy choices are admitted to an expensive
nested comparison.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_KNN,
    CORE_CANDIDATE_REGISTRY,
    CandidateRegistryError,
    build_candidate_pipeline,
    suggest_candidate_parameters,
    supported_feature_policies,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.core_candidate_builders import (  # noqa: E402
    declared_single_thread_parameter,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_LINEAR_EXPANDED,
    feature_policy_output_features,
)
from telco_churn.hpo import extract_continuous_scores  # noqa: E402


SAMPLE_SIZE = 480


class DeterministicSmokeTrial:
    """Small Optuna-compatible object selecting first declared smoke values."""

    def suggest_categorical(self, name: str, choices):
        if not choices:
            raise AssertionError(f"Smoke trial received no choices for {name!r}.")
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, **kwargs):
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise AssertionError(f"Invalid float range for {name!r}: {low}, {high}.")
        return float(low)

    def suggest_int(self, name: str, low: int, high: int, **kwargs):
        if low > high:
            raise AssertionError(f"Invalid integer range for {name!r}: {low}, {high}.")
        return int(low)


def make_small_stratified_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Load development data only and select a deterministic smoke subset."""
    train_df = load_train_data()
    X, y = split_features_target(train_df)

    if len(y) <= SAMPLE_SIZE:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    X_small, _, y_small, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_small.reset_index(drop=True), y_small.reset_index(drop=True)


def assert_valid_scores(
    *,
    candidate_id: str,
    declared_score_kind: str,
    fitted_pipeline,
    X_validation: pd.DataFrame,
) -> None:
    """Verify that routing preserves the candidate's continuous score contract."""
    scores, observed_score_kind = extract_continuous_scores(fitted_pipeline, X_validation)
    scores = np.asarray(scores, dtype=float)

    if observed_score_kind != declared_score_kind:
        raise AssertionError(
            f"{candidate_id} declares {declared_score_kind!r} but exposed "
            f"{observed_score_kind!r} scores."
        )
    if scores.shape != (len(X_validation),):
        raise AssertionError(f"{candidate_id} returned invalid score shape {scores.shape}.")
    if not np.isfinite(scores).all():
        raise AssertionError(f"{candidate_id} returned non-finite continuous scores.")
    if observed_score_kind == "probability" and np.any((scores < 0.0) | (scores > 1.0)):
        raise AssertionError(f"{candidate_id} returned values outside [0, 1].")


def assert_policy_routing(
    *,
    fitted_pipeline,
    policy_id: str,
    X_training: pd.DataFrame,
) -> None:
    """Verify policy identity and exact deterministic post-policy schema."""
    transformer = fitted_pipeline.named_steps.get("feature_policy")
    if transformer is None:
        raise AssertionError("A final-comparison pipeline is missing its feature-policy step.")
    if transformer.policy_id_ != policy_id:
        raise AssertionError(
            f"Pipeline fitted policy {transformer.policy_id_!r}, expected {policy_id!r}."
        )

    expected_columns = feature_policy_output_features(policy_id)
    observed_columns = transformer.get_feature_names_out().tolist()
    if observed_columns != expected_columns:
        raise AssertionError(
            f"Feature policy {policy_id} returned a non-contract schema."
        )

    transformed = transformer.transform(X_training.iloc[:3])
    if list(transformed.columns) != expected_columns:
        raise AssertionError(
            f"Feature policy {policy_id} transform columns differ from its contract."
        )


def run_policy_routing_smoke(X: pd.DataFrame, y: pd.Series) -> None:
    """Fit one low-cost representative pipeline for every compatible route."""
    validate_candidate_registry(CORE_CANDIDATE_REGISTRY)
    X_training, X_validation, y_training, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    trial = DeterministicSmokeTrial()
    routes = [
        (definition, policy_id)
        for definition in CORE_CANDIDATE_REGISTRY
        for policy_id in supported_feature_policies(definition.candidate_id)
    ]

    expected_route_count = 49
    if len(routes) != expected_route_count:
        raise AssertionError(
            f"Expected {expected_route_count} declared candidate-policy routes, "
            f"found {len(routes)}."
        )

    for index, (definition, policy_id) in enumerate(routes, start=1):
        print(
            f"[{index:02d}/{len(routes):02d}] {definition.candidate_id} "
            f"with {policy_id}: {definition.display_name}...",
            flush=True,
        )
        parameters = suggest_candidate_parameters(
            trial,
            candidate_id=definition.candidate_id,
            profile="smoke",
        )
        parameters["feature_policy"] = policy_id

        fitted_pipeline = clone(
            build_candidate_pipeline(
                definition.candidate_id,
                parameters,
                random_state=RANDOM_STATE + index,
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fitted_pipeline.fit(X_training, y_training)

        predictions = np.asarray(fitted_pipeline.predict(X_validation))
        if predictions.shape != (len(X_validation),):
            raise AssertionError(f"{definition.candidate_id} returned invalid labels.")
        if not np.isin(predictions, [0, 1]).all():
            raise AssertionError(
                f"{definition.candidate_id} returned labels outside the binary target space."
            )

        assert_policy_routing(
            fitted_pipeline=fitted_pipeline,
            policy_id=policy_id,
            X_training=X_training,
        )
        assert_valid_scores(
            candidate_id=definition.candidate_id,
            declared_score_kind=definition.score_kind,
            fitted_pipeline=fitted_pipeline,
            X_validation=X_validation,
        )

        classifier = fitted_pipeline.named_steps["classifier"]
        parallelism = declared_single_thread_parameter(
            definition.candidate_id,
            classifier,
        )
        if parallelism is not None:
            name, observed = parallelism
            if observed != 1:
                raise AssertionError(
                    f"{definition.candidate_id} has {name}={observed}; expected one."
                )

    # F2 is intentionally unavailable to distance, probabilistic, tree, boosting,
    # kernel, and neural procedures. This assertion proves incompatible routes fail
    # before expensive estimator fitting begins.
    knn_parameters = suggest_candidate_parameters(
        trial,
        candidate_id=CANDIDATE_KNN,
        profile="smoke",
    )
    knn_parameters["feature_policy"] = FEATURE_POLICY_LINEAR_EXPANDED
    try:
        build_candidate_pipeline(
            CANDIDATE_KNN,
            knn_parameters,
            random_state=RANDOM_STATE,
        )
    except CandidateRegistryError:
        pass
    else:
        raise AssertionError("kNN must reject the F2 linear-only feature policy.")


def main() -> None:
    X, y = make_small_stratified_sample()
    print(
        "Checking all declared final-comparison candidate feature-policy routes "
        "on training data only...",
        flush=True,
    )
    run_policy_routing_smoke(X, y)
    print("Final-comparison feature-policy routing smoke test passed.")


if __name__ == "__main__":
    main()
