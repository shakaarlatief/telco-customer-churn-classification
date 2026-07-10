"""Training-only preflight for every Phase-3 core final-comparison candidate.

The script does not estimate model performance and does not run the held-out test
set. It uses one small stratified sample from ``train.csv`` to verify that every
registered core candidate can:

1. suggest a JSON-safe smoke configuration;
2. construct a fresh fold-safe pipeline;
3. clone, fit, and predict from a training-only split;
4. expose the score type declared in the registry; and
5. keep candidate-level inner parallelism at one where the estimator exposes a
   worker-count parameter.

The persistent interruption-and-resume mechanics are tested separately by
``smoke_test_final_comparison_optuna.py``. Keeping this registry preflight focused
makes it easier to identify which candidate family has a dependency, preprocessing,
or estimator-contract problem before a longer nested-HPO run is started.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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
    CANDIDATE_ADABOOST,
    CANDIDATE_LINEAR_SVM,
    INITIAL_CANDIDATE_REGISTRY,
    build_candidate_pipeline,
    suggest_candidate_parameters,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.core_candidate_builders import (  # noqa: E402
    declared_single_thread_parameter,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.hpo import extract_continuous_scores  # noqa: E402


SAMPLE_SIZE = 480


class DeterministicSmokeTrial:
    """Minimal Optuna-compatible trial that selects the lower-cost representative value.

    This object validates the candidate search-space code without creating persistent
    studies. It intentionally chooses the first categorical option and the lowest
    numeric value from every declared range, which keeps the registry preflight short
    while exercising every parameter decoder and pipeline builder.
    """

    def __init__(self) -> None:
        self.int_ranges: dict[str, list[tuple[int, int, int]]] = {}

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
        step = int(kwargs.get("step", 1))
        if step < 1:
            raise AssertionError(f"Invalid integer step for {name!r}: {step}.")
        if (high - low) % step != 0:
            raise AssertionError(
                f"Integer range for {name!r} is not step-aligned: "
                f"low={low}, high={high}, step={step}."
            )
        self.int_ranges.setdefault(name, []).append((int(low), int(high), step))
        return int(low)


def make_small_stratified_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Load only development data and make a deterministic stratified smoke sample."""
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


def assert_valid_continuous_scores(
    *,
    candidate_id: str,
    declared_score_kind: str,
    fitted_pipeline,
    X_validation: pd.DataFrame,
) -> None:
    """Validate the registry's probability-versus-margin declaration after fitting."""
    scores, observed_score_kind = extract_continuous_scores(fitted_pipeline, X_validation)
    scores = np.asarray(scores, dtype=float)

    if observed_score_kind != declared_score_kind:
        raise AssertionError(
            f"{candidate_id} declares {declared_score_kind!r} but exposes "
            f"{observed_score_kind!r} scores."
        )
    if scores.shape != (len(X_validation),):
        raise AssertionError(f"{candidate_id} returned an invalid score shape: {scores.shape}.")
    if not np.isfinite(scores).all():
        raise AssertionError(f"{candidate_id} returned non-finite continuous scores.")
    if observed_score_kind == "probability" and np.any((scores < 0.0) | (scores > 1.0)):
        raise AssertionError(f"{candidate_id} returned values outside [0, 1].")


def assert_warning_cleanup_contracts() -> None:
    """Verify targeted warning-cleanup contracts before fitting all candidates."""
    trial = DeterministicSmokeTrial()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        parameters = suggest_candidate_parameters(
            trial,
            candidate_id=CANDIDATE_ADABOOST,
            profile="smoke",
        )
    if any("not divisible by `step`" in str(warning.message) for warning in caught_warnings):
        raise AssertionError("C13 smoke n_estimators search space emitted a step warning.")
    n_estimators_ranges = trial.int_ranges.get("n_estimators", [])
    if n_estimators_ranges != [(25, 100, 25)]:
        raise AssertionError(
            "C13 smoke n_estimators range must remain the effective "
            f"step-aligned range [(25, 100, 25)], got {n_estimators_ranges!r}."
        )
    if int(parameters["n_estimators"]) != 25:
        raise AssertionError("C13 deterministic smoke trial should select 25 estimators.")


def assert_linear_svm_max_iter_contract(parameters: dict[str, object], pipeline) -> None:
    """Verify C21 exposes the increased Liblinear iteration budget."""
    if parameters.get("loss") != "squared_hinge":
        raise AssertionError("C21 search space must suggest only squared_hinge loss.")
    if int(parameters.get("max_iter", 0)) != 100_000:
        raise AssertionError("C21 smoke parameters must include max_iter=100_000.")
    classifier = pipeline.named_steps["classifier"]
    if int(getattr(classifier, "max_iter", 0)) != 100_000:
        raise AssertionError("C21 built classifier must expose max_iter=100_000.")


def smoke_test_core_candidate_registry(X: pd.DataFrame, y: pd.Series) -> None:
    """Fit and score one representative training-only pipeline for every candidate."""
    validate_candidate_registry()
    assert_warning_cleanup_contracts()
    X_train, X_validation, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    trial = DeterministicSmokeTrial()

    for index, definition in enumerate(INITIAL_CANDIDATE_REGISTRY, start=1):
        print(
            f"[{index:02d}/{len(INITIAL_CANDIDATE_REGISTRY):02d}] "
            f"Checking {definition.candidate_id}: {definition.display_name}...",
            flush=True,
        )
        parameters = suggest_candidate_parameters(
            trial,
            candidate_id=definition.candidate_id,
            profile="smoke",
        )
        if not parameters:
            raise AssertionError(f"{definition.candidate_id} returned no smoke parameters.")

        pipeline = build_candidate_pipeline(
            definition.candidate_id,
            parameters,
            random_state=RANDOM_STATE + index,
        )
        if definition.candidate_id == CANDIDATE_LINEAR_SVM:
            assert_linear_svm_max_iter_contract(parameters, pipeline)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fitted_pipeline = clone(pipeline).fit(X_train, y_train)

        y_pred = np.asarray(fitted_pipeline.predict(X_validation))
        if y_pred.shape != (len(X_validation),):
            raise AssertionError(
                f"{definition.candidate_id} returned an invalid prediction shape."
            )
        if not np.isin(y_pred, [0, 1]).all():
            raise AssertionError(
                f"{definition.candidate_id} returned labels outside the binary target space."
            )

        assert_valid_continuous_scores(
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
            parameter_name, observed_value = parallelism
            if observed_value != 1:
                raise AssertionError(
                    f"{definition.candidate_id} has {parameter_name}={observed_value}; "
                    "outer-task workers require inner parallelism to be one."
                )


def main() -> None:
    X, y = make_small_stratified_sample()
    print(
        "Checking the complete core candidate registry on a small training-only "
        "stratified sample...",
        flush=True,
    )
    smoke_test_core_candidate_registry(X, y)
    print("Core final-comparison candidate registry smoke test passed.")


if __name__ == "__main__":
    main()
