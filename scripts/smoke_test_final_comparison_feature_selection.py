"""Training-only smoke test for final-comparison feature-selection policies.

The test exercises every nontrivial candidate-policy-selector route admitted by the
registry.  It deliberately reuses the existing core and feature-policy smoke tests for
the no-selection routes, then focuses on the new S1 and S2 branches.  No held-out test
rows are loaded.
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
    CANDIDATE_EXTRA_TREES,
    CORE_CANDIDATE_REGISTRY,
    CandidateRegistryError,
    build_candidate_pipeline,
    candidate_procedure_contract_fingerprint,
    suggest_candidate_parameters,
    supported_feature_policies,
    supported_feature_selection_policies,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_selection import (  # noqa: E402
    FEATURE_SELECTION_L1_LOGISTIC,
    FEATURE_SELECTION_NONE,
    FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
    L1LogisticSelectFromModel,
    VarianceThresholdMutualInfoSelector,
)
from telco_churn.hpo import extract_continuous_scores  # noqa: E402


SAMPLE_SIZE = 480


class DeterministicSmokeTrial:
    """Minimal Optuna-compatible object selecting the first declared model setting."""

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
    """Load only development data and create a deterministic stratified subset."""
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


def make_nontrivial_selection_routes():
    """Return every registry route whose selector is not S0_NONE."""
    routes = []
    for definition in CORE_CANDIDATE_REGISTRY:
        for feature_policy in supported_feature_policies(definition.candidate_id):
            for selection_policy in supported_feature_selection_policies(
                definition.candidate_id,
                feature_policy,
            ):
                if selection_policy != FEATURE_SELECTION_NONE:
                    routes.append((definition, feature_policy, selection_policy))
    return routes


def parameters_for_route(
    *,
    trial: DeterministicSmokeTrial,
    candidate_id: str,
    feature_policy: str,
    selection_policy: str,
) -> dict[str, object]:
    """Create one valid smoke-scale parameter dictionary for a forced selector route."""
    parameters = suggest_candidate_parameters(
        trial,
        candidate_id=candidate_id,
        profile="smoke",
    )
    for name in ("selection_k", "selection_l1_C", "selection_l1_threshold"):
        parameters.pop(name, None)
    parameters["feature_policy"] = feature_policy
    parameters["feature_selection_policy"] = selection_policy

    if selection_policy == FEATURE_SELECTION_VARIANCE_MUTUAL_INFO:
        parameters["selection_k"] = 12
    elif selection_policy == FEATURE_SELECTION_L1_LOGISTIC:
        parameters["selection_l1_C"] = 0.1
        parameters["selection_l1_threshold"] = "mean"
    else:
        raise AssertionError(f"Unexpected nontrivial selection policy {selection_policy!r}.")
    return parameters


def assert_selector_contract(*, fitted_pipeline, selection_policy: str) -> None:
    """Verify selector placement, fitted type, and nonempty selected support."""
    expected_step_order = [
        "feature_policy",
        "preprocessor",
        "feature_selection",
        "classifier",
    ]
    observed_step_order = [name for name, _ in fitted_pipeline.steps]
    if observed_step_order != expected_step_order:
        raise AssertionError(
            f"Unexpected pipeline step order: {observed_step_order!r}."
        )

    selector = fitted_pipeline.named_steps["feature_selection"]
    if selection_policy == FEATURE_SELECTION_VARIANCE_MUTUAL_INFO:
        if not isinstance(selector, VarianceThresholdMutualInfoSelector):
            raise AssertionError("S1 route did not fit the expected mutual-information selector.")
    elif selection_policy == FEATURE_SELECTION_L1_LOGISTIC:
        if not isinstance(selector, L1LogisticSelectFromModel):
            raise AssertionError("S2 route did not fit the expected L1 selector.")
    else:
        raise AssertionError(f"Unexpected selector policy {selection_policy!r}.")

    support = np.asarray(selector.get_support(), dtype=bool)
    if support.ndim != 1 or not support.any():
        raise AssertionError("A fitted selection route must retain at least one feature.")
    if int(selector.n_selected_features_) != int(support.sum()):
        raise AssertionError("Selector selected-column metadata is inconsistent.")


def assert_score_contract(*, candidate_id: str, score_kind: str, fitted_pipeline, X_validation):
    """Check binary predictions and declared probability-versus-margin semantics."""
    predictions = np.asarray(fitted_pipeline.predict(X_validation))
    if predictions.shape != (len(X_validation),) or not np.isin(predictions, [0, 1]).all():
        raise AssertionError(f"{candidate_id} returned invalid binary predictions.")

    scores, observed_kind = extract_continuous_scores(fitted_pipeline, X_validation)
    scores = np.asarray(scores, dtype=float)
    if observed_kind != score_kind:
        raise AssertionError(
            f"{candidate_id} declares {score_kind!r} but returned {observed_kind!r}."
        )
    if scores.shape != (len(X_validation),) or not np.isfinite(scores).all():
        raise AssertionError(f"{candidate_id} returned invalid continuous scores.")
    if observed_kind == "probability" and np.any((scores < 0.0) | (scores > 1.0)):
        raise AssertionError(f"{candidate_id} returned invalid probability values.")


def smoke_test_feature_selection_routes(X: pd.DataFrame, y: pd.Series) -> None:
    """Fit every admitted S1/S2 route on a small training-only split."""
    validate_candidate_registry(CORE_CANDIDATE_REGISTRY)
    X_train, X_validation, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    trial = DeterministicSmokeTrial()
    routes = make_nontrivial_selection_routes()
    expected_route_count = 22
    if len(routes) != expected_route_count:
        raise AssertionError(
            f"Expected {expected_route_count} declared nontrivial selection routes, "
            f"found {len(routes)}."
        )

    for index, (definition, feature_policy, selection_policy) in enumerate(routes, start=1):
        print(
            f"[{index:02d}/{len(routes):02d}] {definition.candidate_id} with "
            f"{feature_policy} and {selection_policy}: {definition.display_name}...",
            flush=True,
        )
        parameters = parameters_for_route(
            trial=trial,
            candidate_id=definition.candidate_id,
            feature_policy=feature_policy,
            selection_policy=selection_policy,
        )
        pipeline = build_candidate_pipeline(
            definition.candidate_id,
            parameters,
            random_state=RANDOM_STATE + index,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fitted_pipeline = clone(pipeline).fit(X_train, y_train)

        assert_selector_contract(
            fitted_pipeline=fitted_pipeline,
            selection_policy=selection_policy,
        )
        assert_score_contract(
            candidate_id=definition.candidate_id,
            score_kind=definition.score_kind,
            fitted_pipeline=fitted_pipeline,
            X_validation=X_validation,
        )

    # Tree ensembles deliberately remain S0-only.  This must fail before estimator fit.
    invalid_parameters = parameters_for_route(
        trial=trial,
        candidate_id=CORE_CANDIDATE_REGISTRY[0].candidate_id,
        feature_policy=supported_feature_policies(CORE_CANDIDATE_REGISTRY[0].candidate_id)[0],
        selection_policy=FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
    )
    invalid_parameters["feature_policy"] = supported_feature_policies(CANDIDATE_EXTRA_TREES)[0]
    invalid_parameters["feature_selection_policy"] = FEATURE_SELECTION_VARIANCE_MUTUAL_INFO
    try:
        build_candidate_pipeline(
            CANDIDATE_EXTRA_TREES,
            invalid_parameters,
            random_state=RANDOM_STATE,
        )
    except CandidateRegistryError:
        pass
    else:
        raise AssertionError("Extra Trees must reject an undeclared S1 selection route.")

    for definition in CORE_CANDIDATE_REGISTRY:
        first = candidate_procedure_contract_fingerprint(definition.candidate_id)
        second = candidate_procedure_contract_fingerprint(definition.candidate_id)
        if first != second or len(first) != 64:
            raise AssertionError("Candidate-procedure contract fingerprint is not stable.")


def main() -> None:
    X, y = make_small_stratified_sample()
    print(
        "Checking all declared nontrivial feature-selection routes on training data only...",
        flush=True,
    )
    smoke_test_feature_selection_routes(X, y)
    print("Final-comparison feature-selection smoke test passed.")


if __name__ == "__main__":
    main()
