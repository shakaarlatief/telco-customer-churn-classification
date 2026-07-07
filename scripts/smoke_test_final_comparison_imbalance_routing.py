"""Training-only smoke test for candidate-specific imbalance routing.

The test verifies that each declared weighting route can fit on a development-only
partition, representative resampling routes fit after the selected representation, and
SMOTENC is admitted only for F0 raw data. No held-out test data is loaded.
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
    CANDIDATE_HYBRID_NAIVE_BAYES,
    CANDIDATE_KNN,
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_SPLINE_LOGISTIC_REGRESSION,
    CANDIDATE_SHRINKAGE_LDA,
    CANDIDATE_REGULARIZED_QDA,
    CANDIDATE_BALANCED_RANDOM_FOREST,
    CANDIDATE_RUSBOOST,
    CANDIDATE_MLP,
    CANDIDATE_RIDGE_CLASSIFIER,
    CANDIDATE_RBF_SVM,
    CANDIDATE_LINEAR_SVM,
    CORE_CANDIDATE_REGISTRY,
    CandidateRegistryError,
    build_candidate_pipeline,
    candidate_procedure_contract,
    suggest_candidate_parameters,
    supported_imbalance_policies,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_LINEAR_EXPANDED,
    FEATURE_POLICY_RAW,
)
from telco_churn.feature_selection import FEATURE_SELECTION_NONE  # noqa: E402
from telco_churn.hpo import extract_continuous_scores  # noqa: E402
from telco_churn.imbalance_routing import suggest_imbalance_configuration  # noqa: E402
from telco_churn.imbalance_policies import (  # noqa: E402
    BalancedSampleWeightClassifier,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_NONE,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
)


SAMPLE_SIZE = 480


class ForcedTrial:
    """Small Optuna-compatible object that forces one declared imbalance policy."""

    def __init__(self, imbalance_policy: str):
        self.imbalance_policy = imbalance_policy

    def suggest_categorical(self, name: str, choices):
        if name.startswith("imbalance_policy__"):
            if self.imbalance_policy not in choices:
                raise AssertionError(
                    f"Forced imbalance policy {self.imbalance_policy!r} is absent from {choices!r}."
                )
            return self.imbalance_policy
        if not choices:
            raise AssertionError(f"Empty choice set for {name!r}.")
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, **kwargs):
        if low > high:
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
        random_state=RANDOM_STATE + 11,
    )


def run_optuna_static_categorical_contract_smoke() -> None:
    """Verify F0 and F1 use different fixed Optuna categorical parameter names.

    A study can visit both feature policies without changing the categorical distribution
    assigned to any individual parameter name. This directly guards against Optuna's
    ``CategoricalDistribution does not support dynamic value space`` error.
    """
    try:
        import optuna
    except ImportError as exc:
        raise AssertionError("Optuna is required for the routing smoke test.") from exc

    study = optuna.create_study(direction="maximize")

    def objective(trial) -> float:
        feature_policy = (
            FEATURE_POLICY_RAW if trial.number == 0 else FEATURE_POLICY_DOMAIN
        )
        configuration = suggest_imbalance_configuration(
            trial,
            candidate_id=CANDIDATE_LINEAR_SVM,
            feature_policy=feature_policy,
            feature_selection_policy=FEATURE_SELECTION_NONE,
            profile="smoke",
        )
        if "imbalance_policy" not in configuration:
            raise AssertionError("Imbalance configuration omitted its canonical policy id.")
        return 0.0

    study.optimize(objective, n_trials=2)
    observed_parameter_names = {
        name
        for trial in study.trials
        for name in trial.params
        if name.startswith("imbalance_policy__")
    }
    expected_parameter_names = {
        "imbalance_policy__f0_raw",
        "imbalance_policy__f1_domain_enriched",
    }
    if observed_parameter_names != expected_parameter_names:
        raise AssertionError(
            "Feature-policy-specific imbalance parameter names are incomplete: "
            f"observed {sorted(observed_parameter_names)!r}."
        )


def parameters_for_route(
    *,
    candidate_id: str,
    feature_policy: str,
    imbalance_policy: str,
) -> dict[str, object]:
    """Build one low-cost smoke configuration for a declared route."""
    parameters = suggest_candidate_parameters(
        ForcedTrial(imbalance_policy),
        candidate_id=candidate_id,
        profile="smoke",
    )
    parameters["feature_policy"] = feature_policy
    parameters["feature_selection_policy"] = FEATURE_SELECTION_NONE
    parameters["imbalance_policy"] = imbalance_policy
    if imbalance_policy in {
        IMBALANCE_RANDOM_OVERSAMPLING,
        IMBALANCE_RANDOM_UNDERSAMPLING,
        IMBALANCE_SMOTENC,
    }:
        parameters["imbalance_sampling_strategy"] = 0.75
    if imbalance_policy == IMBALANCE_SMOTENC:
        parameters["imbalance_smotenc_k_neighbors"] = 3
    return parameters


def fit_and_check(
    *,
    candidate_id: str,
    feature_policy: str,
    imbalance_policy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    seed: int,
) -> None:
    """Fit one declared route and validate topology plus continuous output semantics."""
    parameters = parameters_for_route(
        candidate_id=candidate_id,
        feature_policy=feature_policy,
        imbalance_policy=imbalance_policy,
    )
    pipeline = clone(
        build_candidate_pipeline(
            candidate_id,
            parameters,
            random_state=seed,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        fitted = pipeline.fit(X_train, y_train)

    if imbalance_policy == IMBALANCE_CLASS_WEIGHT_BALANCED:
        weighted = fitted.named_steps["classifier"]
        if not isinstance(weighted, BalancedSampleWeightClassifier):
            raise AssertionError(f"{candidate_id} did not use the I1 classifier adapter.")
        if not np.isclose(
            weighted.class_weight_mapping_[0] * weighted.class_counts_[0],
            weighted.class_weight_mapping_[1] * weighted.class_counts_[1],
        ):
            raise AssertionError(f"{candidate_id} I1 route did not equalize fitted class mass.")
    elif imbalance_policy in {IMBALANCE_RANDOM_OVERSAMPLING, IMBALANCE_RANDOM_UNDERSAMPLING}:
        if "sampler" not in fitted.named_steps or "sampler_imputer" in fitted.named_steps:
            raise AssertionError(f"{candidate_id} has an invalid random-resampling topology.")
    elif imbalance_policy == IMBALANCE_SMOTENC:
        if "sampler" not in fitted.named_steps or "sampler_imputer" not in fitted.named_steps:
            raise AssertionError(f"{candidate_id} has an invalid SMOTENC topology.")

    scores, _ = extract_continuous_scores(fitted, X_validation)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(X_validation),) or not np.isfinite(scores).all():
        raise AssertionError(f"{candidate_id} returned invalid continuous scores.")


def main() -> None:
    validate_candidate_registry(CORE_CANDIDATE_REGISTRY)
    print("Checking Optuna static categorical imbalance distributions...", flush=True)
    run_optuna_static_categorical_contract_smoke()
    X_train, X_validation, y_train, _ = make_partition()

    print("Checking weighted imbalance routes for every declared compatible family...", flush=True)
    weighted_ids = [
        definition.candidate_id
        for definition in CORE_CANDIDATE_REGISTRY
        if IMBALANCE_CLASS_WEIGHT_BALANCED
        in supported_imbalance_policies(
            definition.candidate_id,
            FEATURE_POLICY_RAW,
            FEATURE_SELECTION_NONE,
        )
    ]
    for index, candidate_id in enumerate(weighted_ids, start=1):
        print(f"  [{index:02d}/{len(weighted_ids):02d}] {candidate_id} with I1...", flush=True)
        fit_and_check(
            candidate_id=candidate_id,
            feature_policy=FEATURE_POLICY_RAW,
            imbalance_policy=IMBALANCE_CLASS_WEIGHT_BALANCED,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            seed=RANDOM_STATE + index,
        )

    resampled_secondary_policy = (
        (CANDIDATE_RIDGE_CLASSIFIER, FEATURE_POLICY_LINEAR_EXPANDED),
        (CANDIDATE_LOGISTIC_REGRESSION, FEATURE_POLICY_LINEAR_EXPANDED),
        # C03 intentionally permits only F0 because its raw numeric columns define
        # the spline basis. C04/C05 use the ordinary F1 domain representation.
        (CANDIDATE_SPLINE_LOGISTIC_REGRESSION, FEATURE_POLICY_RAW),
        (CANDIDATE_SHRINKAGE_LDA, FEATURE_POLICY_DOMAIN),
        (CANDIDATE_REGULARIZED_QDA, FEATURE_POLICY_DOMAIN),
        (CANDIDATE_KNN, FEATURE_POLICY_DOMAIN),
        (CANDIDATE_HYBRID_NAIVE_BAYES, FEATURE_POLICY_DOMAIN),
        (CANDIDATE_LINEAR_SVM, FEATURE_POLICY_DOMAIN),
        (CANDIDATE_RBF_SVM, FEATURE_POLICY_DOMAIN),
        (CANDIDATE_MLP, FEATURE_POLICY_DOMAIN),
    )
    cases: list[tuple[str, str, str]] = []
    for candidate_id, secondary_policy in resampled_secondary_policy:
        cases.append((candidate_id, secondary_policy, IMBALANCE_RANDOM_OVERSAMPLING))
        cases.append((candidate_id, secondary_policy, IMBALANCE_RANDOM_UNDERSAMPLING))
        cases.append((candidate_id, FEATURE_POLICY_RAW, IMBALANCE_SMOTENC))

    print("Checking representative random-resampling and SMOTENC routes...", flush=True)
    for index, (candidate_id, feature_policy, imbalance_policy) in enumerate(cases, start=1):
        print(
            f"  [{index:02d}/{len(cases):02d}] {candidate_id} with {feature_policy} and {imbalance_policy}...",
            flush=True,
        )
        fit_and_check(
            candidate_id=candidate_id,
            feature_policy=feature_policy,
            imbalance_policy=imbalance_policy,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            seed=RANDOM_STATE + 100 + index,
        )

    intrinsic_imbalance_ids = (
        CANDIDATE_BALANCED_RANDOM_FOREST,
        CANDIDATE_RUSBOOST,
    )
    print("Checking intrinsic-imbalance candidates expose only I0...", flush=True)
    for index, candidate_id in enumerate(intrinsic_imbalance_ids, start=1):
        policies = supported_imbalance_policies(
            candidate_id,
            FEATURE_POLICY_RAW,
            FEATURE_SELECTION_NONE,
        )
        if policies != (IMBALANCE_NONE,):
            raise AssertionError(
                f"{candidate_id} must expose only I0 because balancing is intrinsic, "
                f"observed {policies!r}."
            )
        fit_and_check(
            candidate_id=candidate_id,
            feature_policy=FEATURE_POLICY_RAW,
            imbalance_policy=IMBALANCE_NONE,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            seed=RANDOM_STATE + 500 + index,
        )

    invalid_parameters = parameters_for_route(
        candidate_id=CANDIDATE_LOGISTIC_REGRESSION,
        feature_policy=FEATURE_POLICY_DOMAIN,
        imbalance_policy=IMBALANCE_SMOTENC,
    )
    try:
        build_candidate_pipeline(
            CANDIDATE_LOGISTIC_REGRESSION,
            invalid_parameters,
            random_state=RANDOM_STATE,
        )
    except (CandidateRegistryError, ValueError):
        pass
    else:
        raise AssertionError("F1 plus I4_SMOTENC must be rejected before fitting.")

    contract = candidate_procedure_contract(CANDIDATE_LOGISTIC_REGRESSION)
    if "imbalance_policies" not in contract:
        raise AssertionError("Persistent study contract omits imbalance routing.")
    print("Final-comparison candidate imbalance-routing smoke test passed.")


if __name__ == "__main__":
    main()
