"""Candidate-specific imbalance routing for final model comparison.

The module contains only immutable compatibility declarations and conditional Optuna
suggestions. It does not fit, resample, or inspect data. Actual fitted-path placement is
owned by ``feature_policy_pipelines.apply_imbalance_policy_to_pipeline``.

One complete candidate procedure selects exactly one imbalance policy. I1 uses
fold-local balanced sample weights. I2-I4 are fit-time-only samplers. I4 is restricted
to F0 because synthesising already-derived F1/F2 coordinates can create internally
inconsistent artificial customer rows.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from telco_churn.feature_selection import validate_feature_selection_policy_id
from telco_churn.feature_policies import FEATURE_POLICY_RAW, validate_feature_policy_id
from telco_churn.imbalance_policies import (
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_NONE,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
    ImbalancePolicyId,
    normalize_imbalance_parameters,
    validate_imbalance_policy_id,
)


class ImbalanceRoutingError(ValueError):
    """Raised when a candidate requests an undeclared imbalance-treatment route."""


# Literal identifiers avoid an import cycle with telco_churn.candidates, which imports
# these routing helpers when it builds a complete procedure.
_CANDIDATE_RIDGE = "C01_RIDGE_CLASSIFIER"
_CANDIDATE_LOGISTIC = "C02_LOGISTIC_REGRESSION"
_CANDIDATE_SPLINE_LOGISTIC = "C03_SPLINE_LOGISTIC_REGRESSION"
_CANDIDATE_SHRINKAGE_LDA = "C04_SHRINKAGE_LDA"
_CANDIDATE_REGULARIZED_QDA = "C05_REGULARIZED_QDA"
_CANDIDATE_KNN = "C06_K_NEAREST_NEIGHBOURS"
_CANDIDATE_HYBRID_NB = "C07_HYBRID_NAIVE_BAYES"
_CANDIDATE_DECISION_TREE = "C08_DECISION_TREE"
_CANDIDATE_EXTRA_TREES = "C09_EXTRA_TREES"
_CANDIDATE_BAGGING = "C10_BAGGING"
_CANDIDATE_RANDOM_FOREST = "C11_RANDOM_FOREST"
_CANDIDATE_BALANCED_RANDOM_FOREST = "C12_BALANCED_RANDOM_FOREST"
_CANDIDATE_ADABOOST = "C13_ADABOOST"
_CANDIDATE_RUSBOOST = "C14_RUSBOOST"
_CANDIDATE_GRADIENT_BOOSTING = "C15_GRADIENT_BOOSTING"
_CANDIDATE_HIST_GRADIENT_BOOSTING = "C16_HIST_GRADIENT_BOOSTING"
_CANDIDATE_XGBOOST = "C17_XGBOOST"
_CANDIDATE_LIGHTGBM = "C18_LIGHTGBM"
_CANDIDATE_CATBOOST = "C19_CATBOOST"
_CANDIDATE_EBM = "C20_EXPLAINABLE_BOOSTING_MACHINE"
_CANDIDATE_LINEAR_SVM = "C21_LINEAR_SVM"
_CANDIDATE_RBF_SVM = "C22_RBF_SVM"
_CANDIDATE_MLP = "C23_MULTILAYER_PERCEPTRON"
_CANDIDATE_TABNET = "C24_TABNET"
_CANDIDATE_FT_TRANSFORMER = "C25_FT_TRANSFORMER"

_WEIGHTED_AND_RESAMPLED: tuple[ImbalancePolicyId, ...] = (
    IMBALANCE_NONE,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
)
_RESAMPLED_ONLY: tuple[ImbalancePolicyId, ...] = (
    IMBALANCE_NONE,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
)
_WEIGHTED_ONLY: tuple[ImbalancePolicyId, ...] = (
    IMBALANCE_NONE,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
)
_INTRINSIC_IMBALANCE_ONLY: tuple[ImbalancePolicyId, ...] = (
    IMBALANCE_NONE,
)

# Tree and boosting procedures use the direct weighted-loss route rather than generic
# resampling. Distance, margin, probabilistic, and MLP procedures also evaluate row-level
# resampling. The matrix is predeclared, not selected from validation or test outcomes.
IMBALANCE_POLICIES_BY_CANDIDATE: dict[str, tuple[ImbalancePolicyId, ...]] = {
    _CANDIDATE_RIDGE: _WEIGHTED_AND_RESAMPLED,
    _CANDIDATE_LOGISTIC: _WEIGHTED_AND_RESAMPLED,
    _CANDIDATE_SPLINE_LOGISTIC: _WEIGHTED_AND_RESAMPLED,
    # LDA and QDA do not support the generic sample-weight adapter, but their
    # represented training matrices can be resampled inside the fitting path.
    _CANDIDATE_SHRINKAGE_LDA: _RESAMPLED_ONLY,
    _CANDIDATE_REGULARIZED_QDA: _RESAMPLED_ONLY,
    _CANDIDATE_KNN: _RESAMPLED_ONLY,
    _CANDIDATE_HYBRID_NB: _RESAMPLED_ONLY,
    _CANDIDATE_LINEAR_SVM: _WEIGHTED_AND_RESAMPLED,
    _CANDIDATE_RBF_SVM: _WEIGHTED_AND_RESAMPLED,
    _CANDIDATE_MLP: _WEIGHTED_AND_RESAMPLED,
    _CANDIDATE_DECISION_TREE: _WEIGHTED_ONLY,
    _CANDIDATE_EXTRA_TREES: _WEIGHTED_ONLY,
    _CANDIDATE_BAGGING: _WEIGHTED_ONLY,
    _CANDIDATE_RANDOM_FOREST: _WEIGHTED_ONLY,
    # C12 and C14 already carry their own imbalance mechanism. Generic external
    # weighting or sampling would create a distinct compound procedure.
    _CANDIDATE_BALANCED_RANDOM_FOREST: _INTRINSIC_IMBALANCE_ONLY,
    _CANDIDATE_ADABOOST: _WEIGHTED_ONLY,
    _CANDIDATE_RUSBOOST: _INTRINSIC_IMBALANCE_ONLY,
    _CANDIDATE_GRADIENT_BOOSTING: _WEIGHTED_ONLY,
    _CANDIDATE_HIST_GRADIENT_BOOSTING: _WEIGHTED_ONLY,
    _CANDIDATE_XGBOOST: _WEIGHTED_ONLY,
    _CANDIDATE_LIGHTGBM: _WEIGHTED_ONLY,
    _CANDIDATE_CATBOOST: _WEIGHTED_ONLY,
    _CANDIDATE_EBM: _WEIGHTED_ONLY,
    _CANDIDATE_TABNET: _WEIGHTED_ONLY,
    _CANDIDATE_FT_TRANSFORMER: _WEIGHTED_ONLY,
}


def supported_imbalance_policies(
    candidate_id: str,
    feature_policy: str,
    feature_selection_policy: str,
) -> tuple[ImbalancePolicyId, ...]:
    """Return the fixed compatible imbalance policies for one full procedure route."""
    feature_policy = validate_feature_policy_id(feature_policy)
    validate_feature_selection_policy_id(feature_selection_policy)
    try:
        policies = IMBALANCE_POLICIES_BY_CANDIDATE[candidate_id]
    except KeyError as exc:
        raise ImbalanceRoutingError(
            f"No imbalance routing is declared for candidate {candidate_id!r}."
        ) from exc
    if feature_policy != FEATURE_POLICY_RAW:
        policies = tuple(policy for policy in policies if policy != IMBALANCE_SMOTENC)
    if not policies:
        raise ImbalanceRoutingError(
            f"No imbalance policy remains for {candidate_id!r} with {feature_policy!r}."
        )
    return policies


def validate_candidate_imbalance_policy(
    candidate_id: str,
    feature_policy: str,
    feature_selection_policy: str,
    imbalance_policy: str,
) -> ImbalancePolicyId:
    """Validate one persisted imbalance-policy choice against the declared matrix."""
    imbalance_policy = validate_imbalance_policy_id(imbalance_policy)
    if imbalance_policy not in supported_imbalance_policies(
        candidate_id,
        feature_policy,
        feature_selection_policy,
    ):
        raise ImbalanceRoutingError(
            f"Imbalance policy {imbalance_policy!r} is not declared for "
            f"{candidate_id!r}, {feature_policy!r}, and {feature_selection_policy!r}."
        )
    return imbalance_policy


def _imbalance_policy_parameter_name(feature_policy: str) -> str:
    """Return the Optuna parameter name with a fixed policy-specific choice set.

    Optuna persists one categorical distribution per parameter name within each study.
    SMOTENC is available only for F0, whereas F1 and F2 omit it. Reusing one generic
    ``imbalance_policy`` parameter would therefore make its choices depend on the earlier
    feature-policy draw and violate Optuna's fixed-distribution contract. A separate,
    deterministic parameter name for every feature policy preserves conditional routing
    without treating structurally incompatible procedures as failed trials.
    """
    feature_policy = validate_feature_policy_id(feature_policy)
    return f"imbalance_policy__{feature_policy.lower()}"


def suggest_imbalance_configuration(
    trial: Any,
    *,
    candidate_id: str,
    feature_policy: str,
    feature_selection_policy: str,
    profile: str,
) -> dict[str, Any]:
    """Suggest one candidate-compatible imbalance branch and only its active controls."""
    if profile not in {"smoke", "full"}:
        raise ImbalanceRoutingError(f"Unsupported search profile {profile!r}.")
    policy_parameter_name = _imbalance_policy_parameter_name(feature_policy)
    policy = trial.suggest_categorical(
        policy_parameter_name,
        list(
            supported_imbalance_policies(
                candidate_id,
                feature_policy,
                feature_selection_policy,
            )
        ),
    )
    policy = validate_candidate_imbalance_policy(
        candidate_id,
        feature_policy,
        feature_selection_policy,
        policy,
    )
    result: dict[str, Any] = {"imbalance_policy": policy}
    if policy in {IMBALANCE_RANDOM_OVERSAMPLING, IMBALANCE_RANDOM_UNDERSAMPLING}:
        result["imbalance_sampling_strategy"] = float(
            trial.suggest_categorical(
                "imbalance_sampling_strategy",
                [0.5, 0.75, 1.0] if profile == "full" else [0.75],
            )
        )
    elif policy == IMBALANCE_SMOTENC:
        result["imbalance_sampling_strategy"] = float(
            trial.suggest_categorical(
                "imbalance_sampling_strategy",
                [0.5, 0.75, 1.0] if profile == "full" else [0.75],
            )
        )
        result["imbalance_smotenc_k_neighbors"] = int(
            trial.suggest_categorical(
                "imbalance_smotenc_k_neighbors",
                [3, 5, 7] if profile == "full" else [3],
            )
        )
    return result


def pop_imbalance_configuration(
    *,
    candidate_id: str,
    feature_policy: str,
    feature_selection_policy: str,
    parameters: MutableMapping[str, Any],
) -> tuple[ImbalancePolicyId, dict[str, object]]:
    """Remove, validate, and return the imbalance branch from persisted parameters."""
    policy = validate_candidate_imbalance_policy(
        candidate_id,
        feature_policy,
        feature_selection_policy,
        parameters.pop("imbalance_policy", IMBALANCE_NONE),
    )
    branch_parameters = {
        name: parameters.pop(name)
        for name in ("imbalance_sampling_strategy", "imbalance_smotenc_k_neighbors")
        if name in parameters
    }
    return policy, normalize_imbalance_parameters(policy, branch_parameters)


def neutralize_estimator_weight_parameters(
    candidate_id: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return parameters with historical estimator-side weighting fixed to neutral values.

    I1 is the project's sole generic balanced-weight mechanism. This defensive helper
    also normalizes a manually supplied or stale persisted parameter dictionary, making
    accidental weight-plus-resampling combinations impossible even outside Optuna.
    """
    result = dict(parameters)
    result.pop("bootstrap_weight_policy", None)
    neutral_keys = {
        _CANDIDATE_RIDGE: "class_weight",
        _CANDIDATE_LOGISTIC: "class_weight",
        _CANDIDATE_SPLINE_LOGISTIC: "class_weight",
        _CANDIDATE_EXTRA_TREES: "class_weight",
        _CANDIDATE_BAGGING: "base_class_weight",
        _CANDIDATE_RANDOM_FOREST: "class_weight",
        _CANDIDATE_LINEAR_SVM: "class_weight",
        _CANDIDATE_RBF_SVM: "class_weight",
    }
    key = neutral_keys.get(candidate_id)
    if key is not None:
        result[key] = "none"
    return result
