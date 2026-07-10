"""Core candidate registry for resumable final-comparison experiments.

The final-comparison protocol evaluates complete candidate procedures rather than
bare estimator names. This registry provides the stable identifiers, continuous-score
semantics, predeclared fold-internal feature policies, Optuna search-space
suggestions, and fresh unfitted pipeline builders for the core classical, tree,
bagging, boosting, SVM, and neural-network library.

The registry keeps three concerns separate:

1. candidate identity and score semantics;
2. Optuna search-space suggestions; and
3. construction of a fresh, unfitted, fold-safe pipeline.

No fitted model, validation score, outer-fold result, or test-set information belongs
in this module. Every returned estimator remains unfitted so preprocessing and model
parameters are learned only from the correct training partition inside the later
nested-CV workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from telco_churn.config import RANDOM_STATE
from telco_churn.feature_selection import (
    FEATURE_SELECTION_L1_LOGISTIC,
    FEATURE_SELECTION_NONE,
    FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
    FeatureSelectionPolicyId,
    validate_feature_selection_policy_id,
)
from telco_churn.feature_policies import (
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_LINEAR_EXPANDED,
    FEATURE_POLICY_RAW,
    FeaturePolicyId,
    validate_feature_policy_id,
)
from telco_churn.feature_policy_pipelines import (
    REPRESENTATION_DENSE_SCALED,
    REPRESENTATION_SPARSE_SCALED,
    REPRESENTATION_SPARSE_UNSCALED,
    apply_imbalance_policy_to_pipeline,
    make_feature_policy_classifier_pipeline,
)
from telco_churn.imbalance_routing import (
    neutralize_estimator_weight_parameters,
    pop_imbalance_configuration,
    suggest_imbalance_configuration,
    supported_imbalance_policies,
)
from telco_churn.models import (
    make_linear_svc_classifier,
    make_logistic_regression_classifier,
    make_mlp_classifier,
)


ScoreKind = Literal["probability", "margin"]


class CandidateRegistryError(ValueError):
    """Raised when the immutable candidate registry is inconsistent."""


@dataclass(frozen=True)
class CandidateDefinition:
    """Immutable description of one candidate procedure family.

    Parameters
    ----------
    candidate_id:
        Stable identifier used in protocols, task keys, study names, and result
        tables. The identifier must not be changed after a run has started.

    display_name:
        Human-readable family name for reports and progress views.

    score_kind:
        ``"probability"`` when the uncalibrated estimator exposes class-one
        probabilities and ``"margin"`` when it exposes a continuous decision
        function. Both score kinds are valid for average precision and ROC-AUC.
        Only probability scores are valid for raw Brier score or log loss.

    representation:
        Human-readable description of the fold-internal input representation.

    search_profile:
        Name of the default full-run search profile. The phase-2 smoke test uses
        the separate ``"smoke"`` profile to validate mechanics without running
        the eventual production-scale budget.
    """

    candidate_id: str
    display_name: str
    score_kind: ScoreKind
    representation: str
    search_profile: str = "full"

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise CandidateRegistryError("candidate_id must not be empty.")
        if not self.display_name.strip():
            raise CandidateRegistryError("display_name must not be empty.")
        if self.score_kind not in {"probability", "margin"}:
            raise CandidateRegistryError(
                "score_kind must be either 'probability' or 'margin'."
            )
        if not self.representation.strip():
            raise CandidateRegistryError("representation must not be empty.")
        if not self.search_profile.strip():
            raise CandidateRegistryError("search_profile must not be empty.")


CANDIDATE_RIDGE_CLASSIFIER = "C01_RIDGE_CLASSIFIER"
CANDIDATE_LOGISTIC_REGRESSION = "C02_LOGISTIC_REGRESSION"
CANDIDATE_SPLINE_LOGISTIC_REGRESSION = "C03_SPLINE_LOGISTIC_REGRESSION"
CANDIDATE_SHRINKAGE_LDA = "C04_SHRINKAGE_LDA"
CANDIDATE_REGULARIZED_QDA = "C05_REGULARIZED_QDA"
CANDIDATE_KNN = "C06_K_NEAREST_NEIGHBOURS"
CANDIDATE_HYBRID_NAIVE_BAYES = "C07_HYBRID_NAIVE_BAYES"
CANDIDATE_DECISION_TREE = "C08_DECISION_TREE"
CANDIDATE_EXTRA_TREES = "C09_EXTRA_TREES"
CANDIDATE_BAGGING = "C10_BAGGING"
CANDIDATE_RANDOM_FOREST = "C11_RANDOM_FOREST"
CANDIDATE_BALANCED_RANDOM_FOREST = "C12_BALANCED_RANDOM_FOREST"
CANDIDATE_ADABOOST = "C13_ADABOOST"
CANDIDATE_RUSBOOST = "C14_RUSBOOST"
CANDIDATE_GRADIENT_BOOSTING = "C15_GRADIENT_BOOSTING"
CANDIDATE_HIST_GRADIENT_BOOSTING = "C16_HIST_GRADIENT_BOOSTING"
CANDIDATE_XGBOOST = "C17_XGBOOST"
CANDIDATE_LIGHTGBM = "C18_LIGHTGBM"
CANDIDATE_CATBOOST = "C19_CATBOOST"
CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE = "C20_EXPLAINABLE_BOOSTING_MACHINE"
CANDIDATE_LINEAR_SVM = "C21_LINEAR_SVM"
CANDIDATE_RBF_SVM = "C22_RBF_SVM"
CANDIDATE_MLP = "C23_MULTILAYER_PERCEPTRON"
CANDIDATE_TABNET = "C24_TABNET"
CANDIDATE_FT_TRANSFORMER = "C25_FT_TRANSFORMER"
CANDIDATE_TABM = "C26_TABM"


FEATURE_POLICIES_GENERAL: tuple[FeaturePolicyId, ...] = (
    FEATURE_POLICY_RAW,
    FEATURE_POLICY_DOMAIN,
)
FEATURE_POLICIES_REGULARIZED_LINEAR: tuple[FeaturePolicyId, ...] = (
    FEATURE_POLICY_RAW,
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_LINEAR_EXPANDED,
)

# This map is part of the immutable candidate-procedure contract.  F2 is deliberately
# restricted to regularized linear procedures.  Nonlinear learners either model
# interactions internally or would receive an unnecessarily high-dimensional input
# representation from the systematic expansion.
FEATURE_POLICIES_BY_CANDIDATE: dict[str, tuple[FeaturePolicyId, ...]] = {
    CANDIDATE_RIDGE_CLASSIFIER: FEATURE_POLICIES_REGULARIZED_LINEAR,
    CANDIDATE_LOGISTIC_REGRESSION: FEATURE_POLICIES_REGULARIZED_LINEAR,
    # C03 keeps raw numeric inputs as its explicit spline-basis variables. F1 already
    # contains selected nonlinear summaries and interactions, so adding splines to F1
    # would blur the intended additive-nonlinear comparison.
    CANDIDATE_SPLINE_LOGISTIC_REGRESSION: (FEATURE_POLICY_RAW,),
    CANDIDATE_SHRINKAGE_LDA: FEATURE_POLICIES_GENERAL,
    CANDIDATE_REGULARIZED_QDA: FEATURE_POLICIES_GENERAL,
    CANDIDATE_KNN: FEATURE_POLICIES_GENERAL,
    CANDIDATE_HYBRID_NAIVE_BAYES: FEATURE_POLICIES_GENERAL,
    CANDIDATE_DECISION_TREE: FEATURE_POLICIES_GENERAL,
    CANDIDATE_EXTRA_TREES: FEATURE_POLICIES_GENERAL,
    CANDIDATE_BAGGING: FEATURE_POLICIES_GENERAL,
    CANDIDATE_RANDOM_FOREST: FEATURE_POLICIES_GENERAL,
    CANDIDATE_BALANCED_RANDOM_FOREST: FEATURE_POLICIES_GENERAL,
    CANDIDATE_ADABOOST: FEATURE_POLICIES_GENERAL,
    CANDIDATE_RUSBOOST: FEATURE_POLICIES_GENERAL,
    CANDIDATE_GRADIENT_BOOSTING: FEATURE_POLICIES_GENERAL,
    CANDIDATE_HIST_GRADIENT_BOOSTING: FEATURE_POLICIES_GENERAL,
    CANDIDATE_XGBOOST: FEATURE_POLICIES_GENERAL,
    CANDIDATE_LIGHTGBM: FEATURE_POLICIES_GENERAL,
    CANDIDATE_CATBOOST: FEATURE_POLICIES_GENERAL,
    CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE: FEATURE_POLICIES_GENERAL,
    CANDIDATE_LINEAR_SVM: FEATURE_POLICIES_GENERAL,
    CANDIDATE_RBF_SVM: FEATURE_POLICIES_GENERAL,
    CANDIDATE_MLP: FEATURE_POLICIES_GENERAL,
    CANDIDATE_TABNET: FEATURE_POLICIES_GENERAL,
    CANDIDATE_FT_TRANSFORMER: FEATURE_POLICIES_GENERAL,
    CANDIDATE_TABM: FEATURE_POLICIES_GENERAL,
}

FEATURE_SELECTION_POLICIES_NONE: tuple[FeatureSelectionPolicyId, ...] = (
    FEATURE_SELECTION_NONE,
)
FEATURE_SELECTION_POLICIES_MUTUAL_INFO: tuple[FeatureSelectionPolicyId, ...] = (
    FEATURE_SELECTION_NONE,
    FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
)
FEATURE_SELECTION_POLICIES_REGULARIZED_LINEAR: tuple[FeatureSelectionPolicyId, ...] = (
    FEATURE_SELECTION_NONE,
    FEATURE_SELECTION_VARIANCE_MUTUAL_INFO,
    FEATURE_SELECTION_L1_LOGISTIC,
)

# Selection is evaluated only for families with a coherent rationale.  Trees and native
# categorical boosting retain their full representation because their own split or
# representation-learning mechanisms already select nonlinear evidence internally.
FEATURE_SELECTION_POLICIES_BY_CANDIDATE: dict[
    str, tuple[FeatureSelectionPolicyId, ...]
] = {
    CANDIDATE_RIDGE_CLASSIFIER: FEATURE_SELECTION_POLICIES_REGULARIZED_LINEAR,
    CANDIDATE_LOGISTIC_REGRESSION: FEATURE_SELECTION_POLICIES_REGULARIZED_LINEAR,
    # C03 already controls basis complexity through L1/L2/elastic-net logistic
    # regularization. C04/C05 estimate covariance structure directly.
    CANDIDATE_SPLINE_LOGISTIC_REGRESSION: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_SHRINKAGE_LDA: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_REGULARIZED_QDA: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_LINEAR_SVM: FEATURE_SELECTION_POLICIES_REGULARIZED_LINEAR,
    CANDIDATE_KNN: FEATURE_SELECTION_POLICIES_MUTUAL_INFO,
    CANDIDATE_RBF_SVM: FEATURE_SELECTION_POLICIES_MUTUAL_INFO,
    CANDIDATE_MLP: FEATURE_SELECTION_POLICIES_MUTUAL_INFO,
    CANDIDATE_HYBRID_NAIVE_BAYES: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_DECISION_TREE: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_EXTRA_TREES: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_BAGGING: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_RANDOM_FOREST: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_BALANCED_RANDOM_FOREST: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_ADABOOST: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_RUSBOOST: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_GRADIENT_BOOSTING: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_HIST_GRADIENT_BOOSTING: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_XGBOOST: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_LIGHTGBM: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_CATBOOST: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_TABNET: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_FT_TRANSFORMER: FEATURE_SELECTION_POLICIES_NONE,
    CANDIDATE_TABM: FEATURE_SELECTION_POLICIES_NONE,
}


def supported_feature_policies(candidate_id: str) -> tuple[FeaturePolicyId, ...]:
    """Return the predeclared representations compatible with one candidate.

    A feature policy is selected only inside the inner HPO loop.  It is not chosen
    from outer-validation or held-out test results.
    """
    try:
        return FEATURE_POLICIES_BY_CANDIDATE[candidate_id]
    except KeyError as exc:
        raise CandidateRegistryError(
            f"No feature-policy route is declared for candidate {candidate_id!r}."
        ) from exc


def validate_candidate_feature_policy(candidate_id: str, policy_id: str) -> FeaturePolicyId:
    """Validate that a persisted feature-policy choice is candidate-compatible."""
    policy_id = validate_feature_policy_id(policy_id)
    if policy_id not in supported_feature_policies(candidate_id):
        raise CandidateRegistryError(
            f"Feature policy {policy_id!r} is not declared for candidate {candidate_id!r}."
        )
    return policy_id


def supported_feature_selection_policies(
    candidate_id: str,
    feature_policy: str,
) -> tuple[FeatureSelectionPolicyId, ...]:
    """Return selection policies declared for one candidate-policy route."""
    validate_candidate_feature_policy(candidate_id, feature_policy)
    try:
        return FEATURE_SELECTION_POLICIES_BY_CANDIDATE[candidate_id]
    except KeyError as exc:
        raise CandidateRegistryError(
            f"No feature-selection policy route is declared for candidate {candidate_id!r}."
        ) from exc


def validate_candidate_feature_selection(
    candidate_id: str,
    feature_policy: str,
    selection_policy: str,
) -> FeatureSelectionPolicyId:
    """Validate that a selection policy is compatible with a candidate-policy route."""
    selection_policy = validate_feature_selection_policy_id(selection_policy)
    if selection_policy not in supported_feature_selection_policies(
        candidate_id,
        feature_policy,
    ):
        raise CandidateRegistryError(
            f"Feature-selection policy {selection_policy!r} is not declared for "
            f"candidate {candidate_id!r} with {feature_policy!r}."
        )
    return selection_policy


def candidate_procedure_contract(candidate_id: str) -> dict[str, object]:
    """Return the routing contract that must bind a persistent inner HPO study.

    The contract now includes the candidate, feature-policy, selector, and imbalance
    matrix. Persistent Optuna studies therefore cannot resume under a changed treatment
    universe where a previous trial may have searched a different fitted procedure.
    """
    feature_policies = supported_feature_policies(candidate_id)
    feature_selection_policies = {
        policy_id: list(supported_feature_selection_policies(candidate_id, policy_id))
        for policy_id in feature_policies
    }
    return {
        "candidate_id": candidate_id,
        "feature_policies": list(feature_policies),
        "feature_selection_policies": feature_selection_policies,
        "imbalance_policies": {
            policy_id: {
                selection_policy: list(
                    supported_imbalance_policies(
                        candidate_id,
                        policy_id,
                        selection_policy,
                    )
                )
                for selection_policy in selection_policies
            }
            for policy_id, selection_policies in feature_selection_policies.items()
        },
    }


def candidate_procedure_contract_fingerprint(candidate_id: str) -> str:
    """Return a deterministic hash of one candidate's routing contract."""
    payload = json.dumps(
        candidate_procedure_contract(candidate_id),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _selection_parameter_suggestions(
    trial: Any,
    *,
    selection_policy: FeatureSelectionPolicyId,
    profile: str,
) -> dict[str, Any]:
    """Suggest JSON-safe parameters only for the selected policy branch."""
    if selection_policy == FEATURE_SELECTION_NONE:
        return {}
    if selection_policy == FEATURE_SELECTION_VARIANCE_MUTUAL_INFO:
        k_choices = [12, 24] if profile == "smoke" else [12, 24, 36, 48, 64, 96]
        return {
            "selection_k": int(
                trial.suggest_categorical("selection_k", k_choices)
            )
        }
    if selection_policy == FEATURE_SELECTION_L1_LOGISTIC:
        return {
            "selection_l1_C": float(
                trial.suggest_float(
                    "selection_l1_C",
                    1e-4 if profile == "smoke" else 1e-5,
                    10.0 if profile == "smoke" else 100.0,
                    log=True,
                )
            ),
            "selection_l1_threshold": trial.suggest_categorical(
                "selection_l1_threshold",
                ["mean", "median"],
            ),
        }
    raise RuntimeError(f"Unexpected validated selection policy {selection_policy!r}.")


def _with_feature_policy(
    trial: Any,
    *,
    candidate_id: str,
    parameters: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Append compatible feature, selection, and imbalance choices for inner HPO."""
    result = dict(parameters)
    feature_policy = trial.suggest_categorical(
        "feature_policy", list(supported_feature_policies(candidate_id))
    )
    result["feature_policy"] = feature_policy
    selection_policy = trial.suggest_categorical(
        "feature_selection_policy",
        list(supported_feature_selection_policies(candidate_id, feature_policy)),
    )
    result["feature_selection_policy"] = selection_policy
    result.update(
        _selection_parameter_suggestions(
            trial,
            selection_policy=selection_policy,
            profile=profile,
        )
    )
    result.update(
        suggest_imbalance_configuration(
            trial,
            candidate_id=candidate_id,
            feature_policy=feature_policy,
            feature_selection_policy=selection_policy,
            profile=profile,
        )
    )
    return result

INITIAL_CANDIDATE_REGISTRY: tuple[CandidateDefinition, ...] = (
    CandidateDefinition(
        candidate_id=CANDIDATE_LOGISTIC_REGRESSION,
        display_name="Regularized logistic regression",
        score_kind="probability",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_EXTRA_TREES,
        display_name="Extra Trees classifier",
        score_kind="probability",
        representation="unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_LINEAR_SVM,
        display_name="Linear support vector machine",
        score_kind="margin",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_MLP,
        display_name="Multilayer perceptron",
        score_kind="probability",
        representation="dense scaled one-hot features",
    ),
)


# Preserve the exact Phase-2 smoke subset for reproducibility documentation.
PHASE2_SMOKE_CANDIDATE_REGISTRY = INITIAL_CANDIDATE_REGISTRY

CORE_CANDIDATE_REGISTRY: tuple[CandidateDefinition, ...] = (
    CandidateDefinition(
        candidate_id=CANDIDATE_RIDGE_CLASSIFIER,
        display_name="Ridge classifier",
        score_kind="margin",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_LOGISTIC_REGRESSION,
        display_name="Regularized logistic regression",
        score_kind="probability",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_SPLINE_LOGISTIC_REGRESSION,
        display_name="Spline logistic regression",
        score_kind="probability",
        representation="dense B-spline numeric basis plus categorical indicators",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_SHRINKAGE_LDA,
        display_name="Shrinkage linear discriminant analysis",
        score_kind="probability",
        representation="dense scaled one-hot features with reference categories dropped",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_REGULARIZED_QDA,
        display_name="Regularized quadratic discriminant analysis",
        score_kind="probability",
        representation="dense scaled one-hot features with reference categories dropped",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_KNN,
        display_name="k-nearest neighbours",
        score_kind="probability",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_HYBRID_NAIVE_BAYES,
        display_name="Hybrid Gaussian-Bernoulli Naive Bayes",
        score_kind="probability",
        representation="numeric-first unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_DECISION_TREE,
        display_name="Decision tree",
        score_kind="probability",
        representation="unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_EXTRA_TREES,
        display_name="Extra Trees classifier",
        score_kind="probability",
        representation="unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_BAGGING,
        display_name="Bagged decision trees",
        score_kind="probability",
        representation="unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_RANDOM_FOREST,
        display_name="Random forest",
        score_kind="probability",
        representation="unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_BALANCED_RANDOM_FOREST,
        display_name="Balanced random forest",
        score_kind="probability",
        representation="unscaled one-hot features with intrinsic balanced samples",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_ADABOOST,
        display_name="AdaBoost",
        score_kind="probability",
        representation="dense unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_RUSBOOST,
        display_name="RUSBoost",
        score_kind="probability",
        representation="unscaled one-hot features with intrinsic iterative undersampling",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_GRADIENT_BOOSTING,
        display_name="Classical gradient boosting",
        score_kind="probability",
        representation="dense unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_HIST_GRADIENT_BOOSTING,
        display_name="Histogram gradient boosting",
        score_kind="probability",
        representation="dense unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_XGBOOST,
        display_name="XGBoost",
        score_kind="probability",
        representation="dense unscaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_LIGHTGBM,
        display_name="LightGBM",
        score_kind="probability",
        representation="native categorical columns",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_CATBOOST,
        display_name="CatBoost",
        score_kind="probability",
        representation="native categorical columns",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE,
        display_name="Explainable Boosting Machine",
        score_kind="probability",
        representation="native categorical string columns",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_LINEAR_SVM,
        display_name="Linear support vector machine",
        score_kind="margin",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_RBF_SVM,
        display_name="RBF support vector machine",
        score_kind="margin",
        representation="scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_MLP,
        display_name="Multilayer perceptron",
        score_kind="probability",
        representation="dense scaled one-hot features",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_TABNET,
        display_name="TabNet",
        score_kind="probability",
        representation="native TabNet categorical embeddings",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_FT_TRANSFORMER,
        display_name="FT-Transformer",
        score_kind="probability",
        representation="native FT-Transformer continuous and categorical embeddings",
    ),
    CandidateDefinition(
        candidate_id=CANDIDATE_TABM,
        display_name="TabM",
        score_kind="probability",
        representation="native TabM continuous and categorical one-hot features",
    ),
)

# Phase-2 public imports continue to work, but the default registry now names the
# complete core library. The fixed Phase-2 smoke task list remains explicit.
INITIAL_CANDIDATE_REGISTRY = CORE_CANDIDATE_REGISTRY
def validate_candidate_registry(
    registry: tuple[CandidateDefinition, ...] = INITIAL_CANDIDATE_REGISTRY,
) -> None:
    """Validate registry identifiers and score-output declarations.

    The validation deliberately happens before an experiment run is created. A
    duplicate or unknown candidate identifier would otherwise make task artifacts
    ambiguous and could invalidate resume safety.
    """
    if not registry:
        raise CandidateRegistryError("The candidate registry must not be empty.")

    candidate_ids = [definition.candidate_id for definition in registry]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CandidateRegistryError("Candidate registry identifiers must be unique.")

    for candidate_id in candidate_ids:
        policy_ids = supported_feature_policies(candidate_id)
        if not policy_ids:
            raise CandidateRegistryError(
                f"Candidate {candidate_id!r} must declare at least one feature policy."
            )
        if len(policy_ids) != len(set(policy_ids)):
            raise CandidateRegistryError(
                f"Candidate {candidate_id!r} has duplicate feature-policy identifiers."
            )
        for policy_id in policy_ids:
            validate_feature_policy_id(policy_id)
            selection_policy_ids = supported_feature_selection_policies(
                candidate_id,
                policy_id,
            )
            if not selection_policy_ids:
                raise CandidateRegistryError(
                    f"Candidate {candidate_id!r} with {policy_id!r} must declare "
                    "at least one feature-selection policy."
                )
            if len(selection_policy_ids) != len(set(selection_policy_ids)):
                raise CandidateRegistryError(
                    f"Candidate {candidate_id!r} with {policy_id!r} has duplicate "
                    "feature-selection policy identifiers."
                )
            for selection_policy_id in selection_policy_ids:
                validate_feature_selection_policy_id(selection_policy_id)


def get_candidate_definition(candidate_id: str) -> CandidateDefinition:
    """Return one immutable definition or raise a clear registry error."""
    validate_candidate_registry()
    for definition in INITIAL_CANDIDATE_REGISTRY:
        if definition.candidate_id == candidate_id:
            return definition
    raise CandidateRegistryError(f"Unknown candidate identifier: {candidate_id!r}")


def _decode_class_weight(value: str) -> str | None:
    """Convert a JSON-safe class-weight choice into scikit-learn input."""
    if value == "none":
        return None
    if value == "balanced":
        return "balanced"
    if value == "balanced_subsample":
        return "balanced_subsample"
    raise CandidateRegistryError(f"Unknown class-weight choice: {value!r}")


def _decode_optional_depth(value: str) -> int | None:
    """Convert a JSON-safe optional tree-depth choice into an estimator value."""
    if value == "none":
        return None
    return int(value)


def _decode_max_features(value: str) -> str | float:
    """Convert categorical Optuna encodings into Extra Trees max_features values."""
    if value in {"sqrt", "log2"}:
        return value
    return float(value)


def suggest_candidate_parameters(
    trial: Any,
    *,
    candidate_id: str,
    profile: str = "full",
) -> dict[str, Any]:
    """Suggest one JSON-compatible hyperparameter configuration.

    ``trial`` is intentionally typed as ``Any`` so this reusable registry does not
    import Optuna at module import time. That keeps ordinary notebooks usable before
    optional HPO dependencies are installed.

    Parameters
    ----------
    trial:
        An Optuna-compatible trial object exposing ``suggest_*`` methods.

    candidate_id:
        Identifier from :data:`INITIAL_CANDIDATE_REGISTRY`.

    profile:
        ``"smoke"`` uses small, fast parameter ranges solely to validate persistent
        nested-HPO mechanics. ``"full"`` is intentionally broader and will be
        frozen in the later full candidate-registry protocol revision.
    """
    definition = get_candidate_definition(candidate_id)
    if profile not in {"smoke", "full"}:
        raise CandidateRegistryError(
            f"Unknown search profile {profile!r} for {definition.candidate_id}."
        )

    if candidate_id == CANDIDATE_LOGISTIC_REGRESSION:
        penalty = trial.suggest_categorical(
            "penalty",
            ["l1", "l2", "elasticnet"],
        )
        parameters: dict[str, Any] = {
            "penalty": penalty,
            "C": float(trial.suggest_float("C", 1e-4, 1e3, log=True)),
            "class_weight": "none",
            "max_iter": 8_000,
        }
        if penalty == "elasticnet":
            parameters["l1_ratio"] = float(trial.suggest_float("l1_ratio", 0.02, 0.98))
        return _with_feature_policy(
            trial, candidate_id=candidate_id, parameters=parameters, profile=profile
        )

    if candidate_id == CANDIDATE_EXTRA_TREES:
        if profile == "smoke":
            n_estimators = int(trial.suggest_int("n_estimators", 25, 80, step=5))
            max_depth_choices = ["none", "4", "8", "14"]
            min_split_high = 20
            min_leaf_high = 10
        else:
            n_estimators = int(trial.suggest_int("n_estimators", 300, 1_500, step=50))
            max_depth_choices = ["none", "4", "6", "10", "16", "24", "32"]
            min_split_high = 80
            min_leaf_high = 40

        bootstrap = bool(trial.suggest_categorical("bootstrap", [False, True]))

        parameters = {
            "n_estimators": n_estimators,
            "criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy", "log_loss"],
            ),
            "max_depth": trial.suggest_categorical("max_depth", max_depth_choices),
            "min_samples_split": int(
                trial.suggest_int("min_samples_split", 2, min_split_high)
            ),
            "min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 1, min_leaf_high)
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", "0.5", "0.75", "1.0"],
            ),
            "bootstrap": bootstrap,
            "class_weight": "none",
            "ccp_alpha": float(trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True)),
        }
        if bootstrap:
            parameters["max_samples"] = float(
                trial.suggest_float("max_samples", 0.5, 1.0)
            )
        else:
            parameters["max_samples"] = None
        return _with_feature_policy(
            trial, candidate_id=candidate_id, parameters=parameters, profile=profile
        )

    if candidate_id == CANDIDATE_LINEAR_SVM:
        return _with_feature_policy(
            trial,
            candidate_id=candidate_id,
            parameters={
                "C": float(trial.suggest_float("C", 1e-4, 1e3, log=True)),
                "loss": trial.suggest_categorical("loss", ["squared_hinge"]),
                "class_weight": "none",
                "max_iter": 100_000,
            },
            profile=profile,
        )

    if candidate_id == CANDIDATE_MLP:
        if profile == "smoke":
            architectures: list[tuple[int, ...]] = [(8,), (16,), (16, 8)]
            max_iter = 100
        else:
            architectures = [
                (8,),
                (16,),
                (32,),
                (64,),
                (16, 8),
                (32, 16),
                (64, 32),
                (32, 16, 8),
            ]
            max_iter = 1_000

        architecture_index = int(
            trial.suggest_int("architecture_index", 0, len(architectures) - 1)
        )
        return _with_feature_policy(
            trial,
            candidate_id=candidate_id,
            parameters={
                "hidden_layer_sizes": list(architectures[architecture_index]),
                "activation": trial.suggest_categorical(
                    "activation",
                    ["relu", "tanh"],
                ),
                "alpha": float(trial.suggest_float("alpha", 1e-6, 1e-1, log=True)),
                "batch_size": int(trial.suggest_categorical("batch_size", [32, 64, 128])),
                "learning_rate_init": float(
                    trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
                ),
                "max_iter": max_iter,
                "validation_fraction": 0.15,
                "n_iter_no_change": 25,
            },
            profile=profile,
        )

    from telco_churn.core_candidate_builders import suggest_core_candidate_parameters

    return _with_feature_policy(
        trial,
        candidate_id=candidate_id,
        parameters=suggest_core_candidate_parameters(
            trial,
            candidate_id=candidate_id,
            profile=profile,
        ),
        profile=profile,
    )



def make_extra_trees_pipeline(
    *,
    n_estimators: int,
    criterion: str,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str | float,
    bootstrap: bool,
    max_samples: float | None,
    class_weight: str | None,
    ccp_alpha: float,
    random_state: int,
    feature_policy: FeaturePolicyId = FEATURE_POLICY_RAW,
    feature_selection_policy: FeatureSelectionPolicyId = FEATURE_SELECTION_NONE,
    feature_selection_parameters: Mapping[str, object] | None = None,
) -> Pipeline:
    """Create an Extra Trees procedure owned by the final-comparison registry.

    Extra Trees is implemented here rather than modifying a historical workflow's
    model-factory module. The procedure belongs to the new final-comparison system
    and keeps its explicit single-thread policy next to the candidate definition.
    The unscaled one-hot preprocessor and the estimator are returned as one
    unfitted pipeline, so all preprocessing remains fold-internal.

    ``max_samples`` is legal only when bootstrap sampling is enabled. The candidate
    registry validates this before constructing the estimator, rather than relying
    on a later scikit-learn error during a long Optuna trial.
    """
    valid_criteria = {"gini", "entropy", "log_loss"}
    if criterion not in valid_criteria:
        raise CandidateRegistryError(
            f"Extra Trees criterion must be one of {sorted(valid_criteria)}."
        )
    if n_estimators < 1:
        raise CandidateRegistryError("Extra Trees n_estimators must be positive.")
    if min_samples_split < 2:
        raise CandidateRegistryError(
            "Extra Trees min_samples_split must be at least two."
        )
    if min_samples_leaf < 1:
        raise CandidateRegistryError(
            "Extra Trees min_samples_leaf must be at least one."
        )
    if ccp_alpha < 0:
        raise CandidateRegistryError("Extra Trees ccp_alpha must be non-negative.")
    if max_samples is not None and not bootstrap:
        raise CandidateRegistryError(
            "Extra Trees max_samples is allowed only when bootstrap is enabled."
        )

    estimator_kwargs: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": max_features,
        "bootstrap": bool(bootstrap),
        "class_weight": class_weight,
        "ccp_alpha": float(ccp_alpha),
        "n_jobs": 1,
        "random_state": int(random_state),
    }
    if bootstrap:
        estimator_kwargs["max_samples"] = max_samples

    return make_feature_policy_classifier_pipeline(
        policy_id=feature_policy,
        representation=REPRESENTATION_SPARSE_UNSCALED,
        classifier=ExtraTreesClassifier(**estimator_kwargs),
        feature_selection_policy=feature_selection_policy,
        feature_selection_parameters=feature_selection_parameters,
        random_state=int(random_state),
    )


def _build_candidate_pipeline_without_imbalance(
    candidate_id: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build one fresh unfitted, fold-safe candidate pipeline.

    Parameters are expected to come from :func:`suggest_candidate_parameters` or
    from a previously persisted JSON result. Values are decoded explicitly rather
    than relying on implicit pandas or JSON conversions.
    """
    get_candidate_definition(candidate_id)
    parameters = dict(parameters)
    feature_policy = validate_candidate_feature_policy(
        candidate_id, parameters.pop("feature_policy", FEATURE_POLICY_RAW)
    )
    feature_selection_policy = validate_candidate_feature_selection(
        candidate_id,
        feature_policy,
        parameters.pop("feature_selection_policy", FEATURE_SELECTION_NONE),
    )
    feature_selection_parameters = {
        name: parameters.pop(name)
        for name in (
            "selection_k",
            "selection_l1_C",
            "selection_l1_threshold",
        )
        if name in parameters
    }

    if candidate_id == CANDIDATE_LOGISTIC_REGRESSION:
        class_weight = _decode_class_weight(str(parameters["class_weight"]))
        classifier = make_logistic_regression_classifier(
            penalty=str(parameters["penalty"]),
            C=float(parameters["C"]),
            class_weight=class_weight,
            l1_ratio=(
                None
                if parameters.get("l1_ratio") is None
                else float(parameters["l1_ratio"])
            ),
            max_iter=int(parameters.get("max_iter", 8_000)),
            random_state=int(random_state),
        )
        return make_feature_policy_classifier_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_SCALED,
            classifier=classifier,
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    if candidate_id == CANDIDATE_EXTRA_TREES:
        return make_extra_trees_pipeline(
            n_estimators=int(parameters["n_estimators"]),
            criterion=str(parameters["criterion"]),
            max_depth=_decode_optional_depth(str(parameters["max_depth"])),
            min_samples_split=int(parameters["min_samples_split"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=_decode_max_features(str(parameters["max_features"])),
            bootstrap=bool(parameters["bootstrap"]),
            max_samples=(
                None
                if parameters.get("max_samples") is None
                else float(parameters["max_samples"])
            ),
            class_weight=_decode_class_weight(str(parameters["class_weight"])),
            ccp_alpha=float(parameters["ccp_alpha"]),
            random_state=int(random_state),
            feature_policy=feature_policy,
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
        )

    if candidate_id == CANDIDATE_LINEAR_SVM:
        classifier = make_linear_svc_classifier(
            C=float(parameters["C"]),
            loss=str(parameters["loss"]),
            class_weight=_decode_class_weight(str(parameters["class_weight"])),
            max_iter=int(parameters["max_iter"]),
            random_state=int(random_state),
        )
        return make_feature_policy_classifier_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_SCALED,
            classifier=classifier,
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    if candidate_id == CANDIDATE_MLP:
        classifier = make_mlp_classifier(
            hidden_layer_sizes=tuple(int(width) for width in parameters["hidden_layer_sizes"]),
            activation=str(parameters["activation"]),
            alpha=float(parameters["alpha"]),
            batch_size=int(parameters["batch_size"]),
            learning_rate_init=float(parameters["learning_rate_init"]),
            max_iter=int(parameters["max_iter"]),
            early_stopping=True,
            validation_fraction=float(parameters["validation_fraction"]),
            n_iter_no_change=int(parameters["n_iter_no_change"]),
            random_state=int(random_state),
        )
        return make_feature_policy_classifier_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_DENSE_SCALED,
            classifier=classifier,
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    from telco_churn.core_candidate_builders import build_core_candidate_pipeline

    return build_core_candidate_pipeline(
        candidate_id,
        parameters,
        random_state=int(random_state),
        feature_policy=feature_policy,
        feature_selection_policy=feature_selection_policy,
        feature_selection_parameters=feature_selection_parameters,
    )


def build_candidate_pipeline(
    candidate_id: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int = RANDOM_STATE,
):
    """Build a fresh candidate procedure including its declared imbalance treatment.

    The base pipeline is constructed first with neutral estimator-side weight settings.
    Exactly one predeclared imbalance policy is then inserted into its fitted training
    path. This avoids combining an estimator's historical ``class_weight`` option with
    I1 balanced sample weighting or a resampling policy.
    """
    get_candidate_definition(candidate_id)
    routed_parameters = neutralize_estimator_weight_parameters(candidate_id, parameters)
    feature_policy = validate_candidate_feature_policy(
        candidate_id,
        routed_parameters.get("feature_policy", FEATURE_POLICY_RAW),
    )
    feature_selection_policy = validate_candidate_feature_selection(
        candidate_id,
        feature_policy,
        routed_parameters.get("feature_selection_policy", FEATURE_SELECTION_NONE),
    )
    imbalance_policy, imbalance_parameters = pop_imbalance_configuration(
        candidate_id=candidate_id,
        feature_policy=feature_policy,
        feature_selection_policy=feature_selection_policy,
        parameters=routed_parameters,
    )
    base_pipeline = _build_candidate_pipeline_without_imbalance(
        candidate_id,
        routed_parameters,
        random_state=int(random_state),
    )
    return apply_imbalance_policy_to_pipeline(
        base_pipeline,
        imbalance_policy=imbalance_policy,
        imbalance_parameters=imbalance_parameters,
        random_state=int(random_state),
    )
