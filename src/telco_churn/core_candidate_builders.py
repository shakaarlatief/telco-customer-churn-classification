"""Core candidate builders for the resumable final-comparison system.

This module extends the initial persistent-HPO registry with the core classical,
tree, bagging, boosting, support-vector-machine, and neural-network procedures
already represented in the project. It deliberately owns the final-comparison
versions of parallel tree and boosting estimators instead of reusing historical
notebook factories whose ``n_jobs=-1`` defaults would create nested parallelism
inside the outer process pool.

Every function returns an unfitted pipeline. A declared feature policy is the first
step, followed by representation-specific imputation, scaling, one-hot encoding, or
native-categorical conversion. All learned transformations are fitted only on the
appropriate inner or outer training partition. The held-out test data is not
referenced anywhere in this module.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from telco_churn.candidates import (
    CANDIDATE_ADABOOST,
    CANDIDATE_BAGGING,
    CANDIDATE_CATBOOST,
    CANDIDATE_DECISION_TREE,
    CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE,
    CANDIDATE_EXTRA_TREES,
    CANDIDATE_FT_TRANSFORMER,
    CANDIDATE_GRADIENT_BOOSTING,
    CANDIDATE_HIST_GRADIENT_BOOSTING,
    CANDIDATE_HYBRID_NAIVE_BAYES,
    CANDIDATE_KNN,
    CANDIDATE_LIGHTGBM,
    CANDIDATE_RANDOM_FOREST,
    CANDIDATE_RBF_SVM,
    CANDIDATE_RIDGE_CLASSIFIER,
    CANDIDATE_TABNET,
    CANDIDATE_TABM,
    CANDIDATE_XGBOOST,
)
from telco_churn.feature_selection import (
    FEATURE_SELECTION_NONE,
    FeatureSelectionPolicyId,
)
from telco_churn.feature_policies import (
    FEATURE_POLICY_RAW,
    FeaturePolicyId,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
)
from telco_churn.feature_policy_pipelines import (
    CloneSafeFeaturePolicyFTTransformerClassifier,
    CloneSafeFeaturePolicyTabMClassifier,
    CloneSafeFeaturePolicyCatBoostClassifier,
    CloneSafeFeaturePolicyTabNetClassifier,
    REPRESENTATION_DENSE_UNSCALED,
    REPRESENTATION_NATIVE_CATEGORICAL_DTYPE,
    REPRESENTATION_NATIVE_CATEGORICAL_STRING,
    REPRESENTATION_SPARSE_SCALED,
    REPRESENTATION_SPARSE_UNSCALED,
    make_feature_policy_classifier_pipeline,
)
from telco_churn.conventional_core_candidates import (
    CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS,
    build_conventional_core_candidate_pipeline,
    declared_conventional_single_thread_parameter,
    suggest_conventional_core_candidate_parameters,
)
from telco_churn.models import HybridGaussianBernoulliNB, make_kernel_svc_classifier


class CoreCandidateBuilderError(ValueError):
    """Raised when core-candidate parameters are internally inconsistent."""


CORE_EXTENSION_CANDIDATE_IDS = frozenset(
    {
        CANDIDATE_RIDGE_CLASSIFIER,
        CANDIDATE_EXTRA_TREES,
        CANDIDATE_KNN,
        CANDIDATE_HYBRID_NAIVE_BAYES,
        CANDIDATE_DECISION_TREE,
    CANDIDATE_EXTRA_TREES,
        CANDIDATE_BAGGING,
        CANDIDATE_RANDOM_FOREST,
        CANDIDATE_ADABOOST,
        CANDIDATE_GRADIENT_BOOSTING,
        CANDIDATE_HIST_GRADIENT_BOOSTING,
        CANDIDATE_XGBOOST,
        CANDIDATE_LIGHTGBM,
        CANDIDATE_CATBOOST,
        CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE,
        CANDIDATE_RBF_SVM,
        CANDIDATE_TABNET,
        CANDIDATE_FT_TRANSFORMER,
        CANDIDATE_TABM,
    }
) | CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS


def _decode_class_weight(value: object) -> str | None:
    """Decode the JSON-safe class-weight values used by Optuna studies."""
    if value in {None, "none"}:
        return None
    if value in {"balanced", "balanced_subsample"}:
        return str(value)
    raise CoreCandidateBuilderError(f"Unsupported class-weight value: {value!r}")


def _decode_optional_positive_int(value: object, *, name: str) -> int | None:
    """Decode ``'none'`` or a positive integer from persisted trial parameters."""
    if value is None or value == "none":
        return None
    result = int(value)
    if result < 1:
        raise CoreCandidateBuilderError(f"{name} must be positive when supplied.")
    return result


def _decode_max_features(value: object) -> str | float:
    """Decode categorical Optuna storage into a scikit-learn tree parameter."""
    if value in {"sqrt", "log2"}:
        return str(value)
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise CoreCandidateBuilderError("max_features fractions must lie in (0, 1].")
    return result


def _suggest_tree_structure(
    trial: Any,
    *,
    profile: str,
    prefix: str,
) -> dict[str, Any]:
    """Suggest shared regularization controls for a tree-based candidate."""
    if profile == "smoke":
        depth_choices = ["none", "3", "6", "10"]
        split_upper = 20
        leaf_upper = 10
        leaf_node_choices = ["none", "8", "16", "32"]
    else:
        depth_choices = ["none", "3", "5", "8", "12", "18", "28"]
        split_upper = 100
        leaf_upper = 50
        leaf_node_choices = ["none", "8", "16", "32", "64", "128"]

    return {
        "criterion": trial.suggest_categorical(
            f"{prefix}criterion",
            ["gini", "entropy", "log_loss"],
        ),
        "max_depth": trial.suggest_categorical(f"{prefix}max_depth", depth_choices),
        "min_samples_split": int(
            trial.suggest_int(f"{prefix}min_samples_split", 2, split_upper)
        ),
        "min_samples_leaf": int(
            trial.suggest_int(f"{prefix}min_samples_leaf", 1, leaf_upper)
        ),
        "max_leaf_nodes": trial.suggest_categorical(
            f"{prefix}max_leaf_nodes",
            leaf_node_choices,
        ),
        "ccp_alpha": float(
            trial.suggest_float(f"{prefix}ccp_alpha", 1e-8, 1e-2, log=True)
        ),
    }


def suggest_core_candidate_parameters(
    trial: Any,
    *,
    candidate_id: str,
    profile: str,
) -> dict[str, Any]:
    """Suggest one complete JSON-safe configuration for a core candidate.

    The public candidate registry owns the identifiers and calls this extension only
    for candidates introduced in Phase 3. Keeping the optional-package imports out of
    this function means search-space validation can run without importing XGBoost,
    LightGBM, or CatBoost before an actual estimator needs to be constructed.
    """
    if profile not in {"smoke", "full"}:
        raise CoreCandidateBuilderError(f"Unsupported search profile: {profile!r}")
    if candidate_id not in CORE_EXTENSION_CANDIDATE_IDS:
        raise CoreCandidateBuilderError(f"Unknown core extension candidate: {candidate_id!r}")

    if candidate_id in CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS:
        return suggest_conventional_core_candidate_parameters(
            trial,
            candidate_id=candidate_id,
            profile=profile,
        )

    if candidate_id == CANDIDATE_RIDGE_CLASSIFIER:
        return {
            "alpha": float(trial.suggest_float("alpha", 1e-5, 1e4, log=True)),
            "class_weight": "none",
        }

    if candidate_id == CANDIDATE_KNN:
        maximum_neighbors = 31 if profile == "smoke" else 201
        return {
            "n_neighbors": int(
                trial.suggest_int("n_neighbors", 3, maximum_neighbors, step=2)
            ),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": int(trial.suggest_categorical("p", [1, 2])),
        }

    if candidate_id == CANDIDATE_HYBRID_NAIVE_BAYES:
        return {
            "alpha": float(trial.suggest_float("alpha", 1e-3, 25.0, log=True)),
            "var_smoothing": float(
                trial.suggest_float("var_smoothing", 1e-12, 1e-5, log=True)
            ),
        }

    if candidate_id == CANDIDATE_DECISION_TREE:
        return _suggest_tree_structure(trial, profile=profile, prefix="")

    if candidate_id == CANDIDATE_BAGGING:
        if profile == "smoke":
            estimators_low, estimators_high, estimators_step = 25, 80, 5
            depth_choices = ["none", "3", "6", "10"]
            leaf_upper = 10
        else:
            estimators_low, estimators_high, estimators_step = 200, 1_500, 50
            depth_choices = ["none", "3", "5", "8", "12", "18", "28"]
            leaf_upper = 50
        return {
            "n_estimators": int(
                trial.suggest_int(
                    "n_estimators",
                    estimators_low,
                    estimators_high,
                    step=estimators_step,
                )
            ),
            "max_samples": float(trial.suggest_float("max_samples", 0.45, 1.0)),
            "max_features": float(trial.suggest_float("max_features", 0.45, 1.0)),
            "base_max_depth": trial.suggest_categorical("base_max_depth", depth_choices),
            "base_min_samples_leaf": int(
                trial.suggest_int("base_min_samples_leaf", 1, leaf_upper)
            ),
            "base_class_weight": "none",
        }

    if candidate_id == CANDIDATE_RANDOM_FOREST:
        if profile == "smoke":
            estimators_low, estimators_high, estimators_step = 25, 80, 5
            depth_choices = ["none", "4", "8", "14"]
            split_upper, leaf_upper = 20, 10
        else:
            estimators_low, estimators_high, estimators_step = 300, 1_500, 50
            depth_choices = ["none", "4", "6", "10", "16", "24", "32"]
            split_upper, leaf_upper = 80, 40
        return {
            "n_estimators": int(
                trial.suggest_int(
                    "n_estimators",
                    estimators_low,
                    estimators_high,
                    step=estimators_step,
                )
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "max_depth": trial.suggest_categorical("max_depth", depth_choices),
            "min_samples_split": int(
                trial.suggest_int("min_samples_split", 2, split_upper)
            ),
            "min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 1, leaf_upper)
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", "0.5", "0.75", "1.0"]
            ),
            "max_samples": float(trial.suggest_float("max_samples", 0.5, 1.0)),
            "class_weight": "none",
            "ccp_alpha": float(trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True)),
        }

    if candidate_id == CANDIDATE_ADABOOST:
        maximum_estimators = 120 if profile == "smoke" else 1_200
        return {
            "base_depth": int(trial.suggest_int("base_depth", 1, 4)),
            "n_estimators": int(
                trial.suggest_int("n_estimators", 25, maximum_estimators, step=25)
            ),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 1.5, log=True)
            ),
        }

    if candidate_id == CANDIDATE_GRADIENT_BOOSTING:
        maximum_estimators = 150 if profile == "smoke" else 1_500
        return {
            "n_estimators": int(
                trial.suggest_int("n_estimators", 50, maximum_estimators, step=25)
            ),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 0.35, log=True)
            ),
            "max_depth": int(trial.suggest_int("max_depth", 1, 5)),
            "min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 1, 25 if profile == "smoke" else 100)
            ),
            "subsample": float(trial.suggest_float("subsample", 0.5, 1.0)),
        }

    if candidate_id == CANDIDATE_HIST_GRADIENT_BOOSTING:
        maximum_iter = 200 if profile == "smoke" else 1_500
        return {
            "max_iter": int(trial.suggest_int("max_iter", 50, maximum_iter, step=25)),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 0.35, log=True)
            ),
            "max_leaf_nodes": int(
                trial.suggest_categorical("max_leaf_nodes", [7, 15, 31, 63, 127])
            ),
            "min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 5, 30 if profile == "smoke" else 150)
            ),
            "l2_regularization": float(
                trial.suggest_float("l2_regularization", 1e-8, 1e2, log=True)
            ),
        }

    if candidate_id == CANDIDATE_XGBOOST:
        maximum_estimators = 200 if profile == "smoke" else 1_500
        return {
            "n_estimators": int(
                trial.suggest_int("n_estimators", 50, maximum_estimators, step=25)
            ),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            ),
            "max_depth": int(trial.suggest_int("max_depth", 2, 10)),
            "min_child_weight": float(
                trial.suggest_float("min_child_weight", 1e-2, 50.0, log=True)
            ),
            "subsample": float(trial.suggest_float("subsample", 0.5, 1.0)),
            "colsample_bytree": float(
                trial.suggest_float("colsample_bytree", 0.5, 1.0)
            ),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-5, 1e3, log=True)),
            "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-8, 1e2, log=True)),
            "gamma": float(trial.suggest_float("gamma", 1e-8, 10.0, log=True)),
        }

    if candidate_id == CANDIDATE_LIGHTGBM:
        maximum_estimators = 200 if profile == "smoke" else 1_500
        return {
            "n_estimators": int(
                trial.suggest_int("n_estimators", 50, maximum_estimators, step=25)
            ),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            ),
            "num_leaves": int(
                trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127])
            ),
            "min_child_samples": int(
                trial.suggest_int("min_child_samples", 5, 50 if profile == "smoke" else 200)
            ),
            "subsample": float(trial.suggest_float("subsample", 0.5, 1.0)),
            "colsample_bytree": float(
                trial.suggest_float("colsample_bytree", 0.5, 1.0)
            ),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-8, 1e3, log=True)),
            "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-8, 1e2, log=True)),
        }

    if candidate_id == CANDIDATE_CATBOOST:
        maximum_iterations = 250 if profile == "smoke" else 1_500
        return {
            "iterations": int(
                trial.suggest_int("iterations", 50, maximum_iterations, step=25)
            ),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            ),
            "depth": int(trial.suggest_int("depth", 3, 10)),
            "l2_leaf_reg": float(
                trial.suggest_float("l2_leaf_reg", 1e-4, 1e3, log=True)
            ),
        }

    if candidate_id == CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE:
        if profile == "smoke":
            return {
                "interactions": trial.suggest_categorical("interactions", [0]),
                "max_rounds": int(trial.suggest_categorical("max_rounds", [64])),
                "outer_bags": int(trial.suggest_categorical("outer_bags", [1])),
                "learning_rate": float(
                    trial.suggest_categorical("learning_rate", [0.03])
                ),
                "max_leaves": int(trial.suggest_categorical("max_leaves", [2])),
                "min_samples_leaf": int(
                    trial.suggest_categorical("min_samples_leaf", [4])
                ),
                "early_stopping_rounds": int(
                    trial.suggest_categorical("early_stopping_rounds", [10])
                ),
            }
        return {
            "interactions": trial.suggest_categorical(
                "interactions",
                [0, 5, 10],
            ),
            "max_rounds": int(trial.suggest_categorical("max_rounds", [2_000])),
            "outer_bags": int(trial.suggest_categorical("outer_bags", [4])),
            "learning_rate": float(
                trial.suggest_categorical(
                    "learning_rate",
                    [0.01, 0.03, 0.05],
                )
            ),
            "max_leaves": int(trial.suggest_categorical("max_leaves", [2, 3])),
            "min_samples_leaf": int(
                trial.suggest_categorical("min_samples_leaf", [2, 4, 8, 16])
            ),
            "early_stopping_rounds": int(
                trial.suggest_categorical("early_stopping_rounds", [100])
            ),
        }

    if candidate_id == CANDIDATE_RBF_SVM:
        return {
            "C": float(trial.suggest_float("C", 1e-4, 1e3, log=True)),
            "gamma": float(trial.suggest_float("gamma", 1e-5, 10.0, log=True)),
            "class_weight": "none",
        }

    if candidate_id == CANDIDATE_TABNET:
        if profile == "smoke":
            return {
                "n_d": int(trial.suggest_categorical("n_d", [8])),
                "n_a": int(trial.suggest_categorical("n_a", [8])),
                "n_steps": int(trial.suggest_categorical("n_steps", [3])),
                "gamma": float(trial.suggest_categorical("gamma", [1.3])),
                "cat_emb_dim": int(trial.suggest_categorical("cat_emb_dim", [1])),
                "n_independent": int(
                    trial.suggest_categorical("n_independent", [1])
                ),
                "n_shared": int(trial.suggest_categorical("n_shared", [1])),
                "lambda_sparse": float(
                    trial.suggest_categorical("lambda_sparse", [0.001])
                ),
                "learning_rate": float(
                    trial.suggest_categorical("learning_rate", [0.02])
                ),
                "weight_decay": float(
                    trial.suggest_categorical("weight_decay", [0.0])
                ),
                "max_epochs": int(trial.suggest_categorical("max_epochs", [8])),
                "patience": int(trial.suggest_categorical("patience", [3])),
                "batch_size": int(trial.suggest_categorical("batch_size", [128])),
                "virtual_batch_size": int(
                    trial.suggest_categorical("virtual_batch_size", [64])
                ),
                "mask_type": trial.suggest_categorical("mask_type", ["sparsemax"]),
            }
        return {
            "n_d": int(trial.suggest_categorical("n_d", [8, 16, 24])),
            "n_a": int(trial.suggest_categorical("n_a", [8, 16, 24])),
            "n_steps": int(trial.suggest_categorical("n_steps", [3, 4, 5])),
            "gamma": float(trial.suggest_categorical("gamma", [1.1, 1.3, 1.5, 1.8])),
            "cat_emb_dim": int(trial.suggest_categorical("cat_emb_dim", [1, 2, 4])),
            "n_independent": int(
                trial.suggest_categorical("n_independent", [1, 2])
            ),
            "n_shared": int(trial.suggest_categorical("n_shared", [1, 2])),
            "lambda_sparse": float(
                trial.suggest_categorical("lambda_sparse", [0.0001, 0.001, 0.01])
            ),
            "learning_rate": float(
                trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.02, 0.03])
            ),
            "weight_decay": float(
                trial.suggest_categorical("weight_decay", [0.0, 0.00001, 0.0001])
            ),
            "max_epochs": int(trial.suggest_categorical("max_epochs", [200])),
            "patience": int(trial.suggest_categorical("patience", [20])),
            "batch_size": int(trial.suggest_categorical("batch_size", [128, 256, 512])),
            "virtual_batch_size": int(
                trial.suggest_categorical("virtual_batch_size", [64])
            ),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        }

    if candidate_id == CANDIDATE_FT_TRANSFORMER:
        if profile == "smoke":
            return {
                "n_blocks": int(trial.suggest_categorical("n_blocks", [1])),
                "d_block": int(trial.suggest_categorical("d_block", [32])),
                "attention_n_heads": int(
                    trial.suggest_categorical("attention_n_heads", [4])
                ),
                "attention_dropout": float(
                    trial.suggest_categorical("attention_dropout", [0.0])
                ),
                "ffn_d_hidden_multiplier": float(
                    trial.suggest_categorical(
                        "ffn_d_hidden_multiplier",
                        [4.0 / 3.0],
                    )
                ),
                "ffn_dropout": float(
                    trial.suggest_categorical("ffn_dropout", [0.0])
                ),
                "residual_dropout": float(
                    trial.suggest_categorical("residual_dropout", [0.0])
                ),
                "learning_rate": float(
                    trial.suggest_categorical("learning_rate", [0.001])
                ),
                "weight_decay": float(
                    trial.suggest_categorical("weight_decay", [0.00001])
                ),
                "max_epochs": int(trial.suggest_categorical("max_epochs", [8])),
                "patience": int(trial.suggest_categorical("patience", [3])),
                "batch_size": int(trial.suggest_categorical("batch_size", [128])),
            }
        return {
            "n_blocks": int(trial.suggest_categorical("n_blocks", [1, 2, 3])),
            "d_block": int(trial.suggest_categorical("d_block", [32, 64, 96])),
            "attention_n_heads": int(
                trial.suggest_categorical("attention_n_heads", [4, 8])
            ),
            "attention_dropout": float(
                trial.suggest_categorical("attention_dropout", [0.0, 0.1, 0.2])
            ),
            "ffn_d_hidden_multiplier": float(
                trial.suggest_categorical(
                    "ffn_d_hidden_multiplier",
                    [1.0, 4.0 / 3.0, 2.0],
                )
            ),
            "ffn_dropout": float(
                trial.suggest_categorical("ffn_dropout", [0.0, 0.05, 0.1, 0.2])
            ),
            "residual_dropout": float(
                trial.suggest_categorical("residual_dropout", [0.0, 0.05, 0.1])
            ),
            "learning_rate": float(
                trial.suggest_categorical("learning_rate", [0.0003, 0.001, 0.003])
            ),
            "weight_decay": float(
                trial.suggest_categorical("weight_decay", [0.0, 0.00001, 0.0001])
            ),
            "max_epochs": int(trial.suggest_categorical("max_epochs", [200])),
            "patience": int(trial.suggest_categorical("patience", [20])),
            "batch_size": int(trial.suggest_categorical("batch_size", [128, 256])),
        }

    if candidate_id == CANDIDATE_TABM:
        if profile == "smoke":
            return {
                "arch_type": trial.suggest_categorical("arch_type", ["tabm-mini"]),
                "n_blocks": int(trial.suggest_categorical("n_blocks", [1])),
                "d_block": int(trial.suggest_categorical("d_block", [32])),
                "dropout": float(trial.suggest_categorical("dropout", [0.0])),
                "k": int(trial.suggest_categorical("k", [4])),
                "learning_rate": float(
                    trial.suggest_categorical("learning_rate", [0.001])
                ),
                "weight_decay": float(
                    trial.suggest_categorical("weight_decay", [0.00001])
                ),
                "max_epochs": int(trial.suggest_categorical("max_epochs", [8])),
                "patience": int(trial.suggest_categorical("patience", [3])),
                "batch_size": int(trial.suggest_categorical("batch_size", [128])),
            }
        return {
            "arch_type": trial.suggest_categorical(
                "arch_type",
                ["tabm-mini", "tabm"],
            ),
            "n_blocks": int(trial.suggest_categorical("n_blocks", [1, 2, 3])),
            "d_block": int(trial.suggest_categorical("d_block", [32, 64, 128])),
            "dropout": float(trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])),
            "k": int(trial.suggest_categorical("k", [4, 8, 16])),
            "learning_rate": float(
                trial.suggest_categorical("learning_rate", [0.0003, 0.001, 0.003])
            ),
            "weight_decay": float(
                trial.suggest_categorical("weight_decay", [0.0, 0.00001, 0.0001])
            ),
            "max_epochs": int(trial.suggest_categorical("max_epochs", [200])),
            "patience": int(trial.suggest_categorical("patience", [20])),
            "batch_size": int(trial.suggest_categorical("batch_size", [128, 256])),
        }

    raise CoreCandidateBuilderError(f"No search space for {candidate_id!r}")


def _make_bagging_classifier(
    *,
    n_estimators: int,
    max_samples: float,
    max_features: float,
    base_max_depth: int | None,
    base_min_samples_leaf: int,
    base_class_weight: str | None,
    random_state: int,
) -> BaggingClassifier:
    """Create single-threaded bagging for an outer-worker execution environment."""
    base_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf,
        class_weight=base_class_weight,
        random_state=random_state,
    )
    kwargs = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "max_features": max_features,
        "bootstrap": True,
        "bootstrap_features": False,
        "oob_score": False,
        "n_jobs": 1,
        "random_state": random_state,
    }
    try:
        return BaggingClassifier(estimator=base_tree, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_tree, **kwargs)


def _make_adaboost_classifier(
    *,
    base_depth: int,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> AdaBoostClassifier:
    """Create AdaBoost with a deterministic shallow tree base learner."""
    base_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=base_depth,
        random_state=random_state,
    )
    kwargs = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "random_state": random_state,
    }
    try:
        return AdaBoostClassifier(estimator=base_tree, **kwargs)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base_tree, **kwargs)


def _make_xgboost_classifier(parameters: Mapping[str, Any], *, random_state: int):
    """Create a one-thread XGBoost classifier without changing notebook factories."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for C17_XGBOOST. Install the project requirements."
        ) from exc

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=int(parameters["n_estimators"]),
        learning_rate=float(parameters["learning_rate"]),
        max_depth=int(parameters["max_depth"]),
        min_child_weight=float(parameters["min_child_weight"]),
        subsample=float(parameters["subsample"]),
        colsample_bytree=float(parameters["colsample_bytree"]),
        reg_lambda=float(parameters["reg_lambda"]),
        reg_alpha=float(parameters["reg_alpha"]),
        gamma=float(parameters["gamma"]),
        random_state=int(random_state),
        n_jobs=1,
        verbosity=0,
    )


def _make_lightgbm_classifier(parameters: Mapping[str, Any], *, random_state: int):
    """Create a one-thread native-categorical LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for C18_LIGHTGBM. Install the project requirements."
        ) from exc

    return LGBMClassifier(
        objective="binary",
        n_estimators=int(parameters["n_estimators"]),
        learning_rate=float(parameters["learning_rate"]),
        num_leaves=int(parameters["num_leaves"]),
        min_child_samples=int(parameters["min_child_samples"]),
        subsample=float(parameters["subsample"]),
        subsample_freq=1,
        colsample_bytree=float(parameters["colsample_bytree"]),
        reg_lambda=float(parameters["reg_lambda"]),
        reg_alpha=float(parameters["reg_alpha"]),
        random_state=int(random_state),
        n_jobs=1,
        verbosity=-1,
    )


def _make_explainable_boosting_classifier(
    parameters: Mapping[str, Any],
    *,
    random_state: int,
    feature_policy: FeaturePolicyId,
):
    """Create a one-thread EBM with explicit native categorical feature types."""
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as exc:
        raise ImportError(
            "interpret-core is required for C20_EXPLAINABLE_BOOSTING_MACHINE. "
            "Install the project requirements."
        ) from exc

    numeric_features = tuple(feature_policy_numeric_features(feature_policy))
    categorical_features = tuple(feature_policy_categorical_features(feature_policy))
    feature_names = [*numeric_features, *categorical_features]
    feature_types = ["continuous"] * len(numeric_features) + ["nominal"] * len(
        categorical_features
    )
    return ExplainableBoostingClassifier(
        feature_names=feature_names,
        feature_types=feature_types,
        max_bins=1024,
        max_interaction_bins=64,
        interactions=parameters["interactions"],
        validation_size=0.15,
        outer_bags=int(parameters["outer_bags"]),
        inner_bags=0,
        learning_rate=float(parameters["learning_rate"]),
        max_rounds=int(parameters["max_rounds"]),
        early_stopping_rounds=int(parameters["early_stopping_rounds"]),
        early_stopping_tolerance=1e-5,
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_leaves=int(parameters["max_leaves"]),
        objective="log_loss",
        n_jobs=1,
        random_state=int(random_state),
    )


def build_core_candidate_pipeline(
    candidate_id: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int,
    feature_policy: FeaturePolicyId = FEATURE_POLICY_RAW,
    feature_selection_policy: FeatureSelectionPolicyId = FEATURE_SELECTION_NONE,
    feature_selection_parameters: Mapping[str, object] | None = None,
) -> Pipeline:
    """Build one fresh single-threaded core-candidate pipeline.

    This function receives only persisted JSON-compatible parameter values and a
    prevalidated feature-policy identifier. It decodes every optional integer,
    class-weight policy, and maximum-feature policy explicitly before constructing the
    estimator, preventing silent differences between a live Optuna trial and a resumed
    task.
    """
    if candidate_id not in CORE_EXTENSION_CANDIDATE_IDS:
        raise CoreCandidateBuilderError(f"Unknown core extension candidate: {candidate_id!r}")
    parameters = dict(parameters)
    if candidate_id in CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS:
        return build_conventional_core_candidate_pipeline(
            candidate_id,
            parameters,
            random_state=int(random_state),
            feature_policy=feature_policy,
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
        )

    make_routed_pipeline = partial(
        make_feature_policy_classifier_pipeline,
        policy_id=feature_policy,
        feature_selection_policy=feature_selection_policy,
        feature_selection_parameters=feature_selection_parameters,
        random_state=int(random_state),
    )

    if candidate_id == CANDIDATE_RIDGE_CLASSIFIER:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_SCALED,
            classifier=RidgeClassifier(
                alpha=float(parameters["alpha"]),
                class_weight=_decode_class_weight(parameters["class_weight"]),
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_KNN:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_SCALED,
            classifier=KNeighborsClassifier(
                n_neighbors=int(parameters["n_neighbors"]),
                weights=str(parameters["weights"]),
                p=int(parameters["p"]),
                metric="minkowski",
                n_jobs=1,
            ),
        )

    if candidate_id == CANDIDATE_HYBRID_NAIVE_BAYES:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_UNSCALED,
            classifier=HybridGaussianBernoulliNB(
                n_numeric_features=len(feature_policy_numeric_features(feature_policy)),
                alpha=float(parameters["alpha"]),
                var_smoothing=float(parameters["var_smoothing"]),
            ),
        )

    if candidate_id == CANDIDATE_DECISION_TREE:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_UNSCALED,
            classifier=DecisionTreeClassifier(
                criterion=str(parameters["criterion"]),
                max_depth=_decode_optional_positive_int(
                    parameters["max_depth"], name="max_depth"
                ),
                min_samples_split=int(parameters["min_samples_split"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                max_leaf_nodes=_decode_optional_positive_int(
                    parameters["max_leaf_nodes"], name="max_leaf_nodes"
                ),
                ccp_alpha=float(parameters["ccp_alpha"]),
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_BAGGING:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_UNSCALED,
            classifier=_make_bagging_classifier(
                n_estimators=int(parameters["n_estimators"]),
                max_samples=float(parameters["max_samples"]),
                max_features=float(parameters["max_features"]),
                base_max_depth=_decode_optional_positive_int(
                    parameters["base_max_depth"], name="base_max_depth"
                ),
                base_min_samples_leaf=int(parameters["base_min_samples_leaf"]),
                base_class_weight=_decode_class_weight(parameters["base_class_weight"]),
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_RANDOM_FOREST:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_UNSCALED,
            classifier=RandomForestClassifier(
                n_estimators=int(parameters["n_estimators"]),
                criterion=str(parameters["criterion"]),
                max_depth=_decode_optional_positive_int(
                    parameters["max_depth"], name="max_depth"
                ),
                min_samples_split=int(parameters["min_samples_split"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                max_features=_decode_max_features(parameters["max_features"]),
                bootstrap=True,
                max_samples=float(parameters["max_samples"]),
                class_weight=_decode_class_weight(parameters["class_weight"]),
                ccp_alpha=float(parameters["ccp_alpha"]),
                n_jobs=1,
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_ADABOOST:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_DENSE_UNSCALED,
            classifier=_make_adaboost_classifier(
                base_depth=int(parameters["base_depth"]),
                n_estimators=int(parameters["n_estimators"]),
                learning_rate=float(parameters["learning_rate"]),
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_GRADIENT_BOOSTING:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_DENSE_UNSCALED,
            classifier=GradientBoostingClassifier(
                loss="log_loss",
                n_estimators=int(parameters["n_estimators"]),
                learning_rate=float(parameters["learning_rate"]),
                max_depth=int(parameters["max_depth"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                subsample=float(parameters["subsample"]),
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_HIST_GRADIENT_BOOSTING:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_DENSE_UNSCALED,
            classifier=HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=int(parameters["max_iter"]),
                learning_rate=float(parameters["learning_rate"]),
                max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                l2_regularization=float(parameters["l2_regularization"]),
                early_stopping=False,
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_XGBOOST:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_DENSE_UNSCALED,
            classifier=_make_xgboost_classifier(parameters, random_state=random_state),
        )

    if candidate_id == CANDIDATE_LIGHTGBM:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_NATIVE_CATEGORICAL_DTYPE,
            classifier=_make_lightgbm_classifier(parameters, random_state=random_state),
        )

    if candidate_id == CANDIDATE_CATBOOST:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_NATIVE_CATEGORICAL_STRING,
            classifier=CloneSafeFeaturePolicyCatBoostClassifier(
                iterations=int(parameters["iterations"]),
                learning_rate=float(parameters["learning_rate"]),
                depth=int(parameters["depth"]),
                l2_leaf_reg=float(parameters["l2_leaf_reg"]),
                categorical_features=tuple(
                    feature_policy_categorical_features(feature_policy)
                ),
                random_state=int(random_state),
                thread_count=1,
            ),
        )

    if candidate_id == CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_NATIVE_CATEGORICAL_STRING,
            classifier=_make_explainable_boosting_classifier(
                parameters,
                random_state=random_state,
                feature_policy=feature_policy,
            ),
        )

    if candidate_id == CANDIDATE_RBF_SVM:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_SPARSE_SCALED,
            classifier=make_kernel_svc_classifier(
                C=float(parameters["C"]),
                kernel="rbf",
                gamma=float(parameters["gamma"]),
                class_weight=_decode_class_weight(parameters["class_weight"]),
                random_state=int(random_state),
            ),
        )

    if candidate_id == CANDIDATE_TABNET:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_NATIVE_CATEGORICAL_STRING,
            classifier=CloneSafeFeaturePolicyTabNetClassifier(
                n_d=int(parameters["n_d"]),
                n_a=int(parameters["n_a"]),
                n_steps=int(parameters["n_steps"]),
                gamma=float(parameters["gamma"]),
                cat_emb_dim=int(parameters["cat_emb_dim"]),
                n_independent=int(parameters["n_independent"]),
                n_shared=int(parameters["n_shared"]),
                lambda_sparse=float(parameters["lambda_sparse"]),
                learning_rate=float(parameters["learning_rate"]),
                weight_decay=float(parameters["weight_decay"]),
                max_epochs=int(parameters["max_epochs"]),
                patience=int(parameters["patience"]),
                batch_size=int(parameters["batch_size"]),
                virtual_batch_size=int(parameters["virtual_batch_size"]),
                mask_type=str(parameters["mask_type"]),
                numeric_features=tuple(feature_policy_numeric_features(feature_policy)),
                categorical_features=tuple(
                    feature_policy_categorical_features(feature_policy)
                ),
                random_state=int(random_state),
                device_name="cpu",
                num_workers=0,
                verbose=0,
            ),
        )

    if candidate_id == CANDIDATE_FT_TRANSFORMER:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_NATIVE_CATEGORICAL_STRING,
            classifier=CloneSafeFeaturePolicyFTTransformerClassifier(
                n_blocks=int(parameters["n_blocks"]),
                d_block=int(parameters["d_block"]),
                attention_n_heads=int(parameters["attention_n_heads"]),
                attention_dropout=float(parameters["attention_dropout"]),
                ffn_d_hidden_multiplier=float(parameters["ffn_d_hidden_multiplier"]),
                ffn_dropout=float(parameters["ffn_dropout"]),
                residual_dropout=float(parameters["residual_dropout"]),
                learning_rate=float(parameters["learning_rate"]),
                weight_decay=float(parameters["weight_decay"]),
                max_epochs=int(parameters["max_epochs"]),
                patience=int(parameters["patience"]),
                batch_size=int(parameters["batch_size"]),
                numeric_features=tuple(feature_policy_numeric_features(feature_policy)),
                categorical_features=tuple(
                    feature_policy_categorical_features(feature_policy)
                ),
                random_state=int(random_state),
                device_name="cpu",
                num_workers=0,
            ),
        )

    if candidate_id == CANDIDATE_TABM:
        return make_routed_pipeline(
            policy_id=feature_policy,
            representation=REPRESENTATION_NATIVE_CATEGORICAL_STRING,
            classifier=CloneSafeFeaturePolicyTabMClassifier(
                arch_type=str(parameters["arch_type"]),
                n_blocks=int(parameters["n_blocks"]),
                d_block=int(parameters["d_block"]),
                dropout=float(parameters["dropout"]),
                k=int(parameters["k"]),
                learning_rate=float(parameters["learning_rate"]),
                weight_decay=float(parameters["weight_decay"]),
                max_epochs=int(parameters["max_epochs"]),
                patience=int(parameters["patience"]),
                batch_size=int(parameters["batch_size"]),
                numeric_features=tuple(feature_policy_numeric_features(feature_policy)),
                categorical_features=tuple(
                    feature_policy_categorical_features(feature_policy)
                ),
                random_state=int(random_state),
                device_name="cpu",
                num_workers=0,
            ),
        )

    raise CoreCandidateBuilderError(f"No pipeline builder for {candidate_id!r}")


def declared_single_thread_parameter(
    candidate_id: str,
    classifier: Any,
) -> tuple[str, int] | None:
    """Return the explicit inner parallelism setting for candidates that expose one.

    The core-registry smoke test calls this helper after fitting representative
    pipelines. Returning a value rather than printing from model constructors keeps
    worker output concise in the actual long-running experiment.
    """
    conventional = declared_conventional_single_thread_parameter(
        candidate_id,
        classifier,
    )
    if conventional is not None:
        return conventional

    if candidate_id in {
        CANDIDATE_EXTRA_TREES,
        CANDIDATE_KNN,
        CANDIDATE_BAGGING,
        CANDIDATE_RANDOM_FOREST,
        CANDIDATE_XGBOOST,
        CANDIDATE_LIGHTGBM,
        CANDIDATE_EXPLAINABLE_BOOSTING_MACHINE,
    }:
        return "n_jobs", int(classifier.get_params()["n_jobs"])
    if candidate_id == CANDIDATE_CATBOOST:
        return "thread_count", int(classifier.get_params()["thread_count"])
    return None
