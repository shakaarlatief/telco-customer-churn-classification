"""Fold-safe conventional core candidates for final model comparison.

This module implements the first conventional expansion that was documented but absent
from the initial 17-family registry:

- C03 spline logistic regression;
- C04 shrinkage LDA;
- C05 regularized QDA;
- C12 balanced random forest; and
- C14 RUSBoost.

The module intentionally uses literal candidate identifiers. Importing the public
registry would create a circular dependency because the registry delegates pipeline
construction to the core builder, which delegates these five identifiers back here.
Every builder returns the standard four-step pipeline topology required by the shared
imbalance-routing adapter:

``raw rows -> feature policy -> representation preprocessor -> selector -> classifier``.

No candidate reads held-out test data, fits a transformation globally, or chooses a
policy from validation outcomes.
"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from telco_churn.feature_policies import (
    FeaturePolicyId,
    FeaturePolicyTransformer,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
)
from telco_churn.feature_policy_pipelines import make_feature_policy_one_hot_preprocessor
from telco_churn.feature_selection import (
    FEATURE_SELECTION_NONE,
    FeatureSelectionPolicyId,
    make_feature_selector,
)
from telco_churn.models import make_logistic_regression_classifier


CANDIDATE_SPLINE_LOGISTIC = "C03_SPLINE_LOGISTIC_REGRESSION"
CANDIDATE_SHRINKAGE_LDA = "C04_SHRINKAGE_LDA"
CANDIDATE_REGULARIZED_QDA = "C05_REGULARIZED_QDA"
CANDIDATE_BALANCED_RANDOM_FOREST = "C12_BALANCED_RANDOM_FOREST"
CANDIDATE_RUSBOOST = "C14_RUSBOOST"

CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS = frozenset(
    {
        CANDIDATE_SPLINE_LOGISTIC,
        CANDIDATE_SHRINKAGE_LDA,
        CANDIDATE_REGULARIZED_QDA,
        CANDIDATE_BALANCED_RANDOM_FOREST,
        CANDIDATE_RUSBOOST,
    }
)


class ConventionalCoreCandidateError(ValueError):
    """Raised when a conventional-core candidate receives invalid persisted parameters."""


def _make_one_hot_encoder(*, dense: bool, drop: str | None = None) -> OneHotEncoder:
    """Create a compatible encoder with optional dropped categorical references."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop=drop,
            sparse_output=not dense,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop=drop,
            sparse=not dense,
        )


def _decode_optional_positive_int(value: object, *, name: str) -> int | None:
    """Decode ``"none"`` or a positive persisted integer."""
    if value is None or value == "none":
        return None
    result = int(value)
    if result < 1:
        raise ConventionalCoreCandidateError(f"{name} must be positive when supplied.")
    return result


def _decode_max_features(value: object) -> str | float:
    """Decode persisted tree-feature subsampling choices."""
    if value in {"sqrt", "log2"}:
        return str(value)
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ConventionalCoreCandidateError("max_features fractions must lie in (0, 1].")
    return result


def _decode_lda_shrinkage(value: object) -> str | float:
    """Decode ``"auto"`` or a fixed unit-interval LDA shrinkage value."""
    if value == "auto":
        return "auto"
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ConventionalCoreCandidateError(
            "LDA shrinkage must lie in [0, 1] or equal 'auto'."
        )
    return result


def _make_spline_preprocessor(
    feature_policy: FeaturePolicyId,
    *,
    n_knots: int,
    degree: int,
) -> ColumnTransformer:
    """Create fold-local numeric B-spline and categorical-indicator preprocessing.

    Quantile-spaced knots are learned within every active training partition. This is
    important because global knot fitting would leak feature-distribution information
    from validation observations into the spline basis. Scaling is deliberately applied
    after the basis expansion so logistic regularization acts comparably across columns.
    """
    if int(n_knots) < 2:
        raise ConventionalCoreCandidateError("Spline models require at least two knots.")
    if int(degree) < 1:
        raise ConventionalCoreCandidateError("Spline degree must be positive.")

    numeric_features = feature_policy_numeric_features(feature_policy)
    categorical_features = feature_policy_categorical_features(feature_policy)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "spline",
                SplineTransformer(
                    n_knots=int(n_knots),
                    degree=int(degree),
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                    sparse_output=False,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder(dense=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


class DenseDiscriminantPreprocessor(BaseEstimator, TransformerMixin):
    """Create a fully standardized dense representation for LDA and QDA.

    The representation first median-imputes numeric variables and mode-imputes
    categorical variables, then one-hot encodes each categorical variable with one
    reference level dropped. The complete resulting dense matrix, including retained
    categorical indicators, is standardized before covariance-based discrimination.

    This is intentionally one estimator rather than an outer scikit-learn
    ``Pipeline(ColumnTransformer -> StandardScaler)``. Generic I2/I3/I4 imbalance
    policies wrap the candidate in an ``imblearn.Pipeline``, whose intermediate steps
    cannot themselves be scikit-learn pipelines. The internal fitted
    ``columnwise_`` and ``scaler_`` objects preserve the original transformation order
    while exposing one outer transformer compatible with that topology.

    Every fitted object is learned from the active training partition only. In
    particular, imputation statistics, category levels, and scaling moments are never
    derived from validation or held-out observations.
    """

    def __init__(self, feature_policy: FeaturePolicyId) -> None:
        """Store only the immutable feature-policy identifier for sklearn cloning."""
        self.feature_policy = feature_policy

    def fit(self, X, y=None):
        """Fit fold-local columnwise encoding and global post-encoding scaling."""
        numeric_features = feature_policy_numeric_features(self.feature_policy)
        categorical_features = feature_policy_categorical_features(self.feature_policy)
        self.columnwise_ = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", _make_one_hot_encoder(dense=True, drop="first")),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        encoded = self.columnwise_.fit_transform(X, y)
        self.scaler_ = StandardScaler().fit(encoded)
        return self

    def transform(self, X):
        """Apply the fitted encoder and the fitted complete-matrix standardizer."""
        encoded = self.columnwise_.transform(X)
        return self.scaler_.transform(encoded)


def _make_discriminant_preprocessor(
    feature_policy: FeaturePolicyId,
) -> DenseDiscriminantPreprocessor:
    """Return one imbalanced-learn-compatible dense discriminant transformer."""
    return DenseDiscriminantPreprocessor(feature_policy=feature_policy)


def _make_standard_pipeline(
    *,
    feature_policy: FeaturePolicyId,
    preprocessor,
    classifier,
    feature_selection_policy: FeatureSelectionPolicyId,
    feature_selection_parameters: Mapping[str, object] | None,
    random_state: int,
) -> Pipeline:
    """Return the standard named topology required by shared imbalance routing."""
    selector = make_feature_selector(
        feature_selection_policy,
        n_numeric_features=len(feature_policy_numeric_features(feature_policy)),
        parameters=feature_selection_parameters,
        random_state=int(random_state),
    )
    return Pipeline(
        steps=[
            ("feature_policy", FeaturePolicyTransformer(policy_id=feature_policy)),
            ("preprocessor", preprocessor),
            ("feature_selection", selector),
            ("classifier", classifier),
        ]
    )


def suggest_conventional_core_candidate_parameters(
    trial: Any,
    *,
    candidate_id: str,
    profile: str,
) -> dict[str, Any]:
    """Suggest one JSON-safe configuration for an expansion candidate.

    The public registry adds feature, selection, and external-imbalance choices after
    this function returns. C12 and C14 still expose their intrinsic imbalance controls
    here because those mechanisms define the model family itself.
    """
    if candidate_id not in CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS:
        raise ConventionalCoreCandidateError(f"Unknown expansion candidate: {candidate_id!r}")
    if profile not in {"smoke", "full"}:
        raise ConventionalCoreCandidateError(f"Unsupported search profile: {profile!r}")

    if candidate_id == CANDIDATE_SPLINE_LOGISTIC:
        parameters: dict[str, Any] = {
            "n_knots": int(
                trial.suggest_categorical(
                    "n_knots",
                    [3, 4] if profile == "smoke" else [3, 4, 5, 6, 8, 10],
                )
            ),
            "degree": int(trial.suggest_categorical("degree", [2, 3])),
            "penalty": trial.suggest_categorical(
                "penalty",
                ["l1", "l2", "elasticnet"],
            ),
            "C": float(trial.suggest_float("C", 1e-4, 1e3, log=True)),
            "class_weight": "none",
            "max_iter": 4_000 if profile == "smoke" else 8_000,
        }
        if parameters["penalty"] == "elasticnet":
            parameters["l1_ratio"] = float(
                trial.suggest_float("l1_ratio", 0.02, 0.98)
            )
        return parameters

    if candidate_id == CANDIDATE_SHRINKAGE_LDA:
        return {
            "shrinkage": trial.suggest_categorical(
                "shrinkage",
                (
                    ["auto", "0.10", "0.40"]
                    if profile == "smoke"
                    else [
                        "auto",
                        "0.01",
                        "0.03",
                        "0.05",
                        "0.10",
                        "0.20",
                        "0.40",
                        "0.60",
                        "0.80",
                    ]
                ),
            )
        }

    if candidate_id == CANDIDATE_REGULARIZED_QDA:
        return {
            "reg_param": float(
                trial.suggest_categorical(
                    "reg_param",
                    (
                        [0.10, 0.40]
                        if profile == "smoke"
                        else [0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95]
                    ),
                )
            )
        }

    if candidate_id == CANDIDATE_BALANCED_RANDOM_FOREST:
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
                "criterion",
                ["gini", "entropy", "log_loss"],
            ),
            "max_depth": trial.suggest_categorical("max_depth", depth_choices),
            "min_samples_split": int(
                trial.suggest_int("min_samples_split", 2, split_upper)
            ),
            "min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 1, leaf_upper)
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", "0.5", "0.75", "1.0"],
            ),
            "ccp_alpha": float(
                trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True)
            ),
        }

    if candidate_id == CANDIDATE_RUSBOOST:
        return {
            "base_criterion": trial.suggest_categorical(
                "base_criterion",
                ["gini", "entropy", "log_loss"],
            ),
            "base_depth": int(trial.suggest_int("base_depth", 1, 3)),
            "base_min_samples_leaf": int(
                trial.suggest_int(
                    "base_min_samples_leaf",
                    1,
                    10 if profile == "smoke" else 50,
                )
            ),
            "n_estimators": int(
                trial.suggest_int(
                    "n_estimators",
                    25 if profile == "smoke" else 100,
                    100 if profile == "smoke" else 1_000,
                    step=25,
                )
            ),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
            ),
            "internal_sampling_strategy": float(
                trial.suggest_categorical(
                    "internal_sampling_strategy",
                    [0.75] if profile == "smoke" else [0.5, 0.75, 1.0],
                )
            ),
        }

    raise ConventionalCoreCandidateError(
        f"No parameter suggestion branch for {candidate_id!r}."
    )


def _make_balanced_random_forest(
    parameters: Mapping[str, Any],
    *,
    random_state: int,
):
    """Construct C12 with intrinsic balanced bootstrap samples and one native worker."""
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for C12_BALANCED_RANDOM_FOREST."
        ) from exc

    return BalancedRandomForestClassifier(
        n_estimators=int(parameters["n_estimators"]),
        criterion=str(parameters["criterion"]),
        max_depth=_decode_optional_positive_int(
            parameters["max_depth"],
            name="max_depth",
        ),
        min_samples_split=int(parameters["min_samples_split"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=_decode_max_features(parameters["max_features"]),
        ccp_alpha=float(parameters["ccp_alpha"]),
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        class_weight=None,
        n_jobs=1,
        random_state=int(random_state),
    )


def _make_rusboost(
    parameters: Mapping[str, Any],
    *,
    random_state: int,
):
    """Construct C14 with shallow trees and intrinsic per-round undersampling."""
    try:
        from imblearn.ensemble import RUSBoostClassifier
    except ImportError as exc:
        raise ImportError("imbalanced-learn is required for C14_RUSBOOST.") from exc

    base_tree = DecisionTreeClassifier(
        criterion=str(parameters["base_criterion"]),
        max_depth=int(parameters["base_depth"]),
        min_samples_leaf=int(parameters["base_min_samples_leaf"]),
        random_state=int(random_state),
    )
    return RUSBoostClassifier(
        estimator=base_tree,
        n_estimators=int(parameters["n_estimators"]),
        learning_rate=float(parameters["learning_rate"]),
        sampling_strategy=float(parameters["internal_sampling_strategy"]),
        replacement=False,
        random_state=int(random_state),
    )


def build_conventional_core_candidate_pipeline(
    candidate_id: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int,
    feature_policy: FeaturePolicyId,
    feature_selection_policy: FeatureSelectionPolicyId = FEATURE_SELECTION_NONE,
    feature_selection_parameters: Mapping[str, object] | None = None,
) -> Pipeline:
    """Build one fresh unfitted conventional-core candidate pipeline."""
    if candidate_id not in CONVENTIONAL_CORE_EXPANSION_CANDIDATE_IDS:
        raise ConventionalCoreCandidateError(f"Unknown expansion candidate: {candidate_id!r}")
    parameters = dict(parameters)

    if candidate_id == CANDIDATE_SPLINE_LOGISTIC:
        return _make_standard_pipeline(
            feature_policy=feature_policy,
            preprocessor=_make_spline_preprocessor(
                feature_policy,
                n_knots=int(parameters["n_knots"]),
                degree=int(parameters["degree"]),
            ),
            classifier=make_logistic_regression_classifier(
                penalty=str(parameters["penalty"]),
                C=float(parameters["C"]),
                class_weight=(
                    None
                    if parameters.get("class_weight") in {None, "none"}
                    else str(parameters["class_weight"])
                ),
                l1_ratio=(
                    None
                    if parameters.get("l1_ratio") is None
                    else float(parameters["l1_ratio"])
                ),
                max_iter=int(parameters["max_iter"]),
                random_state=int(random_state),
            ),
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    if candidate_id == CANDIDATE_SHRINKAGE_LDA:
        return _make_standard_pipeline(
            feature_policy=feature_policy,
            preprocessor=_make_discriminant_preprocessor(feature_policy),
            classifier=LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage=_decode_lda_shrinkage(parameters["shrinkage"]),
            ),
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    if candidate_id == CANDIDATE_REGULARIZED_QDA:
        return _make_standard_pipeline(
            feature_policy=feature_policy,
            preprocessor=_make_discriminant_preprocessor(feature_policy),
            classifier=QuadraticDiscriminantAnalysis(
                reg_param=float(parameters["reg_param"]),
                tol=1e-4,
            ),
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    if candidate_id == CANDIDATE_BALANCED_RANDOM_FOREST:
        return _make_standard_pipeline(
            feature_policy=feature_policy,
            preprocessor=make_feature_policy_one_hot_preprocessor(
                feature_policy,
                scale_numeric=False,
                dense=False,
            ),
            classifier=_make_balanced_random_forest(
                parameters,
                random_state=int(random_state),
            ),
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    if candidate_id == CANDIDATE_RUSBOOST:
        return _make_standard_pipeline(
            feature_policy=feature_policy,
            preprocessor=make_feature_policy_one_hot_preprocessor(
                feature_policy,
                scale_numeric=False,
                dense=False,
            ),
            classifier=_make_rusboost(parameters, random_state=int(random_state)),
            feature_selection_policy=feature_selection_policy,
            feature_selection_parameters=feature_selection_parameters,
            random_state=int(random_state),
        )

    raise ConventionalCoreCandidateError(f"No builder branch for {candidate_id!r}.")


def declared_conventional_single_thread_parameter(
    candidate_id: str,
    classifier: Any,
) -> tuple[str, int] | None:
    """Return C12's explicit native worker count for the shared registry smoke."""
    if candidate_id == CANDIDATE_BALANCED_RANDOM_FOREST:
        return "n_jobs", int(classifier.get_params()["n_jobs"])
    return None
