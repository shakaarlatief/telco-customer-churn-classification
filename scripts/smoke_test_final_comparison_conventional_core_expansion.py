"""Training-only smoke test for the first conventional candidate expansion.

The test checks the specific mechanics that distinguish C03, C04, C05, C12, and C14.
It uses a small stratified sample from development training data only. It does not
estimate model performance and never loads the held-out test set.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_BALANCED_RANDOM_FOREST,
    CANDIDATE_REGULARIZED_QDA,
    CANDIDATE_RUSBOOST,
    CANDIDATE_SHRINKAGE_LDA,
    CANDIDATE_SPLINE_LOGISTIC_REGRESSION,
    build_candidate_pipeline,
    suggest_candidate_parameters,
    supported_feature_policies,
    supported_feature_selection_policies,
    supported_imbalance_policies,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.conventional_core_candidates import (  # noqa: E402
    DenseDiscriminantPreprocessor,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_RAW,
)
from telco_churn.feature_selection import FEATURE_SELECTION_NONE  # noqa: E402
from telco_churn.hpo import extract_continuous_scores  # noqa: E402
from telco_churn.imbalance_policies import (  # noqa: E402
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_NONE,
)


SAMPLE_SIZE = 480
CANDIDATE_IDS = (
    CANDIDATE_SPLINE_LOGISTIC_REGRESSION,
    CANDIDATE_SHRINKAGE_LDA,
    CANDIDATE_REGULARIZED_QDA,
    CANDIDATE_BALANCED_RANDOM_FOREST,
    CANDIDATE_RUSBOOST,
)


class DeterministicSmokeTrial:
    """Minimal Optuna-compatible trial selecting the first declared smoke value."""

    def suggest_categorical(self, name: str, choices):
        if not choices:
            raise AssertionError(f"Empty categorical choice set for {name!r}.")
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, **kwargs):
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise AssertionError(f"Invalid float range for {name!r}.")
        return float(low)

    def suggest_int(self, name: str, low: int, high: int, **kwargs):
        if low > high:
            raise AssertionError(f"Invalid integer range for {name!r}.")
        return int(low)


def make_partition() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build a deterministic development-only training and validation partition."""
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
        random_state=RANDOM_STATE + 71,
    )


def parameters_for(
    trial: DeterministicSmokeTrial,
    candidate_id: str,
    feature_policy: str,
) -> dict[str, object]:
    """Return a forced S0/I0 smoke configuration for one admitted route."""
    parameters = suggest_candidate_parameters(
        trial,
        candidate_id=candidate_id,
        profile="smoke",
    )
    parameters["feature_policy"] = feature_policy
    parameters["feature_selection_policy"] = FEATURE_SELECTION_NONE
    parameters["imbalance_policy"] = IMBALANCE_NONE
    parameters.pop("imbalance_sampling_strategy", None)
    parameters.pop("imbalance_smotenc_k_neighbors", None)
    return parameters


def assert_routes() -> None:
    """Verify the intentionally narrow feature, selector, and imbalance contracts."""
    expected_features = {
        CANDIDATE_SPLINE_LOGISTIC_REGRESSION: (FEATURE_POLICY_RAW,),
        CANDIDATE_SHRINKAGE_LDA: (FEATURE_POLICY_RAW, FEATURE_POLICY_DOMAIN),
        CANDIDATE_REGULARIZED_QDA: (FEATURE_POLICY_RAW, FEATURE_POLICY_DOMAIN),
        CANDIDATE_BALANCED_RANDOM_FOREST: (FEATURE_POLICY_RAW, FEATURE_POLICY_DOMAIN),
        CANDIDATE_RUSBOOST: (FEATURE_POLICY_RAW, FEATURE_POLICY_DOMAIN),
    }
    for candidate_id, expected in expected_features.items():
        if supported_feature_policies(candidate_id) != expected:
            raise AssertionError(f"{candidate_id} has an unexpected feature-policy contract.")
        for feature_policy in expected:
            if supported_feature_selection_policies(
                candidate_id,
                feature_policy,
            ) != (FEATURE_SELECTION_NONE,):
                raise AssertionError(f"{candidate_id} must remain S0-only.")

    for candidate_id in (
        CANDIDATE_BALANCED_RANDOM_FOREST,
        CANDIDATE_RUSBOOST,
    ):
        if supported_imbalance_policies(
            candidate_id,
            FEATURE_POLICY_RAW,
            FEATURE_SELECTION_NONE,
        ) != (IMBALANCE_NONE,):
            raise AssertionError(f"{candidate_id} must expose only intrinsic imbalance handling.")

    for candidate_id in (
        CANDIDATE_SHRINKAGE_LDA,
        CANDIDATE_REGULARIZED_QDA,
    ):
        policies = supported_imbalance_policies(
            candidate_id,
            FEATURE_POLICY_RAW,
            FEATURE_SELECTION_NONE,
        )
        if IMBALANCE_CLASS_WEIGHT_BALANCED in policies:
            raise AssertionError(f"{candidate_id} must not expose unsupported generic I1.")


def assert_probability_output(candidate_id: str, fitted_pipeline, X_validation: pd.DataFrame) -> None:
    """Validate binary labels and finite probability scores."""
    predictions = np.asarray(fitted_pipeline.predict(X_validation))
    if predictions.shape != (len(X_validation),) or not np.isin(predictions, [0, 1]).all():
        raise AssertionError(f"{candidate_id} returned invalid labels.")

    scores, kind = extract_continuous_scores(fitted_pipeline, X_validation)
    scores = np.asarray(scores, dtype=float)
    if kind != "probability":
        raise AssertionError(f"{candidate_id} must expose probabilities, not {kind!r}.")
    if scores.shape != (len(X_validation),) or not np.isfinite(scores).all():
        raise AssertionError(f"{candidate_id} returned invalid continuous scores.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise AssertionError(f"{candidate_id} returned values outside [0, 1].")


def _assert_discriminant_preprocessor(candidate_id: str, fitted_pipeline) -> None:
    """Lock in the fully standardized, imbalanced-learn-compatible C04/C05 topology."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    if not isinstance(preprocessor, DenseDiscriminantPreprocessor):
        raise AssertionError(
            f"{candidate_id} must use the dedicated discriminant transformer rather "
            "than a nested outer sklearn Pipeline."
        )
    if not isinstance(preprocessor.columnwise_, ColumnTransformer):
        raise AssertionError(
            f"{candidate_id} must fit its fold-local columnwise encoder before scaling."
        )
    if not isinstance(preprocessor.scaler_, StandardScaler):
        raise AssertionError(
            f"{candidate_id} must standardize the complete encoded feature matrix."
        )
    if not preprocessor.scaler_.with_mean or not preprocessor.scaler_.with_std:
        raise AssertionError(
            f"{candidate_id} must retain centered, unit-scale post-encoding features."
        )


def assert_candidate_mechanics(candidate_id: str, fitted_pipeline, parameters: dict[str, object]) -> None:
    """Inspect the fitted representation or estimator that defines each procedure."""
    classifier = fitted_pipeline.named_steps["classifier"]

    if candidate_id == CANDIDATE_SPLINE_LOGISTIC_REGRESSION:
        numeric = fitted_pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
        spline = numeric.named_steps["spline"]
        if spline.n_knots != int(parameters["n_knots"]):
            raise AssertionError("C03 did not preserve selected spline knot count.")
        if spline.degree != int(parameters["degree"]):
            raise AssertionError("C03 did not preserve selected spline degree.")
        if spline.knots != "quantile" or spline.include_bias:
            raise AssertionError("C03 must use fold-local quantile knots without a bias basis.")
        return

    if candidate_id == CANDIDATE_SHRINKAGE_LDA:
        if classifier.__class__.__name__ != "LinearDiscriminantAnalysis":
            raise AssertionError("C04 did not build shrinkage LDA.")
        if classifier.solver != "lsqr":
            raise AssertionError("C04 must use the shrinkage-compatible lsqr solver.")
        _assert_discriminant_preprocessor(candidate_id, fitted_pipeline)
        return

    if candidate_id == CANDIDATE_REGULARIZED_QDA:
        if classifier.__class__.__name__ != "QuadraticDiscriminantAnalysis":
            raise AssertionError("C05 did not build regularized QDA.")
        if not np.isclose(classifier.reg_param, float(parameters["reg_param"])):
            raise AssertionError("C05 did not preserve covariance regularization.")
        _assert_discriminant_preprocessor(candidate_id, fitted_pipeline)
        return

    if candidate_id == CANDIDATE_BALANCED_RANDOM_FOREST:
        if classifier.__class__.__name__ != "BalancedRandomForestClassifier":
            raise AssertionError("C12 did not build balanced random forest.")
        if classifier.n_jobs != 1:
            raise AssertionError("C12 must reserve parallelism for outer workers.")
        if classifier.sampling_strategy != "all" or classifier.replacement is not True:
            raise AssertionError("C12 intrinsic sampling contract changed unexpectedly.")
        return

    if candidate_id == CANDIDATE_RUSBOOST:
        if classifier.__class__.__name__ != "RUSBoostClassifier":
            raise AssertionError("C14 did not build RUSBoost.")
        if not np.isclose(
            float(classifier.sampling_strategy),
            float(parameters["internal_sampling_strategy"]),
        ):
            raise AssertionError("C14 did not preserve intrinsic sampling strategy.")
        return

    raise AssertionError(f"Unexpected candidate {candidate_id!r}.")


def assert_rejections(trial: DeterministicSmokeTrial) -> None:
    """Ensure excluded feature and generic-imbalance routes fail before fitting."""
    invalid_spline = parameters_for(
        trial,
        CANDIDATE_SPLINE_LOGISTIC_REGRESSION,
        FEATURE_POLICY_RAW,
    )
    invalid_spline["feature_policy"] = FEATURE_POLICY_DOMAIN
    try:
        build_candidate_pipeline(
            CANDIDATE_SPLINE_LOGISTIC_REGRESSION,
            invalid_spline,
            random_state=RANDOM_STATE,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("C03 must reject F1.")

    for candidate_id in (
        CANDIDATE_BALANCED_RANDOM_FOREST,
        CANDIDATE_RUSBOOST,
    ):
        invalid_intrinsic = parameters_for(
            trial,
            candidate_id,
            FEATURE_POLICY_RAW,
        )
        invalid_intrinsic["imbalance_policy"] = IMBALANCE_CLASS_WEIGHT_BALANCED
        try:
            build_candidate_pipeline(
                candidate_id,
                invalid_intrinsic,
                random_state=RANDOM_STATE,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"{candidate_id} must reject external I1.")


def main() -> None:
    """Run the complete training-only conventional-expansion smoke test."""
    validate_candidate_registry()
    assert_routes()
    X_train, X_validation, y_train, _ = make_partition()
    trial = DeterministicSmokeTrial()
    requested_policy = {
        CANDIDATE_SPLINE_LOGISTIC_REGRESSION: FEATURE_POLICY_RAW,
        CANDIDATE_SHRINKAGE_LDA: FEATURE_POLICY_DOMAIN,
        CANDIDATE_REGULARIZED_QDA: FEATURE_POLICY_DOMAIN,
        CANDIDATE_BALANCED_RANDOM_FOREST: FEATURE_POLICY_DOMAIN,
        CANDIDATE_RUSBOOST: FEATURE_POLICY_DOMAIN,
    }

    for index, candidate_id in enumerate(CANDIDATE_IDS, start=1):
        print(f"[{index:02d}/{len(CANDIDATE_IDS):02d}] Checking {candidate_id}...", flush=True)
        parameters = parameters_for(trial, candidate_id, requested_policy[candidate_id])
        pipeline = build_candidate_pipeline(
            candidate_id,
            parameters,
            random_state=RANDOM_STATE + index,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fitted_pipeline = clone(pipeline).fit(X_train, y_train)
        assert_probability_output(candidate_id, fitted_pipeline, X_validation)
        assert_candidate_mechanics(candidate_id, fitted_pipeline, parameters)

    assert_rejections(trial)
    print("Conventional core-candidate expansion smoke test passed.")


if __name__ == "__main__":
    main()
