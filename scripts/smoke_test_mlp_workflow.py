"""Fast compatibility smoke test for the MLP workflow.

The test uses a small stratified subset of the project's training data and tiny
representative MLP procedures. It validates the reusable dense scaled
preprocessor, shared MLP factories, probability-output path, generic
out-of-fold helper, threshold diagnostics, calibration primitives, and shared
plotting helpers before the slower training-only MLP notebook grid is run.

It is not a performance evaluation and does not read or use the held-out test
set.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    evaluate_threshold_grid,
    get_out_of_fold_predictions,
    make_precision_recall_curve_dataframe,
    make_roc_curve_dataframe,
)
from telco_churn.models import make_mlp_pipeline  # noqa: E402
from telco_churn.preprocessing import make_dense_scaled_preprocessor  # noqa: E402
from telco_churn.visualization import (  # noqa: E402
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
    save_threshold_tradeoff_plot,
)


SAMPLE_SIZE = 400


def make_small_stratified_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Load train.csv and return a small stratified modelling sample only."""
    train_df = load_train_data()
    X, y = split_features_target(train_df)

    if len(y) <= SAMPLE_SIZE:
        return X, y

    X_small, _, y_small, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_small.reset_index(drop=True), y_small.reset_index(drop=True)


def make_smoke_estimators():
    """Create tiny representative MLP procedures from shared project factories."""
    common_kwargs = {
        "batch_size": 32,
        "learning_rate_init": 0.001,
        "max_iter": 50,
        "tol": 1e-4,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 5,
        "random_state": RANDOM_STATE,
    }

    return {
        "shallow_relu": make_mlp_pipeline(
            hidden_layer_sizes=(8,),
            activation="relu",
            alpha=0.001,
            **common_kwargs,
        ),
        "two_layer_tanh": make_mlp_pipeline(
            hidden_layer_sizes=(12, 6),
            activation="tanh",
            alpha=0.01,
            **common_kwargs,
        ),
    }


def smoke_test_dense_scaled_preprocessor(X: pd.DataFrame) -> None:
    """Verify dense MLP preprocessing returns finite numeric arrays."""
    transformed = make_dense_scaled_preprocessor().fit_transform(X)
    transformed_array = np.asarray(transformed, dtype=float)

    if transformed_array.ndim != 2:
        raise AssertionError("Dense MLP preprocessing did not return a 2D array.")
    if transformed_array.shape[0] != len(X):
        raise AssertionError("Dense MLP preprocessing changed the row count.")
    if transformed_array.shape[1] <= X.shape[1]:
        raise AssertionError(
            "Dense MLP preprocessing did not appear to retain one-hot encoded features."
        )
    if not np.isfinite(transformed_array).all():
        raise AssertionError("Dense MLP preprocessing returned non-finite values.")


def smoke_test_estimators(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify MLP factories can clone, fit, predict, score, and cross-validate."""
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    for name, estimator in make_smoke_estimators().items():
        print(f"Checking {name}...", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            cv_result = cross_validate(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                error_score="raise",
            )

        for metric_name in ("test_roc_auc", "test_pr_auc"):
            metric_values = np.asarray(cv_result[metric_name], dtype=float)
            if metric_values.shape != (2,) or not np.isfinite(metric_values).all():
                raise AssertionError(
                    f"{name} returned invalid {metric_name} cross-validation values."
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fitted = clone(estimator).fit(X, y)

        X_probe = X.head(10)
        predictions = fitted.predict(X_probe)
        probabilities = fitted.predict_proba(X_probe)

        if predictions.shape != (10,):
            raise AssertionError(f"{name} returned an unexpected prediction shape.")
        if probabilities.shape != (10, 2):
            raise AssertionError(f"{name} returned an unexpected probability shape.")
        if not np.isfinite(probabilities).all():
            raise AssertionError(f"{name} returned non-finite probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise AssertionError(f"{name} returned invalid probability rows.")

        classifier = fitted.named_steps["classifier"]
        if not hasattr(classifier, "coefs_") or not hasattr(classifier, "intercepts_"):
            raise AssertionError(f"{name} did not expose fitted dense-layer parameters.")
        if classifier.out_activation_ != "logistic":
            raise AssertionError(f"{name} did not use the expected binary logistic output.")
        if classifier.n_iter_ <= 0 or not getattr(classifier, "loss_curve_", []):
            raise AssertionError(f"{name} did not expose a valid optimization history.")
        if classifier.early_stopping and not hasattr(classifier, "validation_scores_"):
            raise AssertionError(
                f"{name} did not expose the expected internal validation history."
            )


def smoke_test_out_of_fold_probability_paths(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify OOF probability, threshold, curve, and calibration diagnostics."""
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    estimator = make_smoke_estimators()["shallow_relu"]

    print("Checking out-of-fold probability diagnostics...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        y_pred, y_score = get_out_of_fold_predictions(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )

    if y_pred.shape != (len(y),):
        raise AssertionError("MLP returned invalid OOF hard predictions.")
    if y_score is None or y_score.shape != (len(y),):
        raise AssertionError("MLP did not return valid OOF probabilities.")
    if not np.isfinite(y_score).all() or np.any((y_score < 0.0) | (y_score > 1.0)):
        raise AssertionError("MLP returned invalid OOF probability values.")

    threshold_df = evaluate_threshold_grid(
        y_true=y,
        y_score=y_score,
        thresholds=np.array([0.25, 0.50, 0.75]),
    )
    required_threshold_columns = {
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
    }
    if not required_threshold_columns.issubset(threshold_df.columns):
        raise AssertionError("MLP threshold table is missing required columns.")
    if threshold_df.shape[0] != 3:
        raise AssertionError("MLP threshold table has an unexpected row count.")

    roc_curve_df = make_roc_curve_dataframe(y_true=y, y_score=y_score)
    pr_curve_df = make_precision_recall_curve_dataframe(y_true=y, y_score=y_score)
    if roc_curve_df.empty or pr_curve_df.empty:
        raise AssertionError("MLP ranking-curve dataframes are unexpectedly empty.")

    observed_frequency, predicted_probability = calibration_curve(
        y,
        y_score,
        n_bins=5,
        strategy="quantile",
    )
    if (
        observed_frequency.size == 0
        or observed_frequency.shape != predicted_probability.shape
        or not np.isfinite(observed_frequency).all()
        or not np.isfinite(predicted_probability).all()
    ):
        raise AssertionError("MLP calibration diagnostic returned invalid values.")

    brier_score = brier_score_loss(y, y_score)
    if not np.isfinite(brier_score):
        raise AssertionError("MLP Brier score is not finite.")


def smoke_test_plots(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify shared ranking and threshold plotting helpers with MLP probabilities."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        fitted = make_smoke_estimators()["shallow_relu"].fit(X, y)

    y_score = fitted.predict_proba(X)[:, 1]
    roc_curve_df = make_roc_curve_dataframe(y_true=y, y_score=y_score)
    pr_curve_df = make_precision_recall_curve_dataframe(y_true=y, y_score=y_score)
    threshold_df = evaluate_threshold_grid(
        y_true=y,
        y_score=y_score,
        thresholds=np.array([0.25, 0.50, 0.75]),
    )

    with tempfile.TemporaryDirectory(prefix="mlp-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        roc_path = temp_path / "roc.png"
        pr_path = temp_path / "precision_recall.png"
        threshold_path = temp_path / "threshold_tradeoff.png"

        save_roc_curve_plot(
            roc_curve_df=roc_curve_df,
            output_path=roc_path,
            title="Smoke-test MLP ROC curve",
        )
        save_precision_recall_curve_plot(
            precision_recall_curve_df=pr_curve_df,
            output_path=pr_path,
            title="Smoke-test MLP precision-recall curve",
            positive_rate=float(y.mean()),
        )
        save_threshold_tradeoff_plot(
            threshold_df=threshold_df,
            output_path=threshold_path,
            title="Smoke-test MLP probability-threshold tradeoff",
            x_label="Predicted churn probability threshold",
            reference_threshold=0.50,
            reference_label="Default probability threshold = 0.50",
        )

        for output_path in (roc_path, pr_path, threshold_path):
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AssertionError(f"Plot was not written successfully: {output_path}")


def main() -> None:
    X, y = make_small_stratified_sample()
    smoke_test_dense_scaled_preprocessor(X)
    smoke_test_estimators(X, y)
    smoke_test_out_of_fold_probability_paths(X, y)
    smoke_test_plots(X, y)
    print("MLP workflow smoke test passed.")


if __name__ == "__main__":
    main()
