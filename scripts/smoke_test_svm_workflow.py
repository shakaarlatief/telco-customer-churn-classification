"""Fast compatibility smoke test for the Support Vector Machine workflow.

The test intentionally uses a small stratified subset of the project's training
data and tiny representative estimators. It validates the same reusable SVM
pipeline factories and scoring paths used by the main notebook before the full,
slower cross-validated grids are executed.

It is not a performance evaluation and does not read or use the held-out test
set.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
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
from telco_churn.models import (  # noqa: E402
    make_kernel_svc_pipeline,
    make_linear_svc_pipeline,
    make_rbf_svc_pipeline,
)
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
    """Create small representative estimators from the shared SVM factories."""
    return {
        "linear_svc_hinge": make_linear_svc_pipeline(
            C=1.0,
            loss="hinge",
            class_weight=None,
        ),
        "linear_svc_squared_hinge_balanced": make_linear_svc_pipeline(
            C=1.0,
            loss="squared_hinge",
            class_weight="balanced",
        ),
        "linear_kernel_svc": make_kernel_svc_pipeline(
            C=1.0,
            kernel="linear",
            gamma="scale",
            degree=3,
            coef0=0.0,
            class_weight=None,
        ),
        "polynomial_kernel_svc": make_kernel_svc_pipeline(
            C=1.0,
            kernel="poly",
            gamma="scale",
            degree=2,
            coef0=1.0,
            class_weight=None,
        ),
        "rbf_kernel_svc": make_rbf_svc_pipeline(
            C=1.0,
            gamma=0.1,
            class_weight="balanced",
        ),
    }


def smoke_test_estimators(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify all SVM factories can clone, fit, predict, score, and cross-validate."""
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    for name, estimator in make_smoke_estimators().items():
        print(f"Checking {name}...")

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

        fitted = clone(estimator).fit(X, y)
        X_probe = X.head(10)

        predictions = fitted.predict(X_probe)
        if predictions.shape != (10,):
            raise AssertionError(f"{name} returned an unexpected prediction shape.")

        decision_scores = fitted.decision_function(X_probe)
        if decision_scores.shape != (10,) or not np.isfinite(decision_scores).all():
            raise AssertionError(
                f"{name} returned invalid decision-function scores."
            )

        classifier = fitted.named_steps["classifier"]
        if name.startswith("linear_svc") and not hasattr(classifier, "coef_"):
            raise AssertionError(f"{name} did not expose expected linear coefficients.")

        if name.endswith("kernel_svc") and not hasattr(classifier, "n_support_"):
            raise AssertionError(f"{name} did not expose expected support-vector counts.")


def smoke_test_out_of_fold_score_paths(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify the project's generic OOF helper uses SVM decision scores correctly."""
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    estimators = {
        "linear_svc": make_linear_svc_pipeline(
            C=1.0,
            loss="squared_hinge",
            class_weight=None,
        ),
        "rbf_svc": make_rbf_svc_pipeline(
            C=1.0,
            gamma=0.1,
            class_weight=None,
        ),
    }

    for name, estimator in estimators.items():
        print(f"Checking out-of-fold decision scores for {name}...")
        y_pred, y_score = get_out_of_fold_predictions(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )

        if y_pred.shape != (len(y),):
            raise AssertionError(f"{name} returned invalid OOF hard predictions.")
        if y_score is None or y_score.shape != (len(y),):
            raise AssertionError(f"{name} did not return valid OOF decision scores.")
        if not np.isfinite(y_score).all():
            raise AssertionError(f"{name} returned non-finite OOF decision scores.")

        threshold_df = evaluate_threshold_grid(
            y_true=y,
            y_score=y_score,
            thresholds=np.array([-0.5, 0.0, 0.5]),
        )
        required_columns = {"threshold", "precision", "recall", "specificity", "f1"}
        if not required_columns.issubset(threshold_df.columns):
            raise AssertionError(f"{name} threshold table is missing required columns.")
        if threshold_df.shape[0] != 3:
            raise AssertionError(f"{name} threshold table has an unexpected row count.")


def smoke_test_plots(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify ROC, PR, and score-threshold plotting helpers with SVM score data."""
    score_source = make_linear_svc_pipeline(
        C=1.0,
        loss="squared_hinge",
        class_weight=None,
    ).fit(X, y)
    y_score = score_source.decision_function(X)

    roc_curve_df = make_roc_curve_dataframe(y_true=y, y_score=y_score)
    precision_recall_curve_df = make_precision_recall_curve_dataframe(
        y_true=y,
        y_score=y_score,
    )
    threshold_df = evaluate_threshold_grid(
        y_true=y,
        y_score=y_score,
        thresholds=np.array([-0.5, 0.0, 0.5]),
    )

    with tempfile.TemporaryDirectory(prefix="svm-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        roc_path = temp_path / "roc.png"
        pr_path = temp_path / "precision_recall.png"
        threshold_path = temp_path / "threshold_tradeoff.png"

        save_roc_curve_plot(
            roc_curve_df=roc_curve_df,
            output_path=roc_path,
            title="Smoke-test SVM ROC curve",
        )
        save_precision_recall_curve_plot(
            precision_recall_curve_df=precision_recall_curve_df,
            output_path=pr_path,
            title="Smoke-test SVM precision-recall curve",
            positive_rate=float(y.mean()),
        )
        save_threshold_tradeoff_plot(
            threshold_df=threshold_df,
            output_path=threshold_path,
            title="Smoke-test SVM score-threshold tradeoff",
            x_label="Decision-function score threshold",
            reference_threshold=0.0,
            reference_label="Natural SVM boundary (score = 0)",
        )

        for output_path in (roc_path, pr_path, threshold_path):
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AssertionError(f"Plot was not written successfully: {output_path}")


def main() -> None:
    X, y = make_small_stratified_sample()
    smoke_test_estimators(X, y)
    smoke_test_out_of_fold_score_paths(X, y)
    smoke_test_plots(X, y)
    print("SVM workflow smoke test passed.")


if __name__ == "__main__":
    main()
