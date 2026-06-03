"""Fast compatibility smoke test for the boosting workflow."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.models import (  # noqa: E402
    make_adaboost_pipeline,
    make_bagging_pipeline,
    make_catboost_pipeline,
    make_decision_tree_pipeline,
    make_gradient_boosting_pipeline,
    make_hist_gradient_boosting_pipeline,
    make_lightgbm_pipeline,
    make_random_forest_pipeline,
    make_xgboost_pipeline,
)
from telco_churn.visualization import (  # noqa: E402
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
)


SAMPLE_SIZE = 400


def make_small_stratified_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Load train.csv and return a small stratified modelling sample."""
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
    """Create tiny estimators from each relevant project factory."""
    return {
        "decision_tree": make_decision_tree_pipeline(
            max_depth=2,
            min_samples_leaf=5,
        ),
        "bagging": make_bagging_pipeline(
            n_estimators=2,
            max_samples=0.7,
            base_max_depth=2,
            base_min_samples_leaf=5,
        ),
        "random_forest": make_random_forest_pipeline(
            n_estimators=2,
            max_depth=3,
            min_samples_leaf=5,
            max_features="sqrt",
        ),
        "adaboost": make_adaboost_pipeline(
            base_depth=1,
            n_estimators=2,
            learning_rate=0.1,
        ),
        "gradient_boosting": make_gradient_boosting_pipeline(
            n_estimators=2,
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=5,
            subsample=1.0,
        ),
        "hist_gradient_boosting": make_hist_gradient_boosting_pipeline(
            max_iter=2,
            learning_rate=0.1,
            max_leaf_nodes=7,
            min_samples_leaf=5,
            l2_regularization=0.0,
        ),
        "xgboost": make_xgboost_pipeline(
            n_estimators=2,
            learning_rate=0.1,
            max_depth=2,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
        ),
        "lightgbm": make_lightgbm_pipeline(
            n_estimators=2,
            learning_rate=0.1,
            num_leaves=7,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
        ),
        "catboost": make_catboost_pipeline(
            iterations=2,
            learning_rate=0.1,
            depth=2,
            l2_leaf_reg=3.0,
        ),
    }


def smoke_test_estimators(X: pd.DataFrame, y: pd.Series) -> None:
    """Verify tiny boosting estimators can clone, fit, predict, and score."""
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator in make_smoke_estimators().items():
        print(f"Checking {name}...")
        cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            error_score="raise",
        )

        fitted = clone(estimator).fit(X, y)
        predictions = fitted.predict(X.head(10))
        if len(predictions) != 10:
            raise AssertionError(f"{name} returned an unexpected prediction length.")

        if hasattr(fitted, "predict_proba"):
            probabilities = fitted.predict_proba(X.head(10))
            if probabilities.shape[0] != 10 or probabilities.shape[1] != 2:
                raise AssertionError(f"{name} returned an unexpected probability shape.")

    catboost_probabilities = cross_val_predict(
        make_smoke_estimators()["catboost"],
        X,
        y,
        cv=cv,
        method="predict_proba",
    )
    if catboost_probabilities.shape != (len(y), 2):
        raise AssertionError("CatBoost cross-validation predict_proba shape is invalid.")


def smoke_test_plots(y: pd.Series) -> None:
    """Verify ROC and PR plotting helpers accept the notebook keyword names."""
    roc_curve_df = pd.DataFrame(
        {
            "false_positive_rate": [0.0, 0.3, 1.0],
            "true_positive_rate": [0.0, 0.8, 1.0],
            "threshold": [float("inf"), 0.5, 0.0],
        }
    )
    precision_recall_curve_df = pd.DataFrame(
        {
            "precision": [float(y.mean()), 0.7, 1.0],
            "recall": [1.0, 0.6, 0.0],
            "threshold": [0.0, 0.5, 1.0],
        }
    )

    with tempfile.TemporaryDirectory(prefix="boosting-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        save_roc_curve_plot(
            roc_curve_df=roc_curve_df,
            output_path=temp_path / "roc.png",
            title="Smoke-test ROC curve",
        )
        save_precision_recall_curve_plot(
            precision_recall_curve_df=precision_recall_curve_df,
            output_path=temp_path / "precision_recall.png",
            title="Smoke-test precision-recall curve",
            positive_rate=float(y.mean()),
        )


def main() -> None:
    X, y = make_small_stratified_sample()
    smoke_test_estimators(X, y)
    smoke_test_plots(y)
    print("Boosting workflow smoke test passed.")


if __name__ == "__main__":
    main()
