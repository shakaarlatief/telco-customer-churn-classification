# %% [markdown]
# # 10 Boosting, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost
#
# Version note: reusable boosting model factories and native-categorical
# preprocessing live in src/telco_churn so the same implementations can be reused
# in the later final model-comparison stage.
#
# ## Purpose
#
# This notebook evaluates boosting models for the Telco Customer Churn project.
# It follows the tree-ensemble section but changes the ensemble principle. Bagging
# and random forests train many trees independently and average them. Boosting
# trains learners sequentially: each new learner is added after the current
# ensemble has revealed mistakes, residuals, or gradient signals.
#
# The deeper reusable theory is documented in:
#
# ```text
# docs/knowledge_notes/models/10_boosting.md
# docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
# ```
#
# The executable section evaluates both classical and modern boosting models:
#
# ```text
# AdaBoostClassifier
# GradientBoostingClassifier
# HistGradientBoostingClassifier
# XGBoost XGBClassifier
# LightGBM LGBMClassifier with native categorical handling
# CatBoost CatBoostClassifier with native categorical handling
# ```

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used in this notebook.
#
# All model-development results are computed from stratified cross-validation
# inside the training set. The selected configurations should therefore be read as
# development-stage candidates selected within the tried grids, not as final
# test-set performance claims.
#
# This section deliberately uses transparent fixed grids. Other tuning strategies,
# such as random search, Optuna-style optimization, successive halving, fold-internal
# early stopping, repeated CV, or nested CV, are valid and may be useful later. The
# goal here is model-family learning and controlled development, not final
# procedure-level model comparison.
#
# Boosting methods can be very strong, and several package implementations may end
# up very close. Small cross-validated differences should not be interpreted as
# statistical proof that one package is truly superior. The final model-family
# comparison remains a later training-only stage, and the final held-out test set
# remains untouched until the end.

# %% [markdown]
# ## Optional package requirements
#
# The scikit-learn models use the existing project environment. The modern GBDT
# part also requires `xgboost`, `lightgbm`, and `catboost`.
#
# Install them in the project environment before executing the full notebook, for
# example:
#
# ```bash
# pip install xgboost lightgbm catboost
# ```

# %%
from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## Import project utilities

# %%
def find_project_root(start: Path | None = None) -> Path:
    """Return the project root by searching upward for project marker files."""
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        has_project_markers = (
            (candidate / "pyproject.toml").exists()
            or (candidate / "README.md").exists()
        )
        has_project_dirs = (
            (candidate / "data").exists()
            and (candidate / "notebooks").exists()
            and (candidate / "src").exists()
        )

        if has_project_markers and has_project_dirs:
            return candidate

    raise FileNotFoundError("Could not locate the project root directory.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# %%
from telco_churn.config import (  # noqa: E402
    ALL_FEATURES,
    FIGURES_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COLUMN,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    compute_binary_classification_metrics,
    evaluate_estimator_cv,
    evaluate_threshold_grid,
    get_out_of_fold_predictions,
    make_confusion_matrix_dataframe,
    make_precision_recall_curve_dataframe,
    make_roc_curve_dataframe,
    make_stratified_kfold,
)
from telco_churn.models import (  # noqa: E402
    make_adaboost_pipeline,
    make_bagging_pipeline,
    make_catboost_pipeline,
    make_decision_tree_pipeline,
    make_gradient_boosting_pipeline,
    make_hist_gradient_boosting_pipeline,
    make_l2_logistic_regression_pipeline,
    make_lightgbm_pipeline,
    make_random_forest_pipeline,
    make_xgboost_pipeline,
)
from telco_churn.visualization import (  # noqa: E402
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
    save_threshold_tradeoff_plot,
)

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", "{:,.4f}".format)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Output paths

# %%
BOOSTING_MODEL_COMPARISON_PATH = TABLES_DIR / "boosting_model_comparison.csv"
BOOSTING_CONFUSION_MATRIX_PATH = TABLES_DIR / "boosting_confusion_matrices.csv"
BOOSTING_CANDIDATE_RESULTS_PATH = TABLES_DIR / "boosting_candidate_results.csv"
BOOSTING_SELECTION_SUMMARY_PATH = TABLES_DIR / "boosting_selection_summary.csv"
BOOSTING_THRESHOLD_RESULTS_PATH = TABLES_DIR / "boosting_threshold_results.csv"
BOOSTING_FEATURE_IMPORTANCE_PATH = TABLES_DIR / "boosting_feature_importance.csv"

ADABOOST_GRID_RESULTS_PATH = TABLES_DIR / "adaboost_grid_results.csv"
GRADIENT_BOOSTING_GRID_RESULTS_PATH = TABLES_DIR / "gradient_boosting_grid_results.csv"
HIST_GRADIENT_BOOSTING_GRID_RESULTS_PATH = TABLES_DIR / "hist_gradient_boosting_grid_results.csv"
XGBOOST_GRID_RESULTS_PATH = TABLES_DIR / "xgboost_grid_results.csv"
LIGHTGBM_GRID_RESULTS_PATH = TABLES_DIR / "lightgbm_grid_results.csv"
CATBOOST_GRID_RESULTS_PATH = TABLES_DIR / "catboost_grid_results.csv"

ADABOOST_GRID_FIGURE_PATH = FIGURES_DIR / "adaboost_pr_auc_grid.png"
GRADIENT_BOOSTING_GRID_FIGURE_PATH = FIGURES_DIR / "gradient_boosting_pr_auc_grid.png"
HIST_GRADIENT_BOOSTING_GRID_FIGURE_PATH = FIGURES_DIR / "hist_gradient_boosting_pr_auc_grid.png"
XGBOOST_GRID_FIGURE_PATH = FIGURES_DIR / "xgboost_pr_auc_grid.png"
LIGHTGBM_GRID_FIGURE_PATH = FIGURES_DIR / "lightgbm_pr_auc_grid.png"
CATBOOST_GRID_FIGURE_PATH = FIGURES_DIR / "catboost_pr_auc_grid.png"
BOOSTING_MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "boosting_model_comparison_pr_auc.png"
BOOSTING_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "boosting_threshold_tradeoff.png"
BOOSTING_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "boosting_roc_curve.png"
BOOSTING_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "boosting_precision_recall_curve.png"
BOOSTING_FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "boosting_feature_importance.png"
BOOSTING_EXECUTION_LOG_PATH = LOGS_DIR / "10_boosting_execution.log"
BOOSTING_EXECUTION_LOG_PATH.write_text("", encoding="utf-8")

output_paths = pd.DataFrame(
    {
        "artifact": [
            "boosting_model_comparison",
            "boosting_confusion_matrices",
            "boosting_candidate_results",
            "adaboost_grid_results",
            "gradient_boosting_grid_results",
            "hist_gradient_boosting_grid_results",
            "xgboost_grid_results",
            "lightgbm_grid_results",
            "catboost_grid_results",
            "boosting_selection_summary",
            "boosting_threshold_results",
            "boosting_feature_importance",
            "adaboost_grid_figure",
            "gradient_boosting_grid_figure",
            "hist_gradient_boosting_grid_figure",
            "xgboost_grid_figure",
            "lightgbm_grid_figure",
            "catboost_grid_figure",
            "boosting_model_comparison_figure",
            "boosting_threshold_figure",
            "boosting_roc_curve_figure",
            "boosting_precision_recall_curve_figure",
            "boosting_feature_importance_figure",
        ],
        "path": [
            BOOSTING_MODEL_COMPARISON_PATH,
            BOOSTING_CONFUSION_MATRIX_PATH,
            BOOSTING_CANDIDATE_RESULTS_PATH,
            ADABOOST_GRID_RESULTS_PATH,
            GRADIENT_BOOSTING_GRID_RESULTS_PATH,
            HIST_GRADIENT_BOOSTING_GRID_RESULTS_PATH,
            XGBOOST_GRID_RESULTS_PATH,
            LIGHTGBM_GRID_RESULTS_PATH,
            CATBOOST_GRID_RESULTS_PATH,
            BOOSTING_SELECTION_SUMMARY_PATH,
            BOOSTING_THRESHOLD_RESULTS_PATH,
            BOOSTING_FEATURE_IMPORTANCE_PATH,
            ADABOOST_GRID_FIGURE_PATH,
            GRADIENT_BOOSTING_GRID_FIGURE_PATH,
            HIST_GRADIENT_BOOSTING_GRID_FIGURE_PATH,
            XGBOOST_GRID_FIGURE_PATH,
            LIGHTGBM_GRID_FIGURE_PATH,
            CATBOOST_GRID_FIGURE_PATH,
            BOOSTING_MODEL_COMPARISON_FIGURE_PATH,
            BOOSTING_THRESHOLD_FIGURE_PATH,
            BOOSTING_ROC_CURVE_FIGURE_PATH,
            BOOSTING_PRECISION_RECALL_CURVE_FIGURE_PATH,
            BOOSTING_FEATURE_IMPORTANCE_FIGURE_PATH,
        ],
    }
)

output_paths

# %% [markdown]
# ## Load training data only

# %%
train_df = load_train_data()
X, y = split_features_target(train_df)

training_overview = pd.DataFrame(
    {
        "item": [
            "training_rows",
            "training_columns",
            "target_column",
            "positive_rate",
            "missing_values",
        ],
        "value": [
            train_df.shape[0],
            train_df.shape[1],
            TARGET_COLUMN,
            y.mean(),
            int(train_df.isna().sum().sum()),
        ],
    }
)

training_overview

# %% [markdown]
# ## Reusable preprocessing and model factories
#
# The boosting section uses two preprocessing branches. The one-hot branch is used
# for scikit-learn AdaBoost, scikit-learn gradient boosting, scikit-learn
# histogram gradient boosting, and XGBoost. It keeps the feature representation
# comparable to earlier model sections. The native-categorical branch is used for
# LightGBM and CatBoost, preserving the original categorical columns so these
# libraries can use their intended categorical handling.
#
# These reusable preprocessing and model factories now live in
# `src/telco_churn/preprocessing.py` and `src/telco_churn/models.py` rather than in
# this notebook. This keeps the notebook focused on the section-specific grids,
# artifacts, and interpretation, while making the same implementations reusable
# for the later final model-comparison stage.

# %% [markdown]
# ## Evaluation helpers

# %%
cv = make_stratified_kfold()

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for compact notebook progress logs."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"


def log_progress(message: str) -> None:
    """Print a timestamped progress message that appears during nbconvert runs."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with BOOSTING_EXECUTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{line}\n")
        log_file.flush()


def log_section_start(section_name: str) -> float:
    """Log the start of a notebook section and return its timer."""
    log_progress(f"Starting {section_name}")
    return time.perf_counter()


def log_section_end(section_name: str, start_time: float) -> None:
    """Log the end of a notebook section with elapsed time."""
    elapsed = time.perf_counter() - start_time
    log_progress(f"Finished {section_name} in {format_elapsed(elapsed)}")


def package_status(package_name: str) -> str:
    """Return a compact availability label for an optional package."""
    if package_name in sys.modules:
        return "imported"
    if importlib.util.find_spec(package_name) is not None:
        return "available"
    return "not available"


def log_preflight_summary() -> None:
    """Print a short runtime preflight summary before expensive grids."""
    log_progress("Boosting notebook preflight")
    log_progress(f"Training data shape: rows={X.shape[0]}, columns={X.shape[1]}")
    log_progress(f"CV folds: {cv.get_n_splits(X, y)}")
    log_progress(
        "Modern boosting packages: "
        f"xgboost={package_status('xgboost')}, "
        f"lightgbm={package_status('lightgbm')}, "
        f"catboost={package_status('catboost')}"
    )
    log_progress("Using train.csv only for all development-stage results.")
    log_progress("Held-out test set is not touched in this notebook.")


log_preflight_summary()


def evaluate_grid_candidate_cv(
    *,
    model_name: str,
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    extra_columns: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate a grid candidate with fold-level cross-validation scores."""
    cv_result = cross_validate(
        clone(estimator),
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=None,
        error_score="raise",
    )

    row: dict[str, object] = {"model": model_name}
    if extra_columns:
        row.update(extra_columns)

    for key, values in cv_result.items():
        if key.startswith("train_") or key.startswith("test_"):
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))

    row["fit_time_mean"] = float(np.mean(cv_result["fit_time"]))
    row["score_time_mean"] = float(np.mean(cv_result["score_time"]))
    return row


def make_metric_columns() -> list[str]:
    """Return the standard metric columns used for display."""
    return [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "predicted_positive_rate",
        "observed_positive_rate",
    ]


def make_grid_metric_columns() -> list[str]:
    """Return compact grid-search metric columns for display."""
    return [
        "model",
        "test_pr_auc_mean",
        "test_roc_auc_mean",
        "test_balanced_accuracy_mean",
        "test_f1_mean",
        "test_recall_mean",
        "test_precision_mean",
        "train_pr_auc_mean",
        "train_balanced_accuracy_mean",
        "fit_time_mean",
    ]


def sort_grid_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Sort grid results by the project selection priorities."""
    return results_df.sort_values(
        by=["test_pr_auc_mean", "test_balanced_accuracy_mean", "test_f1_mean"],
        ascending=False,
    ).reset_index(drop=True)


def evaluate_parameter_grid(
    *,
    family_name: str,
    parameter_grid: dict[str, list[object]],
    estimator_factory,
    model_name_factory,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> pd.DataFrame:
    """Evaluate a transparent fixed grid for one boosting implementation."""
    rows: list[dict[str, object]] = []
    candidates = list(ParameterGrid(parameter_grid))
    family_start = time.perf_counter()

    log_progress(f"Starting {family_name} grid: {len(candidates)} candidates")

    for candidate_index, params in enumerate(candidates, start=1):
        model_name = model_name_factory(params)
        estimator = estimator_factory(**params)
        candidate_start = time.perf_counter()
        rows.append(
            evaluate_grid_candidate_cv(
                model_name=model_name,
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
                extra_columns={"family": family_name, **params},
            )
        )
        candidate_elapsed = time.perf_counter() - candidate_start
        cumulative_elapsed = time.perf_counter() - family_start
        log_progress(
            f"{family_name} grid candidate {candidate_index}/{len(candidates)} "
            f"completed in {format_elapsed(candidate_elapsed)} "
            f"(cumulative {format_elapsed(cumulative_elapsed)}): {model_name}"
        )

    log_section_end(f"{family_name} grid", family_start)
    return sort_grid_results(pd.DataFrame(rows))


def save_top_grid_candidates_plot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 12,
) -> None:
    """Save a compact horizontal bar plot for top grid candidates by PR-AUC."""
    plot_df = (
        results_df.sort_values("test_pr_auc_mean", ascending=False)
        .head(top_n)
        .sort_values("test_pr_auc_mean", ascending=True)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["model"], plot_df["test_pr_auc_mean"])
    ax.set_xlabel("Mean cross-validated PR-AUC")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Save a PR-AUC comparison plot for selected candidates."""
    plot_df = results_df.sort_values("pr_auc", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["model"], plot_df["pr_auc"])
    ax.set_xlabel("Pooled out-of-fold PR-AUC")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(
    *,
    feature_importance_df: pd.DataFrame,
    selected_model: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar plot for the selected model's feature importances."""
    plot_df = feature_importance_df[
        feature_importance_df["model"] == selected_model
    ].copy()

    plot_df = (
        plot_df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top feature importances: {selected_model}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# ## Reference models from previous sections
#
# The boosting models are compared against strong earlier candidates. These
# reference models are not re-tuned here. They are reconstructed from earlier
# section choices and evaluated with the same pooled out-of-fold prediction logic.

# %%
selected_logistic_pipeline = make_l2_logistic_regression_pipeline(C=1.0)

selected_single_tree_pipeline = make_decision_tree_pipeline(
    criterion="gini",
    max_depth=6,
    min_samples_split=25,
    min_samples_leaf=10,
    ccp_alpha=0.0,
)

selected_bagging_pipeline = make_bagging_pipeline(
    n_estimators=200,
    max_samples=0.8,
    base_max_depth=6,
    base_min_samples_leaf=1,
)

selected_random_forest_pipeline = make_random_forest_pipeline(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=10,
    max_features="sqrt",
)

reference_estimators = [
    ("Selected L2 logistic regression", selected_logistic_pipeline),
    ("Selected single decision tree", selected_single_tree_pipeline),
    ("Selected bagged trees", selected_bagging_pipeline),
    ("Selected random forest", selected_random_forest_pipeline),
]

reference_start = log_section_start("reference model evaluation")
reference_results_df = pd.DataFrame(
    [
        evaluate_estimator_cv(model_name=name, estimator=estimator, X=X, y=y, cv=cv)
        for name, estimator in reference_estimators
    ]
).sort_values(by=["pr_auc", "balanced_accuracy", "f1"], ascending=False)
log_section_end("reference model evaluation", reference_start)

reference_results_df[make_metric_columns()]

# %% [markdown]
# ## AdaBoost grid
#
# AdaBoost is the classical sequential reweighting model. The grid compares
# stumps and shallow trees, several learning rates, and several numbers of
# boosting rounds. The primary selection metric is PR-AUC.

# %%
adaboost_grid = {
    "base_depth": [1, 2],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.03, 0.1, 0.3, 1.0],
}

adaboost_grid_results_df = evaluate_parameter_grid(
    family_name="AdaBoost",
    parameter_grid=adaboost_grid,
    estimator_factory=make_adaboost_pipeline,
    model_name_factory=lambda p: (
        "AdaBoost "
        f"depth={p['base_depth']} "
        f"trees={p['n_estimators']} "
        f"lr={p['learning_rate']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

adaboost_grid_results_df.to_csv(ADABOOST_GRID_RESULTS_PATH, index=False)
save_top_grid_candidates_plot(
    results_df=adaboost_grid_results_df,
    output_path=ADABOOST_GRID_FIGURE_PATH,
    title="AdaBoost grid: top PR-AUC candidates",
)

adaboost_grid_results_df.head(12)[
    [
        "model",
        "base_depth",
        "n_estimators",
        "learning_rate",
        *make_grid_metric_columns()[1:],
    ]
]

# %% [markdown]
# ## GradientBoostingClassifier grid
#
# `GradientBoostingClassifier` is the direct scikit-learn implementation of
# stagewise additive gradient boosting. The grid varies shrinkage, number of
# stages, tree depth, leaf-size regularization, and stochastic subsampling.

# %%
gradient_boosting_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3],
    "min_samples_leaf": [10, 25],
    "subsample": [0.8, 1.0],
}

gradient_boosting_grid_results_df = evaluate_parameter_grid(
    family_name="GradientBoostingClassifier",
    parameter_grid=gradient_boosting_grid,
    estimator_factory=make_gradient_boosting_pipeline,
    model_name_factory=lambda p: (
        "GradientBoosting "
        f"trees={p['n_estimators']} "
        f"lr={p['learning_rate']} "
        f"depth={p['max_depth']} "
        f"leaf={p['min_samples_leaf']} "
        f"subsample={p['subsample']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

gradient_boosting_grid_results_df.to_csv(
    GRADIENT_BOOSTING_GRID_RESULTS_PATH, index=False
)
save_top_grid_candidates_plot(
    results_df=gradient_boosting_grid_results_df,
    output_path=GRADIENT_BOOSTING_GRID_FIGURE_PATH,
    title="GradientBoostingClassifier grid: top PR-AUC candidates",
)

gradient_boosting_grid_results_df.head(12)[
    [
        "model",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        *make_grid_metric_columns()[1:],
    ]
]

# %% [markdown]
# ## HistGradientBoostingClassifier grid
#
# Histogram gradient boosting uses binned split search and explicit L2-style leaf
# regularization. The grid keeps early stopping disabled so that the number of
# iterations remains a transparent hyperparameter evaluated by the same outer CV.

# %%
hist_gradient_boosting_grid = {
    "max_iter": [100, 200],
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [10, 25],
    "l2_regularization": [0.0, 1.0],
}

hist_gradient_boosting_grid_results_df = evaluate_parameter_grid(
    family_name="HistGradientBoostingClassifier",
    parameter_grid=hist_gradient_boosting_grid,
    estimator_factory=make_hist_gradient_boosting_pipeline,
    model_name_factory=lambda p: (
        "HistGradientBoosting "
        f"iter={p['max_iter']} "
        f"lr={p['learning_rate']} "
        f"leaves={p['max_leaf_nodes']} "
        f"min_leaf={p['min_samples_leaf']} "
        f"l2={p['l2_regularization']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

hist_gradient_boosting_grid_results_df.to_csv(
    HIST_GRADIENT_BOOSTING_GRID_RESULTS_PATH, index=False
)
save_top_grid_candidates_plot(
    results_df=hist_gradient_boosting_grid_results_df,
    output_path=HIST_GRADIENT_BOOSTING_GRID_FIGURE_PATH,
    title="HistGradientBoostingClassifier grid: top PR-AUC candidates",
)

hist_gradient_boosting_grid_results_df.head(12)[
    [
        "model",
        "max_iter",
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        *make_grid_metric_columns()[1:],
    ]
]

# %% [markdown]
# ## XGBoost grid
#
# XGBoost is evaluated on the dense one-hot representation. This keeps the feature
# representation comparable to the scikit-learn boosting models while adding
# regularized second-order boosted trees.

# %%
xgboost_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3, 4],
    "min_child_weight": [1, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1.0],
}

xgboost_grid_results_df = evaluate_parameter_grid(
    family_name="XGBoost",
    parameter_grid=xgboost_grid,
    estimator_factory=make_xgboost_pipeline,
    model_name_factory=lambda p: (
        "XGBoost "
        f"trees={p['n_estimators']} "
        f"lr={p['learning_rate']} "
        f"depth={p['max_depth']} "
        f"child={p['min_child_weight']} "
        f"sub={p['subsample']} "
        f"col={p['colsample_bytree']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

xgboost_grid_results_df.to_csv(XGBOOST_GRID_RESULTS_PATH, index=False)
save_top_grid_candidates_plot(
    results_df=xgboost_grid_results_df,
    output_path=XGBOOST_GRID_FIGURE_PATH,
    title="XGBoost grid: top PR-AUC candidates",
)

xgboost_grid_results_df.head(12)[
    [
        "model",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        *make_grid_metric_columns()[1:],
    ]
]

# %% [markdown]
# ## LightGBM grid with native categorical features
#
# LightGBM is evaluated with a native-categorical preprocessing branch. This lets
# LightGBM use categorical split handling rather than only one-hot indicators.

# %%
lightgbm_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.1],
    "num_leaves": [15, 31],
    "min_child_samples": [10, 25],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [0.0, 1.0],
}

lightgbm_grid_results_df = evaluate_parameter_grid(
    family_name="LightGBM native categorical",
    parameter_grid=lightgbm_grid,
    estimator_factory=make_lightgbm_pipeline,
    model_name_factory=lambda p: (
        "LightGBM native "
        f"trees={p['n_estimators']} "
        f"lr={p['learning_rate']} "
        f"leaves={p['num_leaves']} "
        f"child={p['min_child_samples']} "
        f"sub={p['subsample']} "
        f"col={p['colsample_bytree']} "
        f"l2={p['reg_lambda']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

lightgbm_grid_results_df.to_csv(LIGHTGBM_GRID_RESULTS_PATH, index=False)
save_top_grid_candidates_plot(
    results_df=lightgbm_grid_results_df,
    output_path=LIGHTGBM_GRID_FIGURE_PATH,
    title="LightGBM native categorical grid: top PR-AUC candidates",
)

lightgbm_grid_results_df.head(12)[
    [
        "model",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        *make_grid_metric_columns()[1:],
    ]
]

# %% [markdown]
# ## CatBoost grid with native categorical features
#
# CatBoost is evaluated on the raw categorical representation. This uses the
# library's native categorical handling and ordered training logic.

# %%
catboost_grid = {
    "iterations": [100, 200],
    "learning_rate": [0.03, 0.1],
    "depth": [3, 4, 6],
    "l2_leaf_reg": [3, 10],
}

catboost_grid_results_df = evaluate_parameter_grid(
    family_name="CatBoost native categorical",
    parameter_grid=catboost_grid,
    estimator_factory=make_catboost_pipeline,
    model_name_factory=lambda p: (
        "CatBoost native "
        f"iter={p['iterations']} "
        f"lr={p['learning_rate']} "
        f"depth={p['depth']} "
        f"l2={p['l2_leaf_reg']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

catboost_grid_results_df.to_csv(CATBOOST_GRID_RESULTS_PATH, index=False)
save_top_grid_candidates_plot(
    results_df=catboost_grid_results_df,
    output_path=CATBOOST_GRID_FIGURE_PATH,
    title="CatBoost native categorical grid: top PR-AUC candidates",
)

catboost_grid_results_df.head(12)[
    [
        "model",
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        *make_grid_metric_columns()[1:],
    ]
]

# %% [markdown]
# ## Select representative boosting candidates
#
# The representative candidate from each boosting family is the highest-PR-AUC
# configuration within that family's development grid. Balanced accuracy and F1
# remain secondary diagnostics.

# %%
grid_tables = {
    "AdaBoost": adaboost_grid_results_df,
    "GradientBoostingClassifier": gradient_boosting_grid_results_df,
    "HistGradientBoostingClassifier": hist_gradient_boosting_grid_results_df,
    "XGBoost": xgboost_grid_results_df,
    "LightGBM native categorical": lightgbm_grid_results_df,
    "CatBoost native categorical": catboost_grid_results_df,
}

selection_summary_rows = []
for family, table in grid_tables.items():
    best = table.iloc[0].to_dict()
    selection_summary_rows.append(
        {
            "family": family,
            "model": best["model"],
            "test_pr_auc_mean": best["test_pr_auc_mean"],
            "test_roc_auc_mean": best["test_roc_auc_mean"],
            "test_balanced_accuracy_mean": best["test_balanced_accuracy_mean"],
            "test_f1_mean": best["test_f1_mean"],
            "train_pr_auc_mean": best["train_pr_auc_mean"],
            "fit_time_mean": best["fit_time_mean"],
        }
    )

selection_summary_df = pd.DataFrame(selection_summary_rows).sort_values(
    by=["test_pr_auc_mean", "test_balanced_accuracy_mean", "test_f1_mean"],
    ascending=False,
)
selection_summary_df.to_csv(BOOSTING_SELECTION_SUMMARY_PATH, index=False)
selection_summary_df

# %%
best_adaboost_row = adaboost_grid_results_df.iloc[0]
best_gradient_boosting_row = gradient_boosting_grid_results_df.iloc[0]
best_hist_gradient_boosting_row = hist_gradient_boosting_grid_results_df.iloc[0]
best_xgboost_row = xgboost_grid_results_df.iloc[0]
best_lightgbm_row = lightgbm_grid_results_df.iloc[0]
best_catboost_row = catboost_grid_results_df.iloc[0]

best_adaboost_pipeline = make_adaboost_pipeline(
    base_depth=int(best_adaboost_row["base_depth"]),
    n_estimators=int(best_adaboost_row["n_estimators"]),
    learning_rate=float(best_adaboost_row["learning_rate"]),
)

best_gradient_boosting_pipeline = make_gradient_boosting_pipeline(
    n_estimators=int(best_gradient_boosting_row["n_estimators"]),
    learning_rate=float(best_gradient_boosting_row["learning_rate"]),
    max_depth=int(best_gradient_boosting_row["max_depth"]),
    min_samples_leaf=int(best_gradient_boosting_row["min_samples_leaf"]),
    subsample=float(best_gradient_boosting_row["subsample"]),
)

best_hist_gradient_boosting_pipeline = make_hist_gradient_boosting_pipeline(
    max_iter=int(best_hist_gradient_boosting_row["max_iter"]),
    learning_rate=float(best_hist_gradient_boosting_row["learning_rate"]),
    max_leaf_nodes=int(best_hist_gradient_boosting_row["max_leaf_nodes"]),
    min_samples_leaf=int(best_hist_gradient_boosting_row["min_samples_leaf"]),
    l2_regularization=float(best_hist_gradient_boosting_row["l2_regularization"]),
)

best_xgboost_pipeline = make_xgboost_pipeline(
    n_estimators=int(best_xgboost_row["n_estimators"]),
    learning_rate=float(best_xgboost_row["learning_rate"]),
    max_depth=int(best_xgboost_row["max_depth"]),
    min_child_weight=float(best_xgboost_row["min_child_weight"]),
    subsample=float(best_xgboost_row["subsample"]),
    colsample_bytree=float(best_xgboost_row["colsample_bytree"]),
    reg_lambda=float(best_xgboost_row["reg_lambda"]),
)

best_lightgbm_pipeline = make_lightgbm_pipeline(
    n_estimators=int(best_lightgbm_row["n_estimators"]),
    learning_rate=float(best_lightgbm_row["learning_rate"]),
    num_leaves=int(best_lightgbm_row["num_leaves"]),
    min_child_samples=int(best_lightgbm_row["min_child_samples"]),
    subsample=float(best_lightgbm_row["subsample"]),
    colsample_bytree=float(best_lightgbm_row["colsample_bytree"]),
    reg_lambda=float(best_lightgbm_row["reg_lambda"]),
)

best_catboost_pipeline = make_catboost_pipeline(
    iterations=int(best_catboost_row["iterations"]),
    learning_rate=float(best_catboost_row["learning_rate"]),
    depth=int(best_catboost_row["depth"]),
    l2_leaf_reg=float(best_catboost_row["l2_leaf_reg"]),
)

# %% [markdown]
# ## Candidate comparison using pooled out-of-fold predictions
#
# The grid tables select one representative candidate per boosting family. These
# candidates are then evaluated with pooled out-of-fold predictions so confusion
# matrices, threshold curves, ROC curves, and precision-recall curves are computed
# consistently with earlier notebook sections.

# %%
boosting_candidate_estimators = [
    *reference_estimators,
    ("Best AdaBoost", best_adaboost_pipeline),
    ("Best GradientBoostingClassifier", best_gradient_boosting_pipeline),
    ("Best HistGradientBoostingClassifier", best_hist_gradient_boosting_pipeline),
    ("Best XGBoost", best_xgboost_pipeline),
    ("Best LightGBM native categorical", best_lightgbm_pipeline),
    ("Best CatBoost native categorical", best_catboost_pipeline),
]

candidate_comparison_start = log_section_start(
    "selected candidate out-of-fold comparison"
)
boosting_candidate_results = [
    evaluate_estimator_cv(model_name=name, estimator=estimator, X=X, y=y, cv=cv)
    for name, estimator in boosting_candidate_estimators
]

boosting_candidate_results_df = pd.DataFrame(boosting_candidate_results)
boosting_model_comparison_df = boosting_candidate_results_df.sort_values(
    by=["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).reset_index(drop=True)

boosting_confusion_matrix_df = (
    make_confusion_matrix_dataframe(boosting_model_comparison_df)
    .set_index("model")
    .loc[boosting_model_comparison_df["model"]]
    .reset_index()
)

boosting_model_comparison_df.to_csv(BOOSTING_MODEL_COMPARISON_PATH, index=False)
boosting_candidate_results_df.to_csv(BOOSTING_CANDIDATE_RESULTS_PATH, index=False)
boosting_confusion_matrix_df.to_csv(BOOSTING_CONFUSION_MATRIX_PATH, index=False)

save_model_comparison_plot(
    results_df=boosting_model_comparison_df,
    output_path=BOOSTING_MODEL_COMPARISON_FIGURE_PATH,
    title="Boosting and reference models: pooled out-of-fold PR-AUC",
)
log_section_end(
    "selected candidate out-of-fold comparison",
    candidate_comparison_start,
)

boosting_model_comparison_df[make_metric_columns()]

# %%
boosting_confusion_matrix_df

# %% [markdown]
# The pooled out-of-fold comparison shows that boosted tree models form the
# strongest group in this development-stage section. The highest pooled
# out-of-fold PR-AUC is obtained by XGBoost:
#
# ```text
# Best XGBoost:
#     PR-AUC = 0.6701
#     ROC-AUC = 0.8498
#     balanced accuracy = 0.7218
#     F1 = 0.5979
# ```
#
# However, the leading boosting implementations are extremely close. The selected
# `GradientBoostingClassifier` reaches PR-AUC 0.6695, and the selected native
# categorical CatBoost model reaches PR-AUC 0.6694. These differences are much
# smaller than the uncertainty one would expect from a single training-set
# development workflow, so the result should not be read as proof that XGBoost is
# intrinsically better than the other leading boosted tree implementations.
#
# The family-selection grid gives the same caution from a different angle. In the
# fixed-grid table, native categorical CatBoost has the highest mean CV PR-AUC
# (0.6728), followed very closely by `GradientBoostingClassifier` (0.6724) and
# XGBoost (0.6724). In the pooled out-of-fold table, XGBoost ranks first. The
# safe conclusion is therefore that boosted tree models are competitive as a
# group, while the ordering among the best boosting implementations remains
# close.
#
# Relative to earlier reference models, boosting gives a modest development-stage
# improvement. The selected bagged trees, selected random forest, and selected
# L2 logistic regression have PR-AUC values of 0.6618, 0.6602, and 0.6584,
# respectively, while the selected single decision tree is lower at 0.6285. The
# improvement over the single tree is clear, but the improvement over the
# strongest previous ensemble references is incremental rather than dramatic.
# Final model-family selection is still deferred to the later training-only final
# comparison stage.

# %% [markdown]
# ## Select model for threshold and curve diagnostics
#
# The highest-PR-AUC candidate in the pooled out-of-fold comparison is selected as
# the representative boosting-section model for threshold and curve diagnostics.

# %%
estimator_lookup = dict(boosting_candidate_estimators)
selected_boosting_name = boosting_model_comparison_df.iloc[0]["model"]
selected_boosting_pipeline = estimator_lookup[selected_boosting_name]
selected_boosting_name

# %%
diagnostics_start = log_section_start("threshold and curve diagnostics")
y_pred_oof, y_score_oof = get_out_of_fold_predictions(
    estimator=selected_boosting_pipeline,
    X=X,
    y=y,
    cv=cv,
)

selected_metrics = compute_binary_classification_metrics(
    y_true=y,
    y_pred=y_pred_oof,
    y_score=y_score_oof,
)

pd.DataFrame(
    [
        {
            "selected_boosting_name": selected_boosting_name,
            "accuracy": selected_metrics.accuracy,
            "balanced_accuracy": selected_metrics.balanced_accuracy,
            "precision": selected_metrics.precision,
            "recall": selected_metrics.recall,
            "specificity": selected_metrics.specificity,
            "f1": selected_metrics.f1,
            "roc_auc": selected_metrics.roc_auc,
            "pr_auc": selected_metrics.pr_auc,
            "predicted_positive_rate": selected_metrics.predicted_positive_rate,
            "observed_positive_rate": selected_metrics.observed_positive_rate,
        }
    ]
)

# %% [markdown]
# ## Threshold behaviour

# %%
thresholds = np.array(
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
)

boosting_threshold_results_df = evaluate_threshold_grid(
    y_true=y,
    y_score=y_score_oof,
    thresholds=thresholds,
)
boosting_threshold_results_df.to_csv(BOOSTING_THRESHOLD_RESULTS_PATH, index=False)

save_threshold_tradeoff_plot(
    threshold_df=boosting_threshold_results_df,
    output_path=BOOSTING_THRESHOLD_FIGURE_PATH,
    title=f"Threshold tradeoff for {selected_boosting_name}",
)

boosting_threshold_results_df

# %% [markdown]
# The selected XGBoost model has the usual threshold tradeoff. At the default
# 0.50 threshold it is relatively conservative:
#
# ```text
# threshold = 0.50
# precision = 0.6711
# recall = 0.5391
# specificity = 0.9046
# F1 = 0.5979
# predicted positive rate = 0.2132
# observed positive rate = 0.2654
# ```
#
# The predicted positive rate is below the observed churn rate, which means the
# default threshold identifies a smaller, higher-risk subset of customers rather
# than attempting to recover most churners. Lower thresholds substantially
# increase recall. For example, threshold 0.25 gives recall 0.8241 and F1
# 0.6315, but precision falls to 0.5118. In the displayed threshold grid, the
# highest F1 occurs around threshold 0.30, with F1 0.6358, recall 0.7712, and
# precision 0.5408.
#
# This section does not choose a final operating threshold. Threshold choice
# depends on the business cost of false positives versus false negatives, and it
# should be handled after final model-family comparison and calibration analysis.

# %% [markdown]
# ## ROC and precision-recall curves

# %%
boosting_roc_curve_df = make_roc_curve_dataframe(y_true=y, y_score=y_score_oof)
boosting_precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=y_score_oof,
)

save_roc_curve_plot(
    roc_curve_df=boosting_roc_curve_df,
    output_path=BOOSTING_ROC_CURVE_FIGURE_PATH,
    title=f"ROC curve for {selected_boosting_name}",
)

save_precision_recall_curve_plot(
    precision_recall_curve_df=boosting_precision_recall_curve_df,
    output_path=BOOSTING_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title=f"Precision-recall curve for {selected_boosting_name}",
    positive_rate=float(y.mean()),
)
log_section_end("threshold and curve diagnostics", diagnostics_start)

boosting_roc_curve_df.head(), boosting_precision_recall_curve_df.head()

# %% [markdown]
# ## Feature importance
#
# Feature importance is computed by refitting selected boosting candidates on the
# full training set. This refit is for interpretation only. It is not used for
# performance reporting.

# %%
def get_one_hot_feature_names(fitted_pipeline: Pipeline) -> np.ndarray:
    """Return dense one-hot feature names from a fitted preprocessing pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def extract_feature_importance(
    *,
    model_name: str,
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Fit an estimator and return feature importances when available."""
    fitted = clone(estimator).fit(X, y)
    classifier = fitted.named_steps["classifier"]

    if model_name in {
        "Best LightGBM native categorical",
        "Best CatBoost native categorical",
    }:
        feature_names = np.array(ALL_FEATURES)
    else:
        feature_names = get_one_hot_feature_names(fitted)

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "get_feature_importance"):
        importances = np.asarray(classifier.get_feature_importance(), dtype=float)
    else:
        return pd.DataFrame(columns=["model", "feature", "importance"])

    return pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)


importance_models = [
    ("Best AdaBoost", best_adaboost_pipeline),
    ("Best GradientBoostingClassifier", best_gradient_boosting_pipeline),
    ("Best HistGradientBoostingClassifier", best_hist_gradient_boosting_pipeline),
    ("Best XGBoost", best_xgboost_pipeline),
    ("Best LightGBM native categorical", best_lightgbm_pipeline),
    ("Best CatBoost native categorical", best_catboost_pipeline),
]

feature_importance_start = log_section_start("feature importance extraction")
feature_importance_df = pd.concat(
    [
        extract_feature_importance(model_name=name, estimator=estimator, X=X, y=y)
        for name, estimator in importance_models
    ],
    ignore_index=True,
)

feature_importance_df.to_csv(BOOSTING_FEATURE_IMPORTANCE_PATH, index=False)

if selected_boosting_name in set(feature_importance_df["model"]):
    selected_importance_model = selected_boosting_name
else:
    selected_importance_model = feature_importance_df.groupby("model")["importance"].sum().idxmax()

save_feature_importance_plot(
    feature_importance_df=feature_importance_df,
    selected_model=selected_importance_model,
    output_path=BOOSTING_FEATURE_IMPORTANCE_FIGURE_PATH,
    top_n=20,
)
log_section_end("feature importance extraction", feature_importance_start)

feature_importance_df.groupby("model").head(12)

# %% [markdown]
# The selected XGBoost feature-importance profile is consistent with the rest of
# the project. The largest importance is attached to the one-hot indicator for
# month-to-month contracts. This is followed by service and support indicators
# such as no online security, fiber-optic internet service, electronic-check
# payment, and no tech support. Tenure also appears among the leading predictors.
#
# This pattern agrees with earlier EDA, logistic regression, decision-tree, and
# random-forest sections: churn risk is repeatedly associated with contract
# structure, tenure, internet-service type, support services, and payment method.
# The exact numerical importances should not be interpreted causally. They are
# model-usage diagnostics that describe how the fitted boosted-tree model uses the
# transformed features to split the training data.

# %% [markdown]
# ## Saved artifacts check

# %%
artifact_check = output_paths.copy()
artifact_check["exists"] = artifact_check["path"].apply(lambda path: Path(path).exists())
artifact_check

# %% [markdown]
# ## Section summary
#
# This section evaluates classical and modern boosting methods under the same
# training-set-only discipline used throughout the project. The best candidates
# from AdaBoost, `GradientBoostingClassifier`, `HistGradientBoostingClassifier`,
# XGBoost, LightGBM, and CatBoost are selected from transparent fixed grids using
# cross-validated PR-AUC as the primary ranking metric.
#
# The strongest fixed-grid family candidate is native categorical CatBoost with
# mean CV PR-AUC 0.6728. The selected `GradientBoostingClassifier` and XGBoost
# candidates are almost indistinguishable by the same criterion, with mean CV
# PR-AUC values of 0.6724 and 0.6724. In the pooled out-of-fold candidate
# comparison, XGBoost ranks first with PR-AUC 0.6701, followed closely by
# `GradientBoostingClassifier` at 0.6695 and native categorical CatBoost at
# 0.6694.
#
# The practical conclusion is that boosting is a strong model family for this
# dataset, but the best boosting implementations are very close in this
# development-stage workflow. Boosting improves clearly over the selected single
# tree and modestly over the strongest previous bagging, random-forest, and
# logistic-regression references. The evidence is not strong enough to claim that
# one modern GBDT package dominates the others.
#
# Native categorical LightGBM and CatBoost run successfully and produce competitive
# results. CatBoost is especially close to the best one-hot XGBoost and
# scikit-learn gradient-boosting candidates, while LightGBM is slightly lower in
# this fixed development grid but still competitive. These native-categorical
# workflows are useful because they preserve the original categorical feature
# representation and can be reused in the later final model-comparison stage.
#
# The train-validation gaps suggest that some boosted models can become more
# flexible than the earlier baselines, especially histogram boosting and larger
# tree ensembles. The grids therefore remain intentionally controlled. More
# extensive random search, Bayesian optimization, early stopping, repeated CV,
# calibration checks, and formal uncertainty-aware comparison are deferred to the
# later final model-selection workflow. The held-out test set remains untouched.
