# %% [markdown]
# # 12 Multilayer Perceptrons and Feed-Forward Neural Networks
#
# ## Purpose
#
# This notebook evaluates feed-forward multilayer perceptrons (MLPs) for the
# Telco Customer Churn project. An MLP learns nonlinear hidden representations
# of the scaled, one-hot encoded input. For binary classification it produces a
# logit z and churn probability p = sigmoid(z), and is trained by minimizing
# binary log loss with L2 regularization.
#
# The corresponding reusable technical reference is:
#
# ```text
# docs/knowledge_notes/models/12_multilayer_perceptrons_and_neural_networks.md
# ```
#
# The workflow covers a small shallow ReLU screen, a limited depth comparison,
# one tanh diagnostic, pooled out-of-fold probabilities, seed sensitivity,
# optimization traces, calibration diagnostics, threshold diagnostics, and a
# contextual comparison with representative prior model families.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not loaded or used. Every candidate is fitted only
# within stratified outer cross-validation, with imputation, standardization,
# and one-hot encoding fitted inside each training fold through a Pipeline.
#
# The selection metric is mean outer-fold PR-AUC, followed by balanced accuracy
# and F1 only as deterministic tie-breakers. Small differences remain
# development-stage evidence, not proof of meaningful superiority.
#
# `MLPClassifier(early_stopping=True)` reserves an internal stratified
# validation fraction and monitors accuracy. This is an optimization control,
# not the project selection metric. Outer-fold PR-AUC remains the metric used
# to compare procedures.

# %%
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import platform
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier


# %% [markdown]
# ## Import project utilities

# %%
def find_project_root(start: Path | None = None) -> Path:
    """Return the project root by searching upward for standard marker files."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        has_markers = (
            (candidate / "pyproject.toml").exists()
            or (candidate / "README.md").exists()
        )
        has_dirs = (
            (candidate / "data").exists()
            and (candidate / "notebooks").exists()
            and (candidate / "src").exists()
        )
        if has_markers and has_dirs:
            return candidate
    raise FileNotFoundError("Could not locate the project root directory.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# %%
from telco_churn.config import FIGURES_DIR, RANDOM_STATE, TABLES_DIR, TARGET_COLUMN  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    compute_binary_classification_metrics,
    evaluate_estimator_cv,
    evaluate_threshold_grid,
    make_confusion_matrix_dataframe,
    make_precision_recall_curve_dataframe,
    make_roc_curve_dataframe,
    make_stratified_kfold,
)
from telco_churn.models import (  # noqa: E402
    make_bagging_pipeline,
    make_classifier_pipeline,
    make_decision_tree_pipeline,
    make_l2_logistic_regression_pipeline,
    make_linear_svc_pipeline,
    make_mlp_pipeline as make_shared_mlp_pipeline,
    make_xgboost_pipeline,
)
from telco_churn.preprocessing import (  # noqa: E402
    make_dense_scaled_preprocessor,
    make_scaled_preprocessor,
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
MLP_SHALLOW_GRID_RESULTS_PATH = TABLES_DIR / "mlp_shallow_grid_results.csv"
MLP_SHALLOW_GRID_FOLDS_PATH = TABLES_DIR / "mlp_shallow_grid_fold_results.csv"
MLP_DEPTH_RESULTS_PATH = TABLES_DIR / "mlp_depth_comparison_results.csv"
MLP_DEPTH_FOLDS_PATH = TABLES_DIR / "mlp_depth_comparison_fold_results.csv"
MLP_ACTIVATION_RESULTS_PATH = TABLES_DIR / "mlp_activation_diagnostic_results.csv"
MLP_ACTIVATION_FOLDS_PATH = TABLES_DIR / "mlp_activation_diagnostic_fold_results.csv"
MLP_SELECTION_SUMMARY_PATH = TABLES_DIR / "mlp_selection_summary.csv"
MLP_SELECTED_FOLDS_PATH = TABLES_DIR / "mlp_selected_fold_results.csv"
MLP_SELECTED_OOF_PATH = TABLES_DIR / "mlp_selected_oof_predictions.csv"
MLP_SEED_SENSITIVITY_PATH = TABLES_DIR / "mlp_seed_sensitivity_results.csv"
MLP_HISTORY_PATH = TABLES_DIR / "mlp_training_history.csv"
MLP_ARCHITECTURE_PATH = TABLES_DIR / "mlp_architecture_summary.csv"
MLP_CALIBRATION_CURVE_PATH = TABLES_DIR / "mlp_calibration_curve.csv"
MLP_CALIBRATION_SUMMARY_PATH = TABLES_DIR / "mlp_calibration_summary.csv"
MLP_THRESHOLD_PATH = TABLES_DIR / "mlp_threshold_results.csv"
MLP_COMPARISON_PATH = TABLES_DIR / "mlp_model_comparison.csv"
MLP_CONFUSION_PATH = TABLES_DIR / "mlp_confusion_matrices.csv"
MLP_ENVIRONMENT_PATH = TABLES_DIR / "mlp_environment_summary.csv"

MLP_SHALLOW_GRID_FIGURE_PATH = FIGURES_DIR / "mlp_shallow_grid_pr_auc.png"
MLP_DEPTH_FIGURE_PATH = FIGURES_DIR / "mlp_depth_comparison_pr_auc.png"
MLP_COMPARISON_FIGURE_PATH = FIGURES_DIR / "mlp_model_comparison_pr_auc.png"
MLP_ROC_FIGURE_PATH = FIGURES_DIR / "mlp_roc_curve.png"
MLP_PR_FIGURE_PATH = FIGURES_DIR / "mlp_precision_recall_curve.png"
MLP_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "mlp_probability_threshold_tradeoff.png"
MLP_CALIBRATION_FIGURE_PATH = FIGURES_DIR / "mlp_calibration_curve.png"
MLP_LOSS_FIGURE_PATH = FIGURES_DIR / "mlp_training_loss_curve.png"
MLP_INTERNAL_VALIDATION_FIGURE_PATH = FIGURES_DIR / "mlp_internal_validation_accuracy_curve.png"
MLP_SEED_FIGURE_PATH = FIGURES_DIR / "mlp_seed_sensitivity_pr_auc.png"
MLP_PROBABILITY_FIGURE_PATH = FIGURES_DIR / "mlp_oof_probability_distribution.png"

MLP_EXECUTION_LOG_PATH = LOGS_DIR / "12_multilayer_perceptrons_execution.log"
MLP_EXECUTION_LOG_PATH.write_text("", encoding="utf-8")

# %% [markdown]
# ## Load training data only

# %%
train_df = load_train_data()
X, y = split_features_target(train_df)

training_overview = pd.DataFrame(
    {
        "item": ["training_rows", "training_columns", "target_column", "positive_rate", "missing_values"],
        "value": [train_df.shape[0], train_df.shape[1], TARGET_COLUMN, y.mean(), int(train_df.isna().sum().sum())],
    }
)
training_overview

# %% [markdown]
# ## Dense scaled input diagnostic
#
# `MLPClassifier` uses gradient-based optimization and is sensitive to feature
# scale. The shared dense scaled preprocessing factory is also used by the MLP
# smoke test. Each candidate receives a fresh preprocessor through its Pipeline.
# The fit below only documents the transformed input width on training data. It
# is not reused by any cross-validation candidate.

# %%
diagnostic_preprocessor = make_dense_scaled_preprocessor()
diagnostic_matrix = diagnostic_preprocessor.fit_transform(X)
if not np.isfinite(np.asarray(diagnostic_matrix, dtype=float)).all():
    raise ValueError("The diagnostic transformed matrix contains non-finite values.")

environment_summary_df = pd.DataFrame(
    {
        "item": [
            "python_version", "platform", "numpy_version", "pandas_version",
            "scikit_learn_version", "training_rows", "raw_feature_count",
            "dense_transformed_feature_count", "dense_transformed_values_finite",
        ],
        "value": [
            platform.python_version(), platform.platform(), np.__version__, pd.__version__,
            sklearn.__version__, int(X.shape[0]), int(X.shape[1]),
            int(diagnostic_matrix.shape[1]),
            bool(np.isfinite(np.asarray(diagnostic_matrix, dtype=float)).all()),
        ],
    }
)
environment_summary_df.to_csv(MLP_ENVIRONMENT_PATH, index=False)
environment_summary_df

# %% [markdown]
# ## Shared MLP factory and cross-validation helpers
#
# The reusable dense scaled preprocessor and Adam MLP estimator factory live in
# `src/telco_churn/`. Before this full workflow is run, execute
# `python scripts/smoke_test_mlp_workflow.py`. That script uses a small
# stratified subset of `train.csv` to verify the same shared factories, outer
# out-of-fold probability path, threshold diagnostics, calibration primitives,
# and plotting helpers without touching the held-out test set.

# %%
cv = make_stratified_kfold()
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE_INIT = 0.001
DEFAULT_MAX_ITER = 500
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_N_ITER_NO_CHANGE = 20
DEFAULT_TOL = 1e-4


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for compact progress messages."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {remainder:02d}s"


def log_progress(message: str) -> None:
    """Write a timestamped message to stdout and the notebook execution log."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with MLP_EXECUTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{line}\n")


def log_section_start(name: str) -> float:
    """Log a section start and return a performance-counter timestamp."""
    log_progress(f"Starting {name}")
    return time.perf_counter()


def log_section_end(name: str, start_time: float) -> None:
    """Log elapsed time since a named section began."""
    log_progress(f"Finished {name} in {format_elapsed(time.perf_counter() - start_time)}")


def normalize_hidden_layer_sizes(value: object) -> tuple[int, ...]:
    """Validate an MLP architecture and return an immutable tuple of widths."""
    if isinstance(value, bool):
        raise TypeError("hidden_layer_sizes cannot be a boolean.")
    if isinstance(value, int):
        value = (value,)
    try:
        result = tuple(int(width) for width in value)
    except TypeError as exc:
        raise TypeError("hidden_layer_sizes must be an integer or iterable of integers.") from exc
    if not result or any(width <= 0 for width in result):
        raise ValueError("Every MLP hidden-layer width must be strictly positive.")
    return result


def format_architecture(hidden_layer_sizes: tuple[int, ...]) -> str:
    """Return a stable compact architecture label."""
    return "(" + ", ".join(str(width) for width in hidden_layer_sizes) + ")"


def make_mlp_pipeline(
    *,
    hidden_layer_sizes: tuple[int, ...] | int,
    activation: str = "relu",
    alpha: float = 0.001,
    random_state: int = RANDOM_STATE,
):
    """Create the notebook's fixed MLP procedure from shared project factories.

    The reusable source factory owns dense fold-safe preprocessing and estimator
    construction. This narrow wrapper fixes the present workflow's batch size,
    Adam learning rate, iteration budget, tolerance, and internal early-stopping
    controls so every grid candidate represents one explicitly documented
    procedure.
    """
    return make_shared_mlp_pipeline(
        hidden_layer_sizes=normalize_hidden_layer_sizes(hidden_layer_sizes),
        activation=activation,
        alpha=alpha,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate_init=DEFAULT_LEARNING_RATE_INIT,
        max_iter=DEFAULT_MAX_ITER,
        shuffle=True,
        tol=DEFAULT_TOL,
        early_stopping=True,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        n_iter_no_change=DEFAULT_N_ITER_NO_CHANGE,
        random_state=random_state,
    )


def count_fitted_mlp_parameters(fitted_pipeline) -> int:
    """Count fitted dense-layer weights and biases."""
    classifier = fitted_pipeline.named_steps["classifier"]
    return int(
        sum(np.asarray(weights).size for weights in classifier.coefs_)
        + sum(np.asarray(bias).size for bias in classifier.intercepts_)
    )


def make_mlp_name(
    *, hidden_layer_sizes: tuple[int, ...], activation: str, alpha: float, random_state: int
) -> str:
    """Build the common label used in tables, logs, and figures."""
    return (
        "MLPClassifier "
        f"hidden={format_architecture(hidden_layer_sizes)} "
        f"activation={activation} alpha={alpha:g} seed={random_state}"
    )


def evaluate_mlp_candidate_cv(
    *,
    model_name: str,
    estimator,
    stage: str,
    candidate_parameters: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    collect_oof: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Evaluate one MLP procedure using a manual outer-CV loop.

    The manual loop preserves MLP-specific fit diagnostics: convergence warnings,
    epochs, final loss, internal validation accuracy, parameter count, and, when
    requested, one honest out-of-fold probability for each training observation.
    Training-fold metrics are stored for diagnosis only. Candidate ranking uses
    outer validation-fold metrics.
    """
    rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    for fold, (train_index, validation_index) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[validation_index], y.iloc[validation_index]
        fitted = clone(estimator)

        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        warning_count = sum(
            issubclass(record.category, ConvergenceWarning)
            for record in caught_warnings
        )

        score_start = time.perf_counter()
        train_score = fitted.predict_proba(X_train)[:, 1]
        valid_score = fitted.predict_proba(X_valid)[:, 1]
        score_seconds = time.perf_counter() - score_start
        train_pred = (train_score >= 0.50).astype(int)
        valid_pred = (valid_score >= 0.50).astype(int)

        classifier = fitted.named_steps["classifier"]
        diagnostics = {
            "fit_time_seconds": float(fit_seconds),
            "score_time_seconds": float(score_seconds),
            "convergence_warning_count": int(warning_count),
            "n_iter": int(classifier.n_iter_),
            "final_loss": float(classifier.loss_),
            "best_internal_validation_accuracy": float(
                getattr(classifier, "best_validation_score_", np.nan)
            ),
            "trainable_parameter_count": count_fitted_mlp_parameters(fitted),
        }

        for split, observed, predicted, scores in [
            ("train", y_train, train_pred, train_score),
            ("validation", y_valid, valid_pred, valid_score),
        ]:
            metrics = compute_binary_classification_metrics(
                y_true=observed,
                y_pred=predicted,
                y_score=scores,
            )
            rows.append(
                {
                    "stage": stage,
                    "model": model_name,
                    "fold": fold,
                    "split": split,
                    **candidate_parameters,
                    **diagnostics,
                    "tp": metrics.tp,
                    "fn": metrics.fn,
                    "fp": metrics.fp,
                    "tn": metrics.tn,
                    "accuracy": metrics.accuracy,
                    "balanced_accuracy": metrics.balanced_accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "specificity": metrics.specificity,
                    "f1": metrics.f1,
                    "roc_auc": metrics.roc_auc,
                    "pr_auc": metrics.pr_auc,
                    "predicted_positive_rate": metrics.predicted_positive_rate,
                    "observed_positive_rate": metrics.observed_positive_rate,
                }
            )

        if collect_oof:
            oof_rows.extend(
                {
                    "row_index": row_index,
                    "fold": fold,
                    "y_true": int(observed),
                    "predicted_probability": float(score),
                    "predicted_class_at_0_50": int(prediction),
                }
                for row_index, observed, score, prediction in zip(
                    X_valid.index,
                    y_valid,
                    valid_score,
                    valid_pred,
                )
            )

    fold_df = pd.DataFrame(rows)
    oof_df = pd.DataFrame(oof_rows) if collect_oof else None
    return fold_df, oof_df


def summarize_mlp_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate train and validation fold results into one row per candidate."""
    validation_df = fold_df.loc[fold_df["split"] == "validation"].copy()
    train_df = fold_df.loc[fold_df["split"] == "train"].copy()
    key_columns = [
        "stage", "model", "architecture", "activation", "alpha",
        "learning_rate_init", "batch_size", "max_iter", "early_stopping",
        "validation_fraction", "n_iter_no_change", "random_state",
    ]
    metrics = [
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
        "f1", "roc_auc", "pr_auc", "predicted_positive_rate",
    ]
    rows: list[dict[str, object]] = []

    for key, candidate_valid in validation_df.groupby(key_columns, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(key_columns, key))
        candidate_train = train_df.loc[train_df["model"] == row["model"]]
        for metric in metrics:
            row[f"validation_{metric}_mean"] = float(candidate_valid[metric].mean())
            row[f"validation_{metric}_std"] = float(candidate_valid[metric].std(ddof=0))
            row[f"train_{metric}_mean"] = float(candidate_train[metric].mean())
            row[f"train_{metric}_std"] = float(candidate_train[metric].std(ddof=0))
        for diagnostic in [
            "fit_time_seconds", "score_time_seconds", "n_iter", "final_loss",
            "best_internal_validation_accuracy", "trainable_parameter_count",
            "convergence_warning_count",
        ]:
            row[f"{diagnostic}_mean"] = float(candidate_valid[diagnostic].mean())
            row[f"{diagnostic}_std"] = float(candidate_valid[diagnostic].std(ddof=0))
        row["convergence_warning_fold_count"] = int(
            (candidate_valid["convergence_warning_count"] > 0).sum()
        )
        row["validation_pr_auc_train_gap"] = (
            row["train_pr_auc_mean"] - row["validation_pr_auc_mean"]
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["validation_pr_auc_mean", "validation_balanced_accuracy_mean", "validation_f1_mean"],
        ascending=False,
    ).reset_index(drop=True)


def evaluate_mlp_grid(
    *, stage: str, parameter_grid: dict[str, list[object]], X: pd.DataFrame, y: pd.Series, cv
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a transparent grid and retain fold-level diagnostics."""
    grids = list(ParameterGrid(parameter_grid))
    grid_start = log_section_start(f"{stage} ({len(grids)} candidates)")
    fold_frames: list[pd.DataFrame] = []

    for candidate_number, parameters in enumerate(grids, start=1):
        architecture = normalize_hidden_layer_sizes(parameters["hidden_layer_sizes"])
        activation = str(parameters.get("activation", "relu"))
        alpha = float(parameters["alpha"])
        random_state = int(parameters.get("random_state", RANDOM_STATE))
        name = make_mlp_name(
            hidden_layer_sizes=architecture,
            activation=activation,
            alpha=alpha,
            random_state=random_state,
        )
        candidate_parameters = {
            "architecture": format_architecture(architecture),
            "activation": activation,
            "alpha": alpha,
            "learning_rate_init": DEFAULT_LEARNING_RATE_INIT,
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_iter": DEFAULT_MAX_ITER,
            "early_stopping": True,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "n_iter_no_change": DEFAULT_N_ITER_NO_CHANGE,
            "random_state": random_state,
        }
        start = time.perf_counter()
        fold_df, _ = evaluate_mlp_candidate_cv(
            model_name=name,
            estimator=make_mlp_pipeline(
                hidden_layer_sizes=architecture,
                activation=activation,
                alpha=alpha,
                random_state=random_state,
            ),
            stage=stage,
            candidate_parameters=candidate_parameters,
            X=X,
            y=y,
            cv=cv,
        )
        fold_frames.append(fold_df)
        log_progress(
            f"{stage} candidate {candidate_number}/{len(grids)} completed in "
            f"{format_elapsed(time.perf_counter() - start)}: {name}"
        )

    combined_folds = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_mlp_folds(combined_folds)
    log_section_end(stage, grid_start)
    return summary, combined_folds


def grid_display_columns() -> list[str]:
    """Return compact grid columns for notebook display."""
    return [
        "model", "architecture", "activation", "alpha",
        "validation_pr_auc_mean", "validation_pr_auc_std",
        "validation_roc_auc_mean", "validation_balanced_accuracy_mean",
        "validation_f1_mean", "train_pr_auc_mean", "validation_pr_auc_train_gap",
        "n_iter_mean", "convergence_warning_fold_count", "fit_time_seconds_mean",
    ]


def save_grid_plot(results_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a horizontal plot of candidate mean outer-fold PR-AUC values."""
    plot_df = (
        results_df.sort_values("validation_pr_auc_mean", ascending=False)
        .head(12)
        .sort_values("validation_pr_auc_mean")
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["model"], plot_df["validation_pr_auc_mean"])
    ax.set_xlabel("Mean outer-fold PR-AUC")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(results_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save pooled OOF PR-AUC values for the selected MLP and references."""
    plot_df = results_df.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["model"], plot_df["pr_auc"])
    ax.set_xlabel("Pooled out-of-fold PR-AUC")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


log_progress("MLP notebook preflight")
log_progress(f"Training data shape: {X.shape}; dense diagnostic width: {diagnostic_matrix.shape[1]}")
log_progress(f"CV folds: {cv.get_n_splits(X, y)}; held-out test set remains untouched.")

# %% [markdown]
# ## Fixed candidate design
#
# The candidate sequence is deliberately staged:
#
# ```text
# 1. Shallow ReLU screen: widths (16,), (32,), (64,) and alpha values
#    0.0001, 0.001, 0.01.
# 2. Depth screen: (32, 16) and (64, 32), using the strongest shallow alpha.
# 3. Activation diagnostic: one tanh fit at the leading ReLU architecture.
# ```
#
# Adam, batch size, learning rate, internal validation fraction, and patience
# remain fixed. This prevents the screen from becoming an opaque search across
# unrelated neural-network controls.

# %%
shallow_relu_grid = {
    "hidden_layer_sizes": [(16,), (32,), (64,)],
    "activation": ["relu"],
    "alpha": [0.0001, 0.001, 0.01],
    "random_state": [RANDOM_STATE],
}

# %% [markdown]
# ## Shallow ReLU width and L2 regularization screen

# %%
shallow_results_df, shallow_folds_df = evaluate_mlp_grid(
    stage="shallow_relu_screen",
    parameter_grid=shallow_relu_grid,
    X=X,
    y=y,
    cv=cv,
)
shallow_results_df.to_csv(MLP_SHALLOW_GRID_RESULTS_PATH, index=False)
shallow_folds_df.to_csv(MLP_SHALLOW_GRID_FOLDS_PATH, index=False)
save_grid_plot(
    shallow_results_df,
    MLP_SHALLOW_GRID_FIGURE_PATH,
    "MLP shallow ReLU screen: mean outer-fold PR-AUC",
)
shallow_results_df[grid_display_columns()]

# %% [markdown]
# ## Two-hidden-layer capacity comparison
#
# The depth candidates use the best observed shallow ReLU regularization scale.
# This tests whether modest extra depth adds useful predictive structure without
# multiplying the width, depth, and alpha search space.

# %%
best_shallow_row = shallow_results_df.iloc[0]
best_shallow_alpha = float(best_shallow_row["alpha"])

depth_grid = {
    "hidden_layer_sizes": [(32, 16), (64, 32)],
    "activation": ["relu"],
    "alpha": [best_shallow_alpha],
    "random_state": [RANDOM_STATE],
}

depth_results_df, depth_folds_df = evaluate_mlp_grid(
    stage="depth_comparison",
    parameter_grid=depth_grid,
    X=X,
    y=y,
    cv=cv,
)
depth_results_df.to_csv(MLP_DEPTH_RESULTS_PATH, index=False)
depth_folds_df.to_csv(MLP_DEPTH_FOLDS_PATH, index=False)
save_grid_plot(
    depth_results_df,
    MLP_DEPTH_FIGURE_PATH,
    "MLP two-hidden-layer comparison: mean outer-fold PR-AUC",
)
depth_results_df[grid_display_columns()]

# %% [markdown]
# ## Controlled tanh activation diagnostic
#
# ReLU is the main hidden activation. Tanh is tested once at the leading ReLU
# architecture and alpha so the workflow can detect a material activation
# sensitivity without performing a broad activation sweep.

# %%
relu_results_df = pd.concat([shallow_results_df, depth_results_df], ignore_index=True)
relu_results_df = relu_results_df.sort_values(
    ["validation_pr_auc_mean", "validation_balanced_accuracy_mean", "validation_f1_mean"],
    ascending=False,
).reset_index(drop=True)
best_relu_row = relu_results_df.iloc[0]


def parse_architecture_label(label: str) -> tuple[int, ...]:
    """Recover a tuple of hidden widths from the stable local display label."""
    values = [part.strip() for part in label.strip("()").split(",") if part.strip()]
    return tuple(int(value) for value in values)


best_relu_architecture = parse_architecture_label(str(best_relu_row["architecture"]))
best_relu_alpha = float(best_relu_row["alpha"])

activation_grid = {
    "hidden_layer_sizes": [best_relu_architecture],
    "activation": ["tanh"],
    "alpha": [best_relu_alpha],
    "random_state": [RANDOM_STATE],
}
activation_results_df, activation_folds_df = evaluate_mlp_grid(
    stage="activation_diagnostic",
    parameter_grid=activation_grid,
    X=X,
    y=y,
    cv=cv,
)
activation_results_df.to_csv(MLP_ACTIVATION_RESULTS_PATH, index=False)
activation_folds_df.to_csv(MLP_ACTIVATION_FOLDS_PATH, index=False)
activation_results_df[grid_display_columns()]

# %% [markdown]
# ## Select one representative MLP candidate
#
# The deterministic selection rule is:
#
# ```text
# highest mean outer-fold PR-AUC,
# then highest mean outer-fold balanced accuracy,
# then highest mean outer-fold F1.
# ```
#
# The result is a representative candidate for diagnostics. It does not claim a
# meaningful or final advantage over close alternatives.

# %%
all_candidate_results_df = pd.concat(
    [shallow_results_df, depth_results_df, activation_results_df],
    ignore_index=True,
).sort_values(
    ["validation_pr_auc_mean", "validation_balanced_accuracy_mean", "validation_f1_mean"],
    ascending=False,
).reset_index(drop=True)

selected_row = all_candidate_results_df.iloc[0]
selected_architecture = parse_architecture_label(str(selected_row["architecture"]))
selected_activation = str(selected_row["activation"])
selected_alpha = float(selected_row["alpha"])
selected_random_state = int(selected_row["random_state"])
selected_name = make_mlp_name(
    hidden_layer_sizes=selected_architecture,
    activation=selected_activation,
    alpha=selected_alpha,
    random_state=selected_random_state,
)
selected_pipeline = make_mlp_pipeline(
    hidden_layer_sizes=selected_architecture,
    activation=selected_activation,
    alpha=selected_alpha,
    random_state=selected_random_state,
)

selection_summary_df = pd.DataFrame(
    {
        "selected_model": [selected_name],
        "selection_rule": ["highest mean outer-fold PR-AUC, then balanced accuracy, then F1"],
        "architecture": [format_architecture(selected_architecture)],
        "activation": [selected_activation],
        "alpha": [selected_alpha],
        "random_state": [selected_random_state],
        "mean_outer_fold_pr_auc": [selected_row["validation_pr_auc_mean"]],
        "std_outer_fold_pr_auc": [selected_row["validation_pr_auc_std"]],
        "mean_outer_fold_roc_auc": [selected_row["validation_roc_auc_mean"]],
        "mean_outer_fold_balanced_accuracy": [selected_row["validation_balanced_accuracy_mean"]],
        "mean_outer_fold_f1": [selected_row["validation_f1_mean"]],
        "convergence_warning_fold_count": [selected_row["convergence_warning_fold_count"]],
    }
)
selection_summary_df.to_csv(MLP_SELECTION_SUMMARY_PATH, index=False)
selection_summary_df

# %% [markdown]
# ## Selected candidate pooled out-of-fold diagnostics
#
# One probability is generated for each training observation by a model that did
# not train on that observation. These pooled OOF values support ranking,
# calibration, threshold, and confusion-matrix diagnostics. They are still
# development-stage evidence, not an independent test evaluation.

# %%
selected_parameters = {
    "architecture": format_architecture(selected_architecture),
    "activation": selected_activation,
    "alpha": selected_alpha,
    "learning_rate_init": DEFAULT_LEARNING_RATE_INIT,
    "batch_size": DEFAULT_BATCH_SIZE,
    "max_iter": DEFAULT_MAX_ITER,
    "early_stopping": True,
    "validation_fraction": DEFAULT_VALIDATION_FRACTION,
    "n_iter_no_change": DEFAULT_N_ITER_NO_CHANGE,
    "random_state": selected_random_state,
}

selected_start = log_section_start("selected MLP pooled OOF diagnostic")
selected_folds_df, selected_oof_df = evaluate_mlp_candidate_cv(
    model_name=selected_name,
    estimator=selected_pipeline,
    stage="selected_candidate_oof",
    candidate_parameters=selected_parameters,
    X=X,
    y=y,
    cv=cv,
    collect_oof=True,
)
log_section_end("selected MLP pooled OOF diagnostic", selected_start)

if selected_oof_df is None:
    raise RuntimeError("Selected candidate OOF predictions were not collected.")

selected_folds_df.to_csv(MLP_SELECTED_FOLDS_PATH, index=False)
selected_oof_df = selected_oof_df.sort_values("row_index").reset_index(drop=True)
selected_oof_df.to_csv(MLP_SELECTED_OOF_PATH, index=False)

selected_oof_metrics = compute_binary_classification_metrics(
    y_true=selected_oof_df["y_true"],
    y_pred=selected_oof_df["predicted_class_at_0_50"],
    y_score=selected_oof_df["predicted_probability"],
)
selected_oof_row = {
    "model": "Selected MLPClassifier",
    "tp": selected_oof_metrics.tp,
    "fn": selected_oof_metrics.fn,
    "fp": selected_oof_metrics.fp,
    "tn": selected_oof_metrics.tn,
    "accuracy": selected_oof_metrics.accuracy,
    "balanced_accuracy": selected_oof_metrics.balanced_accuracy,
    "precision": selected_oof_metrics.precision,
    "recall": selected_oof_metrics.recall,
    "specificity": selected_oof_metrics.specificity,
    "f1": selected_oof_metrics.f1,
    "roc_auc": selected_oof_metrics.roc_auc,
    "pr_auc": selected_oof_metrics.pr_auc,
    "predicted_positive_rate": selected_oof_metrics.predicted_positive_rate,
    "observed_positive_rate": selected_oof_metrics.observed_positive_rate,
}
pd.DataFrame([selected_oof_row])

# %% [markdown]
# ## Seed-sensitivity diagnostic
#
# The fixed candidate grid uses one seed for reproducibility. MLP fitting is
# nevertheless stochastic because initialization, mini-batch order, and the
# internal early-stopping split are random-state dependent. The following runs
# hold architecture and regularization fixed while changing the seed. They are
# a stability diagnostic, not a new architecture search.

# %%
SEED_SENSITIVITY_SEEDS = [7, 19, 2026]
seed_rows = [
    {
        "random_state": selected_random_state,
        "architecture": format_architecture(selected_architecture),
        "activation": selected_activation,
        "alpha": selected_alpha,
        "pooled_oof_pr_auc": selected_oof_row["pr_auc"],
        "pooled_oof_roc_auc": selected_oof_row["roc_auc"],
        "pooled_oof_balanced_accuracy": selected_oof_row["balanced_accuracy"],
        "pooled_oof_f1": selected_oof_row["f1"],
        "note": "selection-seed pooled OOF diagnostic",
    }
]

seed_start = log_section_start("selected MLP seed-sensitivity diagnostic")
for seed in SEED_SENSITIVITY_SEEDS:
    seed_name = make_mlp_name(
        hidden_layer_sizes=selected_architecture,
        activation=selected_activation,
        alpha=selected_alpha,
        random_state=seed,
    )
    seed_parameters = {**selected_parameters, "random_state": seed}
    _, seed_oof_df = evaluate_mlp_candidate_cv(
        model_name=seed_name,
        estimator=make_mlp_pipeline(
            hidden_layer_sizes=selected_architecture,
            activation=selected_activation,
            alpha=selected_alpha,
            random_state=seed,
        ),
        stage="seed_sensitivity",
        candidate_parameters=seed_parameters,
        X=X,
        y=y,
        cv=cv,
        collect_oof=True,
    )
    if seed_oof_df is None:
        raise RuntimeError("Seed-sensitivity OOF predictions were not collected.")
    seed_metrics = compute_binary_classification_metrics(
        y_true=seed_oof_df["y_true"],
        y_pred=seed_oof_df["predicted_class_at_0_50"],
        y_score=seed_oof_df["predicted_probability"],
    )
    seed_rows.append(
        {
            "random_state": seed,
            "architecture": format_architecture(selected_architecture),
            "activation": selected_activation,
            "alpha": selected_alpha,
            "pooled_oof_pr_auc": seed_metrics.pr_auc,
            "pooled_oof_roc_auc": seed_metrics.roc_auc,
            "pooled_oof_balanced_accuracy": seed_metrics.balanced_accuracy,
            "pooled_oof_f1": seed_metrics.f1,
            "note": "additional-seed pooled OOF diagnostic",
        }
    )
log_section_end("selected MLP seed-sensitivity diagnostic", seed_start)

seed_sensitivity_df = pd.DataFrame(seed_rows).sort_values("random_state").reset_index(drop=True)
seed_sensitivity_df.to_csv(MLP_SEED_SENSITIVITY_PATH, index=False)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(seed_sensitivity_df["random_state"].astype(str), seed_sensitivity_df["pooled_oof_pr_auc"], marker="o")
ax.set_xlabel("Random state")
ax.set_ylabel("Pooled out-of-fold PR-AUC")
ax.set_ylim(0, 1)
ax.set_title("Selected MLP architecture: pooled OOF PR-AUC across seeds")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(MLP_SEED_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
seed_sensitivity_df

# %% [markdown]
# ## Full-training optimization diagnostic
#
# This fit trains the selected configuration on the complete training table. It
# is not a performance estimate because it has seen these observations. Its role
# is to document fitted architecture, parameter count, loss by epoch, and the
# internal validation-accuracy history used for early stopping.
#
# Scikit-learn's exposed internal validation history is accuracy, not PR-AUC.

# %%
full_fit_start = log_section_start("selected MLP full-training optimization diagnostic")
selected_full_pipeline = clone(selected_pipeline)
with warnings.catch_warnings(record=True) as full_fit_warnings:
    warnings.simplefilter("always", ConvergenceWarning)
    selected_full_pipeline.fit(X, y)
full_fit_warning_count = sum(
    issubclass(record.category, ConvergenceWarning)
    for record in full_fit_warnings
)
log_section_end("selected MLP full-training optimization diagnostic", full_fit_start)

full_classifier = selected_full_pipeline.named_steps["classifier"]
loss_curve = list(getattr(full_classifier, "loss_curve_", []))
validation_curve = getattr(full_classifier, "validation_scores_", None)
history_rows = []
for epoch, loss_value in enumerate(loss_curve, start=1):
    validation_accuracy = np.nan
    if validation_curve is not None and epoch <= len(validation_curve):
        validation_accuracy = float(validation_curve[epoch - 1])
    history_rows.append(
        {
            "epoch": epoch,
            "training_loss": float(loss_value),
            "internal_validation_accuracy": validation_accuracy,
        }
    )
history_df = pd.DataFrame(history_rows)
history_df.to_csv(MLP_HISTORY_PATH, index=False)

architecture_summary_df = pd.DataFrame(
    {
        "item": [
            "dense_transformed_input_features", "hidden_layer_sizes",
            "hidden_layer_count", "network_layers_including_input_and_output",
            "output_activation", "trainable_parameter_count", "optimizer",
            "hidden_activation", "alpha", "learning_rate_init", "batch_size",
            "early_stopping", "validation_fraction", "n_iter", "final_loss",
            "best_internal_validation_accuracy", "full_training_convergence_warning_count",
        ],
        "value": [
            int(diagnostic_matrix.shape[1]),
            format_architecture(tuple(int(value) for value in full_classifier.hidden_layer_sizes)),
            len(full_classifier.hidden_layer_sizes), full_classifier.n_layers_,
            full_classifier.out_activation_, count_fitted_mlp_parameters(selected_full_pipeline),
            full_classifier.solver, full_classifier.activation, full_classifier.alpha,
            full_classifier.learning_rate_init, full_classifier.batch_size,
            full_classifier.early_stopping, full_classifier.validation_fraction,
            full_classifier.n_iter_, full_classifier.loss_,
            getattr(full_classifier, "best_validation_score_", np.nan),
            full_fit_warning_count,
        ],
    }
)
architecture_summary_df.to_csv(MLP_ARCHITECTURE_PATH, index=False)

if not history_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(history_df["epoch"], history_df["training_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Selected MLP full-training diagnostic: loss by epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MLP_LOSS_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    internal_history = history_df.dropna(subset=["internal_validation_accuracy"])
    if not internal_history.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(internal_history["epoch"], internal_history["internal_validation_accuracy"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Internal validation accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Selected MLP full-training diagnostic: internal validation accuracy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(MLP_INTERNAL_VALIDATION_FIGURE_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)

architecture_summary_df

# %% [markdown]
# ## Threshold, ranking, and calibration diagnostics
#
# These figures use selected-model pooled OOF probabilities. The 0.50 threshold
# is shown as a default probability boundary, not a final business decision.
# The reliability diagram and Brier score do not fit a calibrator. Calibration
# selection remains a later finalist-stage question.

# %%
thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
threshold_df = evaluate_threshold_grid(
    y_true=selected_oof_df["y_true"],
    y_score=selected_oof_df["predicted_probability"],
    thresholds=thresholds,
)
threshold_df.to_csv(MLP_THRESHOLD_PATH, index=False)
save_threshold_tradeoff_plot(
    threshold_df=threshold_df,
    output_path=MLP_THRESHOLD_FIGURE_PATH,
    title="Selected MLP: pooled OOF probability-threshold tradeoff",
    x_label="Predicted churn probability threshold",
    reference_threshold=0.50,
    reference_label="Default probability threshold = 0.50",
)
threshold_df

# %%
roc_curve_df = make_roc_curve_dataframe(
    y_true=selected_oof_df["y_true"],
    y_score=selected_oof_df["predicted_probability"],
)
pr_curve_df = make_precision_recall_curve_dataframe(
    y_true=selected_oof_df["y_true"],
    y_score=selected_oof_df["predicted_probability"],
)
save_roc_curve_plot(
    roc_curve_df=roc_curve_df,
    output_path=MLP_ROC_FIGURE_PATH,
    title="Selected MLP: pooled out-of-fold ROC curve",
)
save_precision_recall_curve_plot(
    precision_recall_curve_df=pr_curve_df,
    output_path=MLP_PR_FIGURE_PATH,
    title="Selected MLP: pooled out-of-fold precision-recall curve",
    positive_rate=float(y.mean()),
)

observed_rate, mean_probability = calibration_curve(
    selected_oof_df["y_true"],
    selected_oof_df["predicted_probability"],
    n_bins=10,
    strategy="quantile",
)
calibration_curve_df = pd.DataFrame(
    {
        "mean_predicted_probability": mean_probability,
        "observed_churn_rate": observed_rate,
    }
)
calibration_summary_df = pd.DataFrame(
    {
        "metric": [
            "pooled_oof_brier_score", "pooled_oof_mean_predicted_probability",
            "observed_positive_rate", "calibration_bins_returned",
        ],
        "value": [
            brier_score_loss(selected_oof_df["y_true"], selected_oof_df["predicted_probability"]),
            float(selected_oof_df["predicted_probability"].mean()),
            float(selected_oof_df["y_true"].mean()),
            int(len(calibration_curve_df)),
        ],
    }
)
calibration_curve_df.to_csv(MLP_CALIBRATION_CURVE_PATH, index=False)
calibration_summary_df.to_csv(MLP_CALIBRATION_SUMMARY_PATH, index=False)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal calibration")
ax.plot(
    calibration_curve_df["mean_predicted_probability"],
    calibration_curve_df["observed_churn_rate"],
    marker="o",
    label="Selected MLP pooled OOF",
)
ax.set_xlabel("Mean predicted churn probability")
ax.set_ylabel("Observed churn rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Selected MLP: pooled out-of-fold calibration diagnostic")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(MLP_CALIBRATION_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for class_value, label in [(0, "Observed no churn"), (1, "Observed churn")]:
    values = selected_oof_df.loc[
        selected_oof_df["y_true"] == class_value,
        "predicted_probability",
    ]
    ax.hist(values, bins=np.linspace(0, 1, 21), density=True, alpha=0.55, label=label)
ax.set_xlabel("Pooled out-of-fold predicted churn probability")
ax.set_ylabel("Density")
ax.set_xlim(0, 1)
ax.set_title("Selected MLP: pooled OOF probability distributions by observed class")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(MLP_PROBABILITY_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

calibration_summary_df

# %% [markdown]
# ## Contextual comparison with earlier model representatives
#
# These estimators are reconstructed at their earlier selected development
# settings. They are not retuned here. The comparison reuses the training set
# and outer CV design, so it is contextual development-stage evidence rather
# than a final model tournament.

# %%
def make_knn_reference_pipeline():
    """Reconstruct the selected kNN pipeline from workflow 06."""
    return make_classifier_pipeline(
        preprocessor=make_scaled_preprocessor(),
        classifier=KNeighborsClassifier(
            n_neighbors=101,
            weights="uniform",
            metric="minkowski",
            p=1,
        ),
    )


reference_estimators = [
    ("Selected L2 logistic regression", make_l2_logistic_regression_pipeline(C=1.0)),
    ("Selected kNN", make_knn_reference_pipeline()),
    (
        "Selected decision tree",
        make_decision_tree_pipeline(
            criterion="gini",
            max_depth=6,
            min_samples_split=25,
            min_samples_leaf=10,
            ccp_alpha=0.0,
        ),
    ),
    (
        "Selected bagged trees",
        make_bagging_pipeline(
            n_estimators=200,
            max_samples=0.8,
            base_max_depth=6,
            base_min_samples_leaf=1,
        ),
    ),
    (
        "Representative XGBoost",
        make_xgboost_pipeline(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=2,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_lambda=1.0,
        ),
    ),
    (
        "Selected LinearSVC",
        make_linear_svc_pipeline(
            C=0.1,
            loss="squared_hinge",
            class_weight="balanced",
        ),
    ),
]

comparison_start = log_section_start("representative earlier-model comparison")
comparison_rows = [selected_oof_row]
for model_name, estimator in reference_estimators:
    start = time.perf_counter()
    comparison_rows.append(
        evaluate_estimator_cv(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )
    )
    log_progress(
        f"Reference comparison completed in {format_elapsed(time.perf_counter() - start)}: {model_name}"
    )
log_section_end("representative earlier-model comparison", comparison_start)

comparison_df = pd.DataFrame(comparison_rows).sort_values(
    ["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).reset_index(drop=True)
confusion_df = make_confusion_matrix_dataframe(comparison_df)
comparison_df.to_csv(MLP_COMPARISON_PATH, index=False)
confusion_df.to_csv(MLP_CONFUSION_PATH, index=False)
save_model_comparison_plot(
    comparison_df,
    MLP_COMPARISON_FIGURE_PATH,
    "Representative models: pooled out-of-fold PR-AUC",
)
comparison_df[
    [
        "model", "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc", "pr_auc", "predicted_positive_rate",
    ]
]

# %% [markdown]
# ## Saved positive-first confusion matrices
#
# The project uses this convention:
#
# ```text
# TP: observed churn, predicted churn
# FN: observed churn, predicted no churn
# FP: observed no churn, predicted churn
# TN: observed no churn, predicted no churn
# ```

# %%
confusion_df

# %% [markdown]
# ## Interpretation of observed development-stage results
#
# All results below were generated from the training set only. Model selection,
# out-of-fold predictions, curves, calibration diagnostics, and reference-model
# comparisons use the project-wide stratified cross-validation design. The
# held-out test set remains untouched.
#
# ### Compact shallow ReLU screen: no material gain from a wider hidden layer
#
# The first screen evaluated one hidden layer with 16, 32, or 64 ReLU units and
# three L2 regularization strengths. Selection used the mean outer-fold PR-AUC,
# with balanced accuracy and F1 as tie-breakers:
#
# | Hidden units | Alpha | Mean outer-fold PR-AUC | Fold SD | Mean train PR-AUC | Train-validation gap |
# |---:|---:|---:|---:|---:|---:|
# | 16 | 0.0001 | 0.6586 | 0.0193 | 0.6665 | 0.0079 |
# | 16 | 0.0010 | 0.6595 | 0.0183 | 0.6691 | 0.0096 |
# | 16 | 0.0100 | 0.6588 | 0.0189 | 0.6687 | 0.0099 |
# | 32 | 0.0001 | 0.6567 | 0.0208 | 0.6739 | 0.0172 |
# | 32 | 0.0010 | 0.6576 | 0.0213 | 0.6734 | 0.0158 |
# | 32 | 0.0100 | 0.6580 | 0.0215 | 0.6795 | 0.0215 |
# | 64 | 0.0001 | 0.6559 | 0.0156 | 0.6942 | 0.0383 |
# | 64 | 0.0010 | 0.6567 | 0.0156 | 0.6952 | 0.0385 |
# | 64 | 0.0100 | 0.6565 | 0.0158 | 0.6867 | 0.0302 |
#
# The representative MLP is therefore:
#
# ```text
# MLPClassifier(hidden_layer_sizes=(16,), activation="relu", alpha=0.001,
#               solver="adam", learning_rate_init=0.001, batch_size=64,
#               early_stopping=True, validation_fraction=0.15,
#               n_iter_no_change=20, random_state=42)
# ```
#
# Its mean outer-fold PR-AUC, 0.6595, is only 0.0007 above the neighbouring
# 16-unit alpha=0.01 candidate and 0.0009 above the 16-unit alpha=0.0001
# candidate. Those differences are much smaller than the ordinary fold-to-fold
# standard deviations of approximately 0.018 to 0.019. The selection rule still
# identifies the 16-unit, alpha=0.001 configuration as the representative model
# within this tried grid, but the result does not establish that it is uniquely
# optimal.
#
# The wider networks did not improve mean validation PR-AUC. At the same time,
# their training PR-AUC was higher and their train-validation gaps widened,
# especially for 64 units. For example, the 64-unit, alpha=0.001 candidate had
# a mean training PR-AUC of 0.6952 but a mean validation PR-AUC of 0.6567. This
# pattern is consistent with extra capacity fitting training-fold detail without
# producing a better unseen churn ranking in this compact search.
#
# ### Depth diagnostic: two hidden layers did not add useful validation signal
#
# The two-layer diagnostic kept alpha=0.001, the strongest observed shallow
# regularization scale, and tested two modest architectures:
#
# | Architecture | Trainable parameters | Mean outer-fold PR-AUC | Fold SD | Train-validation PR-AUC gap |
# |---|---:|---:|---:|
# | (32, 16) | 2,049 | 0.6507 | 0.0196 | 0.0275 |
# | (64, 32) | 5,121 | 0.6523 | 0.0196 | 0.0220 |
# | Selected shallow (16) | 769 | 0.6595 | 0.0183 | 0.0096 |
#
# Neither added-depth candidate exceeds the shallow representative. The observed
# differences, 0.0072 for (64, 32) and 0.0088 for (32, 16), are still modest
# relative to the fold-to-fold uncertainty, so this is not proof that every
# deeper network is worse. It is sufficient evidence that extra depth did not
# justify its additional parameters and greater train-validation gap in this
# deliberately limited development stage.
#
# ### Activation diagnostic: tanh is nearby, not a demonstrated improvement
#
# Replacing ReLU with tanh while retaining the selected shallow architecture and
# alpha produced mean outer-fold PR-AUC 0.6559 with a fold SD of 0.0273. The
# corresponding ReLU result was 0.6595 with a fold SD of 0.0183. The tanh
# estimate is lower by 0.0036, but this difference is small relative to the
# observed fold variation. The correct interpretation is not that ReLU has been
# proven universally better. Rather, ReLU remains the representative activation
# because it was selected by the pre-defined development rule and tanh did not
# reveal a compensating ranking advantage.
#
# ### Selected architecture and optimization behaviour
#
# The dense preprocessing pipeline produced 46 numeric inputs after fold-safe
# imputation, scaling, and one-hot encoding. The selected network has 16 hidden
# ReLU units and one logistic output. Its parameter count is
#
# $$
# (46 \times 16 + 16) + (16 \times 1 + 1) = 769.
# $$
#
# The complete-training diagnostic stopped after 42 epochs, far below the
# maximum of 500. Training loss declined from 0.5851 at the first epoch to
# 0.3960 at the final recorded epoch without numerical instability. The internal
# validation accuracy rose rapidly from approximately 0.779 to a best value of
# 0.8109 and then fluctuated in a narrow range around 0.80. No
# `ConvergenceWarning` occurred in the selected full-data fit or in the selected
# candidate's five outer folds.
#
# These diagnostics do not show a large optimization failure. They also do not
# replace outer-fold model selection. The internal early-stopping monitor is
# validation accuracy, whereas the project selects candidates using outer-fold
# PR-AUC because churn is the minority class. The modest selected-candidate
# train-validation PR-AUC gap of 0.0096 is reassuring relative to the larger
# gaps of several wider and deeper candidates, but it is not a guarantee of
# future generalization.
#
# ### Seed sensitivity: modest stochastic variation
#
# The selected architecture was re-evaluated with random states 7, 19, 42, and
# 2026. Its pooled out-of-fold PR-AUC values were 0.6536, 0.6502, 0.6544, and
# 0.6554. The observed range is therefore only 0.0052. ROC-AUC remains between
# 0.8421 and 0.8440, while balanced accuracy remains between 0.7108 and 0.7167.
#
# This modest variation indicates that the selected result is not driven solely
# by one unusually favourable initialization. It does not eliminate stochastic
# uncertainty: the exact seed-42 value should not be treated as a permanently
# precise estimate, and a later finalist comparison should account for training
# randomness where it is consequential.
#
# ### Pooled out-of-fold ranking and the default 0.50 probability boundary
#
# For the selected seed-42 model, pooled out-of-fold predictions give:
#
# | Metric | Value |
# |---|---:|
# | PR-AUC | 0.6544 |
# | ROC-AUC | 0.8425 |
# | Accuracy | 0.8030 |
# | Balanced accuracy | 0.7108 |
# | Precision at 0.50 | 0.6670 |
# | Recall at 0.50 | 0.5144 |
# | Specificity at 0.50 | 0.9072 |
# | F1 at 0.50 | 0.5808 |
# | Predicted-positive rate at 0.50 | 0.2047 |
# | Observed churn rate | 0.2654 |
#
# At the default probability threshold of 0.50, the MLP identifies 769 of 1,495
# churners and produces 384 false positives. Its predicted-positive rate is
# below the observed churn rate, so this operating point is relatively selective:
# it prioritizes precision and specificity over recovering every potential
# churner.
#
# The precision-recall curve remains well above the positive-rate baseline of
# 0.265, and the ROC curve is clearly above random ranking. The probability
# histograms also show useful separation: observed churners are shifted toward
# higher predicted probabilities, while observed non-churners are concentrated
# near zero. The distributions still overlap substantially, which is expected in
# a real churn task and explains why no single threshold achieves both very high
# precision and very high recall.
#
# ### Threshold diagnostic: illustrate tradeoffs, do not choose a final threshold
#
# The pooled OOF threshold grid illustrates the operating-point tradeoff:
#
# | Probability threshold | Precision | Recall | Specificity | F1 | Predicted-positive rate |
# |---:|---:|---:|---:|---:|---:|
# | 0.10 | 0.397 | 0.941 | 0.484 | 0.559 | 0.629 |
# | 0.20 | 0.495 | 0.829 | 0.695 | 0.620 | 0.444 |
# | 0.30 | 0.559 | 0.728 | 0.792 | 0.632 | 0.346 |
# | 0.50 | 0.667 | 0.514 | 0.907 | 0.581 | 0.205 |
# | 0.70 | 0.771 | 0.221 | 0.976 | 0.343 | 0.076 |
#
# The highest displayed F1 is 0.632 at threshold 0.30, while the highest
# displayed balanced accuracy is 0.7639 at threshold 0.25. Those observations
# are descriptive only. Choosing one of these thresholds directly from the same
# pooled OOF evidence used to study the model would reuse development evidence
# for another tuning decision. A later finalist workflow must define the
# intervention objective, capacity, and relative cost of false positives and
# false negatives before choosing a threshold through a dedicated training-only
# policy-selection procedure.
#
# ### Calibration diagnostic: encouraging but not final calibration evidence
#
# The pooled OOF Brier score is 0.1362. The mean predicted churn probability is
# 0.2588, close to the observed churn rate of 0.2654. Across the ten
# quantile-based reliability bins, the calibration curve stays close to the
# ideal diagonal, especially in the moderate and high-probability range. There
# are still visible low-to-middle probability deviations, so this is not a
# reason to declare the probabilities fully calibrated for deployment.
#
# No calibration model was fitted in this workflow. Calibration is relevant only
# if the MLP becomes a serious finalist and its probabilities are needed for a
# threshold or value calculation. That later assessment must be performed inside
# a distinct training-only calibration and evaluation procedure.
#
# ### Scoped comparison with reconstructed earlier model representatives
#
# The following rows reconstruct earlier selected configurations and evaluate
# them again under the same current outer-fold procedure:
#
# | Model | Pooled OOF PR-AUC | Pooled OOF ROC-AUC |
# |---|---:|---:|
# | Representative XGBoost | 0.6701 | 0.8498 |
# | Selected bagged trees | 0.6618 | 0.8460 |
# | Selected L2 logistic regression | 0.6584 | 0.8456 |
# | Selected LinearSVC | 0.6565 | 0.8448 |
# | Selected MLPClassifier | 0.6544 | 0.8425 |
# | Selected decision tree | 0.6285 | 0.8237 |
# | Selected kNN | 0.6276 | 0.8361 |
#
# Within this scoped comparison, the MLP improves on the single tree and kNN
# representatives but does not produce the largest observed ranking estimate.
# It is below the representative XGBoost, bagging, L2 logistic regression, and
# LinearSVC rows. The gaps are descriptive rather than formal pairwise tests,
# and this table intentionally omits several earlier candidates, including the
# selected random forest and other boosting implementations. It is therefore not
# a final project-wide ranking.
#
# More broadly, the results provide no evidence that this small tabular problem
# requires a deeper neural representation to match or exceed strong tree
# ensembles and simple linear methods. This is a result about the compact MLP
# configurations and preprocessing evaluated here, not a claim that neural
# networks cannot be useful for tabular churn data under other representations,
# tuning budgets, or data sizes.
#
# ### Development-stage conclusion
#
# This workflow successfully evaluates the MLP model family with fold-safe dense
# preprocessing, a transparent compact candidate screen, early stopping,
# optimization diagnostics, seed checks, pooled OOF ranking curves, calibration
# diagnostics, and threshold tradeoffs. The selected shallow ReLU MLP is a
# coherent development-stage candidate with useful churn ranking and apparently
# reasonable probability behaviour.
#
# The screened extra width, extra depth, and tanh activation did not provide a
# material practical improvement. Consequently, adding dropout, batch
# normalization, custom PyTorch training loops, or much deeper architectures is
# not presently justified merely to enlarge the search. Such extensions should
# be motivated later by a specific finalist-stage question rather than by the
# assumption that a more complex neural network must outperform existing tabular
# models.
#
# No test-set performance is reported or implied here. The held-out test set
# remains reserved for the final frozen end-to-end pipeline only after the
# broader training-only finalist, calibration, and threshold-policy decisions
# have been completed.
#
# %% [markdown]
# ## Section summary
#
# The MLP workflow evaluates a shallow feed-forward neural-network family for
# mixed tabular churn classification. The representative model has one hidden
# ReLU layer with 16 units and L2 regularization alpha=0.001. It was selected by
# mean outer-fold PR-AUC within the compact shallow screen, but nearby 16-unit
# regularization settings are effectively tied.
#
# The observed development evidence does not support adding width or a second
# hidden layer. The selected MLP has useful pooled OOF ranking and promising
# calibration diagnostics, yet it does not establish an advantage over the
# strongest reconstructed tree, linear, or maximum-margin references. It is one
# completed model family in the training-only inventory, not the final model.
