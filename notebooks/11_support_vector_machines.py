# %% [markdown]
# # 11 Support Vector Machines, Maximum-Margin Loss, and Kernel SVMs
#
# ## Purpose
#
# This notebook evaluates support vector machines for the Telco Customer Churn
# project. It follows the tree-ensemble and boosting sections, but returns to a
# fundamentally different classification principle: maximum-margin learning.
#
# A linear score model takes the form
#
# $$
# f(x) = w^\top x + b.
# $$
#
# Logistic regression uses this score inside a sigmoid and optimizes log loss.
# SVMs instead use the score to define a separating hyperplane and choose a
# separator that maximizes the margin around that boundary. Kernel SVMs extend
# the same principle to nonlinear boundaries by fitting a linear separator in an
# implicit transformed feature space.
#
# The detailed technical reference is documented in:
#
# ```text
# docs/knowledge_notes/models/11_support_vector_machines.md
# docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
# ```
#
# The executable work in this notebook covers:
#
# ```text
# - a small kernel-family screen: linear, polynomial, and RBF SVC;
# - a fixed development grid for LinearSVC, including hinge and squared-hinge loss;
# - a fixed development grid for RBF-kernel SVC;
# - optional class weighting within the training-only cross-validation workflow;
# - pooled out-of-fold comparison of selected SVM candidates and earlier references;
# - ranking curves and margin-score threshold behaviour;
# - coefficient interpretation for the selected linear SVM;
# - support-vector diagnostics for the selected RBF SVM.
# ```

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used in this notebook.
#
# Every model-development result is produced inside the training set through the
# project's stratified cross-validation design. The fixed grids are deliberately
# transparent and limited: their purpose is to learn how the main SVM controls
# behave, not to claim that this is the final or exhaustive hyperparameter search.
#
# The selected configurations therefore remain development-stage candidates. Small
# differences in cross-validated metrics must not be interpreted as proof that one
# SVM configuration, kernel, or model family is truly superior. Formal final tuning,
# calibration, threshold selection, and the single held-out test evaluation remain
# later steps in the project.
#
# SVM scores are margins, not probabilities. ROC-AUC and PR-AUC remain valid because
# they assess ranking. The threshold diagnostics below use the raw decision-function
# score, so a threshold of zero is the SVM's natural default decision boundary rather
# than a probability threshold of 0.50.

# %%
from __future__ import annotations

from datetime import datetime
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
from sklearn.neighbors import KNeighborsClassifier

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
    FIGURES_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COLUMN,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    evaluate_estimator_cv,
    evaluate_threshold_grid,
    get_out_of_fold_predictions,
    make_confusion_matrix_dataframe,
    make_precision_recall_curve_dataframe,
    make_roc_curve_dataframe,
    make_stratified_kfold,
)
from telco_churn.features import (  # noqa: E402
    clean_transformed_feature_name,
    get_feature_names_from_preprocessor,
)
from telco_churn.models import (  # noqa: E402
    make_bagging_pipeline,
    make_classifier_pipeline,
    make_decision_tree_pipeline,
    make_l2_logistic_regression_pipeline,
    make_linear_svc_pipeline,
    make_kernel_svc_pipeline,
    make_random_forest_pipeline,
    make_rbf_svc_pipeline,
    make_xgboost_pipeline,
)
from telco_churn.preprocessing import make_scaled_preprocessor  # noqa: E402
from telco_churn.visualization import (  # noqa: E402
    save_coefficient_plot,
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

# This keeps output focused on the notebook's own results while preserving
# convergence warnings, which should remain visible if an SVM fails to optimize.
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Output paths

# %%
SVM_MODEL_COMPARISON_PATH = TABLES_DIR / "svm_model_comparison.csv"
SVM_CONFUSION_MATRIX_PATH = TABLES_DIR / "svm_confusion_matrices.csv"
SVM_KERNEL_SCREEN_RESULTS_PATH = TABLES_DIR / "svm_kernel_screen_results.csv"
SVM_LINEAR_GRID_RESULTS_PATH = TABLES_DIR / "svm_linear_grid_results.csv"
SVM_RBF_GRID_RESULTS_PATH = TABLES_DIR / "svm_rbf_grid_results.csv"
SVM_SELECTION_SUMMARY_PATH = TABLES_DIR / "svm_selection_summary.csv"
SVM_SELECTED_CANDIDATE_RESULTS_PATH = TABLES_DIR / "svm_selected_candidate_results.csv"
SVM_THRESHOLD_RESULTS_PATH = TABLES_DIR / "svm_threshold_results.csv"
SVM_LINEAR_COEFFICIENTS_PATH = TABLES_DIR / "svm_linear_coefficients.csv"
SVM_SUPPORT_VECTOR_SUMMARY_PATH = TABLES_DIR / "svm_support_vector_summary.csv"

SVM_KERNEL_SCREEN_FIGURE_PATH = FIGURES_DIR / "svm_kernel_screen_pr_auc.png"
SVM_LINEAR_GRID_FIGURE_PATH = FIGURES_DIR / "svm_linear_grid_pr_auc.png"
SVM_RBF_GRID_FIGURE_PATH = FIGURES_DIR / "svm_rbf_grid_pr_auc.png"
SVM_MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "svm_model_comparison_pr_auc.png"
SVM_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "svm_margin_score_threshold_tradeoff.png"
SVM_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "svm_roc_curve.png"
SVM_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "svm_precision_recall_curve.png"
SVM_LINEAR_COEFFICIENTS_FIGURE_PATH = FIGURES_DIR / "svm_linear_coefficients.png"

SVM_EXECUTION_LOG_PATH = LOGS_DIR / "11_support_vector_machines_execution.log"
SVM_EXECUTION_LOG_PATH.write_text("", encoding="utf-8")

output_paths = pd.DataFrame(
    {
        "artifact": [
            "svm_model_comparison",
            "svm_confusion_matrices",
            "svm_kernel_screen_results",
            "svm_linear_grid_results",
            "svm_rbf_grid_results",
            "svm_selection_summary",
            "svm_selected_candidate_results",
            "svm_threshold_results",
            "svm_linear_coefficients",
            "svm_support_vector_summary",
            "svm_kernel_screen_figure",
            "svm_linear_grid_figure",
            "svm_rbf_grid_figure",
            "svm_model_comparison_figure",
            "svm_threshold_figure",
            "svm_roc_curve_figure",
            "svm_precision_recall_curve_figure",
            "svm_linear_coefficients_figure",
        ],
        "path": [
            SVM_MODEL_COMPARISON_PATH,
            SVM_CONFUSION_MATRIX_PATH,
            SVM_KERNEL_SCREEN_RESULTS_PATH,
            SVM_LINEAR_GRID_RESULTS_PATH,
            SVM_RBF_GRID_RESULTS_PATH,
            SVM_SELECTION_SUMMARY_PATH,
            SVM_SELECTED_CANDIDATE_RESULTS_PATH,
            SVM_THRESHOLD_RESULTS_PATH,
            SVM_LINEAR_COEFFICIENTS_PATH,
            SVM_SUPPORT_VECTOR_SUMMARY_PATH,
            SVM_KERNEL_SCREEN_FIGURE_PATH,
            SVM_LINEAR_GRID_FIGURE_PATH,
            SVM_RBF_GRID_FIGURE_PATH,
            SVM_MODEL_COMPARISON_FIGURE_PATH,
            SVM_THRESHOLD_FIGURE_PATH,
            SVM_ROC_CURVE_FIGURE_PATH,
            SVM_PRECISION_RECALL_CURVE_FIGURE_PATH,
            SVM_LINEAR_COEFFICIENTS_FIGURE_PATH,
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
# ## SVM implementation choices used here
#
# Reusable `LinearSVC` and kernel-`SVC` factories live in
# `src/telco_churn/models.py`. This notebook therefore focuses on the
# SVM-specific fixed grids, diagnostics, saved artifacts, and interpretation,
# while the same leakage-safe scaled pipelines can later be reconstructed in the
# final model-comparison stage.
#
# The SVM note develops maximum-margin learning with labels in $\{-1,+1\}$ and
# the standard hinge loss
#
# $$
# \ell_i = \max\{0, 1-y_i f(x_i)\}.
# $$
#
# Scikit-learn exposes two relevant linear loss choices through `LinearSVC`:
#
# ```text
# loss="hinge"          standard hinge loss
# loss="squared_hinge"  squared hinge loss, the package default
# ```
#
# The notebook evaluates both deliberately. The squared hinge loss still penalizes
# examples within the margin, but penalizes large violations more heavily. This is
# a package-specific modelling choice, rather than a reason to treat it as a wholly
# separate model family.
#
# The nonlinear SVM uses `SVC(kernel="rbf")`. It does not use `probability=True`.
# The model's `decision_function` values are sufficient for ROC-AUC, PR-AUC, ROC and
# PR curves, and score-threshold diagnostics. Avoiding internal probability fitting
# also keeps the training procedure transparent and avoids an additional internal
# calibration procedure inside each outer cross-validation fit.

# %% [markdown]
# ## Progress logging and evaluation helpers

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
    """Format elapsed seconds for compact progress messages."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"


def log_progress(message: str) -> None:
    """Write a timestamped progress message to stdout and the execution log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with SVM_EXECUTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{line}\n")
        log_file.flush()


def log_section_start(section_name: str) -> float:
    """Log a section start and return the corresponding performance-counter time."""
    log_progress(f"Starting {section_name}")
    return time.perf_counter()


def log_section_end(section_name: str, start_time: float) -> None:
    """Log elapsed time since a section start."""
    elapsed = time.perf_counter() - start_time
    log_progress(f"Finished {section_name} in {format_elapsed(elapsed)}")


def log_preflight_summary() -> None:
    """Write a compact preflight summary before the expensive grids start."""
    log_progress("SVM notebook preflight")
    log_progress(f"Training data shape: rows={X.shape[0]}, columns={X.shape[1]}")
    log_progress(f"CV folds: {cv.get_n_splits(X, y)}")
    log_progress("All model-development results use train.csv only.")
    log_progress("The held-out test set is not touched in this notebook.")
    log_progress(
        "SVM ranking metrics use decision_function scores; score threshold 0 is "
        "the model's natural margin boundary."
    )


log_preflight_summary()


def make_knn_reference_pipeline():
    """Reconstruct the selected kNN reference from section 06."""
    return make_classifier_pipeline(
        preprocessor=make_scaled_preprocessor(),
        classifier=KNeighborsClassifier(
            n_neighbors=101,
            weights="uniform",
            metric="minkowski",
            p=1,
        ),
    )


def evaluate_grid_candidate_cv(
    *,
    model_name: str,
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    extra_columns: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate one fixed-grid candidate with fold-level training and validation scores."""
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


def sort_grid_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Sort fixed-grid candidates by the project selection priorities."""
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
    """Evaluate a transparent fixed grid with progress messages per candidate."""
    rows: list[dict[str, object]] = []
    candidates = list(ParameterGrid(parameter_grid))
    family_start = time.perf_counter()

    log_progress(f"Starting {family_name} grid: {len(candidates)} candidates")

    for candidate_index, params in enumerate(candidates, start=1):
        model_name = model_name_factory(params)
        candidate_start = time.perf_counter()
        estimator = estimator_factory(**params)

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
            f"{family_name} candidate {candidate_index}/{len(candidates)} "
            f"completed in {format_elapsed(candidate_elapsed)} "
            f"(cumulative {format_elapsed(cumulative_elapsed)}): {model_name}"
        )

    log_section_end(f"{family_name} grid", family_start)
    return sort_grid_results(pd.DataFrame(rows))


def make_grid_metric_columns() -> list[str]:
    """Return the compact fixed-grid metrics shown in notebook tables."""
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


def save_top_grid_candidates_plot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 12,
) -> None:
    """Save a horizontal PR-AUC plot for the strongest grid candidates."""
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
    """Save pooled out-of-fold PR-AUC values for selected models."""
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


def extract_linear_svm_coefficients(*, fitted_pipeline) -> pd.DataFrame:
    """Return coefficient directions for a fitted linear-SVM pipeline.

    Unlike logistic-regression coefficients, SVM coefficients are not log-odds
    coefficients and must not be exponentiated into odds ratios. Their sign only
    indicates whether a transformed feature increases or decreases the SVM's
    positive-class decision score, holding other transformed features fixed.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise AttributeError("The fitted classifier does not expose linear coefficients.")

    feature_names = [
        clean_transformed_feature_name(name)
        for name in get_feature_names_from_preprocessor(preprocessor)
    ]
    coefficients = np.ravel(classifier.coef_)

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Number of transformed feature names does not match number of coefficients."
        )

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )
    coefficient_df["direction"] = np.where(
        coefficient_df["coefficient"] >= 0,
        "higher churn decision score",
        "lower churn decision score",
    )

    return coefficient_df.sort_values(
        "absolute_coefficient", ascending=False
    ).reset_index(drop=True)


def make_support_vector_summary(*, fitted_rbf_pipeline, n_training_rows: int) -> pd.DataFrame:
    """Summarize support-vector counts after fitting an RBF SVC on training data.

    The support-vector counts describe the full-training fitted diagnostic model.
    They are not an out-of-fold performance metric and are not used to select the
    final model.
    """
    classifier = fitted_rbf_pipeline.named_steps["classifier"]

    if not hasattr(classifier, "n_support_"):
        raise AttributeError("The fitted classifier does not expose support-vector counts.")

    rows = []
    for class_label, count in zip(classifier.classes_, classifier.n_support_):
        rows.append(
            {
                "class": int(class_label),
                "support_vector_count": int(count),
                "support_vector_share_of_training_rows": float(count / n_training_rows),
            }
        )

    total_count = int(np.sum(classifier.n_support_))
    rows.append(
        {
            "class": "total",
            "support_vector_count": total_count,
            "support_vector_share_of_training_rows": float(total_count / n_training_rows),
        }
    )

    return pd.DataFrame(rows)

# %% [markdown]
# ## Cross-validation and leakage-safe preprocessing
#
# SVMs are not scale invariant. The numeric branch is standardized and the
# categorical branch is one-hot encoded inside each pipeline. Because the
# pipeline is fitted separately inside each cross-validation training fold,
# statistics used for scaling are never learned from the validation fold.

# %%
cv_check = pd.DataFrame(
    {
        "item": ["strategy", "n_splits", "shuffle", "random_state"],
        "value": ["StratifiedKFold", cv.n_splits, True, RANDOM_STATE],
    }
)

cv_check

# %% [markdown]
# ## Kernel-family screen
#
# Before tuning the principal linear and RBF candidates, this screen gives a
# compact view of the role played by the kernel. It is not a final exhaustive
# search for every possible polynomial coefficient or degree.
#
# The screen includes:
#
# ```text
# LinearSVC with standard hinge loss
# LinearSVC with squared-hinge loss
# SVC with a linear kernel
# SVC with a quadratic polynomial kernel
# SVC with an RBF kernel
# ```
#
# The linear SVC and linear-kernel SVC use different solvers and implementation
# details. They should therefore be expected to be close rather than necessarily
# identical. The screen keeps the settings deliberately simple so the main fixed
# grids can focus on the two most relevant development candidates: linear and RBF.

# %%
kernel_screen_estimators = {
    "LinearSVC hinge C=1": make_linear_svc_pipeline(
        C=1.0,
        loss="hinge",
        class_weight=None,
    ),
    "LinearSVC squared hinge C=1": make_linear_svc_pipeline(
        C=1.0,
        loss="squared_hinge",
        class_weight=None,
    ),
    "SVC linear C=1": make_kernel_svc_pipeline(
        C=1.0,
        kernel="linear",
    ),
    "SVC polynomial degree=2 C=1": make_kernel_svc_pipeline(
        C=1.0,
        kernel="poly",
        degree=2,
        gamma="scale",
        coef0=1.0,
    ),
    "SVC RBF C=1 gamma=0.01": make_kernel_svc_pipeline(
        C=1.0,
        kernel="rbf",
        gamma=0.01,
    ),
}

kernel_screen_start = log_section_start("kernel-family screen")
kernel_screen_results_df = pd.DataFrame(
    [
        evaluate_estimator_cv(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )
        for model_name, estimator in kernel_screen_estimators.items()
    ]
).sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)
log_section_end("kernel-family screen", kernel_screen_start)

kernel_screen_results_df.to_csv(SVM_KERNEL_SCREEN_RESULTS_PATH, index=False)
save_model_comparison_plot(
    results_df=kernel_screen_results_df,
    output_path=SVM_KERNEL_SCREEN_FIGURE_PATH,
    title="SVM kernel-family screen: pooled out-of-fold PR-AUC",
)

kernel_screen_results_df[
    [
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
    ]
]

# %% [markdown]
# ## Linear SVM fixed grid
#
# The linear SVM grid varies three controls:
#
# ```text
# C:
#     inverse regularization strength. Smaller C tolerates more margin violations;
#     larger C penalizes violations more strongly.
#
# loss:
#     standard hinge loss versus scikit-learn's squared-hinge default.
#
# class_weight:
#     None versus "balanced". The balanced version increases the effective penalty
#     for misclassifying churn observations relative to non-churn observations.
# ```
#
# This grid is a transparent development-stage screen. It is intentionally not a
# random search or a final nested-CV procedure.

# %%
linear_svm_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "loss": ["hinge", "squared_hinge"],
    "class_weight": [None, "balanced"],
}

linear_svm_grid_results_df = evaluate_parameter_grid(
    family_name="LinearSVC",
    parameter_grid=linear_svm_grid,
    estimator_factory=make_linear_svc_pipeline,
    model_name_factory=lambda p: (
        "LinearSVC "
        f"loss={p['loss']} "
        f"C={p['C']} "
        f"class_weight={p['class_weight']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

linear_svm_grid_results_df.to_csv(SVM_LINEAR_GRID_RESULTS_PATH, index=False)
save_top_grid_candidates_plot(
    results_df=linear_svm_grid_results_df,
    output_path=SVM_LINEAR_GRID_FIGURE_PATH,
    title="LinearSVC grid: top PR-AUC candidates",
)

linear_svm_grid_results_df.head(16)[
    ["model", "C", "loss", "class_weight", *make_grid_metric_columns()[1:]]
]

# %% [markdown]
# ## RBF-kernel SVM fixed grid
#
# The RBF kernel is
#
# $$
# K(x,z)=\exp\left(-\gamma\lVert x-z\rVert^2\right).
# $$
#
# The RBF grid varies:
#
# ```text
# C:
#     the regularization / violation-penalty tradeoff.
#
# gamma:
#     the locality of the RBF influence. Small gamma produces smoother decision
#     regions; large gamma creates more local and potentially higher-variance regions.
#
# class_weight:
#     ordinary versus imbalance-aware fitting.
# ```
#
# The chosen values cover a compact logarithmic range around the scale expected
# after standardization. Very large C or gamma values are deliberately not included
# at this stage because they can create highly flexible boundaries, increase runtime,
# and would broaden the development search without yet serving the educational goal.

# %%
rbf_svm_grid = {
    "C": [0.1, 1.0, 10.0],
    "gamma": [0.001, 0.01, 0.1],
    "class_weight": [None, "balanced"],
}

rbf_svm_grid_results_df = evaluate_parameter_grid(
    family_name="RBF SVC",
    parameter_grid=rbf_svm_grid,
    estimator_factory=make_rbf_svc_pipeline,
    model_name_factory=lambda p: (
        "RBF SVC "
        f"C={p['C']} "
        f"gamma={p['gamma']} "
        f"class_weight={p['class_weight']}"
    ),
    X=X,
    y=y,
    cv=cv,
)

rbf_svm_grid_results_df.to_csv(SVM_RBF_GRID_RESULTS_PATH, index=False)
save_top_grid_candidates_plot(
    results_df=rbf_svm_grid_results_df,
    output_path=SVM_RBF_GRID_FIGURE_PATH,
    title="RBF SVC grid: top PR-AUC candidates",
)

rbf_svm_grid_results_df.head(18)[
    ["model", "C", "gamma", "class_weight", *make_grid_metric_columns()[1:]]
]

# %% [markdown]
# ## Select one linear candidate and one RBF candidate
#
# Each family selects its strongest observed configuration by mean validation
# PR-AUC, then balanced accuracy, then F1. This gives one representative linear
# maximum-margin candidate and one representative nonlinear kernel candidate for
# the pooled out-of-fold comparison below.

# %%
best_linear_svm_row = linear_svm_grid_results_df.iloc[0]
best_rbf_svm_row = rbf_svm_grid_results_df.iloc[0]

selected_linear_svm_params = {
    "C": float(best_linear_svm_row["C"]),
    "loss": str(best_linear_svm_row["loss"]),
    "class_weight": (
        None
        if pd.isna(best_linear_svm_row["class_weight"])
        else str(best_linear_svm_row["class_weight"])
    ),
}
selected_rbf_svm_params = {
    "C": float(best_rbf_svm_row["C"]),
    "gamma": float(best_rbf_svm_row["gamma"]),
    "class_weight": (
        None
        if pd.isna(best_rbf_svm_row["class_weight"])
        else str(best_rbf_svm_row["class_weight"])
    ),
}

selected_linear_svm_pipeline = make_linear_svc_pipeline(**selected_linear_svm_params)
selected_rbf_svm_pipeline = make_rbf_svc_pipeline(**selected_rbf_svm_params)

svm_selection_summary_df = pd.DataFrame(
    {
        "family": ["LinearSVC", "RBF SVC"],
        "selected_model": [best_linear_svm_row["model"], best_rbf_svm_row["model"]],
        "selection_rule": [
            "highest mean CV PR-AUC, then balanced accuracy, then F1",
            "highest mean CV PR-AUC, then balanced accuracy, then F1",
        ],
        "selected_C": [selected_linear_svm_params["C"], selected_rbf_svm_params["C"]],
        "selected_loss": [selected_linear_svm_params["loss"], pd.NA],
        "selected_gamma": [pd.NA, selected_rbf_svm_params["gamma"]],
        "selected_class_weight": [
            selected_linear_svm_params["class_weight"],
            selected_rbf_svm_params["class_weight"],
        ],
        "mean_cv_pr_auc": [
            best_linear_svm_row["test_pr_auc_mean"],
            best_rbf_svm_row["test_pr_auc_mean"],
        ],
        "mean_cv_roc_auc": [
            best_linear_svm_row["test_roc_auc_mean"],
            best_rbf_svm_row["test_roc_auc_mean"],
        ],
        "mean_cv_balanced_accuracy": [
            best_linear_svm_row["test_balanced_accuracy_mean"],
            best_rbf_svm_row["test_balanced_accuracy_mean"],
        ],
        "mean_cv_f1": [
            best_linear_svm_row["test_f1_mean"],
            best_rbf_svm_row["test_f1_mean"],
        ],
    }
)

svm_selection_summary_df.to_csv(SVM_SELECTION_SUMMARY_PATH, index=False)
svm_selection_summary_df

# %% [markdown]
# ## Pooled out-of-fold comparison of selected SVM candidates
#
# The following comparison obtains one out-of-fold score for every training
# observation from each selected SVM candidate. This is a development-stage
# diagnostic, not an independent validation set. It is useful because it places
# the selected linear and RBF SVMs on the same pooled out-of-fold metric scale.

# %%
selected_svm_estimators = [
    (
        "Selected LinearSVC",
        selected_linear_svm_pipeline,
    ),
    (
        "Selected RBF SVC",
        selected_rbf_svm_pipeline,
    ),
]

selected_svm_start = log_section_start("selected SVM candidate comparison")
selected_svm_candidate_results_df = pd.DataFrame(
    [
        evaluate_estimator_cv(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )
        for model_name, estimator in selected_svm_estimators
    ]
).sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)
log_section_end("selected SVM candidate comparison", selected_svm_start)

selected_svm_candidate_results_df.to_csv(
    SVM_SELECTED_CANDIDATE_RESULTS_PATH,
    index=False,
)

selected_svm_candidate_results_df[
    [
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
]

# %% [markdown]
# ## Earlier-model references
#
# The SVM candidates are compared with representative earlier models reconstructed
# from the choices already made in previous model-family sections. They are not
# re-tuned here. The comparison remains development-stage evidence because the
# same training set and cross-validation structure are reused.

# %%
selected_logistic_pipeline = make_l2_logistic_regression_pipeline(C=1.0)
selected_knn_pipeline = make_knn_reference_pipeline()
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
selected_xgboost_pipeline = make_xgboost_pipeline(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=1.0,
)

reference_estimators = [
    ("Selected L2 logistic regression", selected_logistic_pipeline),
    ("Selected kNN", selected_knn_pipeline),
    ("Selected single decision tree", selected_single_tree_pipeline),
    ("Selected bagged trees", selected_bagging_pipeline),
    ("Selected random forest", selected_random_forest_pipeline),
    ("Best XGBoost reference", selected_xgboost_pipeline),
]

reference_start = log_section_start("reference model evaluation")
reference_results_df = pd.DataFrame(
    [
        evaluate_estimator_cv(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )
        for model_name, estimator in reference_estimators
    ]
)
log_section_end("reference model evaluation", reference_start)

svm_model_comparison_df = pd.concat(
    [selected_svm_candidate_results_df, reference_results_df],
    ignore_index=True,
).sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)

svm_model_comparison_df.to_csv(SVM_MODEL_COMPARISON_PATH, index=False)
svm_confusion_matrix_df = make_confusion_matrix_dataframe(svm_model_comparison_df)
svm_confusion_matrix_df.to_csv(SVM_CONFUSION_MATRIX_PATH, index=False)

save_model_comparison_plot(
    results_df=svm_model_comparison_df,
    output_path=SVM_MODEL_COMPARISON_FIGURE_PATH,
    title="Selected SVMs and reference models: pooled out-of-fold PR-AUC",
)

svm_model_comparison_df[
    [
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
]

# %%
svm_confusion_matrix_df

# %% [markdown]
# ## Select the representative SVM for score diagnostics
#
# The selected linear and RBF candidates are ranked by pooled out-of-fold PR-AUC.
# The first model under that ranking becomes the representative SVM for ROC, PR,
# and score-threshold diagnostics. This is a within-section development decision,
# not a final project-wide model decision.

# %%
selected_svm_result_row = selected_svm_candidate_results_df.iloc[0]
selected_svm_name = str(selected_svm_result_row["model"])

if selected_svm_name == "Selected LinearSVC":
    selected_svm_pipeline = selected_linear_svm_pipeline
else:
    selected_svm_pipeline = selected_rbf_svm_pipeline

selected_svm_summary_df = pd.DataFrame(
    {
        "item": [
            "representative_SVM_selection_rule",
            "selected_representative_SVM",
            "pooled_oof_pr_auc",
            "pooled_oof_roc_auc",
            "pooled_oof_balanced_accuracy",
            "pooled_oof_f1",
        ],
        "value": [
            "highest pooled out-of-fold PR-AUC among the selected linear and RBF candidates",
            selected_svm_name,
            selected_svm_result_row["pr_auc"],
            selected_svm_result_row["roc_auc"],
            selected_svm_result_row["balanced_accuracy"],
            selected_svm_result_row["f1"],
        ],
    }
)

selected_svm_summary_df

# %% [markdown]
# ## ROC curve, precision-recall curve, and margin-score threshold behaviour
#
# The SVM decision function produces a signed score. Positive values lie on the
# positive side of the fitted boundary and negative values lie on the negative side.
# The default classifier threshold is therefore score $0$.
#
# Unlike a calibrated probability, the numerical magnitude of an SVM margin score
# has no direct probability interpretation. The threshold grid is still useful for
# seeing the precision/recall/specificity tradeoff, but its horizontal axis must be
# read as a score threshold rather than a probability threshold.

# %%
diagnostics_start = log_section_start("selected SVM ranking and threshold diagnostics")
y_pred_oof, y_score_oof = get_out_of_fold_predictions(
    estimator=selected_svm_pipeline,
    X=X,
    y=y,
    cv=cv,
)

if y_score_oof is None:
    raise RuntimeError("The selected SVM did not expose an out-of-fold decision score.")

svm_roc_curve_df = make_roc_curve_dataframe(y_true=y, y_score=y_score_oof)
svm_precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=y_score_oof,
)

save_roc_curve_plot(
    roc_curve_df=svm_roc_curve_df,
    output_path=SVM_ROC_CURVE_FIGURE_PATH,
    title=f"ROC curve for {selected_svm_name}",
)
save_precision_recall_curve_plot(
    precision_recall_curve_df=svm_precision_recall_curve_df,
    output_path=SVM_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title=f"Precision-recall curve for {selected_svm_name}",
    positive_rate=float(y.mean()),
)

# Quantile-based thresholds spread the diagnostic across the observed score range.
# Score zero is included explicitly because it is the SVM's natural boundary.
score_quantiles = np.linspace(0.05, 0.95, 13)
margin_score_thresholds = np.unique(
    np.concatenate(
        [
            np.quantile(y_score_oof, score_quantiles),
            np.array([0.0]),
        ]
    )
)
margin_score_thresholds.sort()

svm_threshold_results_df = evaluate_threshold_grid(
    y_true=y,
    y_score=y_score_oof,
    thresholds=margin_score_thresholds,
)
svm_threshold_results_df.insert(1, "threshold_scale", "decision_function_score")
svm_threshold_results_df.to_csv(SVM_THRESHOLD_RESULTS_PATH, index=False)

save_threshold_tradeoff_plot(
    threshold_df=svm_threshold_results_df,
    output_path=SVM_THRESHOLD_FIGURE_PATH,
    title=f"Margin-score threshold tradeoff for {selected_svm_name}",
    x_label="Decision-function score threshold",
    reference_threshold=0.0,
    reference_label="Natural SVM boundary (score = 0)",
)
log_section_end("selected SVM ranking and threshold diagnostics", diagnostics_start)

svm_roc_curve_df.head(), svm_precision_recall_curve_df.head()

# %%
svm_threshold_results_df[
    [
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
        "predicted_positive_rate",
    ]
]

# %% [markdown]
# ## Linear-SVM coefficient interpretation
#
# The selected linear SVM is fitted once on the complete training set only to
# inspect its coefficient directions. This fit is not used for performance claims.
#
# A positive coefficient increases the positive-class SVM decision score. A negative
# coefficient decreases it. Unlike logistic regression, these values are not
# log-odds coefficients and must not be transformed into odds ratios.

# %%
linear_interpretation_start = log_section_start("linear SVM coefficient extraction")
fitted_linear_svm_pipeline = clone(selected_linear_svm_pipeline).fit(X, y)
svm_linear_coefficients_df = extract_linear_svm_coefficients(
    fitted_pipeline=fitted_linear_svm_pipeline,
)
svm_linear_coefficients_df.to_csv(SVM_LINEAR_COEFFICIENTS_PATH, index=False)

save_coefficient_plot(
    coefficient_df=svm_linear_coefficients_df,
    output_path=SVM_LINEAR_COEFFICIENTS_FIGURE_PATH,
    title="Selected linear SVM: largest absolute decision-score coefficients",
    top_n=20,
)
log_section_end("linear SVM coefficient extraction", linear_interpretation_start)

svm_linear_coefficients_df.head(20)

# %% [markdown]
# ## RBF support-vector diagnostics
#
# `LinearSVC` is optimized for scalable linear fitting and does not expose an
# explicit support-vector list. `SVC`, including the RBF SVC, does. The next table
# therefore fits the selected RBF candidate on the entire training set and reports
# how many observations become support vectors.
#
# This is a structural diagnostic rather than a performance evaluation. A high
# support-vector share means that many training observations remain close enough to
# the margin, or inside it, to influence the fitted nonlinear decision function.

# %%
support_vector_start = log_section_start("RBF support-vector diagnostics")
fitted_rbf_svm_pipeline = clone(selected_rbf_svm_pipeline).fit(X, y)
svm_support_vector_summary_df = make_support_vector_summary(
    fitted_rbf_pipeline=fitted_rbf_svm_pipeline,
    n_training_rows=len(X),
)
svm_support_vector_summary_df.to_csv(SVM_SUPPORT_VECTOR_SUMMARY_PATH, index=False)
log_section_end("RBF support-vector diagnostics", support_vector_start)

svm_support_vector_summary_df

# %% [markdown]
# ## Interpretation of observed development-stage results
#
# ### Kernel-family screen: no meaningful nonlinear advantage
#
# The compact kernel screen placed all five initial candidates in a narrow pooled
# out-of-fold PR-AUC range from $0.6463$ to $0.6537$:
#
# | Candidate | Pooled OOF PR-AUC | Pooled OOF ROC-AUC |
# |---|---:|---:|
# | LinearSVC, squared hinge, $C=1$ | 0.6537 | 0.8423 |
# | RBF SVC, $C=1$, $\gamma=0.01$ | 0.6522 | 0.8348 |
# | LinearSVC, hinge, $C=1$ | 0.6483 | 0.8336 |
# | Linear-kernel SVC, $C=1$ | 0.6481 | 0.8335 |
# | Quadratic polynomial SVC, $C=1$ | 0.6463 | 0.8185 |
#
# The initial screen therefore did not reveal a material advantage from either
# the polynomial or RBF kernel. The linear implementations were very close to
# one another, which is consistent with the feature space already containing
# one-hot encoded categories and with the principal predictive signal being
# adequately captured by an approximately linear decision rule.
#
# ### Fixed-grid selection: linear and RBF candidates are effectively tied
#
# The strongest observed linear candidate was:
#
# ```text
# LinearSVC(loss="squared_hinge", C=0.1, class_weight="balanced")
# ```
#
# Its mean fold-level validation PR-AUC was $0.6594$ with a fold standard
# deviation of $0.0184$. The strongest observed RBF candidate was:
#
# ```text
# SVC(kernel="rbf", C=10.0, gamma=0.001, class_weight="balanced")
# ```
#
# Its mean fold-level validation PR-AUC was $0.6595$ with a fold standard
# deviation of $0.0161$. The difference is only $0.0001$, far smaller than the
# ordinary fold-to-fold variation observed for either configuration. Under this
# deliberately limited development grid, the results do not establish a
# meaningful nonlinear RBF improvement.
#
# The squared-hinge linear candidates were also notably stable across the tested
# values of $C$. With class weighting enabled, their mean validation PR-AUC
# ranged only from $0.6584$ to $0.6594$ over $C \in \{0.01, 0.1, 1, 10\}$.
# This suggests that the selected solution is not highly sensitive to the
# regularization values examined here. By contrast, the class-weighted standard
# hinge candidates deteriorated as $C$ increased, with PR-AUC declining from
# $0.6582$ at $C=0.01$ to $0.5860$ at $C=10$.
#
# The RBF grid also illustrates the bias-variance role of $\gamma$. The selected
# RBF solution uses a small $\gamma=0.001$, which gives broad, smooth similarity
# regions. At the more local $\gamma=0.1$ setting with $C=10$ and no class
# weighting, mean training PR-AUC rose to $0.8569$ while mean validation PR-AUC
# fell to $0.5722$. This large generalization gap is evidence that a highly local
# RBF boundary can memorize fold-specific structure rather than improve unseen
# churn ranking.
#
# ### Class weighting mainly changes the operating point
#
# Comparing the squared-hinge candidates at $C=0.1$ illustrates the role of
# `class_weight="balanced"`. The weighted and unweighted versions had closely
# similar mean validation PR-AUC values, $0.6594$ and $0.6574$, respectively.
# The weighted configuration nevertheless increased mean recall from $0.5324$
# to $0.8067$ and mean balanced accuracy from $0.7154$ to $0.7648$, while
# decreasing mean precision from $0.6546$ to $0.5127$.
#
# Thus, class weighting did not create a large ranking improvement. It shifted
# the default decision boundary toward identifying more potential churners. That
# is a legitimate design choice when missed churners are relatively costly, but
# the appropriate cost tradeoff has not yet been fixed for this project.
#
# ### Mean fold-level PR-AUC and pooled OOF PR-AUC answer different questions
#
# Grid selection used the mean of PR-AUC values calculated separately in each
# validation fold. This is the pre-specified selection statistic and selected
# the RBF candidate by a negligible margin. The later pooled out-of-fold
# diagnostic instead concatenates the score from every held-out observation and
# calculates one PR-AUC across the combined vector of scores.
#
# For the selected candidates, the pooled diagnostic was:
#
# | Candidate | Pooled OOF PR-AUC | Pooled OOF ROC-AUC | Balanced accuracy | F1 |
# |---|---:|---:|---:|---:|
# | Selected LinearSVC | 0.6565 | 0.8448 | 0.7648 | 0.6268 |
# | Selected RBF SVC | 0.6426 | 0.8383 | 0.7464 | 0.6003 |
#
# These statistics are not expected to match exactly. PR-AUC is nonlinear, and
# raw SVM decision-function scores can have different numerical scales across
# fold-specific fitted models. The pooled result is therefore a descriptive
# diagnostic, not an alternative hyperparameter-selection rule. Taken together,
# the evidence supports treating linear and RBF SVMs as practically close in
# fold-level selection, while preferring the linear SVM as the representative
# SVM for diagnostics because it is faster, interpretable, and stronger on the
# pooled out-of-fold diagnostic.
#
# ### Comparison with earlier selected references
#
# The representative linear SVM was competitive but not a clear leader among the
# earlier selected references:
#
# | Model | Pooled OOF PR-AUC |
# |---|---:|
# | Best XGBoost reference | 0.6701 |
# | Selected bagged trees | 0.6618 |
# | Selected random forest | 0.6602 |
# | Selected L2 logistic regression | 0.6584 |
# | Selected LinearSVC | 0.6565 |
# | Selected RBF SVC | 0.6426 |
# | Selected single decision tree | 0.6285 |
# | Selected kNN | 0.6276 |
#
# The linear SVM is therefore especially informative as a maximum-margin
# alternative to L2 logistic regression. Its ranking performance was very close
# to logistic regression, but its class-weighted default operating point produced
# substantially higher recall and a higher predicted-positive rate. This chart
# is not a project-wide final ranking because it intentionally includes only
# representative earlier models and does not yet include all boosting finalists.
#
# ### Margin-score threshold behaviour
#
# The selected linear SVM uses raw decision-function scores rather than calibrated
# probabilities. At the natural SVM boundary, score $0$, the pooled out-of-fold
# results were:
#
# | Score threshold | Precision | Recall | Specificity | F1 | Predicted-positive rate |
# |---:|---:|---:|---:|---:|---:|
# | 0.000 | 0.5125 | 0.8067 | 0.7229 | 0.6268 | 0.4176 |
#
# Lowering the threshold to approximately $-0.304$ increased recall to $0.9237$
# but reduced precision to $0.4264$. Raising it to approximately $0.455$
# increased precision to $0.6708$ while reducing recall to $0.5057$. The highest
# displayed F1 value, $0.6380$, occurred around score $0.135$.
#
# These values illustrate the available tradeoff only. A raw SVM margin is not a
# churn probability, and the threshold should not be selected here from the same
# out-of-fold diagnostic used to study the model. A later finalist workflow must
# define the business objective and, when probability interpretation is required,
# evaluate calibration and threshold selection using a dedicated training-only
# procedure.
#
# ### Linear coefficient directions and RBF complexity
#
# The selected linear SVM was fitted once on the complete training data for
# structural interpretation only. The largest positive decision-score directions
# were Fiber optic internet service ($+0.2789$), a month-to-month contract
# ($+0.2383$), StreamingMovies=Yes ($+0.1067$), StreamingTV=Yes ($+0.1020$),
# electronic check payment ($+0.0940$), and OnlineSecurity=No ($+0.0825$).
# The strongest negative directions were tenure ($-0.3223$), MonthlyCharges
# ($-0.2445$), DSL internet service ($-0.2388$), and a two-year contract
# ($-0.2227$).
#
# These are conditional score directions in a correlated one-hot encoded feature
# representation. They are neither causal effects nor log-odds effects, and they
# must not be exponentiated into odds ratios. In particular, the negative
# MonthlyCharges coefficient is conditional on tenure, contract type, internet
# service, and the correlated service indicators. It should not be read as a
# marginal claim that higher charges reduce churn.
#
# The selected RBF diagnostic model used 3,062 support vectors, equal to 54.35%
# of the 5,633 training observations. That support-vector share is not a
# performance metric and is not inherently undesirable. It does, however,
# emphasize that the RBF solution is materially more complex than the linear
# representation without delivering a clear fold-level ranking benefit.
#
# ### Execution note
#
# Several non-selected LinearSVC fits emitted `ConvergenceWarning` messages in
# the executed notebook, especially among some stronger-penalty configurations.
# The selected squared-hinge, $C=0.1$, class-weighted configuration did not emit
# a warning in its selected-candidate evaluation or full-training coefficient
# fit. The selected-model interpretation above therefore remains grounded in a
# converged selected workflow, while the non-selected warning-producing
# configurations should not be overinterpreted.

# %% [markdown]
# ## Section summary
#
# This section evaluated linear maximum-margin classification and nonlinear
# kernel SVMs under the same training-only cross-validation discipline used
# throughout the project. The strongest observed linear and RBF configurations
# were effectively tied on mean fold-level validation PR-AUC. The RBF kernel did
# not demonstrate a meaningful nonlinear advantage within the fixed grid.
#
# The representative linear SVM was retained for detailed diagnostics because it
# is substantially faster, interpretable through score coefficients, and stronger
# in the pooled out-of-fold comparison of the two selected SVM candidates. It is
# competitive with the selected L2 logistic, bagging, and random-forest
# references, but it is not a final project-wide winner. The threshold analysis
# uses raw margins rather than probabilities, so calibration and final
# decision-policy selection remain separate later tasks.
