# %% [markdown]
# # 09 Bagging and Random Forests
#
# Version note: v4 keeps the v3 random-forest feature-importance export fix and
# adds explicit wording about why this section uses transparent fixed grids rather
# than random search, Optuna-style optimization, repeated CV, or nested CV.
#
# ## Purpose
#
# This notebook evaluates bagged decision trees and random forests for the Telco
# Customer Churn project.
#
# Section 08 showed that a single decision tree is useful and interpretable, but
# that an unrestricted tree overfits while a carefully regularized tree performs
# much better. Bagging and random forests build directly on this result. Instead
# of relying on one fitted tree, they average predictions from many trees fitted
# on perturbed versions of the training data. The goal is to reduce variance and
# improve ranking stability while retaining the nonlinear split-based structure
# of tree models.
#
# The deeper reusable theory is documented in:
#
# ```text
# docs/knowledge_notes/models/08_decision_trees.md
# docs/knowledge_notes/models/09_bagging_and_random_forests.md
# docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
# ```
#
# This notebook focuses on the executable workflow, cross-validated ensemble
# experiments, saved artifacts, and result inspection.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used in this notebook.
#
# All model-development results are computed from stratified cross-validation
# inside the training set. The selected configurations should therefore be read
# as development-stage candidates selected within the tried grids, not as final
# test-set performance claims.
#
# The grid-search tables use fold-level cross-validation scores from
# `cross_validate`. The final candidate-comparison table uses pooled out-of-fold
# predictions so that confusion matrices, threshold curves, ROC curves, and
# precision-recall curves can be computed consistently with earlier model
# sections.
#
# Out-of-bag diagnostics are reported only as supplementary training-set
# diagnostics for fitted bagging/random-forest models. They are not used as final
# performance estimates and do not replace the cross-validation results.
#
# Hyperparameter tuning can be organized in several valid ways. Common choices
# include manually specified fixed grids, randomized search over predefined
# distributions, Bayesian or Optuna-style optimization, successive-halving or
# other adaptive search procedures, repeated cross-validation for more stable
# tuning, and nested cross-validation for evaluating a complete tune-and-select
# procedure. This section deliberately uses transparent fixed grids. That choice
# keeps the model-family learning workflow readable: each varied hyperparameter
# has a direct interpretation, the search budget is easy to see, and the figures
# can show how ensemble behaviour changes across a small number of controlled
# settings. It should not be read as a claim that fixed grids are always the best
# tuning strategy. Later, after all model families have been implemented, the
# final comparison stage can use stronger training-only procedures such as
# repeated CV, nested CV, paired comparisons, or more adaptive search for serious
# candidate models.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

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
    compute_binary_classification_metrics,
    evaluate_estimator_cv,
    evaluate_threshold_grid,
    get_out_of_fold_predictions,
    make_confusion_matrix_dataframe,
    make_metric_dataframe,
    make_precision_recall_curve_dataframe,
    make_roc_curve_dataframe,
    make_stratified_kfold,
)
from telco_churn.models import make_classifier_pipeline  # noqa: E402
from telco_churn.preprocessing import make_unscaled_preprocessor  # noqa: E402
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

# %% [markdown]
# ## Output paths

# %%
ENSEMBLE_MODEL_COMPARISON_PATH = TABLES_DIR / "bagging_random_forest_model_comparison.csv"
ENSEMBLE_CONFUSION_MATRIX_PATH = TABLES_DIR / "bagging_random_forest_confusion_matrices.csv"
BAGGING_GRID_RESULTS_PATH = TABLES_DIR / "bagging_grid_results.csv"
RANDOM_FOREST_GRID_RESULTS_PATH = TABLES_DIR / "random_forest_grid_results.csv"
ENSEMBLE_CANDIDATE_RESULTS_PATH = TABLES_DIR / "bagging_random_forest_candidate_results.csv"
ENSEMBLE_SELECTION_SUMMARY_PATH = TABLES_DIR / "bagging_random_forest_selection_summary.csv"
ENSEMBLE_THRESHOLD_RESULTS_PATH = TABLES_DIR / "bagging_random_forest_threshold_results.csv"
RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = TABLES_DIR / "random_forest_feature_importance.csv"
OOB_DIAGNOSTICS_PATH = TABLES_DIR / "bagging_random_forest_oob_diagnostics.csv"

BAGGING_PR_AUC_FIGURE_PATH = FIGURES_DIR / "bagging_pr_auc_by_estimators.png"
BAGGING_BALANCED_ACCURACY_FIGURE_PATH = FIGURES_DIR / "bagging_balanced_accuracy_by_estimators.png"
RANDOM_FOREST_PR_AUC_FIGURE_PATH = FIGURES_DIR / "random_forest_pr_auc_by_depth.png"
RANDOM_FOREST_BALANCED_ACCURACY_FIGURE_PATH = FIGURES_DIR / "random_forest_balanced_accuracy_by_depth.png"
ENSEMBLE_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "bagging_random_forest_threshold_tradeoff.png"
ENSEMBLE_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "bagging_random_forest_roc_curve.png"
ENSEMBLE_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "bagging_random_forest_precision_recall_curve.png"
RANDOM_FOREST_FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "random_forest_feature_importance.png"

output_paths = pd.DataFrame(
    {
        "artifact": [
            "bagging_random_forest_model_comparison",
            "bagging_random_forest_confusion_matrices",
            "bagging_grid_results",
            "random_forest_grid_results",
            "bagging_random_forest_candidate_results",
            "bagging_random_forest_selection_summary",
            "bagging_random_forest_threshold_results",
            "random_forest_feature_importance",
            "bagging_random_forest_oob_diagnostics",
            "bagging_pr_auc_figure",
            "bagging_balanced_accuracy_figure",
            "random_forest_pr_auc_figure",
            "random_forest_balanced_accuracy_figure",
            "bagging_random_forest_threshold_figure",
            "bagging_random_forest_roc_curve_figure",
            "bagging_random_forest_precision_recall_curve_figure",
            "random_forest_feature_importance_figure",
        ],
        "path": [
            ENSEMBLE_MODEL_COMPARISON_PATH,
            ENSEMBLE_CONFUSION_MATRIX_PATH,
            BAGGING_GRID_RESULTS_PATH,
            RANDOM_FOREST_GRID_RESULTS_PATH,
            ENSEMBLE_CANDIDATE_RESULTS_PATH,
            ENSEMBLE_SELECTION_SUMMARY_PATH,
            ENSEMBLE_THRESHOLD_RESULTS_PATH,
            RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
            OOB_DIAGNOSTICS_PATH,
            BAGGING_PR_AUC_FIGURE_PATH,
            BAGGING_BALANCED_ACCURACY_FIGURE_PATH,
            RANDOM_FOREST_PR_AUC_FIGURE_PATH,
            RANDOM_FOREST_BALANCED_ACCURACY_FIGURE_PATH,
            ENSEMBLE_THRESHOLD_FIGURE_PATH,
            ENSEMBLE_ROC_CURVE_FIGURE_PATH,
            ENSEMBLE_PRECISION_RECALL_CURVE_FIGURE_PATH,
            RANDOM_FOREST_FEATURE_IMPORTANCE_FIGURE_PATH,
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
# ## Model factories
#
# Trees and tree ensembles do not need numeric standardization, so the unscaled
# preprocessing pipeline is used. Categorical variables are still one-hot encoded
# because the scikit-learn tree implementations operate on numeric arrays.

# %%
def is_missing_value(value: object) -> bool:
    """Return whether a value is missing in a pandas/NumPy-compatible sense."""
    return value is None or pd.isna(value)


def optional_int(value: object) -> int | None:
    """Convert pandas scalar values to optional integer hyperparameters.

    Grid-search results are stored in pandas dataframes. A column that contains
    both integers and ``None`` can be coerced so that integers appear as floats
    and ``None`` appears as ``NaN``. Scikit-learn tree hyperparameters such as
    ``max_depth`` require either a Python ``int`` or ``None``. This helper keeps
    estimator reconstruction robust after reading values back from dataframe
    rows.
    """
    if is_missing_value(value):
        return None
    return int(value)


def optional_float(value: object) -> float | None:
    """Convert pandas scalar values to optional float hyperparameters."""
    if is_missing_value(value):
        return None
    return float(value)


def make_tree_classifier(
    *,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
    random_state: int = RANDOM_STATE,
) -> DecisionTreeClassifier:
    """Create a decision-tree classifier with normalized hyperparameters."""
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=optional_int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        ccp_alpha=float(ccp_alpha),
        random_state=random_state,
    )


def make_tree_pipeline(
    *,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
) -> object:
    """Create a preprocessing plus single-tree classifier pipeline."""
    return make_classifier_pipeline(
        classifier=make_tree_classifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
        ),
        preprocessor=make_unscaled_preprocessor(),
    )


def make_bagging_classifier(
    *,
    n_estimators: int,
    max_samples: float,
    base_max_depth: int | None,
    base_min_samples_leaf: int,
    oob_score: bool = False,
) -> BaggingClassifier:
    """Create a bagged-tree classifier.

    The constructor supports both newer scikit-learn versions, where the base
    learner argument is called ``estimator``, and older versions, where it is
    called ``base_estimator``.
    """
    base_tree = make_tree_classifier(
        criterion="gini",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf,
        random_state=RANDOM_STATE,
    )

    common_kwargs = {
        "n_estimators": int(n_estimators),
        "max_samples": float(max_samples),
        "bootstrap": True,
        "oob_score": bool(oob_score),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    try:
        return BaggingClassifier(estimator=base_tree, **common_kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_tree, **common_kwargs)


def make_bagging_pipeline(
    *,
    n_estimators: int,
    max_samples: float,
    base_max_depth: int | None,
    base_min_samples_leaf: int,
    oob_score: bool = False,
) -> object:
    """Create a preprocessing plus bagged-tree classifier pipeline."""
    return make_classifier_pipeline(
        classifier=make_bagging_classifier(
            n_estimators=n_estimators,
            max_samples=max_samples,
            base_max_depth=base_max_depth,
            base_min_samples_leaf=base_min_samples_leaf,
            oob_score=oob_score,
        ),
        preprocessor=make_unscaled_preprocessor(),
    )


def make_random_forest_classifier(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    oob_score: bool = False,
) -> RandomForestClassifier:
    """Create a random-forest classifier with normalized hyperparameters."""
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        criterion="gini",
        max_depth=optional_int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=True,
        oob_score=bool(oob_score),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_random_forest_pipeline(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    oob_score: bool = False,
) -> object:
    """Create a preprocessing plus random-forest classifier pipeline."""
    return make_classifier_pipeline(
        classifier=make_random_forest_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score,
        ),
        preprocessor=make_unscaled_preprocessor(),
    )

# %% [markdown]
# ## Evaluation helpers

# %%
cv = make_stratified_kfold()

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


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
    )

    row: dict[str, object] = {"model": model_name}
    if extra_columns:
        row.update(extra_columns)

    for key, values in cv_result.items():
        if key.startswith("train_") or key.startswith("test_"):
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))

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
    ]


def sort_grid_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Sort grid results by the project selection priorities."""
    return results_df.sort_values(
        by=["test_pr_auc_mean", "test_balanced_accuracy_mean", "test_f1_mean"],
        ascending=False,
    ).reset_index(drop=True)


def depth_label(max_depth: object) -> str:
    """Return a readable label for optional depth values."""
    if is_missing_value(max_depth):
        return "None"
    return str(int(max_depth))


def depth_sort_value(max_depth: object) -> int:
    """Return a numeric sort value for optional depth values."""
    if is_missing_value(max_depth):
        return 10_000
    return int(max_depth)


def save_bagging_grid_plot(
    *,
    results_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    """Save a line plot for bagging grid metrics over n_estimators."""
    plot_df = results_df.copy()
    plot_df["base_max_depth_label"] = plot_df["base_max_depth"].apply(depth_label)
    plot_df["base_max_depth_sort"] = plot_df["base_max_depth"].apply(depth_sort_value)
    plot_df = plot_df.sort_values(
        ["base_min_samples_leaf", "max_samples", "base_max_depth_sort", "n_estimators"]
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for (max_samples, base_max_depth, min_leaf), group_df in plot_df.groupby(
        ["max_samples", "base_max_depth_label", "base_min_samples_leaf"],
        dropna=False,
    ):
        label = (
            f"max_samples={max_samples}, "
            f"depth={base_max_depth}, leaf={min_leaf}"
        )
        ax.plot(group_df["n_estimators"], group_df[metric], marker="o", label=label)

    ax.set_xlabel("Number of trees")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_random_forest_grid_plot(
    *,
    results_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    """Save a line plot for random-forest grid metrics over max_depth."""
    plot_df = results_df.copy()
    plot_df["max_depth_label"] = plot_df["max_depth"].apply(depth_label)
    plot_df["max_depth_sort"] = plot_df["max_depth"].apply(depth_sort_value)
    plot_df = plot_df.sort_values(
        ["max_features", "min_samples_leaf", "n_estimators", "max_depth_sort"]
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for (max_features, min_leaf, n_estimators), group_df in plot_df.groupby(
        ["max_features", "min_samples_leaf", "n_estimators"],
        dropna=False,
    ):
        label = f"max_features={max_features}, leaf={min_leaf}, trees={n_estimators}"
        ax.plot(group_df["max_depth_label"], group_df[metric], marker="o", label=label)

    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(
    *,
    feature_importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar plot of random-forest feature importances."""
    plot_df = (
        feature_importance_df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_xlabel("Impurity-based feature importance")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# ## Reference models
#
# The selected single decision tree from Section 08 is included as a reference.
# The default random forest is also evaluated as a baseline ensemble before grid
# search.

# %%
selected_single_tree_pipeline = make_tree_pipeline(
    criterion="gini",
    max_depth=6,
    min_samples_split=25,
    min_samples_leaf=10,
    ccp_alpha=0.0,
)

reference_bagging_pipeline = make_bagging_pipeline(
    n_estimators=100,
    max_samples=1.0,
    base_max_depth=None,
    base_min_samples_leaf=1,
)

default_random_forest_pipeline = make_random_forest_pipeline(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
)

reference_results = [
    evaluate_estimator_cv(
        model_name="Selected single decision tree",
        estimator=selected_single_tree_pipeline,
        X=X,
        y=y,
        cv=cv,
    ),
    evaluate_estimator_cv(
        model_name="Bagged full trees, 100 estimators",
        estimator=reference_bagging_pipeline,
        X=X,
        y=y,
        cv=cv,
    ),
    evaluate_estimator_cv(
        model_name="Default random forest, 100 estimators",
        estimator=default_random_forest_pipeline,
        X=X,
        y=y,
        cv=cv,
    ),
]

reference_results_df = pd.DataFrame(reference_results).sort_values(
    by=["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).reset_index(drop=True)

reference_results_df[make_metric_columns()]

# %% [markdown]
# **Initial interpretation.** The reference table compares one regularized single
# tree with two simple ensemble baselines. The actual interpretation should focus
# on whether averaging many trees improves PR-AUC and ROC-AUC relative to the
# selected single tree, and whether the ensemble changes the default-threshold
# precision-recall tradeoff.

# %% [markdown]
# ## Bagging grid
#
# Bagging is expected to help most when the base learners are unstable. The grid
# therefore includes unrestricted trees and moderately regularized base trees.
# This is a controlled fixed grid rather than an exhaustive optimization over all
# possible bagging settings. The grid varies the number of trees, the bootstrap
# sample size, the base-tree maximum depth, and the base-tree minimum leaf size so
# that each change has a clear interpretation.

# %%
bagging_rows: list[dict[str, object]] = []

for n_estimators in [50, 100, 200]:
    for max_samples in [0.632, 0.8, 1.0]:
        for base_max_depth in [None, 6]:
            for base_min_samples_leaf in [1, 10]:
                model_name = (
                    "Bagging "
                    f"trees={n_estimators} "
                    f"samples={max_samples} "
                    f"depth={depth_label(base_max_depth)} "
                    f"leaf={base_min_samples_leaf}"
                )
                estimator = make_bagging_pipeline(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    base_max_depth=base_max_depth,
                    base_min_samples_leaf=base_min_samples_leaf,
                )
                bagging_rows.append(
                    evaluate_grid_candidate_cv(
                        model_name=model_name,
                        estimator=estimator,
                        X=X,
                        y=y,
                        cv=cv,
                        extra_columns={
                            "n_estimators": n_estimators,
                            "max_samples": max_samples,
                            "base_max_depth": base_max_depth,
                            "base_min_samples_leaf": base_min_samples_leaf,
                        },
                    )
                )

bagging_grid_results_df = sort_grid_results(pd.DataFrame(bagging_rows))
bagging_grid_results_df.to_csv(BAGGING_GRID_RESULTS_PATH, index=False)

bagging_grid_results_df.head(12)[
    [
        "model",
        "n_estimators",
        "max_samples",
        "base_max_depth",
        "base_min_samples_leaf",
        *make_grid_metric_columns()[1:],
    ]
]

# %%
save_bagging_grid_plot(
    results_df=bagging_grid_results_df,
    metric="test_pr_auc_mean",
    output_path=BAGGING_PR_AUC_FIGURE_PATH,
    title="Bagged trees: cross-validated PR-AUC",
)

save_bagging_grid_plot(
    results_df=bagging_grid_results_df,
    metric="test_balanced_accuracy_mean",
    output_path=BAGGING_BALANCED_ACCURACY_FIGURE_PATH,
    title="Bagged trees: cross-validated balanced accuracy",
)

# %% [markdown]
# The bagging grid shows that averaging many trees substantially improves the
# ranking performance of the single decision tree from section 08. The best
# development-grid bagging configuration is:
#
# ```text
# Bagging trees=200 samples=0.8 depth=6 leaf=1
# ```
#
# Its mean cross-validated PR-AUC is about 0.668 and its mean ROC-AUC is about
# 0.847. The corresponding training PR-AUC is about 0.745, which is higher than
# the validation estimate but much less extreme than an unrestricted single tree.
# This is consistent with the role of bagging: it reduces variance by averaging
# many bootstrap-fitted trees, but it does not remove all model complexity.
#
# The figure shows that the number of trees mainly stabilizes the ensemble rather
# than creating a large performance jump across 50, 100, and 200 trees. The best
# grid point uses 200 trees, but nearby configurations are close enough that this
# should be interpreted as a representative strong setting within the tried grid,
# not as proof that exactly 200 trees is uniquely optimal.

# %% [markdown]
# ## Random-forest grid
#
# Random forests add feature subsampling to bagging. This decorrelates the trees:
# each split considers only a random subset of features, so the ensemble is less
# likely to build many nearly identical trees around the same dominant predictors.
# This is again a controlled fixed grid rather than a claim of exhaustive random-
# forest optimization. The grid varies the number of trees, maximum depth, minimum
# leaf size, and the number of candidate features considered at each split.

# %%
random_forest_rows: list[dict[str, object]] = []

for n_estimators in [100, 200]:
    for max_depth in [None, 6, 10]:
        for min_samples_leaf in [1, 10]:
            for max_features in ["sqrt", "log2"]:
                model_name = (
                    "Random forest "
                    f"trees={n_estimators} "
                    f"depth={depth_label(max_depth)} "
                    f"leaf={min_samples_leaf} "
                    f"max_features={max_features}"
                )
                estimator = make_random_forest_pipeline(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                )
                random_forest_rows.append(
                    evaluate_grid_candidate_cv(
                        model_name=model_name,
                        estimator=estimator,
                        X=X,
                        y=y,
                        cv=cv,
                        extra_columns={
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                        },
                    )
                )

random_forest_grid_results_df = sort_grid_results(pd.DataFrame(random_forest_rows))
random_forest_grid_results_df.to_csv(RANDOM_FOREST_GRID_RESULTS_PATH, index=False)

random_forest_grid_results_df.head(12)[
    [
        "model",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        *make_grid_metric_columns()[1:],
    ]
]

# %%
save_random_forest_grid_plot(
    results_df=random_forest_grid_results_df,
    metric="test_pr_auc_mean",
    output_path=RANDOM_FOREST_PR_AUC_FIGURE_PATH,
    title="Random forests: cross-validated PR-AUC",
)

save_random_forest_grid_plot(
    results_df=random_forest_grid_results_df,
    metric="test_balanced_accuracy_mean",
    output_path=RANDOM_FOREST_BALANCED_ACCURACY_FIGURE_PATH,
    title="Random forests: cross-validated balanced accuracy",
)

# %% [markdown]
# The best random-forest configuration in this development grid is:
#
# ```text
# Random forest trees=200 depth=10 leaf=10 max_features=sqrt
# ```
#
# It reaches mean cross-validated PR-AUC about 0.664 and mean ROC-AUC about
# 0.847. Its balanced accuracy and F1 are very close to the best bagged-tree
# configuration, but its PR-AUC is slightly lower in the grid summary.
#
# The grid does not show a large advantage from feature subsampling on this
# dataset. The strongest random forest and the strongest bagged-tree ensemble are
# close. This should be read as development-stage evidence that both tree-ensemble
# variants are competitive, not as statistical proof that one variant is truly
# better. A plausible interpretation is that ordinary bagging already captures
# most of the gain from averaging tree predictions here. The Telco data has a few
# dominant churn predictors, so forcing each split to consider only a subset of
# features does not necessarily add much beyond bootstrap averaging in this grid.

# %% [markdown]
# ## Select representative ensemble candidates
#
# The primary selection metric is PR-AUC, with balanced accuracy and F1 used as
# secondary criteria. This follows the earlier sections because churn is the
# minority class and positive-class retrieval matters.

# %%
best_bagging_row = bagging_grid_results_df.iloc[0]
best_random_forest_row = random_forest_grid_results_df.iloc[0]

best_bagging_pipeline = make_bagging_pipeline(
    n_estimators=int(best_bagging_row["n_estimators"]),
    max_samples=float(best_bagging_row["max_samples"]),
    base_max_depth=optional_int(best_bagging_row["base_max_depth"]),
    base_min_samples_leaf=int(best_bagging_row["base_min_samples_leaf"]),
)

best_random_forest_pipeline = make_random_forest_pipeline(
    n_estimators=int(best_random_forest_row["n_estimators"]),
    max_depth=optional_int(best_random_forest_row["max_depth"]),
    min_samples_leaf=int(best_random_forest_row["min_samples_leaf"]),
    max_features=best_random_forest_row["max_features"],
)

selection_summary_df = pd.DataFrame(
    [
        {
            "selected_variant": "best_bagging_grid_candidate",
            "model": best_bagging_row["model"],
            "n_estimators": best_bagging_row["n_estimators"],
            "max_samples": best_bagging_row["max_samples"],
            "base_max_depth": depth_label(best_bagging_row["base_max_depth"]),
            "base_min_samples_leaf": best_bagging_row["base_min_samples_leaf"],
            "test_pr_auc_mean": best_bagging_row["test_pr_auc_mean"],
            "test_roc_auc_mean": best_bagging_row["test_roc_auc_mean"],
            "test_balanced_accuracy_mean": best_bagging_row[
                "test_balanced_accuracy_mean"
            ],
            "test_f1_mean": best_bagging_row["test_f1_mean"],
        },
        {
            "selected_variant": "best_random_forest_grid_candidate",
            "model": best_random_forest_row["model"],
            "n_estimators": best_random_forest_row["n_estimators"],
            "max_depth": depth_label(best_random_forest_row["max_depth"]),
            "min_samples_leaf": best_random_forest_row["min_samples_leaf"],
            "max_features": best_random_forest_row["max_features"],
            "test_pr_auc_mean": best_random_forest_row["test_pr_auc_mean"],
            "test_roc_auc_mean": best_random_forest_row["test_roc_auc_mean"],
            "test_balanced_accuracy_mean": best_random_forest_row[
                "test_balanced_accuracy_mean"
            ],
            "test_f1_mean": best_random_forest_row["test_f1_mean"],
        },
    ]
)

selection_summary_df.to_csv(ENSEMBLE_SELECTION_SUMMARY_PATH, index=False)
selection_summary_df

# %% [markdown]
# ## Candidate comparison using pooled out-of-fold predictions
#
# The grid tables above are useful for hyperparameter selection, but the report
# also needs confusion matrices, threshold diagnostics, and curves. For those, the
# strongest bagging and random-forest candidates are evaluated again with pooled
# out-of-fold predictions.

# %%
candidate_estimators = [
    (
        "Selected single decision tree",
        selected_single_tree_pipeline,
    ),
    (
        "Best bagged trees",
        best_bagging_pipeline,
    ),
    (
        "Default random forest, 100 estimators",
        default_random_forest_pipeline,
    ),
    (
        "Best random forest",
        best_random_forest_pipeline,
    ),
]

candidate_results = [
    evaluate_estimator_cv(model_name=name, estimator=estimator, X=X, y=y, cv=cv)
    for name, estimator in candidate_estimators
]

candidate_results_df = pd.DataFrame(candidate_results)
model_comparison_df = candidate_results_df.sort_values(
    by=["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).reset_index(drop=True)

confusion_matrix_df = (
    make_confusion_matrix_dataframe(model_comparison_df)
    .set_index("model")
    .loc[model_comparison_df["model"]]
    .reset_index()
)

model_comparison_df.to_csv(ENSEMBLE_MODEL_COMPARISON_PATH, index=False)
confusion_matrix_df.to_csv(ENSEMBLE_CONFUSION_MATRIX_PATH, index=False)
candidate_results_df.to_csv(ENSEMBLE_CANDIDATE_RESULTS_PATH, index=False)

model_comparison_df[make_metric_columns()]

# %%
confusion_matrix_df

# %% [markdown]
# The candidate comparison confirms the main ensemble result. Both tuned
# ensembles improve materially over the selected single decision tree in ranking
# metrics:
#
# ```text
# Selected single decision tree:
#     ROC-AUC about 0.824
#     PR-AUC about 0.628
#
# Best bagged trees:
#     ROC-AUC about 0.846
#     PR-AUC about 0.662
#
# Best random forest:
#     ROC-AUC about 0.847
#     PR-AUC about 0.660
# ```
#
# The selected bagging and random-forest candidates are therefore stronger
# ranking models than the single tree, while still using the same basic
# tree-partitioning principle. The improvement is especially clear in PR-AUC,
# which is the most important ranking metric in this imbalanced churn setting.
#
# The two tuned ensemble candidates are very close to each other. The bagged-tree
# candidate has the highest pooled out-of-fold PR-AUC, while the random forest has
# a slightly higher pooled ROC-AUC, accuracy, precision, specificity, balanced
# accuracy, and F1. These small differences should not be overinterpreted. For
# this section, the best bagged trees are used as the representative highest
# PR-AUC ensemble, and the best random forest is retained as an important close
# comparator and interpretation model.

# %% [markdown]
# ## Select model for threshold and curve diagnostics
#
# The highest-PR-AUC ensemble from the candidate-comparison table is used for
# threshold, ROC, and precision-recall diagnostics. Feature importance is reported
# separately for the best random forest, even if the bagged-tree ensemble has the
# highest PR-AUC, because random forests expose a direct impurity-importance
# vector and are a central model family in this section.

# %%
best_ensemble_name = model_comparison_df.iloc[0]["model"]

if best_ensemble_name == "Best random forest":
    selected_ensemble_pipeline = best_random_forest_pipeline
    selected_ensemble_type = "random_forest"
elif best_ensemble_name == "Default random forest, 100 estimators":
    selected_ensemble_pipeline = default_random_forest_pipeline
    selected_ensemble_type = "random_forest"
elif best_ensemble_name == "Best bagged trees":
    selected_ensemble_pipeline = best_bagging_pipeline
    selected_ensemble_type = "bagging"
else:
    selected_ensemble_pipeline = selected_single_tree_pipeline
    selected_ensemble_type = "single_tree"

selected_ensemble_name = best_ensemble_name
selected_ensemble_name, selected_ensemble_type

# %%
y_pred_oof, y_score_oof = get_out_of_fold_predictions(
    estimator=selected_ensemble_pipeline,
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
            "selected_ensemble_name": selected_ensemble_name,
            "selected_ensemble_type": selected_ensemble_type,
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
thresholds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70])

threshold_results_df = evaluate_threshold_grid(
    y_true=y,
    y_score=y_score_oof,
    thresholds=thresholds,
)
threshold_results_df.to_csv(ENSEMBLE_THRESHOLD_RESULTS_PATH, index=False)

save_threshold_tradeoff_plot(
    threshold_df=threshold_results_df,
    output_path=ENSEMBLE_THRESHOLD_FIGURE_PATH,
    title=f"Threshold tradeoff for {selected_ensemble_name}",
)

threshold_results_df

# %% [markdown]
# The threshold curve again shows the usual churn-detection tradeoff. At the
# default 0.50 threshold, the selected bagged-tree ensemble is relatively
# conservative: precision is about 0.662, recall is about 0.504, and specificity
# is about 0.907.
#
# Lowering the threshold increases recall but creates more false positives. For
# example:
#
# ```text
# threshold = 0.25:
#     recall about 0.817
#     precision about 0.505
#     specificity about 0.711
#     F1 about 0.625
#
# threshold = 0.20:
#     recall about 0.864
#     precision about 0.476
#     specificity about 0.656
#     F1 about 0.614
# ```
#
# This is useful for understanding the score distribution, but no final threshold
# is selected here. Threshold selection is a model-selection decision and is
# deferred to the later final comparison and decision-policy stage.

# %% [markdown]
# ## ROC and precision-recall curves

# %%
roc_curve_df = make_roc_curve_dataframe(y_true=y, y_score=y_score_oof)
precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=y_score_oof,
)

save_roc_curve_plot(
    roc_curve_df=roc_curve_df,
    output_path=ENSEMBLE_ROC_CURVE_FIGURE_PATH,
    title=f"ROC curve for {selected_ensemble_name}",
)

save_precision_recall_curve_plot(
    precision_recall_curve_df=precision_recall_curve_df,
    output_path=ENSEMBLE_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title=f"Precision-recall curve for {selected_ensemble_name}",
    positive_rate=float(y.mean()),
)

curve_summary_df = pd.DataFrame(
    [
        {
            "model": selected_ensemble_name,
            "roc_auc": roc_auc_score(y, y_score_oof),
            "pr_auc": average_precision_score(y, y_score_oof),
            "positive_rate_baseline": float(y.mean()),
        }
    ]
)

curve_summary_df

# %% [markdown]
# ## Out-of-bag diagnostics
#
# Out-of-bag predictions are available because bootstrap sampling leaves some
# training observations out of each tree's bootstrap sample. These predictions are
# useful as an internal ensemble diagnostic. They are not treated as final
# performance estimates in this project, because the main development comparison
# uses stratified cross-validation.

# %%
def compute_oob_diagnostics(
    *,
    model_name: str,
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, object]:
    """Fit an ensemble with OOB enabled and compute OOB diagnostics."""
    fitted = clone(estimator).fit(X, y)
    classifier = fitted.named_steps["classifier"]

    row: dict[str, object] = {
        "model": model_name,
        "oob_score_accuracy": getattr(classifier, "oob_score_", np.nan),
    }

    oob_decision_function = getattr(classifier, "oob_decision_function_", None)
    if oob_decision_function is not None:
        oob_score = np.asarray(oob_decision_function)[:, 1]
        valid_mask = ~np.isnan(oob_score)
        row["oob_valid_observations"] = int(valid_mask.sum())
        row["oob_pr_auc"] = average_precision_score(y[valid_mask], oob_score[valid_mask])
        row["oob_roc_auc"] = roc_auc_score(y[valid_mask], oob_score[valid_mask])
    else:
        row["oob_valid_observations"] = 0
        row["oob_pr_auc"] = np.nan
        row["oob_roc_auc"] = np.nan

    return row


best_bagging_oob_pipeline = make_bagging_pipeline(
    n_estimators=int(best_bagging_row["n_estimators"]),
    max_samples=float(best_bagging_row["max_samples"]),
    base_max_depth=optional_int(best_bagging_row["base_max_depth"]),
    base_min_samples_leaf=int(best_bagging_row["base_min_samples_leaf"]),
    oob_score=True,
)

best_random_forest_oob_pipeline = make_random_forest_pipeline(
    n_estimators=int(best_random_forest_row["n_estimators"]),
    max_depth=optional_int(best_random_forest_row["max_depth"]),
    min_samples_leaf=int(best_random_forest_row["min_samples_leaf"]),
    max_features=best_random_forest_row["max_features"],
    oob_score=True,
)

oob_diagnostics_df = pd.DataFrame(
    [
        compute_oob_diagnostics(
            model_name="Best bagged trees",
            estimator=best_bagging_oob_pipeline,
            X=X,
            y=y,
        ),
        compute_oob_diagnostics(
            model_name="Best random forest",
            estimator=best_random_forest_oob_pipeline,
            X=X,
            y=y,
        ),
    ]
)

oob_diagnostics_df.to_csv(OOB_DIAGNOSTICS_PATH, index=False)
oob_diagnostics_df

# %% [markdown]
# ## Random-forest feature importance
#
# Impurity-based feature importance is extracted from a final training-set refit
# of the best random forest. This refit is used only for interpretation, not for
# estimating performance. The best bagged-tree ensemble is still used for the
# threshold and curve diagnostics when it is the highest-PR-AUC candidate.

# %%
fitted_rf_pipeline = clone(best_random_forest_pipeline).fit(X, y)
preprocessor = fitted_rf_pipeline.named_steps["preprocessor"]
classifier = fitted_rf_pipeline.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
cleaned_feature_names = [name.split("__", 1)[-1] for name in feature_names]

feature_importance_df = pd.DataFrame(
    {
        "feature": cleaned_feature_names,
        "importance": classifier.feature_importances_,
    }
).sort_values("importance", ascending=False)

feature_importance_df.to_csv(RANDOM_FOREST_FEATURE_IMPORTANCE_PATH, index=False)
save_feature_importance_plot(
    feature_importance_df=feature_importance_df,
    output_path=RANDOM_FOREST_FEATURE_IMPORTANCE_FIGURE_PATH,
    title="Best random forest: impurity-based feature importance",
)

feature_importance_df.head(20)

# %% [markdown]
# The feature-importance table is computed for the best random forest, even
# though the highest-PR-AUC representative ensemble in this run is the bagged-tree
# model. This gives an ensemble-level interpretability diagnostic for the
# random-forest family without changing the performance comparison.
#
# These importances are impurity-based split importances. They show which
# transformed features were useful for reducing node impurity across the forest.
# They should not be interpreted as causal effects, and they can be biased toward
# variables with more split opportunities. Their role here is descriptive:
# comparing the variables emphasized by the forest with earlier EDA, logistic
# regression coefficients, Naive Bayes behaviour, and the selected single tree.

# %% [markdown]
# ## Saved artifacts

# %%
saved_artifacts = output_paths.copy()
saved_artifacts["exists"] = saved_artifacts["path"].apply(lambda path: Path(path).exists())
saved_artifacts

# %% [markdown]
# ## Section summary
#
# Bagging and random forests improve the single-tree results from section 08.
# The selected single decision tree had PR-AUC about 0.628 and ROC-AUC about
# 0.824, while the best bagged-tree ensemble reaches pooled out-of-fold PR-AUC
# about 0.662 and ROC-AUC about 0.846. This confirms the main modelling
# motivation for bagging: averaging many tree predictors reduces the instability
# of one fitted tree and produces smoother, more useful ranking scores.
#
# The best bagging configuration in the fixed development grid is:
#
# ```text
# n_estimators = 200
# max_samples = 0.8
# base_max_depth = 6
# base_min_samples_leaf = 1
# ```
#
# The best random forest is:
#
# ```text
# n_estimators = 200
# max_depth = 10
# min_samples_leaf = 10
# max_features = sqrt
# ```
#
# The random forest is extremely close to the bagged-tree candidate. It has
# slightly higher pooled ROC-AUC, accuracy, precision, specificity, balanced
# accuracy, and F1, while the bagged-tree candidate has slightly higher pooled
# PR-AUC. Because these differences are small and come from development-stage
# cross-validation, they should be interpreted cautiously.
#
# Compared with previous model families, the best ensemble reaches approximately
# the same ROC-AUC level as logistic regression and slightly improves PR-AUC over
# the earlier logistic-regression benchmark. This makes ensembles serious
# candidates for the later final comparison stage, while logistic regression
# remains the simpler and more interpretable benchmark.
#
# The threshold analysis shows that the default threshold is conservative and
# gives precision above 0.66 but recall near 0.50. Lower thresholds can recover
# many more churners, but at the cost of more false positives. This reinforces
# the project-wide rule that threshold selection should be handled later as a
# separate decision-policy problem.
#
# The next section studies boosting. Boosting is a natural continuation because
# it also combines trees, but it does so sequentially rather than independently:
# later weak learners focus on correcting errors or gradients from earlier
# learners. This allows the project to compare variance-reduction ensembles
# against sequential error-correction ensembles.
