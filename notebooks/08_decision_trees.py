# %% [markdown]
# # 08 Decision Trees
#
# ## Purpose
#
# This notebook evaluates single decision-tree classifiers for the Telco Customer
# Churn project.
#
# The previous learned-model sections introduced:
#
# - logistic regression as a discriminative linear probability model;
# - k-nearest neighbours as a non-parametric distance-based classifier;
# - Naive Bayes as a probabilistic generative classifier.
#
# Decision trees introduce a different modelling idea: recursive partitioning of
# the feature space. A fitted tree sends each customer through a sequence of split
# rules and assigns the customer to a terminal leaf. The leaf then determines the
# predicted class and the predicted churn probability.
#
# The deeper reusable theory is documented in:
#
# ```text
# docs/knowledge_notes/models/08_decision_trees.md
# docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
# ```
#
# This notebook focuses on the executable workflow, cross-validated tree
# experiments, saved artifacts, and result inspection.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used here.
#
# All development-stage results are computed from stratified cross-validation
# inside the training set.
#
# Decision trees do not require numeric feature scaling, because split rules depend
# on feature order and thresholds rather than on Euclidean distances or coefficient
# penalties. However, scikit-learn trees still require a numeric feature matrix.
# Therefore, categorical variables are one-hot encoded and numeric variables are
# median-imputed inside the pipeline.
#
# Cost-complexity pruning is treated as a tree-complexity hyperparameter in this
# notebook. It is selected using training-set cross-validation, together with the
# other tree-complexity controls. This is appropriate for development-stage model
# selection. A nested validation design would be needed for an unbiased estimate of
# the full tune-and-select procedure, but that stricter comparison is deferred to
# the later final model-comparison stage.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
DECISION_TREE_MODEL_COMPARISON_PATH = TABLES_DIR / "decision_tree_model_comparison.csv"
DECISION_TREE_CONFUSION_MATRIX_PATH = TABLES_DIR / "decision_tree_confusion_matrices.csv"
DECISION_TREE_PREPRUNED_GRID_RESULTS_PATH = TABLES_DIR / "decision_tree_prepruned_grid_results.csv"
DECISION_TREE_CCP_ALPHA_RESULTS_PATH = TABLES_DIR / "decision_tree_ccp_alpha_results.csv"
DECISION_TREE_CANDIDATE_RESULTS_PATH = TABLES_DIR / "decision_tree_candidate_results.csv"
DECISION_TREE_SELECTION_SUMMARY_PATH = TABLES_DIR / "decision_tree_selection_summary.csv"
DECISION_TREE_THRESHOLD_RESULTS_PATH = TABLES_DIR / "decision_tree_threshold_results.csv"
DECISION_TREE_FEATURE_IMPORTANCE_PATH = TABLES_DIR / "decision_tree_feature_importance.csv"

DECISION_TREE_PREPRUNED_PR_AUC_FIGURE_PATH = FIGURES_DIR / "decision_tree_prepruned_pr_auc_by_depth.png"
DECISION_TREE_PREPRUNED_BALANCED_ACCURACY_FIGURE_PATH = FIGURES_DIR / "decision_tree_prepruned_balanced_accuracy_by_depth.png"
DECISION_TREE_CCP_ALPHA_FIGURE_PATH = FIGURES_DIR / "decision_tree_ccp_alpha_metrics.png"
DECISION_TREE_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "decision_tree_threshold_tradeoff.png"
DECISION_TREE_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "decision_tree_roc_curve.png"
DECISION_TREE_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "decision_tree_precision_recall_curve.png"
DECISION_TREE_FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "decision_tree_feature_importance.png"
DECISION_TREE_STRUCTURE_FIGURE_PATH = FIGURES_DIR / "decision_tree_selected_structure.png"

output_paths = pd.DataFrame(
    {
        "artifact": [
            "decision_tree_model_comparison",
            "decision_tree_confusion_matrices",
            "decision_tree_prepruned_grid_results",
            "decision_tree_ccp_alpha_results",
            "decision_tree_candidate_results",
            "decision_tree_selection_summary",
            "decision_tree_threshold_results",
            "decision_tree_feature_importance",
            "decision_tree_prepruned_pr_auc_figure",
            "decision_tree_prepruned_balanced_accuracy_figure",
            "decision_tree_ccp_alpha_figure",
            "decision_tree_threshold_figure",
            "decision_tree_roc_curve_figure",
            "decision_tree_precision_recall_curve_figure",
            "decision_tree_feature_importance_figure",
            "decision_tree_structure_figure",
        ],
        "path": [
            DECISION_TREE_MODEL_COMPARISON_PATH,
            DECISION_TREE_CONFUSION_MATRIX_PATH,
            DECISION_TREE_PREPRUNED_GRID_RESULTS_PATH,
            DECISION_TREE_CCP_ALPHA_RESULTS_PATH,
            DECISION_TREE_CANDIDATE_RESULTS_PATH,
            DECISION_TREE_SELECTION_SUMMARY_PATH,
            DECISION_TREE_THRESHOLD_RESULTS_PATH,
            DECISION_TREE_FEATURE_IMPORTANCE_PATH,
            DECISION_TREE_PREPRUNED_PR_AUC_FIGURE_PATH,
            DECISION_TREE_PREPRUNED_BALANCED_ACCURACY_FIGURE_PATH,
            DECISION_TREE_CCP_ALPHA_FIGURE_PATH,
            DECISION_TREE_THRESHOLD_FIGURE_PATH,
            DECISION_TREE_ROC_CURVE_FIGURE_PATH,
            DECISION_TREE_PRECISION_RECALL_CURVE_FIGURE_PATH,
            DECISION_TREE_FEATURE_IMPORTANCE_FIGURE_PATH,
            DECISION_TREE_STRUCTURE_FIGURE_PATH,
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

# %%
target_distribution = (
    y.value_counts(normalize=False)
    .rename("count")
    .to_frame()
    .assign(percentage=lambda df: 100 * df["count"] / df["count"].sum())
    .rename_axis(TARGET_COLUMN)
    .reset_index()
)

target_distribution

# %% [markdown]
# ## Decision-tree theory needed for this notebook
#
# A classification tree recursively partitions the feature space. At each internal
# node, the algorithm chooses a split that makes the child nodes more class-pure
# than the parent node.
#
# For a node containing class proportions \(p_0\) and \(p_1\), the Gini impurity is:
#
# $$
# G = 1 - p_0^2 - p_1^2.
# $$
#
# Entropy is another impurity measure:
#
# $$
# H = -\sum_{k} p_k \log_2(p_k).
# $$
#
# A split is useful when the weighted impurity of the children is lower than the
# impurity of the parent. The tree is built greedily: it chooses the best local
# split at each node rather than globally searching over all possible trees.
#
# The predicted churn probability for a customer is the churn proportion in the
# terminal leaf reached by that customer. Therefore, a tree can be evaluated not
# only by hard classification metrics but also by ranking metrics such as ROC-AUC
# and PR-AUC. The ranking is stepwise because all customers in the same leaf
# receive the same score.

# %% [markdown]
# ## Cross-validation and preprocessing

# %%
cv = make_stratified_kfold()
unscaled_preprocessor = make_unscaled_preprocessor()

cv_check = pd.DataFrame(
    {
        "item": ["strategy", "n_splits", "shuffle", "random_state", "numeric_scaling"],
        "value": ["StratifiedKFold", cv.n_splits, True, RANDOM_STATE, "not used"],
    }
)

cv_check

# %% [markdown]
# ## Helper functions for decision-tree experiments

# %%
def make_decision_tree_pipeline(
    *,
    criterion: str = "gini",
    max_depth: object = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_leaf_nodes: object = None,
    ccp_alpha: float = 0.0,
):
    """Create an unscaled-preprocessing plus decision-tree pipeline.

    Grid-search results are stored in pandas dataframes. Integer hyperparameters
    such as ``max_depth`` can therefore return from a selected row as floats,
    for example ``6.0`` instead of ``6``. Missing values that originally
    represented ``None`` can also return as ``NaN``. scikit-learn validates these
    parameters strictly, so this factory normalizes dataframe-returned values
    before constructing the estimator.
    """
    classifier = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=normalize_optional_positive_int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_leaf_nodes=normalize_optional_positive_int(max_leaf_nodes),
        ccp_alpha=float(ccp_alpha),
        random_state=RANDOM_STATE,
    )

    return make_classifier_pipeline(
        preprocessor=make_unscaled_preprocessor(),
        classifier=classifier,
    )


def is_missing_depth(max_depth: object) -> bool:
    """Return whether a depth value represents an unrestricted tree depth.

    The pre-pruning grid includes ``max_depth=None`` to represent an unrestricted
    depth. When the grid results are stored in a pandas dataframe, ``None`` can be
    converted to ``NaN`` because the column also contains numeric values. The plot
    helper therefore needs to treat both Python ``None`` and pandas/NumPy missing
    values as the same unrestricted-depth setting.
    """
    return max_depth is None or pd.isna(max_depth)


def normalize_optional_positive_int(value: object) -> int | None:
    """Normalize optional integer hyperparameters recovered from pandas rows.

    scikit-learn expects parameters such as ``max_depth`` and ``max_leaf_nodes``
    to be Python integers or ``None``. During grid-result storage, pandas may
    represent unrestricted values as ``NaN`` and integer values as floats. This
    helper converts ``None``/``NaN`` to ``None`` and values such as ``6.0`` to
    ``6``.
    """
    if value is None or pd.isna(value):
        return None
    return int(value)


def depth_label(max_depth: object) -> str:
    """Return a readable label for a tree depth value."""
    if is_missing_depth(max_depth):
        return "None"
    return str(int(max_depth))


def max_depth_sort_value(max_depth: object) -> int:
    """Return a numeric sort value for max_depth, placing None/NaN last."""
    if is_missing_depth(max_depth):
        return 10_000
    return int(max_depth)


def make_metric_columns() -> list[str]:
    """Return the standard metric columns used in display tables."""
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


def select_best_candidate(results_df: pd.DataFrame) -> pd.Series:
    """Select a representative candidate using PR-AUC, then balanced accuracy and F1."""
    return results_df.sort_values(
        ["pr_auc", "balanced_accuracy", "f1"],
        ascending=False,
    ).iloc[0]


def save_tree_grid_plot(
    *,
    results_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    """Save a pre-pruning grid metric plot by depth and leaf-size setting."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_df = results_df.copy()
    plot_df["max_depth_label"] = plot_df["max_depth"].apply(depth_label)
    plot_df["max_depth_sort"] = plot_df["max_depth"].apply(max_depth_sort_value)
    plot_df = plot_df.sort_values(["min_samples_leaf", "max_depth_sort"])

    for min_samples_leaf, group_df in plot_df.groupby("min_samples_leaf"):
        collapsed_df = (
            group_df.groupby(["max_depth_label", "max_depth_sort"], as_index=False)[metric]
            .max()
            .sort_values("max_depth_sort")
        )
        ax.plot(
            collapsed_df["max_depth_label"],
            collapsed_df[metric],
            marker="o",
            label=f"min_samples_leaf={min_samples_leaf}",
        )

    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_ccp_alpha_metric_plot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    metric_columns: list[str] | None = None,
) -> None:
    """Save metrics over a cost-complexity pruning-alpha grid."""
    if metric_columns is None:
        metric_columns = ["pr_auc", "roc_auc", "balanced_accuracy", "f1"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_df = results_df.sort_values("ccp_alpha")

    for metric in metric_columns:
        ax.plot(plot_df["ccp_alpha"], plot_df[metric], marker="o", label=metric)

    positive_alphas = plot_df.loc[plot_df["ccp_alpha"] > 0, "ccp_alpha"]
    if not positive_alphas.empty:
        ax.set_xscale("symlog", linthresh=float(positive_alphas.min()))

    ax.set_xlabel("Cost-complexity pruning alpha")
    ax.set_ylabel("Cross-validated metric")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_preprocessed_feature_names(fitted_preprocessor) -> np.ndarray:
    """Return readable feature names from a fitted ColumnTransformer."""
    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = np.array([f"feature_{i}" for i in range(fitted_preprocessor.transformers_[0][1].shape[1])])

    cleaned_names = []
    for name in feature_names:
        cleaned = str(name)
        cleaned = cleaned.replace("numeric__", "")
        cleaned = cleaned.replace("categorical__", "")
        cleaned_names.append(cleaned)

    return np.array(cleaned_names)


def save_feature_importance_plot(
    *,
    feature_importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar plot of the largest decision-tree feature importances."""
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


def save_selected_tree_plot(
    *,
    fitted_pipeline,
    output_path: Path,
    title: str,
    max_depth: int = 3,
) -> None:
    """Save a readable top-level plot of the selected fitted decision tree."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = get_preprocessed_feature_names(preprocessor)

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=["No churn", "Churn"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        max_depth=max_depth,
        ax=ax,
    )
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def derive_ccp_alpha_candidates(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    max_candidates: int = 12,
) -> list[float]:
    """Derive a compact cost-complexity alpha grid from the training data only.

    The pruning path can contain many possible alpha values. This helper keeps the
    grid compact by taking quantiles of the positive path values and always
    including 0.0 for the unpruned reference tree.
    """
    preprocessor = make_unscaled_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    path_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    pruning_path = path_tree.cost_complexity_pruning_path(X_transformed, y)
    path_alphas = pruning_path.ccp_alphas

    positive_alphas = np.unique(path_alphas[path_alphas > 0])

    if positive_alphas.size == 0:
        return [0.0]

    if positive_alphas.size <= max_candidates - 1:
        selected_positive_alphas = positive_alphas
    else:
        quantiles = np.linspace(0.05, 0.95, max_candidates - 1)
        selected_positive_alphas = np.quantile(positive_alphas, quantiles)

    alpha_candidates = np.unique(np.concatenate([[0.0], selected_positive_alphas]))
    alpha_candidates = np.round(alpha_candidates, decimals=10)
    return sorted(float(alpha) for alpha in alpha_candidates)

# %% [markdown]
# ## Baseline decision-tree models
#
# The first comparison evaluates two simple reference trees:
#
# 1. a **decision stump**, which is a depth-one tree with a single split;
# 2. a **default decision tree**, which is intentionally flexible and therefore
#    likely to overfit if not regularized.
#
# The stump is useful as an interpretable low-capacity tree. The default tree is
# useful as a warning: an unconstrained tree can fit local training irregularities
# too closely.

# %%
stump_pipeline = make_decision_tree_pipeline(max_depth=1)
default_tree_pipeline = make_decision_tree_pipeline()

baseline_tree_estimators = {
    "Decision stump max_depth=1": stump_pipeline,
    "Default decision tree": default_tree_pipeline,
}

baseline_tree_results = []

for model_name, estimator in baseline_tree_estimators.items():
    result = evaluate_estimator_cv(
        model_name=model_name,
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    result["variant_source"] = "baseline_tree"
    result["criterion"] = estimator.named_steps["classifier"].criterion
    result["max_depth"] = estimator.named_steps["classifier"].max_depth
    result["min_samples_split"] = estimator.named_steps["classifier"].min_samples_split
    result["min_samples_leaf"] = estimator.named_steps["classifier"].min_samples_leaf
    result["ccp_alpha"] = estimator.named_steps["classifier"].ccp_alpha
    baseline_tree_results.append(result)

baseline_tree_results_df = pd.DataFrame(baseline_tree_results)
baseline_tree_results_df[make_metric_columns()]

# %% [markdown]
# ## Baseline-tree interpretation
#
# The two baseline trees show why tree complexity must be controlled.
#
# The depth-one stump is extremely simple. At the default threshold, it predicts no
# customers as churners, giving recall \(0.000\), precision \(0.000\), and balanced
# accuracy \(0.500\). However, its ROC-AUC is about \(0.726\) and its PR-AUC is
# about \(0.413\), which shows an important distinction: although the stump is too
# coarse for default-threshold classification, its single split still creates a
# non-random risk ranking.
#
# The default unconstrained tree has the opposite problem. It predicts positives
# at about the observed churn rate, but its ranking performance is weak: ROC-AUC is
# about \(0.648\) and PR-AUC is about \(0.371\). This is worse than the stump by
# PR-AUC and much worse than the tuned tree candidates below. The default tree is
# therefore a useful overfitting warning. A flexible tree can create many small
# leaves and fit local training irregularities, but those detailed splits do not
# necessarily generalize in cross-validation.

# %% [markdown]
# ## Pre-pruned decision-tree grid
#
# Pre-pruning controls tree complexity while the tree is being grown. The grid
# varies:
#
# - the impurity criterion;
# - maximum depth;
# - minimum samples required to split an internal node;
# - minimum samples required in a terminal leaf.
#
# These controls directly affect the bias-variance tradeoff. Shallow trees and
# larger leaves are more stable but less flexible. Deeper trees and smaller leaves
# can model more interactions but are more sensitive to local noise.

# %%
CRITERION_GRID = ["gini", "entropy"]
MAX_DEPTH_GRID = [2, 3, 4, 5, 6, None]
MIN_SAMPLES_SPLIT_GRID = [2, 10, 25]
MIN_SAMPLES_LEAF_GRID = [1, 10, 25, 50]

prepruned_grid_rows = []

for criterion in CRITERION_GRID:
    for max_depth in MAX_DEPTH_GRID:
        for min_samples_split in MIN_SAMPLES_SPLIT_GRID:
            for min_samples_leaf in MIN_SAMPLES_LEAF_GRID:
                if min_samples_split < 2 * min_samples_leaf and min_samples_leaf > 1:
                    # This combination is allowed by scikit-learn, but it is less
                    # interpretable as a split-size/leaf-size pair. Skipping it keeps
                    # the development grid compact and clearer.
                    continue

                model_name = (
                    f"Tree {criterion} depth={depth_label(max_depth)} "
                    f"split={min_samples_split} leaf={min_samples_leaf}"
                )
                estimator = make_decision_tree_pipeline(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    ccp_alpha=0.0,
                )
                result = evaluate_estimator_cv(
                    model_name=model_name,
                    estimator=estimator,
                    X=X,
                    y=y,
                    cv=cv,
                )
                result["variant_source"] = "prepruned_grid"
                result["criterion"] = criterion
                result["max_depth"] = max_depth
                result["min_samples_split"] = min_samples_split
                result["min_samples_leaf"] = min_samples_leaf
                result["ccp_alpha"] = 0.0
                prepruned_grid_rows.append(result)

prepruned_grid_results_df = pd.DataFrame(prepruned_grid_rows)

prepruned_grid_display_df = (
    prepruned_grid_results_df.sort_values(
        ["pr_auc", "balanced_accuracy", "f1"],
        ascending=False,
    )
    .reset_index(drop=True)
)

prepruned_grid_display_df[
    [
        "model",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
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
].head(20)

# %%
save_tree_grid_plot(
    results_df=prepruned_grid_results_df,
    metric="pr_auc",
    output_path=DECISION_TREE_PREPRUNED_PR_AUC_FIGURE_PATH,
    title="Decision-tree pre-pruning grid: PR-AUC by depth",
)

save_tree_grid_plot(
    results_df=prepruned_grid_results_df,
    metric="balanced_accuracy",
    output_path=DECISION_TREE_PREPRUNED_BALANCED_ACCURACY_FIGURE_PATH,
    title="Decision-tree pre-pruning grid: balanced accuracy by depth",
)

prepruned_grid_results_df.to_csv(DECISION_TREE_PREPRUNED_GRID_RESULTS_PATH, index=False)

# %% [markdown]
# ## Pre-pruning grid interpretation
#
# The pre-pruning grid shows that moderate tree complexity works best for ranking
# churners. The strongest candidate by PR-AUC is:
#
# ```text
# Tree gini depth=6 split=25 leaf=10
# ```
#
# Its development-stage cross-validated PR-AUC is about \(0.628\), ROC-AUC about
# \(0.824\), balanced accuracy about \(0.701\), and \(F_1\) about \(0.564\).
#
# The broader pattern is more important than the exact winning row. The best rows
# are concentrated around depths five and six, often with `min_samples_split=25`
# and `min_samples_leaf=10`. This suggests that the tree needs enough depth to
# model interactions among contract, tenure, internet-service, and payment
# variables, but also needs leaf-size constraints to avoid overly noisy terminal
# leaves.
#
# The unrestricted-depth settings are clearly weaker by PR-AUC, especially with
# very small leaves. This confirms the expected bias-variance pattern for single
# trees: unconstrained trees can fit very specific local patterns, but their
# leaf-level churn proportions become unstable and produce weaker out-of-fold
# rankings.
#
# Balanced accuracy tells a slightly different story. Very shallow depth-two trees
# have the highest balanced accuracy in this grid because their default-threshold
# class predictions are more recall-oriented. However, their PR-AUC is much lower,
# around \(0.496\). Since the project uses PR-AUC as the primary development
# ranking metric for churn retrieval, the selected tree is a moderate-depth tree
# rather than the depth-two default-threshold classifier.

# %% [markdown]
# ## Cost-complexity pruning grid
#
# Cost-complexity pruning is a post-pruning method. A large tree is grown first,
# and then subtrees are compared using a complexity penalty. In scikit-learn, the
# pruning strength is controlled by `ccp_alpha`.
#
# The candidate alpha values below are derived from the cost-complexity pruning
# path on the training data only. They are then evaluated using the same
# cross-validation procedure as the other tree settings.
#
# This is still development-stage hyperparameter selection. The selected alpha is
# useful for choosing a representative pruned tree within this section, but the
# winning cross-validated score should not be interpreted as a final independent
# estimate of the whole pruning-search procedure.

# %%
CCP_ALPHA_GRID = derive_ccp_alpha_candidates(X=X, y=y, max_candidates=12)

ccp_alpha_grid_df = pd.DataFrame({"ccp_alpha": CCP_ALPHA_GRID})
ccp_alpha_grid_df

# %%
ccp_alpha_rows = []

for ccp_alpha in CCP_ALPHA_GRID:
    model_name = f"CCP-pruned tree alpha={ccp_alpha:.10g}"
    estimator = make_decision_tree_pipeline(ccp_alpha=ccp_alpha)
    result = evaluate_estimator_cv(
        model_name=model_name,
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    result["variant_source"] = "ccp_alpha_grid"
    result["criterion"] = "gini"
    result["max_depth"] = None
    result["min_samples_split"] = 2
    result["min_samples_leaf"] = 1
    result["ccp_alpha"] = ccp_alpha
    ccp_alpha_rows.append(result)

ccp_alpha_results_df = pd.DataFrame(ccp_alpha_rows)

ccp_alpha_display_df = (
    ccp_alpha_results_df.sort_values(
        ["pr_auc", "balanced_accuracy", "f1"],
        ascending=False,
    )
    .reset_index(drop=True)
)

ccp_alpha_display_df[
    [
        "model",
        "ccp_alpha",
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

# %%
save_ccp_alpha_metric_plot(
    results_df=ccp_alpha_results_df,
    output_path=DECISION_TREE_CCP_ALPHA_FIGURE_PATH,
    title="Cost-complexity pruning grid for decision trees",
)

ccp_alpha_results_df.to_csv(DECISION_TREE_CCP_ALPHA_RESULTS_PATH, index=False)

# %% [markdown]
# ## Cost-complexity pruning interpretation
#
# Cost-complexity pruning improves substantially over the default unpruned tree.
# With `ccp_alpha=0`, the default tree has PR-AUC about \(0.371\), ROC-AUC about
# \(0.648\), and \(F_1\) about \(0.483\). The best cost-complexity candidate in
# this grid uses:
#
# ```text
# ccp_alpha ≈ 0.000698
# ```
#
# and reaches PR-AUC about \(0.615\), ROC-AUC about \(0.822\), balanced accuracy
# about \(0.710\), and \(F_1\) about \(0.577\).
#
# The pruning curve therefore illustrates the intended role of post-pruning: a
# complexity penalty removes unstable branches and greatly improves generalization
# relative to the unpruned tree. However, the best post-pruned tree remains below
# the best pre-pruned tree by PR-AUC in this development grid. For this dataset,
# directly constraining depth and leaf size gives the strongest single-tree
# ranking result among the tried candidates.

# %% [markdown]
# ## Select representative decision-tree candidate
#
# The representative decision-tree candidate is selected from the stump/default
# comparison, the pre-pruned grid, and the cost-complexity pruning grid.
#
# The selection rule is the same style used in earlier model sections:
#
# ```text
# highest development-stage cross-validated PR-AUC
# secondary tie-breakers: balanced accuracy and F1
# ```
#
# This is a development-stage selection rule, not a final performance claim.

# %%
decision_tree_candidate_results_df = pd.concat(
    [
        baseline_tree_results_df,
        prepruned_grid_results_df,
        ccp_alpha_results_df,
    ],
    ignore_index=True,
)

best_tree_row = select_best_candidate(decision_tree_candidate_results_df)

selection_summary = pd.DataFrame(
    {
        "item": [
            "selection_rule",
            "selected_model",
            "selected_variant_source",
            "selected_criterion",
            "selected_max_depth",
            "selected_min_samples_split",
            "selected_min_samples_leaf",
            "selected_ccp_alpha",
            "selected_pr_auc",
            "selected_roc_auc",
            "selected_balanced_accuracy",
            "selected_f1",
        ],
        "value": [
            "highest cross-validated PR-AUC among decision-tree candidates; balanced accuracy and F1 as secondary criteria",
            best_tree_row["model"],
            best_tree_row["variant_source"],
            best_tree_row["criterion"],
            best_tree_row["max_depth"],
            best_tree_row["min_samples_split"],
            best_tree_row["min_samples_leaf"],
            best_tree_row["ccp_alpha"],
            best_tree_row["pr_auc"],
            best_tree_row["roc_auc"],
            best_tree_row["balanced_accuracy"],
            best_tree_row["f1"],
        ],
    }
)

selection_summary

# %%
selected_tree_pipeline = make_decision_tree_pipeline(
    criterion=best_tree_row["criterion"],
    max_depth=best_tree_row["max_depth"],
    min_samples_split=best_tree_row["min_samples_split"],
    min_samples_leaf=best_tree_row["min_samples_leaf"],
    ccp_alpha=best_tree_row["ccp_alpha"],
)

selected_tree_result = evaluate_estimator_cv(
    model_name="Selected decision tree",
    estimator=selected_tree_pipeline,
    X=X,
    y=y,
    cv=cv,
)

selected_tree_result_df = pd.DataFrame([selected_tree_result])
selected_tree_result_df[make_metric_columns()]

# %% [markdown]
# ## Selected-tree interpretation
#
# The representative decision tree selected by the predefined rule is:
#
# ```text
# Tree gini depth=6 split=25 leaf=10
# ```
#
# This is a pre-pruned tree rather than an unconstrained or post-pruned tree. The
# selected settings mean that the tree can learn several levels of interactions,
# but it cannot keep splitting into very small leaves. The selected tree therefore
# represents a moderate-complexity compromise: more expressive than a stump or a
# depth-two tree, but much less variable than a fully grown default tree.
#
# Its development-stage cross-validated metrics are:
#
# ```text
# accuracy ≈ 0.789
# balanced accuracy ≈ 0.701
# precision ≈ 0.624
# recall ≈ 0.514
# specificity ≈ 0.888
# F1 ≈ 0.564
# ROC-AUC ≈ 0.824
# PR-AUC ≈ 0.628
# ```
#
# The selected tree is useful, but it is not the strongest model family so far.
# Its PR-AUC is close to the selected kNN model and higher than the selected Naive
# Bayes model, but still below the logistic-regression benchmark from section 05.
# This supports the interpretation that a single tree captures meaningful churn
# structure, but a single greedy partition is less effective than a well-regularized
# linear probability model for this transformed Telco feature space.

# %% [markdown]
# ## Model comparison tables
#
# The main comparison table keeps a small set of interpretable rows:
#
# - the decision stump;
# - the default tree;
# - the best pre-pruned tree;
# - the best cost-complexity-pruned tree;
# - the selected representative decision-tree candidate.

# %%
best_prepruned_row = select_best_candidate(prepruned_grid_results_df)
best_ccp_row = select_best_candidate(ccp_alpha_results_df)

comparison_rows = []

for row in baseline_tree_results_df.to_dict("records"):
    comparison_rows.append(row)

best_prepruned_record = best_prepruned_row.to_dict()
best_prepruned_record["model"] = "Best pre-pruned decision tree"
comparison_rows.append(best_prepruned_record)

best_ccp_record = best_ccp_row.to_dict()
best_ccp_record["model"] = "Best cost-complexity-pruned decision tree"
comparison_rows.append(best_ccp_record)

selected_record = selected_tree_result.copy()
selected_record["variant_source"] = "selected_representative"
selected_record["criterion"] = best_tree_row["criterion"]
selected_record["max_depth"] = best_tree_row["max_depth"]
selected_record["min_samples_split"] = best_tree_row["min_samples_split"]
selected_record["min_samples_leaf"] = best_tree_row["min_samples_leaf"]
selected_record["ccp_alpha"] = best_tree_row["ccp_alpha"]
comparison_rows.append(selected_record)

decision_tree_model_comparison_df = pd.DataFrame(comparison_rows)

decision_tree_metric_df = (
    decision_tree_model_comparison_df[make_metric_columns()]
    .sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)
    .reset_index(drop=True)
)

decision_tree_confusion_df = (
    make_confusion_matrix_dataframe(decision_tree_model_comparison_df)
    .set_index("model")
    .loc[decision_tree_metric_df["model"]]
    .reset_index()
)

decision_tree_metric_df

# %%
decision_tree_confusion_df

# %%
decision_tree_candidate_results_df.to_csv(DECISION_TREE_CANDIDATE_RESULTS_PATH, index=False)
selection_summary.to_csv(DECISION_TREE_SELECTION_SUMMARY_PATH, index=False)
decision_tree_metric_df.to_csv(DECISION_TREE_MODEL_COMPARISON_PATH, index=False)
decision_tree_confusion_df.to_csv(DECISION_TREE_CONFUSION_MATRIX_PATH, index=False)

# %% [markdown]
# ## Threshold behaviour for the selected tree
#
# A classification tree produces predicted probabilities from leaf class
# proportions. These scores can be thresholded at different levels to trade recall
# against precision and specificity.
#
# Because all observations in the same leaf receive the same score, the threshold
# curve can be stepwise. A shallow tree may have only a small number of distinct
# probability values.

# %%
selected_y_pred_oof, selected_y_score_oof = get_out_of_fold_predictions(
    estimator=selected_tree_pipeline,
    X=X,
    y=y,
    cv=cv,
)

if selected_y_score_oof is None:
    raise ValueError("The selected decision tree did not produce probability scores.")

unique_score_summary = pd.DataFrame(
    {
        "item": [
            "unique_probability_scores",
            "minimum_score",
            "median_score",
            "maximum_score",
        ],
        "value": [
            len(np.unique(np.round(selected_y_score_oof, decimals=10))),
            float(np.min(selected_y_score_oof)),
            float(np.median(selected_y_score_oof)),
            float(np.max(selected_y_score_oof)),
        ],
    }
)

unique_score_summary

# %%
threshold_grid = np.round(np.linspace(0.05, 0.95, 19), 2)
decision_tree_threshold_df = evaluate_threshold_grid(
    y_true=y,
    y_score=selected_y_score_oof,
    thresholds=threshold_grid,
)

decision_tree_threshold_df

# %%
decision_tree_threshold_df.to_csv(DECISION_TREE_THRESHOLD_RESULTS_PATH, index=False)

save_threshold_tradeoff_plot(
    threshold_df=decision_tree_threshold_df,
    output_path=DECISION_TREE_THRESHOLD_FIGURE_PATH,
    title="Threshold tradeoff for the selected decision tree",
)

# %% [markdown]
# ## ROC and precision-recall curves
#
# ROC-AUC and PR-AUC evaluate the ranking induced by the tree's leaf-probability
# scores. The tree is not trained directly to optimize these ranking metrics, so
# they must be inspected explicitly.

# %%
decision_tree_roc_curve_df = make_roc_curve_dataframe(
    y_true=y,
    y_score=selected_y_score_oof,
)

decision_tree_precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=selected_y_score_oof,
)

save_roc_curve_plot(
    roc_curve_df=decision_tree_roc_curve_df,
    output_path=DECISION_TREE_ROC_CURVE_FIGURE_PATH,
    title="ROC curve for the selected decision tree",
)

save_precision_recall_curve_plot(
    precision_recall_curve_df=decision_tree_precision_recall_curve_df,
    output_path=DECISION_TREE_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title="Precision-recall curve for the selected decision tree",
    positive_rate=float(y.mean()),
)

curve_summary = pd.DataFrame(
    {
        "item": ["roc_auc", "pr_auc", "positive_rate_baseline"],
        "value": [
            selected_tree_result["roc_auc"],
            selected_tree_result["pr_auc"],
            float(y.mean()),
        ],
    }
)

curve_summary

# %% [markdown]
# ## Fit selected tree for interpretation
#
# The cross-validated metrics above come from out-of-fold predictions. For
# interpretation, the selected tree is fitted once on the full training set. This
# fitted object is used only to inspect the selected structure and feature
# importances. The resulting tree plot is not a test-set evaluation.

# %%
fitted_selected_tree_pipeline = clone(selected_tree_pipeline)
fitted_selected_tree_pipeline.fit(X, y)

fitted_preprocessor = fitted_selected_tree_pipeline.named_steps["preprocessor"]
fitted_tree = fitted_selected_tree_pipeline.named_steps["classifier"]
feature_names = get_preprocessed_feature_names(fitted_preprocessor)

feature_importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": fitted_tree.feature_importances_,
    }
).sort_values("importance", ascending=False)

feature_importance_df = feature_importance_df[feature_importance_df["importance"] > 0].reset_index(drop=True)
feature_importance_df.to_csv(DECISION_TREE_FEATURE_IMPORTANCE_PATH, index=False)

feature_importance_df.head(30)

# %%
save_feature_importance_plot(
    feature_importance_df=feature_importance_df,
    output_path=DECISION_TREE_FEATURE_IMPORTANCE_FIGURE_PATH,
    title="Selected decision tree: impurity-based feature importance",
    top_n=20,
)

save_selected_tree_plot(
    fitted_pipeline=fitted_selected_tree_pipeline,
    output_path=DECISION_TREE_STRUCTURE_FIGURE_PATH,
    title="Selected decision tree structure, truncated to top levels",
    max_depth=3,
)

# %% [markdown]
# ## Interpretation artifacts
#
# The feature-importance table and tree-structure plot are interpretation aids.
# They should be read cautiously:
#
# - impurity-based feature importance can favour variables with many possible
#   split points or many one-hot indicators;
# - one-hot encoding means that categorical variables appear through individual
#   indicator columns rather than as multiway categorical splits;
# - a single fitted tree can be unstable, so the exact top-level splits should not
#   be treated as universal causal rules.
#
# These artifacts are still useful because decision trees are transparent models:
# they reveal the kinds of threshold and indicator rules that the development
# workflow found useful.

# %% [markdown]
# ## Section summary
#
# This section evaluated single decision trees as transparent recursive
# partitioning classifiers.
#
# The baseline comparison shows two extremes. The decision stump is highly
# interpretable and has non-random ranking signal, but it is too coarse for
# default-threshold classification because it predicts no churners at threshold
# \(0.50\). The default unconstrained tree is more flexible, but its
# cross-validated ranking metrics are weak, indicating overfitting from overly
# detailed local splits.
#
# The pre-pruning grid gives the strongest single-tree candidate. The selected
# tree is a Gini tree with maximum depth \(6\), minimum split size \(25\), and
# minimum leaf size \(10\). It reaches PR-AUC about \(0.628\) and ROC-AUC about
# \(0.824\). Nearby depth-five and depth-six configurations perform similarly, so
# the exact winning row should not be overinterpreted. The robust conclusion is
# that moderately deep, leaf-regularized trees generalize better than either very
# shallow trees or unrestricted trees.
#
# Cost-complexity pruning also works as expected. It improves the default tree
# from PR-AUC about \(0.371\) to about \(0.615\), showing that pruning removes many
# unstable branches. In this grid, however, the best pre-pruned tree remains
# slightly stronger by PR-AUC than the best post-pruned tree.
#
# Threshold analysis shows that the default \(0.50\) threshold is not necessarily
# the best operating point. At threshold \(0.50\), the selected tree has precision
# about \(0.621\) and recall about \(0.514\). Lowering the threshold to around
# \(0.30\)-\(0.35\) substantially increases recall and gives the highest \(F_1\)
# values in the threshold grid, around \(0.618\)-\(0.620\). This is consistent with
# earlier sections: threshold choice is a separate operating decision and should
# be deferred to the later model-comparison stage.
#
# The ranking curves confirm that the selected tree learns useful churn ordering:
# ROC-AUC is about \(0.824\), and PR-AUC is about \(0.632\) on the saved curve
# calculation, well above the positive-rate baseline of about \(0.265\). The curve
# is still below the strongest logistic-regression ranking results from section
# 05.
#
# The fitted-tree interpretation artifacts show that the most important split
# variables are consistent with the EDA and previous models. The largest
# impurity-based importance belongs to `Contract_Month-to-month`, followed by
# `InternetService_Fiber optic`, `tenure`, `TotalCharges`, `MonthlyCharges`, and
# `PaymentMethod_Electronic check`. These importances should be read as descriptive
# model diagnostics rather than causal effects. They are impurity-based, depend on
# the one-hot representation, and come from one fitted tree.
#
# Overall, single decision trees add an important modelling perspective. They are
# interpretable rule-based classifiers, naturally handle nonlinear interactions,
# and form the foundation for later tree ensembles. In this project, the selected
# single tree is useful but not dominant. Final model-family conclusions remain
# deferred until later sections evaluate bagging, random forests, boosting, and
# the final comparison workflow.
