# %% [markdown]
# # 06 k-Nearest Neighbours
#
# ## Purpose
#
# This notebook evaluates k-nearest neighbours (kNN) for the Telco Customer Churn project.
#
# Section 05 introduced learned linear classifiers. kNN is different: it is an
# instance-based, distance-based, non-parametric classifier. Instead of learning a
# small vector of coefficients, it stores the training observations and predicts a
# new customer's class from nearby training customers.
#
# The deeper reusable theory is documented in:
#
# ```text
# docs/knowledge_notes/models/06_knn.md
# docs/knowledge_notes/methodology/evaluation_metrics.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# ```
#
# This notebook focuses on the executable workflow, cross-validated tuning, saved
# artifacts, and result inspection.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used here.
#
# All development-stage results are computed from stratified cross-validation
# inside the training set.
#
# kNN is sensitive to feature scales, so preprocessing must remain inside the
# pipeline. This ensures that scaling and one-hot encoding are fitted only on each
# fold's training data.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from telco_churn.models import make_classifier_pipeline  # noqa: E402
from telco_churn.preprocessing import make_scaled_preprocessor  # noqa: E402
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
KNN_GRID_RESULTS_PATH = TABLES_DIR / "knn_grid_results.csv"
KNN_MODEL_COMPARISON_PATH = TABLES_DIR / "knn_model_comparison.csv"
KNN_CONFUSION_MATRIX_PATH = TABLES_DIR / "knn_confusion_matrices.csv"
KNN_THRESHOLD_RESULTS_PATH = TABLES_DIR / "knn_threshold_results.csv"

KNN_PR_AUC_BY_K_FIGURE_PATH = FIGURES_DIR / "knn_pr_auc_by_k.png"
KNN_BALANCED_ACCURACY_BY_K_FIGURE_PATH = FIGURES_DIR / "knn_balanced_accuracy_by_k.png"
KNN_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "knn_threshold_tradeoff.png"
KNN_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "knn_roc_curve.png"
KNN_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "knn_precision_recall_curve.png"

output_paths = pd.DataFrame(
    {
        "artifact": [
            "knn_grid_results",
            "knn_model_comparison",
            "knn_confusion_matrices",
            "knn_threshold_results",
            "knn_pr_auc_by_k_figure",
            "knn_balanced_accuracy_by_k_figure",
            "knn_threshold_figure",
            "knn_roc_curve_figure",
            "knn_precision_recall_curve_figure",
        ],
        "path": [
            KNN_GRID_RESULTS_PATH,
            KNN_MODEL_COMPARISON_PATH,
            KNN_CONFUSION_MATRIX_PATH,
            KNN_THRESHOLD_RESULTS_PATH,
            KNN_PR_AUC_BY_K_FIGURE_PATH,
            KNN_BALANCED_ACCURACY_BY_K_FIGURE_PATH,
            KNN_THRESHOLD_FIGURE_PATH,
            KNN_ROC_CURVE_FIGURE_PATH,
            KNN_PRECISION_RECALL_CURVE_FIGURE_PATH,
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
# ## kNN theory needed for this notebook
#
# For a customer with feature vector \(x\), kNN finds the \(k\) nearest training
# observations according to a distance function. The unweighted predicted churn
# probability is:
#
# $$
# \hat{p}(Y=1 \mid X=x)
# =
# \frac{1}{k}
# \sum_{i \in \mathcal{N}_k(x)}
# y_i.
# $$
#
# The most important hyperparameter is \(k\). Small \(k\) gives local, flexible,
# high-variance predictions. Large \(k\) gives smoother, higher-bias predictions.
#
# Distances depend strongly on feature scaling. Therefore, this notebook uses the
# scaled preprocessing pipeline.

# %% [markdown]
# ## Cross-validation and preprocessing

# %%
cv = make_stratified_kfold()
scaled_preprocessor = make_scaled_preprocessor()

cv_check = pd.DataFrame(
    {
        "item": ["strategy", "n_splits", "shuffle", "random_state"],
        "value": ["StratifiedKFold", cv.n_splits, True, RANDOM_STATE],
    }
)

cv_check

# %% [markdown]
# ## Helper functions for kNN experiments

# %%
def make_knn_pipeline(
    *,
    n_neighbors: int,
    weights: str,
    p: int,
):
    """Create a scaled-preprocessing plus kNN pipeline."""
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="minkowski",
        p=p,
    )

    return make_classifier_pipeline(
        preprocessor=make_scaled_preprocessor(),
        classifier=classifier,
    )


def distance_name_from_p(p: int) -> str:
    """Return the common distance name for a Minkowski p value."""
    if p == 1:
        return "Manhattan"
    if p == 2:
        return "Euclidean"
    return f"Minkowski p={p}"


def save_knn_metric_by_k_plot(
    *,
    results_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    """Save a kNN grid-search metric plot with one line per weighting/distance setting."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_df = results_df.sort_values("n_neighbors")

    for (weights, p), group_df in plot_df.groupby(["weights", "p"]):
        label = f"{weights}, {distance_name_from_p(int(p))}"
        ax.plot(
            group_df["n_neighbors"],
            group_df[metric],
            marker="o",
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of neighbours, k")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# ## Baseline kNN model
#
# The baseline kNN model uses:
#
# ```text
# n_neighbors = 5
# weights = "uniform"
# p = 2
# ```
#
# This is the standard Euclidean kNN classifier with equal voting weights.

# %%
baseline_knn_pipeline = make_knn_pipeline(
    n_neighbors=5,
    weights="uniform",
    p=2,
)

baseline_knn_result = evaluate_estimator_cv(
    model_name="kNN k=5 uniform Euclidean",
    estimator=baseline_knn_pipeline,
    X=X,
    y=y,
    cv=cv,
)

baseline_knn_result_df = pd.DataFrame([baseline_knn_result])
baseline_knn_result_df[
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
# ## Baseline kNN interpretation
#
# The default kNN model with \(k=5\), uniform weights, and Euclidean distance gives a useful first distance-based benchmark, but it is clearly not the strongest kNN configuration.
#
# The baseline achieves accuracy around \(0.765\), balanced accuracy around \(0.691\), ROC-AUC around \(0.782\), and PR-AUC around \(0.505\). Its recall is about \(0.532\), meaning it detects a little over half of actual churners at the default threshold, while its precision is about \(0.561\).
#
# This is better than a dummy classifier, but it is weaker than the logistic regression models from section 05. The result already suggests that naive local voting with only five neighbours is too variable for this dataset.

# %% [markdown]
# ## kNN hyperparameter grid
#
# This section uses a transparent grid search rather than Optuna.
#
# The grid varies:
#
# - \(k\), the number of neighbours;
# - `weights`, uniform versus distance-weighted voting;
# - `p`, Manhattan versus Euclidean distance.
#
# The validation procedure is still cross-validation inside the training set.

# %%
N_NEIGHBORS_GRID = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
WEIGHTS_GRID = ["uniform", "distance"]
P_GRID = [1, 2]

grid_rows = []

for n_neighbors in N_NEIGHBORS_GRID:
    for weights in WEIGHTS_GRID:
        for p in P_GRID:
            model_name = (
                f"kNN k={n_neighbors} {weights} {distance_name_from_p(p)}"
            )
            estimator = make_knn_pipeline(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p,
            )
            result = evaluate_estimator_cv(
                model_name=model_name,
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
            )
            result["n_neighbors"] = n_neighbors
            result["weights"] = weights
            result["p"] = p
            result["distance"] = distance_name_from_p(p)
            grid_rows.append(result)

knn_grid_results_df = pd.DataFrame(grid_rows)

grid_display_columns = [
    "model",
    "n_neighbors",
    "weights",
    "distance",
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

knn_grid_results_df.sort_values(
    ["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
)[grid_display_columns].head(15)

# %% [markdown]
# ## Grid-search interpretation
#
# The grid search shows a clear pattern: very small values of \(k\) perform poorly, while larger neighbourhoods perform substantially better.
#
# For PR-AUC, performance rises strongly from \(k=1\) to larger values and is best at the largest tested neighbourhoods. The best configuration by PR-AUC is:
#
# ```text
# k = 101
# weights = uniform
# distance = Manhattan
# ```
#
# with PR-AUC about \(0.628\), ROC-AUC about \(0.836\), and balanced accuracy about \(0.722\).
#
# The balanced-accuracy plot shows a slightly different pattern. Balanced accuracy peaks around intermediate-to-large \(k\), especially near \(k=51\) for uniform Manhattan distance, but PR-AUC continues to slightly favour \(k=101\). This difference is useful: PR-AUC evaluates ranking quality over thresholds, while balanced accuracy is tied to the default classification threshold.
#
# Uniform weighting performs better than distance weighting for the best configurations. This suggests that, in this transformed feature space, allowing very close neighbours to dominate does not improve generalization. A smoother average over many neighbours appears more stable.

# %% [markdown]
# ## Select representative kNN model
#
# The first selection rule chooses the kNN configuration with the highest
# cross-validated PR-AUC. PR-AUC is useful because churn is the minority class and
# the quality of positive churn retrieval matters.
#
# After inspecting the outputs, this rule can be reconsidered if it selects a
# model with an undesirable threshold behaviour.

# %%
best_knn_row = knn_grid_results_df.sort_values(
    ["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).iloc[0]

selected_knn_params = {
    "n_neighbors": int(best_knn_row["n_neighbors"]),
    "weights": str(best_knn_row["weights"]),
    "p": int(best_knn_row["p"]),
}

selected_knn_pipeline = make_knn_pipeline(**selected_knn_params)

selection_summary = pd.DataFrame(
    {
        "item": [
            "selection_rule",
            "selected_n_neighbors",
            "selected_weights",
            "selected_distance",
            "selected_pr_auc",
            "selected_roc_auc",
            "selected_balanced_accuracy",
            "selected_f1",
        ],
        "value": [
            "highest cross-validated PR-AUC among kNN grid",
            selected_knn_params["n_neighbors"],
            selected_knn_params["weights"],
            distance_name_from_p(selected_knn_params["p"]),
            best_knn_row["pr_auc"],
            best_knn_row["roc_auc"],
            best_knn_row["balanced_accuracy"],
            best_knn_row["f1"],
        ],
    }
)

selection_summary

# %% [markdown]
# ## Selected kNN model interpretation
#
# Using the predefined selection rule, the selected model is:
#
# ```text
# Selected kNN k=101 uniform Manhattan
# ```
#
# This selected model is not a small-neighbour local classifier. It is a relatively smooth kNN model. That makes sense because smaller \(k\) values appear too noisy in this one-hot encoded tabular feature space.
#
# The selected model's PR-AUC is about \(0.628\), which is well above the positive-rate baseline of about \(0.265\). However, it is still below the logistic regression PR-AUC from section 05, which was around \(0.658\).
#
# This means kNN learns useful churn-ranking structure, but it does not outperform logistic regression on this training-set cross-validation evaluation.

# %% [markdown]
# ## Main kNN comparison table
#
# This table compares the baseline kNN model with the selected kNN model. Later,
# the report should compare the selected kNN model against logistic regression
# from section 05.

# %%
selected_knn_result = evaluate_estimator_cv(
    model_name=(
        "Selected kNN "
        f"k={selected_knn_params['n_neighbors']} "
        f"{selected_knn_params['weights']} "
        f"{distance_name_from_p(selected_knn_params['p'])}"
    ),
    estimator=selected_knn_pipeline,
    X=X,
    y=y,
    cv=cv,
)

knn_comparison_df = (
    pd.DataFrame([baseline_knn_result, selected_knn_result])
    .drop_duplicates(subset=["model"])
    .sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)
    .reset_index(drop=True)
)

knn_metric_columns = [
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

knn_comparison_df[knn_metric_columns]

# %%
knn_confusion_df = make_confusion_matrix_dataframe(knn_comparison_df)
knn_confusion_df

# %% [markdown]
# ## kNN comparison interpretation
#
# The selected kNN model improves materially over the baseline \(k=5\) model.
#
# At the default threshold, the selected model has:
#
# ```text
# accuracy ≈ 0.792
# balanced accuracy ≈ 0.722
# precision ≈ 0.617
# recall ≈ 0.571
# specificity ≈ 0.872
# F1 ≈ 0.593
# ROC-AUC ≈ 0.836
# PR-AUC ≈ 0.628
# ```
#
# The confusion matrix shows:
#
# ```text
# TP = 854
# FN = 641
# FP = 530
# TN = 3609
# ```
#
# Compared with the \(k=5\) baseline, the selected kNN model has more true positives, fewer false negatives, fewer false positives, and more true negatives. So the tuning step improves both churn detection and false-positive control.
#
# However, compared with logistic regression from section 05, kNN is slightly weaker in ranking metrics. Logistic regression had ROC-AUC around \(0.846\) and PR-AUC around \(0.658\), while selected kNN has ROC-AUC around \(0.836\) and PR-AUC around \(0.628\). This suggests that the global weighted feature structure learned by logistic regression is more useful than pure distance-based local similarity for this dataset.

# %% [markdown]
# ## Threshold tradeoff for selected kNN
#
# The selected kNN model gives predicted probabilities from neighbour labels. We
# use out-of-fold probabilities to study threshold behaviour.
#
# This does not choose a final threshold. It only shows how recall, precision, and
# specificity change when the probability cutoff changes.

# %%
_, selected_knn_oof_probability = get_out_of_fold_predictions(
    estimator=selected_knn_pipeline,
    X=X,
    y=y,
    cv=cv,
)

thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

knn_threshold_df = evaluate_threshold_grid(
    y_true=y,
    y_score=selected_knn_oof_probability,
    thresholds=thresholds,
)

knn_threshold_df

# %% [markdown]
# ## Threshold tradeoff interpretation
#
# The selected kNN model has the usual threshold tradeoff.
#
# Lower thresholds produce high recall but many false positives. For example, around threshold \(0.25\), recall is about \(0.862\), but precision is only about \(0.455\), and roughly half of all customers are flagged as churn risks.
#
# At threshold \(0.40\), the model has a more balanced operating point:
#
# ```text
# precision ≈ 0.548
# recall ≈ 0.742
# specificity ≈ 0.779
# F1 ≈ 0.631
# balanced accuracy ≈ 0.761
# ```
#
# At the default threshold \(0.50\), precision improves to about \(0.617\), but recall falls to about \(0.571\). This default threshold is more conservative: it flags fewer customers and misses more churners.
#
# This reinforces the broader project lesson that probability models should not be judged only at threshold \(0.50\). Threshold choice changes the business meaning of the model, but final threshold selection should still wait until later model comparison and should never use the held-out test set.

# %% [markdown]
# ## ROC and precision-recall curves for selected kNN

# %%
knn_roc_curve_df = make_roc_curve_dataframe(
    y_true=y,
    y_score=selected_knn_oof_probability,
)

knn_precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=selected_knn_oof_probability,
)

curve_summary = pd.DataFrame(
    {
        "curve": ["ROC", "Precision-recall"],
        "rows": [len(knn_roc_curve_df), len(knn_precision_recall_curve_df)],
        "baseline_reference": [
            "diagonal random-ranking line",
            f"positive-rate baseline = {y.mean():.4f}",
        ],
    }
)

curve_summary

# %% [markdown]
# ## ROC and precision-recall curve interpretation
#
# The selected kNN ROC curve has AUC about \(0.836\). This is clearly above the random-ranking diagonal and shows that kNN ranks churners above non-churners much better than chance.
#
# The precision-recall curve has AUC about \(0.631\), also well above the positive-rate baseline of about \(0.265\). This confirms that the model has useful positive-class retrieval ability.
#
# However, both ranking metrics are below the section-05 logistic regression model. The gap is not enormous, but it is consistent. kNN is useful and educational here, but logistic regression remains the stronger model so far.
#
# The likely reason is that kNN relies on distances in the scaled and one-hot encoded feature space. In this space, similarity is not necessarily as informative as the learned feature weights from logistic regression.

# %% [markdown]
# ## Save tables and figures

# %%
knn_grid_results_df.to_csv(KNN_GRID_RESULTS_PATH, index=False)
knn_comparison_df[knn_metric_columns].to_csv(KNN_MODEL_COMPARISON_PATH, index=False)
knn_confusion_df.to_csv(KNN_CONFUSION_MATRIX_PATH, index=False)
knn_threshold_df.to_csv(KNN_THRESHOLD_RESULTS_PATH, index=False)

save_knn_metric_by_k_plot(
    results_df=knn_grid_results_df,
    metric="pr_auc",
    output_path=KNN_PR_AUC_BY_K_FIGURE_PATH,
    title="kNN PR-AUC across neighbour counts",
)

save_knn_metric_by_k_plot(
    results_df=knn_grid_results_df,
    metric="balanced_accuracy",
    output_path=KNN_BALANCED_ACCURACY_BY_K_FIGURE_PATH,
    title="kNN balanced accuracy across neighbour counts",
)

knn_threshold_plot_df = knn_threshold_df[knn_threshold_df["threshold"] <= 0.80].copy()

save_threshold_tradeoff_plot(
    threshold_df=knn_threshold_plot_df,
    output_path=KNN_THRESHOLD_FIGURE_PATH,
    title="Selected kNN Threshold Tradeoff",
)

save_roc_curve_plot(
    roc_curve_df=knn_roc_curve_df,
    output_path=KNN_ROC_CURVE_FIGURE_PATH,
    title="Selected kNN ROC Curve",
)

save_precision_recall_curve_plot(
    precision_recall_curve_df=knn_precision_recall_curve_df,
    output_path=KNN_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title="Selected kNN Precision-Recall Curve",
    positive_rate=float(y.mean()),
)

saved_artifacts = pd.DataFrame(
    {
        "artifact": [
            "knn_grid_results",
            "knn_model_comparison",
            "knn_confusion_matrices",
            "knn_threshold_results",
            "knn_pr_auc_by_k_figure",
            "knn_balanced_accuracy_by_k_figure",
            "knn_threshold_figure",
            "knn_roc_curve_figure",
            "knn_precision_recall_curve_figure",
        ],
        "exists": [
            KNN_GRID_RESULTS_PATH.exists(),
            KNN_MODEL_COMPARISON_PATH.exists(),
            KNN_CONFUSION_MATRIX_PATH.exists(),
            KNN_THRESHOLD_RESULTS_PATH.exists(),
            KNN_PR_AUC_BY_K_FIGURE_PATH.exists(),
            KNN_BALANCED_ACCURACY_BY_K_FIGURE_PATH.exists(),
            KNN_THRESHOLD_FIGURE_PATH.exists(),
            KNN_ROC_CURVE_FIGURE_PATH.exists(),
            KNN_PRECISION_RECALL_CURVE_FIGURE_PATH.exists(),
        ],
        "path": [
            KNN_GRID_RESULTS_PATH,
            KNN_MODEL_COMPARISON_PATH,
            KNN_CONFUSION_MATRIX_PATH,
            KNN_THRESHOLD_RESULTS_PATH,
            KNN_PR_AUC_BY_K_FIGURE_PATH,
            KNN_BALANCED_ACCURACY_BY_K_FIGURE_PATH,
            KNN_THRESHOLD_FIGURE_PATH,
            KNN_ROC_CURVE_FIGURE_PATH,
            KNN_PRECISION_RECALL_CURVE_FIGURE_PATH,
        ],
    }
)

saved_artifacts

# %% [markdown]
# ## Section summary
#
# The kNN experiment shows that distance-based classification learns meaningful churn structure, but it does not outperform logistic regression.
#
# The most important conclusions are:
#
# ```text
# - kNN performance is very sensitive to the number of neighbours.
# - Small k values are too noisy for this dataset.
# - Larger neighbourhoods perform better, suggesting that smoothing helps.
# - Uniform weighting outperforms distance weighting in the strongest configurations.
# - Manhattan and Euclidean distances are close, with Manhattan slightly preferred by the selected PR-AUC rule.
# - The selected kNN model improves clearly over the default k=5 baseline.
# - The selected kNN model remains slightly weaker than logistic regression in ROC-AUC and PR-AUC.
# ```
#
# The selected kNN model is useful as a non-parametric comparison model. It demonstrates local, distance-based learning and the role of the bias-variance tradeoff through \(k\). However, logistic regression remains the stronger model family so far for this transformed Telco churn dataset.
#
# The next model family should introduce a different probabilistic assumption or inductive bias, such as Naive Bayes, before moving toward tree-based models.