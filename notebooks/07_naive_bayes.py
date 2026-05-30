# %% [markdown]
# # 07 Naive Bayes
#
# ## Purpose
#
# This notebook evaluates Naive Bayes classifiers for the Telco Customer Churn
# project.
#
# The previous learned-model sections introduced:
#
# - logistic regression as a discriminative linear probability model;
# - k-nearest neighbours as a non-parametric distance-based classifier.
#
# Naive Bayes introduces a different modelling philosophy. It is a probabilistic
# generative classifier: it models class priors and class-conditional feature
# likelihoods, then uses Bayes' rule to form posterior class probabilities.
#
# The deeper reusable theory is documented in:
#
# ```text
# docs/knowledge_notes/models/07_naive_bayes.md
# docs/knowledge_notes/methodology/evaluation_metrics.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# ```
#
# This notebook focuses on the executable workflow, model variants, cross-validated
# outputs, and result inspection.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used here.
#
# All development-stage results are computed from stratified cross-validation
# inside the training set.
#
# Naive Bayes variants require model-specific preprocessing:
#
# - GaussianNB is used with numeric features and continuous Gaussian likelihoods.
# - BernoulliNB is used with one-hot encoded categorical features.
# - Hybrid Gaussian-Bernoulli Naive Bayes combines Gaussian numeric likelihoods
#   with Bernoulli one-hot categorical likelihoods.
# - A full transformed GaussianNB variant is retained as a simple comparison, even
#   though Gaussian likelihoods are not theoretically ideal for one-hot indicators.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

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
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
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
from telco_churn.models import (  # noqa: E402
    make_classifier_pipeline,
    make_hybrid_gaussian_bernoulli_nb_classifier,
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

# %% [markdown]
# ## Output paths

# %%
NAIVE_BAYES_MODEL_COMPARISON_PATH = TABLES_DIR / "naive_bayes_model_comparison.csv"
NAIVE_BAYES_CONFUSION_MATRIX_PATH = TABLES_DIR / "naive_bayes_confusion_matrices.csv"
BERNOULLI_NB_ALPHA_RESULTS_PATH = TABLES_DIR / "bernoulli_nb_alpha_results.csv"
NAIVE_BAYES_THRESHOLD_RESULTS_PATH = TABLES_DIR / "naive_bayes_threshold_results.csv"

BERNOULLI_NB_ALPHA_FIGURE_PATH = FIGURES_DIR / "bernoulli_nb_alpha_metrics.png"
NAIVE_BAYES_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "naive_bayes_threshold_tradeoff.png"
NAIVE_BAYES_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "naive_bayes_roc_curve.png"
NAIVE_BAYES_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "naive_bayes_precision_recall_curve.png"

output_paths = pd.DataFrame(
    {
        "artifact": [
            "naive_bayes_model_comparison",
            "naive_bayes_confusion_matrices",
            "bernoulli_nb_alpha_results",
            "naive_bayes_threshold_results",
            "bernoulli_nb_alpha_figure",
            "naive_bayes_threshold_figure",
            "naive_bayes_roc_curve_figure",
            "naive_bayes_precision_recall_curve_figure",
        ],
        "path": [
            NAIVE_BAYES_MODEL_COMPARISON_PATH,
            NAIVE_BAYES_CONFUSION_MATRIX_PATH,
            BERNOULLI_NB_ALPHA_RESULTS_PATH,
            NAIVE_BAYES_THRESHOLD_RESULTS_PATH,
            BERNOULLI_NB_ALPHA_FIGURE_PATH,
            NAIVE_BAYES_THRESHOLD_FIGURE_PATH,
            NAIVE_BAYES_ROC_CURVE_FIGURE_PATH,
            NAIVE_BAYES_PRECISION_RECALL_CURVE_FIGURE_PATH,
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
            "numeric_features",
            "categorical_features",
        ],
        "value": [
            train_df.shape[0],
            train_df.shape[1],
            TARGET_COLUMN,
            y.mean(),
            int(train_df.isna().sum().sum()),
            len(NUMERIC_FEATURES),
            len(CATEGORICAL_FEATURES),
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
# ## Naive Bayes theory needed for this notebook
#
# The ideal Bayes classifier predicts the class with the largest true posterior
# probability:
#
# $$
# h^\star(x)
# =
# \arg\max_{y \in \{0,1\}}
# P(Y=y \mid X=x).
# $$
#
# The true posterior is unknown. Naive Bayes approximates it by estimating the
# class prior and class-conditional likelihoods:
#
# $$
# P(Y=y),
# \qquad
# P(X=x \mid Y=y).
# $$
#
# The "naive" assumption is conditional independence:
#
# $$
# P(X=x \mid Y=y)
# =
# \prod_{j=1}^{p}
# P(X_j=x_j \mid Y=y).
# $$
#
# This assumption is not literally true for the Telco features, but it can still
# produce useful classification and ranking performance.

# %% [markdown]
# ## Cross-validation

# %%
cv = make_stratified_kfold()

cv_check = pd.DataFrame(
    {
        "item": ["strategy", "n_splits", "shuffle", "random_state"],
        "value": ["StratifiedKFold", cv.n_splits, True, RANDOM_STATE],
    }
)

cv_check

# %% [markdown]
# ## Helper functions for Naive Bayes preprocessing
#
# Different Naive Bayes variants need different feature representations.
#
# GaussianNB expects dense continuous features. BernoulliNB works naturally with
# binary indicator features, so it is suitable for one-hot encoded categorical
# variables.

# %%
def make_one_hot_encoder_for_nb(*, sparse_output: bool):
    """Create a one-hot encoder compatible with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_output)


def to_dense_array(X):
    """Convert sparse matrices to dense arrays for estimators that require dense input."""
    if sparse.issparse(X):
        return X.toarray()
    return X


def make_numeric_only_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numeric-only GaussianNB."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_categorical_onehot_preprocessor(*, sparse_output: bool = True) -> ColumnTransformer:
    """Create one-hot preprocessing for categorical-only BernoulliNB."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder_for_nb(sparse_output=sparse_output)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_full_onehot_preprocessor() -> ColumnTransformer:
    """Create full-feature preprocessing with numeric columns followed by one-hot columns.

    The resulting matrix has the numeric features first and the one-hot encoded
    categorical features after them. This ordering is required by the hybrid
    Gaussian-Bernoulli Naive Bayes classifier.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder_for_nb(sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_full_dense_onehot_preprocessor() -> Pipeline:
    """Create dense full-feature preprocessing for GaussianNB.

    Numeric features are median-imputed and categorical features are one-hot
    encoded. The combined matrix is converted to dense form because GaussianNB
    does not accept sparse matrices.
    """
    return Pipeline(
        steps=[
            ("columns", make_full_onehot_preprocessor()),
            ("dense", FunctionTransformer(to_dense_array, accept_sparse=True)),
        ]
    )


def make_gaussian_numeric_nb_pipeline() -> Pipeline:
    """Create numeric-only GaussianNB pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_numeric_only_preprocessor(),
        classifier=GaussianNB(),
    )


def make_bernoulli_categorical_nb_pipeline(*, alpha: float = 1.0) -> Pipeline:
    """Create categorical-only BernoulliNB pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_categorical_onehot_preprocessor(sparse_output=True),
        classifier=BernoulliNB(alpha=alpha),
    )


def make_gaussian_full_nb_pipeline() -> Pipeline:
    """Create full transformed GaussianNB pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_full_dense_onehot_preprocessor(),
        classifier=GaussianNB(),
    )


def make_hybrid_gaussian_bernoulli_nb_pipeline(
    *,
    alpha: float = 1.0,
    var_smoothing: float = 1e-9,
) -> Pipeline:
    """Create hybrid Gaussian numeric plus Bernoulli categorical Naive Bayes pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_full_onehot_preprocessor(),
        classifier=make_hybrid_gaussian_bernoulli_nb_classifier(
            n_numeric_features=len(NUMERIC_FEATURES),
            alpha=alpha,
            var_smoothing=var_smoothing,
        ),
    )


def save_alpha_metric_plot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    metric_columns: list[str] | None = None,
) -> None:
    """Save BernoulliNB smoothing metrics over alpha."""
    if metric_columns is None:
        metric_columns = ["pr_auc", "roc_auc", "balanced_accuracy", "f1"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_df = results_df.sort_values("alpha")

    for metric in metric_columns:
        ax.plot(plot_df["alpha"], plot_df[metric], marker="o", label=metric)

    ax.set_xscale("log")
    ax.set_xlabel("Additive smoothing alpha")
    ax.set_ylabel("Cross-validated metric")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# ## Naive Bayes model variants
#
# We evaluate four transparent variants:
#
# 1. **GaussianNB numeric only**: uses the three numeric features.
# 2. **BernoulliNB categorical only**: uses one-hot encoded categorical features.
# 3. **Hybrid Gaussian-BernoulliNB**: uses Gaussian likelihoods for numeric
#    features and Bernoulli likelihoods for one-hot categorical indicators.
# 4. **GaussianNB full transformed**: uses numeric features plus one-hot encoded
#    categorical indicators as a simple full-feature GaussianNB comparison.
#
# The hybrid model is the most natural Naive Bayes specification for this mixed
# feature space. The full transformed GaussianNB variant is retained as a useful
# benchmark, but it is not the cleanest theoretical likelihood for one-hot
# indicators.

# %%
nb_estimators = {
    "GaussianNB numeric only": make_gaussian_numeric_nb_pipeline(),
    "BernoulliNB categorical only alpha=1": make_bernoulli_categorical_nb_pipeline(alpha=1.0),
    "Hybrid Gaussian-BernoulliNB alpha=1": make_hybrid_gaussian_bernoulli_nb_pipeline(alpha=1.0),
    "GaussianNB full transformed": make_gaussian_full_nb_pipeline(),
}

nb_results = []

for model_name, estimator in nb_estimators.items():
    result = evaluate_estimator_cv(
        model_name=model_name,
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    nb_results.append(result)

nb_results_df = pd.DataFrame(nb_results)

nb_metric_columns = [
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

nb_metric_df = (
    nb_results_df[nb_metric_columns]
    .sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)
    .reset_index(drop=True)
)

nb_metric_df

# %%
nb_confusion_df = (
    make_confusion_matrix_dataframe(nb_results_df)
    .set_index("model")
    .loc[nb_metric_df["model"]]
    .reset_index()
)

nb_confusion_df

# %% [markdown]
# ## Naive Bayes variant interpretation
#
# After adding the hybrid model, the Naive Bayes comparison becomes more theoretically complete.
#
# The strongest Naive Bayes variant by cross-validated PR-AUC is now:
#
# ```text
# Hybrid Gaussian-BernoulliNB alpha=1
# ```
#
# Its main metrics are:
#
# ```text
# accuracy ≈ 0.727
# balanced accuracy ≈ 0.753
# precision ≈ 0.491
# recall ≈ 0.809
# specificity ≈ 0.697
# F1 ≈ 0.611
# ROC-AUC ≈ 0.822
# PR-AUC ≈ 0.615
# ```
#
# This is the most natural Naive Bayes model for the mixed Telco feature space, because it uses Gaussian likelihoods for the numeric features and Bernoulli likelihoods for one-hot encoded categorical indicators.
#
# The hybrid model improves over the previous full transformed GaussianNB model. The old full GaussianNB treated one-hot binary indicators as if they were continuous Gaussian variables. It achieved PR-AUC around \(0.605\), while the hybrid model improves PR-AUC to about \(0.615\). The improvement is not enormous, but it is directionally correct and theoretically cleaner.
#
# The categorical-only BernoulliNB model remains close, with PR-AUC around \(0.596\), showing that the categorical variables contain most of the Naive Bayes signal. The numeric-only GaussianNB model has higher ordinary accuracy, but weaker recall and weaker ranking performance. This means numeric variables alone are useful, but they miss much of the churn structure captured by categorical service and contract features.

# %% [markdown]
# ## BernoulliNB smoothing grid
#
# BernoulliNB uses additive smoothing. Smoothing prevents zero probabilities and
# controls how strongly rare indicator patterns affect the likelihood.
#
# We tune alpha only for the categorical-only BernoulliNB model.

# %%
ALPHA_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

alpha_results = []

for alpha in ALPHA_GRID:
    estimator = make_bernoulli_categorical_nb_pipeline(alpha=alpha)
    result = evaluate_estimator_cv(
        model_name=f"BernoulliNB categorical only alpha={alpha}",
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    result["alpha"] = alpha
    alpha_results.append(result)

bernoulli_alpha_results_df = pd.DataFrame(alpha_results).sort_values("alpha").reset_index(drop=True)

alpha_display_columns = [
    "alpha",
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

bernoulli_alpha_results_df[alpha_display_columns]

# %% [markdown]
# ## Smoothing-grid interpretation
#
# The BernoulliNB smoothing grid is almost flat across the tested alpha values.
#
# PR-AUC stays near \(0.596\), ROC-AUC near \(0.815\), and balanced accuracy near \(0.751\). This means the categorical-only BernoulliNB model is not very sensitive to additive smoothing in this dataset.
#
# That is useful because it suggests that the categorical one-hot likelihoods are stable and not dominated by rare-category zero-probability problems.
#
# The smoothing grid is still useful pedagogically, but it does not materially change model selection here. The best overall Naive Bayes candidate is the hybrid Gaussian-Bernoulli model, not a differently smoothed categorical-only Bernoulli model.

# %% [markdown]
# ## Select representative Naive Bayes model
#
# The representative Naive Bayes model is selected by cross-validated PR-AUC, with
# balanced accuracy and F1 as secondary tie-breakers. This is the same selection
# logic used in the kNN section.

# %%
candidate_rows = []

# Main untuned variants.
for _, row in nb_results_df.iterrows():
    candidate = row.to_dict()
    candidate["variant_source"] = "main_variant"
    candidate["alpha"] = np.nan
    candidate_rows.append(candidate)

# Bernoulli smoothing variants.
for _, row in bernoulli_alpha_results_df.iterrows():
    candidate = row.to_dict()
    candidate["variant_source"] = "bernoulli_alpha_grid"
    candidate_rows.append(candidate)

nb_candidate_results_df = pd.DataFrame(candidate_rows)

best_nb_row = nb_candidate_results_df.sort_values(
    ["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).iloc[0]

selection_summary = pd.DataFrame(
    {
        "item": [
            "selection_rule",
            "selected_model",
            "selected_variant_source",
            "selected_alpha",
            "selected_pr_auc",
            "selected_roc_auc",
            "selected_balanced_accuracy",
            "selected_f1",
        ],
        "value": [
            "highest cross-validated PR-AUC among Naive Bayes candidates",
            best_nb_row["model"],
            best_nb_row["variant_source"],
            best_nb_row["alpha"],
            best_nb_row["pr_auc"],
            best_nb_row["roc_auc"],
            best_nb_row["balanced_accuracy"],
            best_nb_row["f1"],
        ],
    }
)

selection_summary

# %% [markdown]
# ## Selected Naive Bayes model interpretation
#
# Using the predefined selection rule, the representative Naive Bayes model is:
#
# ```text
# Hybrid Gaussian-BernoulliNB alpha=1
# ```
#
# It is selected because it has the highest cross-validated PR-AUC among the Naive Bayes candidates, approximately \(0.615\).
#
# This is an important correction compared with the earlier version of this section. The selected model is now also the most theoretically natural Naive Bayes specification for this dataset:
#
# ```text
# numeric features       -> Gaussian likelihoods
# one-hot categorical features -> Bernoulli likelihoods
# ```
#
# The selected Naive Bayes model is still not the strongest model family so far. Logistic regression and kNN both had stronger PR-AUC and ROC-AUC. However, the hybrid Naive Bayes model gives a distinct operating profile: high recall, moderate precision, and relatively broad churn flagging.

# %% [markdown]
# ## Recreate selected Naive Bayes pipeline

# %%
selected_model_name = str(best_nb_row["model"])

if selected_model_name.startswith("GaussianNB numeric only"):
    selected_nb_pipeline = make_gaussian_numeric_nb_pipeline()
elif selected_model_name.startswith("GaussianNB full transformed"):
    selected_nb_pipeline = make_gaussian_full_nb_pipeline()
elif selected_model_name.startswith("Hybrid Gaussian-BernoulliNB"):
    selected_alpha = float(best_nb_row["alpha"])
    if np.isnan(selected_alpha):
        selected_alpha = 1.0
    selected_nb_pipeline = make_hybrid_gaussian_bernoulli_nb_pipeline(alpha=selected_alpha)
elif selected_model_name.startswith("BernoulliNB categorical only"):
    selected_alpha = float(best_nb_row["alpha"])
    if np.isnan(selected_alpha):
        selected_alpha = 1.0
    selected_nb_pipeline = make_bernoulli_categorical_nb_pipeline(alpha=selected_alpha)
else:
    raise ValueError(f"Unknown selected model: {selected_model_name}")

selected_nb_result = evaluate_estimator_cv(
    model_name=f"Selected {selected_model_name}",
    estimator=selected_nb_pipeline,
    X=X,
    y=y,
    cv=cv,
)

selected_nb_result_df = pd.DataFrame([selected_nb_result])
selected_nb_result_df[nb_metric_columns]

# %% [markdown]
# ## Selected model metric interpretation
#
# At the default threshold \(0.50\), the selected hybrid Naive Bayes model predicts churn for about \(43.7\%\) of customers:
#
# ```text
# predicted positive rate ≈ 0.437
# observed positive rate ≈ 0.265
# ```
#
# This explains the model's behaviour. It detects many actual churners, but it also flags many customers who do not churn.
#
# The confusion matrix is:
#
# ```text
# TP = 1209
# FN = 286
# FP = 1253
# TN = 2886
# ```
#
# So the model detects \(1209\) of \(1495\) churners and misses \(286\). It also incorrectly flags \(1253\) non-churners.
#
# Compared with the previous full transformed GaussianNB model, the hybrid model is less aggressive:
#
# ```text
# Full GaussianNB:
#     TP = 1267
#     FP = 1475
#
# Hybrid Gaussian-BernoulliNB:
#     TP = 1209
#     FP = 1253
# ```
#
# So the hybrid model gives up some recall but removes substantially more false positives. That is why it has better precision, F1, and PR-AUC.

# %% [markdown]
# ## Threshold tradeoff for selected Naive Bayes
#
# The selected Naive Bayes model produces predicted probabilities. Because Naive
# Bayes can be overconfident when features are correlated, threshold behaviour is
# important. The threshold analysis uses out-of-fold predicted probabilities.

# %%
_, selected_nb_oof_probability = get_out_of_fold_predictions(
    estimator=selected_nb_pipeline,
    X=X,
    y=y,
    cv=cv,
)

thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

nb_threshold_df = evaluate_threshold_grid(
    y_true=y,
    y_score=selected_nb_oof_probability,
    thresholds=thresholds,
)

nb_threshold_df

# %% [markdown]
# ## Threshold-tradeoff interpretation
#
# The threshold curve remains relatively flat compared with logistic regression and kNN, but it is slightly less extreme than the previous full transformed GaussianNB curve.
#
# At threshold \(0.50\):
#
# ```text
# precision ≈ 0.491
# recall ≈ 0.809
# specificity ≈ 0.697
# F1 ≈ 0.611
# ```
#
# At threshold \(0.80\):
#
# ```text
# precision ≈ 0.527
# recall ≈ 0.768
# specificity ≈ 0.751
# F1 ≈ 0.625
# ```
#
# The threshold \(0.80\) gives a slightly better F1 and better precision/specificity while retaining high recall. However, final threshold selection should not happen yet, because it should be considered only after all model families are compared and the business cost tradeoff is clearer.
#
# The broad flatness of the threshold curve still suggests that many Naive Bayes scores are high. This is consistent with the conditional-independence assumption being imperfect for correlated Telco features, although the hybrid model behaves better than the full GaussianNB variant.

# %% [markdown]
# ## ROC and precision-recall curves for selected Naive Bayes

# %%
nb_roc_curve_df = make_roc_curve_dataframe(
    y_true=y,
    y_score=selected_nb_oof_probability,
)

nb_precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=selected_nb_oof_probability,
)

curve_summary = pd.DataFrame(
    {
        "curve": ["ROC", "Precision-recall"],
        "rows": [len(nb_roc_curve_df), len(nb_precision_recall_curve_df)],
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
# The selected hybrid Naive Bayes ROC curve has AUC about \(0.822\), which is clearly better than random ranking. The precision-recall curve has AUC about \(0.615\), also well above the positive-rate baseline of about \(0.265\).
#
# So the hybrid model learns useful churn-ranking structure.
#
# However, the ranking metrics are still weaker than the previous learned models:
#
# ```text
# Logistic regression:
#     ROC-AUC ≈ 0.846
#     PR-AUC ≈ 0.658
#
# Selected kNN:
#     ROC-AUC ≈ 0.836
#     PR-AUC ≈ 0.628
#
# Selected hybrid Naive Bayes:
#     ROC-AUC ≈ 0.822
#     PR-AUC ≈ 0.615
# ```
#
# The hybrid Naive Bayes model is therefore a useful and theoretically coherent generative baseline, but it is not the strongest model so far for this dataset.

# %% [markdown]
# ## Save tables and figures

# %%
nb_metric_df.to_csv(NAIVE_BAYES_MODEL_COMPARISON_PATH, index=False)
nb_confusion_df.to_csv(NAIVE_BAYES_CONFUSION_MATRIX_PATH, index=False)
bernoulli_alpha_results_df.to_csv(BERNOULLI_NB_ALPHA_RESULTS_PATH, index=False)
nb_threshold_df.to_csv(NAIVE_BAYES_THRESHOLD_RESULTS_PATH, index=False)

save_alpha_metric_plot(
    results_df=bernoulli_alpha_results_df,
    output_path=BERNOULLI_NB_ALPHA_FIGURE_PATH,
    title="Bernoulli Naive Bayes smoothing metrics",
)

nb_threshold_plot_df = nb_threshold_df[nb_threshold_df["threshold"] <= 0.80].copy()

save_threshold_tradeoff_plot(
    threshold_df=nb_threshold_plot_df,
    output_path=NAIVE_BAYES_THRESHOLD_FIGURE_PATH,
    title="Selected Naive Bayes Threshold Tradeoff",
)

save_roc_curve_plot(
    roc_curve_df=nb_roc_curve_df,
    output_path=NAIVE_BAYES_ROC_CURVE_FIGURE_PATH,
    title="Selected Naive Bayes ROC Curve",
)

save_precision_recall_curve_plot(
    precision_recall_curve_df=nb_precision_recall_curve_df,
    output_path=NAIVE_BAYES_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title="Selected Naive Bayes Precision-Recall Curve",
    positive_rate=float(y.mean()),
)

saved_artifacts = pd.DataFrame(
    {
        "artifact": [
            "naive_bayes_model_comparison",
            "naive_bayes_confusion_matrices",
            "bernoulli_nb_alpha_results",
            "naive_bayes_threshold_results",
            "bernoulli_nb_alpha_figure",
            "naive_bayes_threshold_figure",
            "naive_bayes_roc_curve_figure",
            "naive_bayes_precision_recall_curve_figure",
        ],
        "exists": [
            NAIVE_BAYES_MODEL_COMPARISON_PATH.exists(),
            NAIVE_BAYES_CONFUSION_MATRIX_PATH.exists(),
            BERNOULLI_NB_ALPHA_RESULTS_PATH.exists(),
            NAIVE_BAYES_THRESHOLD_RESULTS_PATH.exists(),
            BERNOULLI_NB_ALPHA_FIGURE_PATH.exists(),
            NAIVE_BAYES_THRESHOLD_FIGURE_PATH.exists(),
            NAIVE_BAYES_ROC_CURVE_FIGURE_PATH.exists(),
            NAIVE_BAYES_PRECISION_RECALL_CURVE_FIGURE_PATH.exists(),
        ],
        "path": [
            NAIVE_BAYES_MODEL_COMPARISON_PATH,
            NAIVE_BAYES_CONFUSION_MATRIX_PATH,
            BERNOULLI_NB_ALPHA_RESULTS_PATH,
            NAIVE_BAYES_THRESHOLD_RESULTS_PATH,
            BERNOULLI_NB_ALPHA_FIGURE_PATH,
            NAIVE_BAYES_THRESHOLD_FIGURE_PATH,
            NAIVE_BAYES_ROC_CURVE_FIGURE_PATH,
            NAIVE_BAYES_PRECISION_RECALL_CURVE_FIGURE_PATH,
        ],
    }
)

saved_artifacts

# %% [markdown]
# ## Section summary
#
# The revised Naive Bayes experiment introduces a generative probabilistic classifier based on class priors, class-conditional likelihoods, and the conditional-independence assumption.
#
# The most important conclusions are:
#
# ```text
# - The hybrid Gaussian-BernoulliNB model is the most theoretically natural Naive Bayes model for this mixed tabular dataset.
# - The hybrid model has the highest PR-AUC among the tested Naive Bayes candidates.
# - It improves over the full transformed GaussianNB model, which incorrectly treats one-hot indicators as Gaussian continuous variables.
# - The categorical-only BernoulliNB model remains close, showing that categorical features carry most of the Naive Bayes signal.
# - Numeric-only GaussianNB is more conservative and has weaker ranking performance.
# - BernoulliNB smoothing has almost no effect across the tested alpha values.
# - Selected hybrid Naive Bayes has high recall, moderate precision, and a broad churn-flagging profile.
# - Naive Bayes learns useful churn structure, but it is weaker than logistic regression and kNN in ROC-AUC and PR-AUC.
# ```
#
# The selected Naive Bayes model is useful as a generative probabilistic comparison model. It demonstrates how Bayes' rule and class-conditional likelihoods can be used for classification, while also showing the practical consequences of simplifying independence assumptions.
#
# The next stage can move toward tree-based models, starting with decision stumps and decision trees, where interactions and nonlinear feature splits are learned directly rather than imposed through a likelihood factorization.