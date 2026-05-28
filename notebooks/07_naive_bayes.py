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
# - A full transformed GaussianNB variant is included as a simple comparison, even
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
from telco_churn.models import make_classifier_pipeline  # noqa: E402
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


def make_full_dense_onehot_preprocessor() -> Pipeline:
    """Create dense full-feature preprocessing for GaussianNB.

    Numeric features are median-imputed and categorical features are one-hot
    encoded. The combined matrix is converted to dense form because GaussianNB
    does not accept sparse matrices.
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

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("columns", column_transformer),
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
# We evaluate three transparent variants:
#
# 1. **GaussianNB numeric only**: uses the three numeric features.
# 2. **BernoulliNB categorical only**: uses one-hot encoded categorical features.
# 3. **GaussianNB full transformed**: uses numeric features plus one-hot encoded
#    categorical indicators as a simple full-feature GaussianNB comparison.
#
# The third variant is not the cleanest theoretical likelihood for one-hot
# indicators, but it is useful as a simple full-feature benchmark.

# %%
nb_estimators = {
    "GaussianNB numeric only": make_gaussian_numeric_nb_pipeline(),
    "BernoulliNB categorical only alpha=1": make_bernoulli_categorical_nb_pipeline(alpha=1.0),
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
# The three Naive Bayes variants behave quite differently.
#
# The selected highest-PR-AUC variant is the **full transformed GaussianNB** model. It uses the numeric features together with the one-hot encoded categorical features. It obtains approximately:
#
# ```text
# accuracy ≈ 0.698
# balanced accuracy ≈ 0.746
# precision ≈ 0.462
# recall ≈ 0.847
# specificity ≈ 0.644
# F1 ≈ 0.598
# ROC-AUC ≈ 0.819
# PR-AUC ≈ 0.605
# ```
#
# This is a high-recall model: it detects most churners, but it also produces many false positives.
#
# The categorical-only BernoulliNB model is very close. It has slightly lower PR-AUC, but slightly higher balanced accuracy and F1 in some settings. This shows that the categorical variables contain most of the predictive signal used by Naive Bayes.
#
# The numeric-only GaussianNB model has higher ordinary accuracy, but lower recall and weaker ranking performance. It is more conservative and flags fewer customers as churners. This confirms that the numeric features alone contain useful information, but not enough to match the categorical information.

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
# That is a useful result. It suggests that there are no severe rare-category zero-probability problems driving the performance. The categorical signal is stable across reasonable smoothing strengths.
#
# Because the selected overall Naive Bayes model is the full transformed GaussianNB variant, the smoothing grid is mainly used to understand the Bernoulli categorical baseline, not to choose the final representative Naive Bayes model.

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
# GaussianNB full transformed
# ```
#
# This model is selected because it has the highest cross-validated PR-AUC among the Naive Bayes candidates, approximately \(0.605\).
#
# This selection should be interpreted carefully. The model is not theoretically perfect because Gaussian likelihoods are not natural for one-hot encoded indicator variables. However, it is useful as a full-feature Naive Bayes comparison and gives the strongest positive-class ranking among the tested Naive Bayes variants.
#
# The selected model is still not the best model family so far. Logistic regression and kNN both had stronger PR-AUC and ROC-AUC. However, Naive Bayes gives a different tradeoff: it produces high recall at the default threshold, but with relatively low precision.

# %% [markdown]
# ## Recreate selected Naive Bayes pipeline

# %%
selected_model_name = str(best_nb_row["model"])

if selected_model_name.startswith("GaussianNB numeric only"):
    selected_nb_pipeline = make_gaussian_numeric_nb_pipeline()
elif selected_model_name.startswith("GaussianNB full transformed"):
    selected_nb_pipeline = make_gaussian_full_nb_pipeline()
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
# At the default threshold \(0.50\), the selected full transformed GaussianNB model predicts churn for almost half of all customers:
#
# ```text
# predicted positive rate ≈ 0.487
# observed positive rate ≈ 0.265
# ```
#
# This explains its high recall and low precision. It detects many actual churners, but it also flags many non-churners.
#
# The confusion matrix is:
#
# ```text
# TP = 1267
# FN = 228
# FP = 1475
# TN = 2664
# ```
#
# So the model detects \(1267\) of \(1495\) churners, but also incorrectly flags \(1475\) non-churners.
#
# This is similar in spirit to the EDA-inspired rule baseline: broad churn detection with many false positives. Naive Bayes is more probabilistic and model-based than the EDA rule, but its default operating point is still aggressive.

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
# The threshold curve is unusually flat compared with logistic regression and kNN. Recall remains high across the whole plotted threshold range, while precision improves only gradually.
#
# At threshold \(0.50\):
#
# ```text
# precision ≈ 0.462
# recall ≈ 0.847
# specificity ≈ 0.644
# F1 ≈ 0.598
# ```
#
# At threshold \(0.95\):
#
# ```text
# precision ≈ 0.492
# recall ≈ 0.813
# specificity ≈ 0.696
# F1 ≈ 0.613
# ```
#
# Even at a very high threshold, the model still predicts churn for about \(43.9\%\) of customers. This suggests that many predicted probabilities are pushed toward high values, which is consistent with a known limitation of Naive Bayes: when correlated features are treated as conditionally independent, the model can double-count related evidence and become overconfident.
#
# Therefore, Naive Bayes probabilities should not be treated as well-calibrated without a later calibration check.

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
# The selected Naive Bayes ROC curve has AUC about \(0.819\), which is clearly better than random ranking. The precision-recall curve has AUC about \(0.604\), also well above the positive-rate baseline of about \(0.265\).
#
# So Naive Bayes does learn useful churn-ranking structure.
#
# However, the ranking metrics are weaker than the previous learned models:
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
# Selected Naive Bayes:
#     ROC-AUC ≈ 0.819
#     PR-AUC ≈ 0.605
# ```
#
# The gap is meaningful. Naive Bayes is useful for learning generative probabilistic classification, but it is not the strongest model so far for this dataset.

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
# The Naive Bayes experiment introduces a generative probabilistic classifier based on class priors, class-conditional likelihoods, and the conditional-independence assumption.
#
# The most important conclusions are:
#
# ```text
# - The full transformed GaussianNB model has the highest PR-AUC among the tested Naive Bayes variants.
# - The categorical-only BernoulliNB model is very close, showing that categorical features carry most of the Naive Bayes signal.
# - Numeric-only GaussianNB is more conservative and has weaker ranking performance.
# - BernoulliNB smoothing has almost no effect across the tested alpha values.
# - Selected Naive Bayes has high recall but relatively low precision.
# - Its probabilities appear overconfident, likely because correlated one-hot features violate the conditional-independence assumption.
# - Naive Bayes learns useful churn structure, but it is weaker than logistic regression and kNN in ROC-AUC and PR-AUC.
# ```
#
# The selected Naive Bayes model is useful as a generative probabilistic comparison model. It demonstrates how Bayes' rule and class-conditional likelihoods can be used for classification, while also showing the practical consequences of unrealistic independence assumptions.
#
# The next stage can move toward tree-based models, starting with decision stumps and decision trees, where interactions and nonlinear feature splits are learned directly rather than imposed through a likelihood factorization.