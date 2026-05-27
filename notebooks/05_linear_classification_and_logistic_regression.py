# %% [markdown]
# # 05 Linear Classification and Logistic Regression
#
# ## Purpose
#
# This notebook introduces the first learned classification models in the Telco
# Customer Churn project.
#
# The previous section established:
#
# - training-only evaluation discipline;
# - stratified cross-validation;
# - positive-first confusion-matrix terminology;
# - classification metrics;
# - simple dummy and EDA-inspired baselines.
#
# This section now fits learned linear classifiers:
#
# - regularized least-squares classification through `RidgeClassifier`;
# - logistic regression with L2 regularization;
# - logistic regression with L1 regularization;
# - class-weighted logistic regression as a simple imbalance-aware variant.
#
# The deeper reusable theory is documented in:
#
# ```text
# docs/knowledge_notes/models/05_linear_classification_and_logistic_regression.md
# docs/knowledge_notes/methodology/hyperparameter_tuning.md
# docs/knowledge_notes/methodology/evaluation_metrics.md
# ```
#
# This notebook focuses on the executable workflow and the model outputs.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is not used here.
#
# All reported development-stage performance is based on out-of-fold predictions
# from stratified cross-validation inside the training set.
#
# Preprocessing is inside the scikit-learn pipeline, so scaling and one-hot
# encoding are fitted only on each fold's training data.

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
    TRAIN_DATA_PATH,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    evaluate_estimator_cv,
    evaluate_threshold_grid,
    get_out_of_fold_predictions,
    make_confusion_matrix_dataframe,
    make_metric_dataframe,
    make_precision_recall_curve_dataframe,
    make_roc_curve_dataframe,
    make_stratified_kfold,
)
from telco_churn.features import extract_linear_model_coefficients  # noqa: E402
from telco_churn.models import (  # noqa: E402
    make_classifier_pipeline,
    make_l1_logistic_regression_classifier,
    make_l2_logistic_regression_classifier,
    make_ridge_classifier,
)
from telco_churn.preprocessing import make_scaled_preprocessor  # noqa: E402
from telco_churn.visualization import (  # noqa: E402
    save_coefficient_plot,
    save_precision_recall_curve_plot,
    save_regularization_metric_plot,
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
LINEAR_MODEL_COMPARISON_PATH = TABLES_DIR / "linear_model_comparison.csv"
LINEAR_MODEL_CONFUSION_MATRIX_PATH = TABLES_DIR / "linear_model_confusion_matrices.csv"
LOGISTIC_L2_REGULARIZATION_RESULTS_PATH = TABLES_DIR / "logistic_l2_regularization_results.csv"
LOGISTIC_L1_REGULARIZATION_RESULTS_PATH = TABLES_DIR / "logistic_l1_regularization_results.csv"
LOGISTIC_TOP_COEFFICIENTS_PATH = TABLES_DIR / "logistic_top_coefficients.csv"
LOGISTIC_THRESHOLD_RESULTS_PATH = TABLES_DIR / "logistic_threshold_results.csv"

LOGISTIC_L2_REGULARIZATION_FIGURE_PATH = FIGURES_DIR / "logistic_l2_regularization_metrics.png"
LOGISTIC_L1_REGULARIZATION_FIGURE_PATH = FIGURES_DIR / "logistic_l1_regularization_metrics.png"
LOGISTIC_TOP_COEFFICIENTS_FIGURE_PATH = FIGURES_DIR / "logistic_top_coefficients.png"
LOGISTIC_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "logistic_threshold_tradeoff.png"
LOGISTIC_ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "logistic_roc_curve.png"
LOGISTIC_PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "logistic_precision_recall_curve.png"

output_paths = pd.DataFrame(
    {
        "artifact": [
            "linear_model_comparison",
            "linear_model_confusion_matrices",
            "logistic_l2_regularization_results",
            "logistic_l1_regularization_results",
            "logistic_top_coefficients",
            "logistic_threshold_results",
            "logistic_l2_regularization_figure",
            "logistic_l1_regularization_figure",
            "logistic_top_coefficients_figure",
            "logistic_threshold_figure",
            "logistic_roc_curve_figure",
            "logistic_precision_recall_curve_figure",
        ],
        "path": [
            LINEAR_MODEL_COMPARISON_PATH,
            LINEAR_MODEL_CONFUSION_MATRIX_PATH,
            LOGISTIC_L2_REGULARIZATION_RESULTS_PATH,
            LOGISTIC_L1_REGULARIZATION_RESULTS_PATH,
            LOGISTIC_TOP_COEFFICIENTS_PATH,
            LOGISTIC_THRESHOLD_RESULTS_PATH,
            LOGISTIC_L2_REGULARIZATION_FIGURE_PATH,
            LOGISTIC_L1_REGULARIZATION_FIGURE_PATH,
            LOGISTIC_TOP_COEFFICIENTS_FIGURE_PATH,
            LOGISTIC_THRESHOLD_FIGURE_PATH,
            LOGISTIC_ROC_CURVE_FIGURE_PATH,
            LOGISTIC_PRECISION_RECALL_CURVE_FIGURE_PATH,
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
# ## Linear classification theory needed for this notebook
#
# A linear classifier starts from a score:
#
# $$
# f(x) = w^\\top x + b.
# $$
#
# The decision boundary is:
#
# $$
# w^\\top x + b = 0.
# $$
#
# Logistic regression uses the same linear score but maps it to a probability:
#
# $$
# \\hat{p}(Y=1 \\mid X=x)
# =
# \\frac{1}{1+\\exp[-(w^\\top x+b)]}.
# $$
#
# The default class prediction uses threshold $0.5$, but later threshold analysis
# can change the recall/precision/specificity tradeoff.
#
# This notebook uses scaled numeric features and one-hot encoded categorical
# features. Scaling is important because L1 and L2 regularization penalize
# coefficient magnitudes.

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
# ## Main linear model comparison
#
# The first comparison uses a small set of representative linear classifiers.
#
# `RidgeClassifier` represents regularized least-squares classification.
#
# Logistic regression is evaluated with L2 and L1 regularization.
#
# Class-weighted L2 logistic regression is included as a simple imbalance-aware
# variant. It changes the training objective by giving more weight to the
# minority class. It does not resample the data.

# %%
linear_estimators = {
    "RidgeClassifier alpha=1": make_classifier_pipeline(
        preprocessor=scaled_preprocessor,
        classifier=make_ridge_classifier(alpha=1.0),
    ),
    "Logistic L2 C=1": make_classifier_pipeline(
        preprocessor=scaled_preprocessor,
        classifier=make_l2_logistic_regression_classifier(C=1.0),
    ),
    "Logistic L2 balanced C=1": make_classifier_pipeline(
        preprocessor=scaled_preprocessor,
        classifier=make_l2_logistic_regression_classifier(C=1.0, class_weight="balanced"),
    ),
    "Logistic L1 C=1": make_classifier_pipeline(
        preprocessor=scaled_preprocessor,
        classifier=make_l1_logistic_regression_classifier(C=1.0),
    ),
}

linear_results = []

for model_name, estimator in linear_estimators.items():
    result = evaluate_estimator_cv(
        model_name=model_name,
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    linear_results.append(result)

linear_results_df = pd.DataFrame(linear_results)

linear_metric_columns = [
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

linear_metric_df = (
    linear_results_df[linear_metric_columns]
    .sort_values(["pr_auc", "balanced_accuracy", "f1"], ascending=False)
    .reset_index(drop=True)
)

linear_confusion_df = (
    make_confusion_matrix_dataframe(linear_results_df)
    .set_index("model")
    .loc[linear_metric_df["model"]]
    .reset_index()
)

linear_metric_df

# %%
linear_confusion_df

# %% [markdown]
# ## Interpretation: main linear model comparison
#
# The learned linear models clearly improve on the dummy baselines from the
# previous section. The standard L1 and L2 logistic regression models are almost
# indistinguishable in this first comparison. Both reach an ROC-AUC of about
# 0.846 and a PR-AUC of about 0.659, which is far above the positive-rate
# baseline of approximately 0.265.
#
# The class-weighted L2 logistic regression has a different operating behaviour.
# It increases recall substantially, from about 0.54 to about 0.80, but this is
# achieved by flagging many more customers as churn risks. Its predicted
# positive rate is about 0.41, compared with about 0.22 for the ordinary L1/L2
# logistic regressions and an observed churn rate of about 0.27. This explains
# why its precision is lower even though its balanced accuracy is higher.
#
# Ridge classification is a useful regularized least-squares baseline, but it is
# slightly weaker than logistic regression on the ranking metrics. This is
# expected: RidgeClassifier is a linear classifier, but it is not fitted by
# maximizing a Bernoulli likelihood and does not naturally model probabilities.

# %% [markdown]
# ## What to look for after running
#
# Compare these learned models with the simple baselines from section 04.
#
# Important questions:
#
# - Do the linear models beat dummy baselines?
# - Do they improve precision and specificity compared with the broad EDA rule?
# - Does class weighting increase recall but reduce precision?
# - Does the L1 model behave similarly to L2, or does sparsity change results?
# - Are ROC-AUC and PR-AUC meaningfully above the baseline PR-AUC implied by the
#   churn rate?

# %% [markdown]
# ## L2 logistic regression regularization grid
#
# Scikit-learn's `C` is inverse regularization strength:
#
# ```text
# smaller C = stronger regularization
# larger C  = weaker regularization
# ```
#
# We use a small log-scale grid. This is deliberately transparent and easier to
# interpret than automated optimization for this first learned model section.

# %%
C_GRID_L2 = [0.001, 0.01, 0.1, 1, 10, 100]

l2_results = []

for C in C_GRID_L2:
    estimator = make_classifier_pipeline(
        preprocessor=scaled_preprocessor,
        classifier=make_l2_logistic_regression_classifier(C=C),
    )
    result = evaluate_estimator_cv(
        model_name=f"Logistic L2 C={C}",
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    result["C"] = C
    result["penalty"] = "l2"
    result["class_weight"] = "none"
    l2_results.append(result)

l2_results_df = pd.DataFrame(l2_results).sort_values("C").reset_index(drop=True)

l2_display_columns = [
    "C",
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

l2_results_df[l2_display_columns]

# %% [markdown]
# ## Interpretation: L2 regularization path
#
# The L2 grid shows that very strong regularization is too restrictive. At
# `C = 0.001`, the model is conservative: recall is only about 0.22, although
# specificity is very high. As `C` increases, the model becomes less constrained
# and recall, F1, balanced accuracy, ROC-AUC, and PR-AUC all improve.
#
# After approximately `C = 1`, the ranking metrics barely change. This means
# that weakening L2 regularization further does not materially improve the
# fitted ranking of churners versus non-churners. The best PR-AUC in the grid is
# reached around the larger `C` values, but the gains are very small. Therefore,
# the exact choice between `C = 1`, `C = 10`, and `C = 100` should not be
# overinterpreted.

# %% [markdown]
# ## L1 logistic regression regularization grid
#
# L1 regularization can shrink some coefficients exactly to zero. This makes it
# useful for sparse interpretation.
#
# The grid is smaller than the L2 grid because L1 fitting is somewhat more
# sensitive and expensive.

# %%
C_GRID_L1 = [0.001, 0.01, 0.1, 1, 10]

l1_results = []

for C in C_GRID_L1:
    estimator = make_classifier_pipeline(
        preprocessor=scaled_preprocessor,
        classifier=make_l1_logistic_regression_classifier(C=C),
    )
    result = evaluate_estimator_cv(
        model_name=f"Logistic L1 C={C}",
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    result["C"] = C
    result["penalty"] = "l1"
    result["class_weight"] = "none"
    l1_results.append(result)

l1_results_df = pd.DataFrame(l1_results).sort_values("C").reset_index(drop=True)

l1_results_df[l2_display_columns]

# %% [markdown]
# ## Interpretation: L1 regularization path
#
# The L1 path shows the sparsity effect more clearly. At `C = 0.001`, the L1
# penalty is so strong that the model effectively predicts no churners at the
# default threshold. This gives zero recall and an F1-score of zero.
#
# Once the penalty is relaxed, performance quickly approaches the L2 logistic
# regression results. Around `C = 1`, the L1 model reaches the best PR-AUC in
# the current linear-model comparison. The difference from L2 is extremely
# small, so the main practical conclusion is not that L1 is definitively better,
# but that both regularized logistic models learn a similar linear signal.
#
# L1 remains useful because it can simplify coefficient interpretation by
# shrinking some less useful feature effects to exactly zero.

# %% [markdown]
# ## Select representative logistic regression model for interpretation
#
# The first notebook version selects the L2 logistic regression setting with the
# highest cross-validated PR-AUC. After inspecting results, we can decide whether
# this selection rule should remain or whether another metric is more appropriate.
#
# This selected model is fitted on the full training set only for coefficient
# extraction and later final-model preparation. Its in-sample fitted performance
# is not reported as validation performance.

# %%
best_l2_row = l2_results_df.sort_values(
    ["pr_auc", "balanced_accuracy", "f1"],
    ascending=False,
).iloc[0]

best_l2_C = float(best_l2_row["C"])

selected_logistic_pipeline = make_classifier_pipeline(
    preprocessor=make_scaled_preprocessor(),
    classifier=make_l2_logistic_regression_classifier(C=best_l2_C),
)

selected_logistic_pipeline.fit(X, y)

selection_summary = pd.DataFrame(
    {
        "item": [
            "selected_model",
            "selection_metric",
            "selected_C",
            "selected_pr_auc",
            "selected_roc_auc",
            "selected_balanced_accuracy",
        ],
        "value": [
            "L2 logistic regression",
            "highest cross-validated PR-AUC among L2 grid",
            best_l2_C,
            best_l2_row["pr_auc"],
            best_l2_row["roc_auc"],
            best_l2_row["balanced_accuracy"],
        ],
    }
)

selection_summary

# %% [markdown]
# ## Interpretation: selected logistic model
#
# For coefficient interpretation, the notebook selects the L2 logistic
# regression model with the highest cross-validated PR-AUC among the L2 grid.
# This gives a stable probabilistic linear model for interpretation.
#
# The model is refitted on the full training set only after cross-validation has
# selected the hyperparameter. This fitted model is used to extract coefficients
# and prepare artifacts. Its fitted values on the full training set are not used
# as validation performance.

# %% [markdown]
# ## Coefficient extraction
#
# Coefficients are extracted from the selected model fitted on the full training
# data.
#
# Interpretation rules:
#
# - positive coefficient: higher fitted churn score;
# - negative coefficient: lower fitted churn score;
# - numeric coefficients correspond to standardized numeric features;
# - categorical coefficients correspond to one-hot encoded indicators;
# - coefficients describe fitted associations, not causal effects.

# %%
coefficient_df = extract_linear_model_coefficients(
    fitted_pipeline=selected_logistic_pipeline,
    top_n=None,
    sort_by_absolute=True,
)

top_coefficient_df = coefficient_df.head(30)

top_coefficient_df

# %% [markdown]
# ## Interpretation: coefficient pattern
#
# The largest coefficients agree with the exploratory analysis. Shorter tenure
# is strongly associated with higher churn risk: the standardized `tenure`
# coefficient is the most negative coefficient, meaning longer tenure lowers the
# fitted churn score. Contract type is also important: `Contract_Two year` lowers
# the fitted churn score, while `Contract_Month-to-month` raises it.
#
# `InternetService_Fiber optic` has a positive coefficient, while
# `InternetService_DSL` has a negative coefficient. This means that, after the
# model accounts for the other included variables, fiber-optic customers are
# assigned higher churn scores than otherwise comparable customers in the linear
# model.
#
# `TotalCharges` is positive while `tenure` is negative. This combination should
# be interpreted carefully because accumulated charges are mechanically related
# to tenure and monthly charges. Coefficients are conditional associations inside
# this fitted linear model, not causal effects.
#
# The `No internet service` indicator levels have similar negative coefficients
# across several internet add-on variables. This reflects the structural
# category created by customers who do not have internet service, not ordinary
# missingness.

# %% [markdown]
# ## Threshold tradeoff for selected logistic regression
#
# The selected logistic regression model gives predicted probabilities.
#
# The default threshold is 0.5, but the threshold controls the tradeoff between
# recall, precision, and specificity.
#
# We use cross-validated out-of-fold probabilities for this threshold analysis,
# not the held-out test set.

# %%
selected_cv_pipeline = make_classifier_pipeline(
    preprocessor=make_scaled_preprocessor(),
    classifier=make_l2_logistic_regression_classifier(C=best_l2_C),
)

_, selected_oof_probability = get_out_of_fold_predictions(
    estimator=selected_cv_pipeline,
    X=X,
    y=y,
    cv=cv,
)

thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

threshold_df = evaluate_threshold_grid(
    y_true=y,
    y_score=selected_oof_probability,
    thresholds=thresholds,
)

threshold_df

# %%
threshold_plot_df = threshold_df.loc[threshold_df["threshold"] <= 0.80].copy()

threshold_plot_df

# %% [markdown]
# ## Interpretation: threshold tradeoff
#
# The default threshold of `0.50` gives a balanced but conservative operating
# point: precision is about 0.65, recall is about 0.54, and specificity is about
# 0.90. This means that positive churn predictions are relatively reliable, but
# many churners are still missed.
#
# Lowering the threshold increases recall. For example, thresholds around
# `0.25` to `0.35` recover many more churners and produce the strongest F1-scores
# in this grid, but they also increase the number of false positives. This is
# exactly the expected threshold tradeoff: broader targeting catches more
# churners but also flags more customers who would not churn.
#
# Very high thresholds are not practically useful here because the model flags
# almost nobody. Those rows remain in the saved threshold table, but the plotted
# threshold tradeoff is restricted to thresholds up to `0.80` to keep the
# operationally relevant part readable.

# %% [markdown]
# ## ROC and precision-recall curves
#
# Logistic regression produces predicted probabilities. This allows us to study
# ranking performance across all possible classification thresholds.
#
# The ROC curve shows the tradeoff between true positive rate and false positive
# rate. The precision-recall curve shows how reliable positive churn predictions
# remain as recall increases.
#
# Because churn is the minority class, the precision-recall curve is especially
# important. The horizontal reference line in the PR plot is the observed churn
# rate, which is the expected precision of a non-informative random ranking.

# %%
roc_curve_df = make_roc_curve_dataframe(
    y_true=y,
    y_score=selected_oof_probability,
)

precision_recall_curve_df = make_precision_recall_curve_dataframe(
    y_true=y,
    y_score=selected_oof_probability,
)

curve_summary = pd.DataFrame(
    {
        "curve": ["ROC", "Precision-recall"],
        "rows": [len(roc_curve_df), len(precision_recall_curve_df)],
        "baseline_reference": [
            "diagonal random-ranking line",
            f"positive-rate baseline = {y.mean():.4f}",
        ],
    }
)

curve_summary

# %% [markdown]
# ## Interpretation: ROC and precision-recall curves
#
# The ROC curve lies well above the diagonal random-ranking reference line, with
# ROC-AUC around 0.846. This indicates that the selected logistic regression
# model has strong ranking ability: it usually assigns higher churn probabilities
# to actual churners than to non-churners.
#
# The precision-recall curve is especially important because churn is the
# minority class. The PR-AUC is about 0.658, compared with the positive-rate
# baseline of about 0.265. This confirms that the model is much better than a
# non-informative ranking at concentrating actual churners near the top of the
# predicted-risk list.
#
# The curve also shows the practical tradeoff: high precision can be achieved
# when recall is low, but precision gradually falls as the model attempts to
# recover more churners. This supports later threshold tuning or cost-sensitive
# decision analysis, but no final threshold is selected in this section.

# %% [markdown]
# ## Save tables and figures

# %%
linear_metric_df.to_csv(LINEAR_MODEL_COMPARISON_PATH, index=False)
linear_confusion_df.to_csv(LINEAR_MODEL_CONFUSION_MATRIX_PATH, index=False)
l2_results_df.to_csv(LOGISTIC_L2_REGULARIZATION_RESULTS_PATH, index=False)
l1_results_df.to_csv(LOGISTIC_L1_REGULARIZATION_RESULTS_PATH, index=False)
coefficient_df.to_csv(LOGISTIC_TOP_COEFFICIENTS_PATH, index=False)
threshold_df.to_csv(LOGISTIC_THRESHOLD_RESULTS_PATH, index=False)

save_regularization_metric_plot(
    results_df=l2_results_df,
    output_path=LOGISTIC_L2_REGULARIZATION_FIGURE_PATH,
    title="L2 Logistic Regression Regularization Metrics",
)

save_regularization_metric_plot(
    results_df=l1_results_df,
    output_path=LOGISTIC_L1_REGULARIZATION_FIGURE_PATH,
    title="L1 Logistic Regression Regularization Metrics",
)

save_coefficient_plot(
    coefficient_df=coefficient_df,
    output_path=LOGISTIC_TOP_COEFFICIENTS_FIGURE_PATH,
    title="Top Logistic Regression Coefficients",
    top_n=25,
)

save_threshold_tradeoff_plot(
    threshold_df=threshold_plot_df,
    output_path=LOGISTIC_THRESHOLD_FIGURE_PATH,
    title="Selected Logistic Regression Threshold Tradeoff",
)

save_roc_curve_plot(
    roc_curve_df=roc_curve_df,
    output_path=LOGISTIC_ROC_CURVE_FIGURE_PATH,
    title="Selected Logistic Regression ROC Curve",
)

save_precision_recall_curve_plot(
    precision_recall_curve_df=precision_recall_curve_df,
    output_path=LOGISTIC_PRECISION_RECALL_CURVE_FIGURE_PATH,
    title="Selected Logistic Regression Precision-Recall Curve",
    positive_rate=float(y.mean()),
)

saved_artifacts = pd.DataFrame(
    {
        "artifact": [
            "linear_model_comparison",
            "linear_model_confusion_matrices",
            "logistic_l2_regularization_results",
            "logistic_l1_regularization_results",
            "logistic_top_coefficients",
            "logistic_threshold_results",
            "logistic_l2_regularization_figure",
            "logistic_l1_regularization_figure",
            "logistic_top_coefficients_figure",
            "logistic_threshold_figure",
            "logistic_roc_curve_figure",
            "logistic_precision_recall_curve_figure",
        ],
        "exists": [
            LINEAR_MODEL_COMPARISON_PATH.exists(),
            LINEAR_MODEL_CONFUSION_MATRIX_PATH.exists(),
            LOGISTIC_L2_REGULARIZATION_RESULTS_PATH.exists(),
            LOGISTIC_L1_REGULARIZATION_RESULTS_PATH.exists(),
            LOGISTIC_TOP_COEFFICIENTS_PATH.exists(),
            LOGISTIC_THRESHOLD_RESULTS_PATH.exists(),
            LOGISTIC_L2_REGULARIZATION_FIGURE_PATH.exists(),
            LOGISTIC_L1_REGULARIZATION_FIGURE_PATH.exists(),
            LOGISTIC_TOP_COEFFICIENTS_FIGURE_PATH.exists(),
            LOGISTIC_THRESHOLD_FIGURE_PATH.exists(),
            LOGISTIC_ROC_CURVE_FIGURE_PATH.exists(),
            LOGISTIC_PRECISION_RECALL_CURVE_FIGURE_PATH.exists(),
        ],
        "path": [
            LINEAR_MODEL_COMPARISON_PATH,
            LINEAR_MODEL_CONFUSION_MATRIX_PATH,
            LOGISTIC_L2_REGULARIZATION_RESULTS_PATH,
            LOGISTIC_L1_REGULARIZATION_RESULTS_PATH,
            LOGISTIC_TOP_COEFFICIENTS_PATH,
            LOGISTIC_THRESHOLD_RESULTS_PATH,
            LOGISTIC_L2_REGULARIZATION_FIGURE_PATH,
            LOGISTIC_L1_REGULARIZATION_FIGURE_PATH,
            LOGISTIC_TOP_COEFFICIENTS_FIGURE_PATH,
            LOGISTIC_THRESHOLD_FIGURE_PATH,
            LOGISTIC_ROC_CURVE_FIGURE_PATH,
            LOGISTIC_PRECISION_RECALL_CURVE_FIGURE_PATH,
        ],
    }
)

saved_artifacts

# %% [markdown]
# ## Section summary
#
# This section fitted the first learned classifiers in the project. The main
# conclusion is that regularized logistic regression is a strong and interpretable
# first learned model for this dataset.
#
# The standard L1 and L2 logistic regressions perform almost identically. Both
# improve clearly over the simple dummy baselines and produce useful probability
# rankings, with ROC-AUC around 0.846 and PR-AUC around 0.658. Class weighting
# increases recall substantially, but at the cost of many more false positives
# and lower precision.
#
# The regularization grids show that extremely strong regularization underfits,
# while moderate and weak regularization produce stable results. Coefficients
# identify the same broad churn patterns found in the EDA: shorter tenure,
# month-to-month contracts, fiber-optic internet service, and electronic checks
# are associated with higher fitted churn scores, while longer contracts and
# longer tenure are associated with lower fitted churn scores.
#
# The threshold analysis shows that the default `0.50` threshold is not the only
# possible operating point. Lower thresholds around `0.25` to `0.35` can recover
# more churners and improve F1, but this comes with more false positives. Final
# threshold selection is postponed because it depends on business costs and the
# later comparison with other model families.
#
# Saved artifacts:
#
# ```text
# linear_model_comparison.csv
# linear_model_confusion_matrices.csv
# logistic_l2_regularization_results.csv
# logistic_l1_regularization_results.csv
# logistic_top_coefficients.csv
# logistic_threshold_results.csv
# logistic_l2_regularization_metrics.png
# logistic_l1_regularization_metrics.png
# logistic_top_coefficients.png
# logistic_threshold_tradeoff.png
# logistic_roc_curve.png
# logistic_precision_recall_curve.png
# ```
#
# The next step is to write the polished LaTeX report section and then continue
# to the next model family.
