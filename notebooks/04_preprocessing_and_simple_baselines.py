# %% [markdown]
# # 04 Preprocessing, Evaluation, and Simple Baselines
#
# ## Purpose
#
# This notebook starts the modelling workflow for the Telco Customer Churn
# classification project.
#
# The previous workflow stages created:
#
# 1. a corrected interim dataset;
# 2. a clean supervised modelling table;
# 3. a held-out stratified train-test split;
# 4. exploratory analysis using only the training set.
#
# This notebook has two roles.
#
# First, it establishes reusable modelling infrastructure:
#
# - project constants;
# - data-loading helpers;
# - feature and target splitting;
# - preprocessing objects;
# - cross-validation;
# - binary-classification evaluation metrics.
#
# Second, it evaluates simple baselines:
#
# - majority-class baseline;
# - prior-probability baseline;
# - stratified random baseline;
# - uniform random baseline;
# - transparent EDA-inspired rule baseline.
#
# Learned models such as logistic regression, k-nearest neighbours, Naive Bayes,
# decision trees, support vector machines, and neural networks are deliberately
# saved for later sections.

# %% [markdown]
# ## Methodological discipline
#
# The held-out test set is reserved for final evaluation. It should not be used
# while choosing preprocessing strategies, model families, hyperparameters,
# feature engineering rules, decision thresholds, or calibration methods.
#
# All evaluations in this notebook are based on cross-validation inside the
# training set. For simple dummy and rule-based baselines, cross-validation is
# still useful because it makes the evaluation procedure identical to later model
# sections and produces out-of-fold predictions.
#
# An out-of-fold prediction means that the prediction for an observation is
# produced by a model fitted without that observation. This is the core idea that
# keeps validation estimates more honest than evaluating on the same data used
# for fitting.

# %%
from pathlib import Path
import sys

import pandas as pd

# %% [markdown]
# ## Import project utilities
#
# The repository uses a `src/` layout. When the package has not been installed in
# editable mode, the notebook adds the project `src` directory to `sys.path`.
# This keeps the notebook runnable from either the repository root or the
# `notebooks/` directory.

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
    CATEGORICAL_FEATURES,
    CV_N_SPLITS,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SIMPLE_BASELINE_COMPARISON_PATH,
    SIMPLE_BASELINE_CONFUSION_MATRIX_PATH,
    TABLES_DIR,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    evaluate_estimator_cv,
    make_confusion_matrix_dataframe,
    make_metric_dataframe,
    make_results_dataframe,
    make_stratified_kfold,
)
from telco_churn.models import (  # noqa: E402
    make_eda_inspired_rule_classifier,
    make_most_frequent_dummy_classifier,
    make_prior_probability_dummy_classifier,
    make_stratified_dummy_classifier,
    make_uniform_dummy_classifier,
)
from telco_churn.preprocessing import (  # noqa: E402
    make_scaled_preprocessor,
    make_unscaled_preprocessor,
)

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 160)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:,.4f}".format)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Configuration check

# %%
configuration_check = pd.DataFrame(
    {
        "item": [
            "project_root",
            "train_data_path",
            "train_data_path_exists",
            "target_column",
            "number_of_features",
            "number_of_numeric_features",
            "number_of_categorical_features",
            "cv_n_splits",
            "random_state",
            "simple_baseline_metrics_output_path",
            "simple_baseline_confusion_output_path",
        ],
        "value": [
            str(PROJECT_ROOT),
            str(TRAIN_DATA_PATH),
            TRAIN_DATA_PATH.exists(),
            TARGET_COLUMN,
            len(ALL_FEATURES),
            len(NUMERIC_FEATURES),
            len(CATEGORICAL_FEATURES),
            CV_N_SPLITS,
            RANDOM_STATE,
            str(SIMPLE_BASELINE_COMPARISON_PATH),
            str(SIMPLE_BASELINE_CONFUSION_MATRIX_PATH),
        ],
    }
)

configuration_check

# %% [markdown]
# ## Load training data only
#
# This notebook loads only the training split. The held-out test split is not
# inspected.

# %%
train_df = load_train_data()

training_overview = pd.DataFrame(
    {
        "item": [
            "training_rows",
            "training_columns",
            "missing_values",
            "target_positive_rate",
        ],
        "value": [
            train_df.shape[0],
            train_df.shape[1],
            int(train_df.isna().sum().sum()),
            train_df[TARGET_COLUMN].mean(),
        ],
    }
)

training_overview

# %% [markdown]
# ## Split features and target

# %%
X, y = split_features_target(train_df)

feature_group_check = pd.DataFrame(
    {
        "feature_group": ["numeric", "categorical", "all"],
        "count": [len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES), len(ALL_FEATURES)],
        "features": [
            ", ".join(NUMERIC_FEATURES),
            ", ".join(CATEGORICAL_FEATURES),
            ", ".join(ALL_FEATURES),
        ],
    }
)

feature_group_check

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
# ## Validation strategy
#
# We use stratified 5-fold cross-validation inside the training set.
# Stratification preserves the churn proportion within each fold, which is
# important because the positive class represents only about a quarter of the
# data.
#
# Cross-validation gives a more stable development-stage estimate than one
# arbitrary validation split. The held-out test set remains unused.

# %%
cv = make_stratified_kfold()

cv_check = pd.DataFrame(
    {
        "item": ["strategy", "n_splits", "shuffle", "random_state"],
        "value": [
            "StratifiedKFold",
            cv.n_splits,
            True,
            RANDOM_STATE,
        ],
    }
)

cv_check

# %% [markdown]
# # Evaluation theory for binary classification
#
# Before fitting more models, we need a clear evaluation language. This section
# is intentionally detailed because the same ideas will be used throughout the
# rest of the project.
#
# ## Positive and negative class
#
# A binary classification problem has two classes. Many evaluation metrics are
# defined relative to the **positive** class.
#
# A medical example is useful:
#
# - positive class: the person has cancer;
# - negative class: the person does not have cancer;
# - predicted positive: the test says cancer;
# - predicted negative: the test says no cancer.
#
# For this churn project:
#
# - positive class: the customer churns;
# - negative class: the customer does not churn;
# - predicted positive: the model flags the customer as likely to churn;
# - predicted negative: the model does not flag the customer.
#
# The positive class is not necessarily the most frequent class. It is the event
# of interest. In this project, churn is the minority class but it is the class
# we care about detecting.

# %% [markdown]
# ## Confusion matrix
#
# The positive-first confusion matrix puts the actual class on the rows, the
# predicted class on the columns, and lists the positive class first:
#
# ```text
#                       predicted
#                  positive    negative
# actual positive     TP          FN
# actual negative     FP          TN
# ```
#
# This differs from scikit-learn's internal numeric array order, which uses
# negative first when labels are `[0, 1]`. In this project we compute with
# scikit-learn, then explicitly rename and display the counts in the
# positive-first order.
#
# In the medical example:
#
# - true positive: the test says cancer, and the person really has cancer;
# - false negative: the test says no cancer, but the person really has cancer;
# - false positive: the test says cancer, but the person does not have cancer;
# - true negative: the test says no cancer, and the person does not have cancer.
#
# In churn terms:
#
# - true positive: the model predicts churn, and the customer churns;
# - false negative: the model predicts no churn, but the customer churns;
# - false positive: the model predicts churn, but the customer does not churn;
# - true negative: the model predicts no churn, and the customer does not churn.
#
# These four counts are important because false positives and false negatives
# have different practical meanings. In churn prediction, a false positive may
# mean wasting retention effort on a customer who would have stayed anyway. A
# false negative may mean missing a customer who actually leaves.

# %% [markdown]
# ## Accuracy
#
# Accuracy is the proportion of all predictions that are correct:
#
# ```text
# accuracy = (TP + TN) / (TP + FP + TN + FN)
# ```
#
# Accuracy is useful when the classes are balanced and the costs of false
# positives and false negatives are similar. It can be misleading when one class
# is much more common than the other.
#
# In this project, about 73.46% of training customers do not churn. A model that
# always predicts no churn can therefore obtain about 73.46% accuracy while
# detecting none of the churners. This is why accuracy alone is not sufficient.

# %% [markdown]
# ## Recall, specificity, precision, and F1
#
# Recall, also called true positive rate, asks:
#
# ```text
# Of all actual positives, how many did we catch?
# ```
#
# ```text
# recall = TP / (TP + FN)
# ```
#
# In churn terms, recall measures the fraction of real churners that the model
# detects.
#
# Specificity, also called true negative rate, asks:
#
# ```text
# Of all actual negatives, how many did we correctly reject?
# ```
#
# ```text
# specificity = TN / (TN + FP)
# ```
#
# In churn terms, specificity measures the fraction of non-churners that the
# model correctly does not flag.
#
# Precision asks:
#
# ```text
# Of all predicted positives, how many were actually positive?
# ```
#
# ```text
# precision = TP / (TP + FP)
# ```
#
# In churn terms, precision measures how reliable the churn alerts are.
#
# F1-score combines precision and recall:
#
# ```text
# F1 = 2 * precision * recall / (precision + recall)
# ```
#
# F1 is useful as a compact summary when both precision and recall matter, but it
# hides the separate practical meanings of false positives and false negatives.

# %% [markdown]
# ## Balanced accuracy
#
# Balanced accuracy averages recall and specificity:
#
# ```text
# balanced accuracy = (recall + specificity) / 2
# ```
#
# This makes it useful under class imbalance. A majority-class classifier can
# have high ordinary accuracy, but its balanced accuracy will reveal that it is
# ignoring the minority class.
#
# For a binary task, a classifier that always predicts one class has balanced
# accuracy equal to 0.5, because it performs perfectly on one class and
# completely fails on the other.

# %% [markdown]
# ## ROC-AUC and PR-AUC
#
# Many classifiers produce a score or probability, not only a hard class label.
# By changing the threshold used to turn scores into classes, we get different
# tradeoffs between false positives and false negatives.
#
# The ROC curve plots:
#
# ```text
# x-axis: false positive rate = FP / (FP + TN)
# y-axis: true positive rate = TP / (TP + FN)
# ```
#
# ROC-AUC summarizes how well the model ranks positives above negatives across
# all possible thresholds.
#
# The precision-recall curve plots:
#
# ```text
# x-axis: recall
# y-axis: precision
# ```
#
# PR-AUC summarizes how precise the model remains as it tries to recover more
# positives.
#
# ROC-AUC and PR-AUC often point in the same direction when one model is clearly
# better. They can differ under class imbalance. A model can have a reasonable
# ROC-AUC but weak precision if many predicted positives are false positives.
# Since churn is the minority class and the positive class is the focus, PR-AUC
# is an important complement to ROC-AUC.

# %% [markdown]
# ## Thresholds and calibration
#
# A probabilistic classifier often predicts:
#
# ```text
# p(churn | customer features)
# ```
#
# The default class threshold is usually 0.5:
#
# ```text
# predict churn if p(churn) >= 0.5
# ```
#
# But 0.5 is not automatically the right threshold. In churn prediction, the best
# threshold depends on the relative cost of missing a churner versus incorrectly
# targeting a customer who would not churn.
#
# Calibration is a different question. A model is calibrated if predicted
# probabilities behave like real probabilities. For example, among customers
# assigned a churn probability near 0.30, about 30% should actually churn.
#
# Threshold tuning and calibration will become more important once we fit real
# probability models. For this simple-baseline section, the main goal is to
# define the language and metrics.

# %% [markdown]
# ## Class imbalance strategies and why they are not applied yet
#
# This section evaluates baselines under the natural training distribution.
# Resampling strategies such as random oversampling, random undersampling, and
# SMOTE are not applied yet because they are training interventions for learned
# models.
#
# The validation and test distributions should remain realistic. If we use
# resampling later, it must happen only inside each cross-validation training
# fold. The validation fold must remain untouched. The incorrect workflow would
# be:
#
# ```text
# oversample or SMOTE the full training set
# then cross-validate on the modified data
# ```
#
# because validation-fold information can leak into the resampled training data.
# The correct workflow is:
#
# ```text
# for each cross-validation split:
#     fit preprocessing on the training fold only
#     resample the training fold only
#     fit the model on the resampled training fold
#     evaluate on the untouched validation fold
# ```
#
# In Python, this usually requires an imbalanced-learn pipeline rather than an
# ordinary scikit-learn pipeline, because the sampler changes both `X` and `y`.
# Resampling, class weights, and threshold tuning will be considered later after
# learned models have been introduced.

# %% [markdown]
# ## Preprocessing infrastructure
#
# Most simple baselines in this notebook do not need feature preprocessing. The
# dummy classifiers ignore the feature matrix, and the EDA-inspired rule works
# directly with a few original categorical columns.
#
# Nevertheless, this notebook introduces the reusable preprocessing objects
# because later learned models will depend on them.

# %%
scaled_preprocessor = make_scaled_preprocessor()
unscaled_preprocessor = make_unscaled_preprocessor()

preprocessor_check = pd.DataFrame(
    {
        "preprocessor": ["scaled_preprocessor", "unscaled_preprocessor"],
        "intended_use": [
            "linear, distance-based, margin-based, and neural models",
            "tree-based models and models not requiring numeric scaling",
        ],
    }
)

preprocessor_check

# %% [markdown]
# ## Define simple baseline models
#
# The baselines in this section are deliberately simple. They tell us what kind
# of performance can be achieved without fitting a flexible model.
#
# ### Majority-class baseline
#
# Always predicts the most frequent class in the training fold.
#
# ### Prior-probability baseline
#
# Uses the empirical class probabilities from the training fold. Its hard class
# prediction is still the majority class, but its probability output is the
# class prior.
#
# ### Stratified random baseline
#
# Randomly samples labels according to the class distribution in the training
# fold.
#
# ### Uniform random baseline
#
# Randomly samples labels uniformly. In binary classification, each class has
# probability 0.5.
#
# ### EDA-inspired rule baseline
#
# Uses a transparent risk score based on high-risk patterns from training-set
# EDA. It predicts churn when at least two of these conditions are true:
#
# - month-to-month contract;
# - electronic check payment;
# - fiber optic internet service;
# - no online security;
# - no tech support.

# %%
simple_baseline_estimators = {
    "Dummy: majority class": make_most_frequent_dummy_classifier(),
    "Dummy: prior probability": make_prior_probability_dummy_classifier(),
    "Dummy: stratified random": make_stratified_dummy_classifier(
        random_state=RANDOM_STATE
    ),
    "Dummy: uniform random": make_uniform_dummy_classifier(
        random_state=RANDOM_STATE
    ),
    "EDA-inspired rule": make_eda_inspired_rule_classifier(risk_threshold=2),
}

simple_baseline_estimators

# %% [markdown]
# ## Cross-validated simple-baseline evaluation

# %%
simple_baseline_results = []

for model_name, estimator in simple_baseline_estimators.items():
    result = evaluate_estimator_cv(
        model_name=model_name,
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    simple_baseline_results.append(result)

simple_baseline_results_df = make_results_dataframe(simple_baseline_results)
simple_baseline_metric_df = make_metric_dataframe(simple_baseline_results_df)
simple_baseline_confusion_df = make_confusion_matrix_dataframe(
    simple_baseline_results_df
)

simple_baseline_metric_df

# %% [markdown]
# ## Confusion-matrix counts in positive-first order
#
# The table below uses the positive-first confusion-matrix order:
#
# ```text
#                       predicted
#                  positive    negative
# actual positive     TP          FN
# actual negative     FP          TN
# ```
#
# Each row can therefore be read as the four cells of this matrix:
# `TP, FN, FP, TN`.

# %%
simple_baseline_confusion_df

# %% [markdown]
# ## Save simple-baseline tables

# %%
simple_baseline_metric_df.to_csv(SIMPLE_BASELINE_COMPARISON_PATH, index=False)
simple_baseline_confusion_df.to_csv(
    SIMPLE_BASELINE_CONFUSION_MATRIX_PATH,
    index=False,
)

save_check = pd.DataFrame(
    {
        "item": [
            "simple_baseline_model_comparison_path",
            "simple_baseline_model_comparison_exists",
            "simple_baseline_confusion_matrix_path",
            "simple_baseline_confusion_matrix_exists",
        ],
        "value": [
            str(SIMPLE_BASELINE_COMPARISON_PATH),
            SIMPLE_BASELINE_COMPARISON_PATH.exists(),
            str(SIMPLE_BASELINE_CONFUSION_MATRIX_PATH),
            SIMPLE_BASELINE_CONFUSION_MATRIX_PATH.exists(),
        ],
    }
)

save_check

# %% [markdown]
# ## Interpretation guide
#
# The majority-class and prior-probability baselines are expected to have the
# same hard predictions. They should have relatively high accuracy because
# non-churn is the majority class, but they should fail completely at detecting
# churners under the default class prediction. Their recall should therefore be
# zero.
#
# The stratified random baseline is expected to have nonzero recall only because
# it randomly predicts churn for some customers. It is not feature-informed.
#
# The uniform random baseline is a pure random benchmark. It may have higher
# recall than the majority-class model because it predicts churn more often, but
# this comes at the cost of many false positives.
#
# The EDA-inspired rule baseline is the only baseline here that uses feature
# information. If it improves balanced accuracy, recall, ROC-AUC, or PR-AUC
# relative to the dummy baselines, that confirms that the strongest EDA patterns
# contain real predictive signal. Its limitation is that it uses only a few
# manually chosen conditions and cannot learn subtler feature combinations.
#
# The later learned models should improve on these baselines in a more balanced
# way. In particular, a useful churn model should detect churners while avoiding
# an excessive number of false positives.

# %% [markdown]
# ## Summary
#
# This notebook established the evaluation foundation for the rest of the
# project:
#
# - the held-out test set remains untouched;
# - stratified cross-validation is used for development-stage evaluation;
# - confusion-matrix terminology is defined in the positive-first order;
# - accuracy, recall, specificity, precision, F1, balanced accuracy, ROC-AUC,
#   PR-AUC, thresholds, and calibration are explained;
# - resampling methods are deferred until learned models and must later be used
#   inside cross-validation training folds only;
# - reusable preprocessing infrastructure is initialized for later models;
# - simple dummy and rule-based baselines are evaluated.
#
# The next modelling section should start the learned-model sequence with linear
# classification:
#
# - linear decision boundaries;
# - least-squares classification;
# - logistic regression;
# - regularized logistic regression;
# - coefficient and probability interpretation.
