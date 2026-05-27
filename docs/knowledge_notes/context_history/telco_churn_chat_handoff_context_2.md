# Telco Customer Churn Classification: Chat Handoff Context 2

## Purpose of this handoff file

This file is an updated continuation of the first handoff context file.

Use it when the chat becomes too long, when the conversation becomes slow, or when a new chat is started inside the Data Projects folder. It captures the project state, methodology decisions, documentation system, completed sections, modelling results, coding conventions, and immediate next steps.

This file should be treated as the current project memory. It supersedes the earlier handoff file where project status has changed, but the earlier file is still useful for detailed background on the first three notebooks and the early EDA process.

## Project identity

This project is the Telco Customer Churn binary classification project.

GitHub repository:

```text
https://github.com/shakaarlatief/telco-customer-churn-classification
```

The project is part of the broader Data Projects portfolio folder. The aim is not only to produce a good churn model, but to build a professional, portfolio-ready, reusable classification reference project.

The project deliberately applies and preserves knowledge from the Machine Learning material in a practical Python workflow. The user wants the project to become a reusable reference for later classification projects. This means using and documenting many model families, even if some are not ultimately the best model.

The report should still be written as a standalone technical report. Do not write “in the course” or “in the slides” inside report sections. The slides can guide the knowledge and modelling plan, but the final report should read professionally.

The project should preserve and explain:

```text
supervised learning framing
train, validation, and test discipline
preprocessing
missing values
disguised missing values
outliers
feature transformations
feature selection
class imbalance
baseline models
linear classifiers
logistic regression
kNN
Naive Bayes
decision trees
decision stumps
bagging
random forests
boosting
AdaBoost
gradient boosting
SVMs
RBF kernels
MLPs
evaluation metrics
threshold tuning
calibration
model comparison
```

## User preferences and style rules

The user prefers:

```text
professional, portfolio-ready work
deep technical explanations when discussing models
standalone mathematical explanations
clear code and reusable project modules
notebooks that are professional and not unnecessarily long
LaTeX report sections with polished explanations and results
knowledge notes for deep reusable theory
no emojis in technical/professional responses
no em dashes in output
explicit mention of changes made to code
no silent removal of good notes
```

When updating code, be careful not to overwrite good existing code. If the current file is not available, ask the user to upload it or provide patch-style snippets that the user can paste. This was explicitly appreciated by the user.

## Current repository structure and documentation system

The project now has a structured documentation system.

Recommended structure:

```text
docs/
  knowledge_notes/
    00_documentation_workflow.md
    01_model_inventory_and_roadmap.md
    current_notebook_documentation_audit.md

    methodology/
      evaluation_metrics.md
      hyperparameter_tuning.md

    models/
      05_linear_classification_and_logistic_regression.md
      06_knn.md
```

The documentation workflow is:

```text
knowledge note -> notebook/code -> run outputs -> notebook interpretation -> LaTeX report -> optional knowledge-note update
```

Document roles:

```text
knowledge notes:
    deep reusable theory, mathematics, assumptions, implementation implications

notebooks:
    executable workflow, concise explanation, outputs, figures, tables, result interpretation

LaTeX report:
    polished portfolio-ready explanation, selected math, method decisions, results, limitations

source modules:
    reusable functions, professional docstrings, no report narrative
```

Knowledge notes can live in `docs/knowledge_notes/` for now. They do not need to be hidden from the repo yet.

## Important methodology decisions

The full raw file may be inspected before splitting only for schema and data-quality issues:

```text
rows and columns
data types
duplicates
standard missing values
disguised missing values
target label validity
identifier checks
deterministic raw representation corrections
```

Feature-target EDA must not be done before the split.

The held-out test set must remain unused until final evaluation. It must not be used for:

```text
feature-target EDA
preprocessing design
feature engineering decisions
feature selection
model-family selection
hyperparameter tuning
threshold tuning
calibration decisions
resampling decisions
choosing metrics
deciding whether results are good enough
```

The project uses `data/processed/train.csv` for development and `data/processed/test.csv` only for the final test evaluation.

All current modelling development uses training-set cross-validation only.

## Data audit and splitting decisions

The raw audit found that `TotalCharges` contained 11 blank strings. These rows all had `tenure = 0`. The correction was deterministic:

```text
convert TotalCharges to numeric
set blank TotalCharges to 0.0 only where tenure = 0
```

This was not mean imputation, median imputation, model-based imputation, or target-informed correction.

`customerID` is excluded from modelling because it is a unique identifier and does not represent a generalizable customer characteristic.

The binary target is:

```text
Churn_binary = 1 if Churn = Yes
Churn_binary = 0 if Churn = No
```

Positive class:

```text
churn
```

Train-test split:

```text
test_size = 0.20
random_state = 42
stratified by Churn_binary
```

Training set:

```text
5634 rows
churn rate approximately 26.54%
```

Test set:

```text
1409 rows
churn rate approximately 26.54%
```

Saved processed outputs:

```text
data/processed/train.csv
data/processed/test.csv
```

## Completed sections and commits

The project has completed and committed:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
```

Recent commit messages included:

```text
Add evaluation methodology and simple baseline workflow
Add linear classification and logistic regression section
```

The report title subtitle was updated to:

```latex
\large A Training-Only Workflow from Data Audit to Linear Classification
```

This better matches the report scope.

## Current notebooks

Current notebook workflow files:

```text
notebooks/01_raw_data_audit.py
notebooks/01_raw_data_audit.ipynb

notebooks/02_cleaning_and_splitting.py
notebooks/02_cleaning_and_splitting.ipynb

notebooks/03_training_set_eda.py
notebooks/03_training_set_eda.ipynb

notebooks/04_preprocessing_and_simple_baselines.py
notebooks/04_preprocessing_and_simple_baselines.ipynb

notebooks/05_linear_classification_and_logistic_regression.py
notebooks/05_linear_classification_and_logistic_regression.ipynb
```

The `.py` files are the source workflow files. The `.ipynb` files are also kept for readability, running cells, saved outputs, and sharing results.

The notebooks are allowed to contain explanation and interpretation, but should not become giant textbooks. The deepest reusable theory belongs in knowledge notes; the polished final explanation belongs in the LaTeX report.

## Current LaTeX report structure

Current LaTeX sections:

```text
reports/latex/main.tex
reports/latex/main.pdf

reports/latex/sections/01_raw_data_audit.tex
reports/latex/sections/02_cleaning_and_splitting.tex
reports/latex/sections/03_training_set_eda.tex
reports/latex/sections/04_preprocessing_and_simple_baselines.tex
reports/latex/sections/05_linear_classification_and_logistic_regression.tex
```

Section `05` is integrated into `main.tex`.

The report compiles locally with TinyTeX / TeX Live 2024.

## Local LaTeX setup

The local compiler is TinyTeX / TeX Live 2024, not MiKTeX.

`pdflatex` path:

```text
C:\Users\shaka\AppData\Roaming\TinyTeX\bin\windows\pdflatex.exe
```

`tlmgr.bat` path:

```text
C:\Users\shaka\AppData\Roaming\TinyTeX\bin\windows\tlmgr.bat
```

The TeX Live 2024 frozen repository was set with:

```bash
/c/Users/shaka/AppData/Roaming/TinyTeX/bin/windows/tlmgr.bat option repository https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/tlnet-final
```

To install packages:

```bash
/c/Users/shaka/AppData/Roaming/TinyTeX/bin/windows/tlmgr.bat install PACKAGE_NAME
```

Packages installed:

```text
enumitem
microtype
```

Do not use MiKTeX Console for this project unless fully switching TeX distributions later.

## Section 03 EDA summary

The EDA uses the training set only.

Target distribution:

```text
Churn_binary = 0: 73.46%
Churn_binary = 1: 26.54%
```

Numeric features:

```text
tenure
MonthlyCharges
TotalCharges
```

Important numeric patterns:

```text
churners have lower tenure
churners tend to have higher MonthlyCharges
churners tend to have lower TotalCharges, closely related to shorter tenure
tenure and TotalCharges are strongly correlated
scatter matrix shows patterns but no clean pairwise separation
```

Pearson correlations with churn:

```text
tenure: about -0.35
MonthlyCharges: about 0.20
TotalCharges: about -0.19
```

Strong feature correlations:

```text
tenure and TotalCharges: about 0.83
MonthlyCharges and TotalCharges: about 0.65
```

Important categorical churn-rate differences:

```text
Contract
InternetService
PaymentMethod
OnlineSecurity
TechSupport
PaperlessBilling
SeniorCitizen
```

Selected churn rates:

```text
Month-to-month contract: 42.75%
One-year contract: 11.08%
Two-year contract: 2.87%
Fiber optic internet: 42.09%
DSL: 18.69%
No internet service: 7.25%
Electronic check: 45.74%
Paperless billing Yes: 33.80%
Paperless billing No: 16.02%
SeniorCitizen 1: 41.09%
SeniorCitizen 0: 23.70%
```

Important interpretation:

```text
These are associations in the training data, not causal effects.
No internet service and No phone service are structural categories, not missing values.
Weak marginal separation does not automatically mean a variable should be removed.
```

## Figure standards

The project uses centralized figure style defaults as a starting point, but figures are still judged individually.

Workflow:

```text
start with centralized report-style defaults
generate figures
inspect each figure individually
override specific settings only where needed
keep overrides local and intentional
```

Preference:

```text
readability over small text
professional report figures
not oversized or cramped
```

For large categorical grid figures, figures were split into Part I and Part II for readability.

## Section 04: preprocessing, evaluation, and simple baselines

Section `04` created the preprocessing and evaluation framework.

Important concepts introduced:

```text
ColumnTransformer
Pipeline
training-set cross-validation
positive and negative class definition
positive-first confusion matrix
accuracy
recall
specificity
precision
F1
balanced accuracy
ROC-AUC
PR-AUC
thresholds
calibration
class imbalance
dummy baselines
EDA-inspired rule baseline
```

Positive-first confusion matrix layout:

```text
                         predicted
                    positive    negative
actual positive        TP          FN
actual negative        FP          TN
```

Section `04` outputs include:

```text
reports/tables/simple_baseline_model_comparison.csv
reports/tables/simple_baseline_confusion_matrices.csv
reports/latex/sections/04_preprocessing_and_simple_baselines.tex
```

Knowledge note created:

```text
docs/knowledge_notes/methodology/evaluation_metrics.md
```

This note is a living reference and should later be updated if the project introduces:

```text
threshold optimization based on business costs
calibration curves
Brier score
cost-sensitive evaluation
confidence intervals for metrics
bootstrap uncertainty
formal paired model-comparison tests
final test-set reporting template
```

Current section-04 results interpretation:

```text
majority and prior baselines achieve high ordinary accuracy because no churn is the majority class
majority and prior baselines have zero recall and detect no churners
stratified and uniform dummy baselines are weak and noisy
EDA-inspired rule has high recall but many false positives
learned models should beat dummy baselines and improve precision/specificity compared with the broad EDA rule
```

## Section 05: linear classification and logistic regression

Section `05` is complete and committed.

Knowledge note:

```text
docs/knowledge_notes/models/05_linear_classification_and_logistic_regression.md
```

Notebook:

```text
notebooks/05_linear_classification_and_logistic_regression.py
notebooks/05_linear_classification_and_logistic_regression.ipynb
```

Report section:

```text
reports/latex/sections/05_linear_classification_and_logistic_regression.tex
```

Section `05` covers:

```text
linear score functions
decision boundaries
least-squares classification
RidgeClassifier
logistic regression
Bernoulli likelihood
log loss
L1 regularization
L2 regularization
class-weighted logistic regression
coefficient interpretation
threshold tradeoff
ROC curve
precision-recall curve
```

Section `05` outputs include:

```text
reports/tables/linear_model_comparison.csv
reports/tables/linear_model_confusion_matrices.csv
reports/tables/logistic_l2_regularization_results.csv
reports/tables/logistic_l1_regularization_results.csv
reports/tables/logistic_top_coefficients.csv
reports/tables/logistic_threshold_results.csv

reports/figures/logistic_l2_regularization_metrics.png
reports/figures/logistic_l1_regularization_metrics.png
reports/figures/logistic_top_coefficients.png
reports/figures/logistic_threshold_tradeoff.png
reports/figures/logistic_roc_curve.png
reports/figures/logistic_precision_recall_curve.png
```

Main model comparison results:

```text
Logistic L1 C=1:
  PR-AUC 0.6590
  ROC-AUC 0.8455
  balanced accuracy 0.7196

Logistic L2 C=1:
  PR-AUC 0.6584
  ROC-AUC 0.8456
  balanced accuracy 0.7195

Logistic L2 balanced C=1:
  PR-AUC 0.6569
  ROC-AUC 0.8455
  balanced accuracy 0.7654
  recall 0.8013

RidgeClassifier alpha=1:
  PR-AUC 0.6494
  ROC-AUC 0.8385
  balanced accuracy 0.7119
```

Interpretation:

```text
standard L1 and L2 logistic regression are almost identical
class-weighted logistic regression increases recall strongly but creates many more false positives
RidgeClassifier is slightly weaker but still a reasonable linear baseline
logistic regression clearly improves on dummy baselines
the EDA rule still has higher recall, but logistic regression is more balanced and precise
```

L2 regularization path:

```text
performance improves from very strong regularization C=0.001 to C=0.01
performance becomes stable around C=0.1 to C=100
logistic regression is not extremely sensitive once regularization is not too strong
```

L1 regularization path:

```text
very strong L1 regularization underfits
performance improves substantially at C=0.01 and C=0.1
higher C values are very similar
L1 and L2 performance are almost identical in this dataset
```

Coefficient interpretation:

Top positive churn-score associations:

```text
InternetService_Fiber optic
Contract_Month-to-month
TotalCharges
StreamingMovies_Yes
StreamingTV_Yes
PaymentMethod_Electronic check
```

Top negative churn-score associations:

```text
tenure
Contract_Two year
InternetService_DSL
MonthlyCharges
PaperlessBilling_No
No internet service indicators
```

Important note:

```text
Coefficients are fitted associations, not causal effects.
Numeric coefficients correspond to standardized numeric features.
Categorical coefficients correspond to one-hot encoded indicators.
```

ROC/PR results:

```text
ROC-AUC about 0.846
PR-AUC about 0.658
positive-rate baseline about 0.265
```

Interpretation:

```text
ROC curve indicates strong ranking ability versus random ranking
PR curve is far above the positive-rate baseline
PR curve is especially important because churn is the minority class
```

Threshold tradeoff:

```text
default 0.5 threshold is not necessarily optimal
lower thresholds increase recall but reduce precision and specificity
higher thresholds improve precision and specificity but miss more churners
final threshold tuning should be postponed until later model comparison and business-cost reasoning
```

## Hyperparameter tuning methodology

Knowledge note:

```text
docs/knowledge_notes/methodology/hyperparameter_tuning.md
```

The note covers:

```text
parameters vs hyperparameters
why tuning can overfit validation data
test-set discipline
cross-validation for hyperparameter selection
final refit after CV
grid search
random search
successive halving / multi-fidelity search
coarse-to-fine tuning
Bayesian optimization
Optuna
scikit-learn search tools
Ray Tune
KerasTuner
experiment tracking with MLflow / Weights & Biases
appropriate scales
parallelization
choosing tuning metrics
threshold tuning as hyperparameter tuning
resampling and tuning
early stopping
bias-variance perspective
honest reporting
```

Important project policy:

```text
simple models:
    small transparent grids

complex models:
    random search or Optuna can be considered

section 05:
    no Optuna, use grid search

later ensembles, SVMs, MLPs:
    Optuna may be considered if search space becomes large

test set:
    never used for tuning
```

## Source modules

The project has reusable modules in:

```text
src/telco_churn/
```

Key modules:

```text
config.py
data.py
preprocessing.py
models.py
evaluation.py
features.py
visualization.py
```

Important additions from sections 04 and 05:

```text
make_stratified_kfold
compute_binary_classification_metrics
get_out_of_fold_predictions
evaluate_estimator_cv
evaluate_threshold_grid
make_roc_curve_dataframe
make_precision_recall_curve_dataframe
make_confusion_matrix_dataframe
make_metric_dataframe

make_classifier_pipeline
make_ridge_classifier
make_l2_logistic_regression_classifier
make_l1_logistic_regression_classifier
EDAInspiredRuleClassifier

extract_linear_model_coefficients

save_regularization_metric_plot
save_coefficient_plot
save_threshold_tradeoff_plot
save_roc_curve_plot
save_precision_recall_curve_plot
```

Important coding note:

The scikit-learn logistic regression API produced deprecation warnings about `penalty`. A helper was added in `models.py` to support newer scikit-learn versions using `l1_ratio` for L1/L2 behaviour when needed.

If current file content is unknown, ask the user for the file before rewriting it. The user prefers patch snippets or complete files only when safe.

## Current immediate state

The most recent completed action:

```text
docs/knowledge_notes/models/06_knn.md was created and added by the user.
```

The next modelling stage is:

```text
06 k-nearest neighbours
```

## Section 06: kNN knowledge note

Knowledge note created:

```text
docs/knowledge_notes/models/06_knn.md
```

It covers:

```text
kNN as lazy / instance-based learning
training data storage
local classification
neighbour sets
majority voting
probability estimates from neighbour proportions
Euclidean distance
Manhattan distance
Minkowski distance
feature scaling
one-hot encoding and distance geometry
weighted voting
role of k
bias-variance tradeoff
nonlinear decision boundaries
curse of dimensionality
class imbalance
planned hyperparameter grid
notebook plan
report plan
```

kNN planned grid:

```text
n_neighbors = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
weights = ["uniform", "distance"]
p = [1, 2]
```

Current decision:

```text
No Optuna for kNN unless needed later.
No SMOTE yet.
No test set.
Use training-set stratified cross-validation only.
Use scaled preprocessing pipeline.
```

Next task:

```text
Create notebooks/06_knn.py
```

Expected notebook contents:

```text
load training data only
build scaled preprocessing pipeline
define kNN model factory if needed
evaluate baseline kNN
grid-search k, weights, and p using training-set CV
save full grid results
plot performance versus k
select representative kNN model
compute out-of-fold probabilities for selected model
save confusion matrix and metric tables
save ROC and precision-recall curves
save threshold tradeoff table and plot
interpret whether kNN adds value relative to logistic regression
```

Expected report section later:

```text
reports/latex/sections/06_knn.tex
```

## Important external references for later

External references planned for future sections:

```text
Kaggle notebook: Resampling strategies for imbalanced datasets
scikit-learn feature selection documentation
```

Feature selection topics to cover later:

```text
VarianceThreshold
SelectKBest
SelectPercentile
chi2
f_classif
mutual_info_classif
RFE
RFECV
SelectFromModel
L1-based feature selection
```

Class imbalance topics to cover later:

```text
class weights
random oversampling
random undersampling
SMOTE
resampling inside cross-validation only
```

## How to continue from this handoff

If starting from this file in a new chat:

1. Load or ask for the current repository files if code modifications are needed.
2. Do not assume the current content of source files if not available.
3. Continue with section `06_knn.py`.
4. Use patch-style snippets if unsure about overwriting source files.
5. After the user runs notebook 06, request the results and figures.
6. Update notebook 06 with concise interpretation.
7. Write `reports/latex/sections/06_knn.tex`.
8. Ask the user to compile and inspect the PDF.
9. Commit when the section is clean.
