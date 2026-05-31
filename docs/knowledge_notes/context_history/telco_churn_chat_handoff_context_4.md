# Telco Customer Churn Classification: Chat Handoff Context 4

## Purpose of this handoff

Use this file when continuing the Telco Customer Churn classification project in a new chat.

This handoff is intended to be standalone. It supersedes earlier handoff files for the current state, while earlier files remain useful for detailed background on raw data audit, cleaning, splitting, EDA, methodology cleanup, and earlier model sections.

## Project identity

Project:

```text
Telco Customer Churn binary classification
```

Repository:

```text
https://github.com/shakaarlatief/telco-customer-churn-classification
```

Goal:

```text
Build a professional, portfolio-ready classification project and use it as a reusable reference for machine-learning classification knowledge.
```

The project deliberately applies many model families, not only the best one, because it is meant to preserve modelling knowledge for future projects.

## User preferences and constraints

The user prefers:

```text
professional, portfolio-ready work
deep technical explanations
standalone mathematical explanations
long report sections when useful
notebooks that are educational but not overloaded
knowledge notes for reusable deep theory
LaTeX report for polished public explanation
no emojis in technical/professional responses
no em dashes
explicit mention of changes made
no silent deletion of useful content
```

Do not write report sections as if they are lecture notes. The source slides and course material can guide the work, but the report should read as a standalone technical report.

When updating code or files, be careful not to overwrite good current content. If the exact current file is not available, ask the user for it or provide exact replacement files/snippets.

## Latest confirmed GitHub state at time of this handoff update

Latest confirmed commit at the time this handoff was last updated:

```text
d91fa9bf086bdfba58e4118d371c882e71452b05
Update project status after decision trees
```

Important note:

```text
The previous status-update commit was pushed after the decision-tree modelling commit, but the replacement files used there still contained stale checkpoint text in some places. This version corrects those live status and roadmap references.
```

The actual section 08 modelling commit is:

```text
08fb64873d4c8a929cfde529638d2e1ed49fcd5d
Add decision tree modelling section
```

That commit includes:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
reports/latex/main.tex
reports/latex/main.pdf
reports/figures/decision_tree_*.png
reports/tables/decision_tree_*.csv
```

When starting a later chat from this handoff, first check GitHub for newer commits. Treat the commit listed above as the latest confirmed checkpoint only as of the time this handoff was written.

## Dataset state

Clean modelling dataset:

```text
7043 observations
```

Training set:

```text
5634 observations
about 26.54% churn
```

Test set:

```text
1409 observations
about 26.54% churn
```

Target:

```text
Churn_binary
```

Positive class:

```text
1 = churn
0 = no churn
```

The held-out test set must remain unused until final evaluation.

## Feature groups

Numeric:

```text
tenure
MonthlyCharges
TotalCharges
```

Categorical:

```text
SeniorCitizen
gender
Partner
Dependents
PhoneService
PaperlessBilling
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaymentMethod
```

Excluded from modelling:

```text
customerID
```

Reason:

```text
unique identifier, not a generalizable customer characteristic
```

## Important data-audit decisions

`TotalCharges` had 11 blank strings. These rows had `tenure = 0`. The correction was deterministic:

```text
convert TotalCharges to numeric
set blank TotalCharges to 0.0 only when tenure = 0
```

This was not target-informed imputation.

Train-test split:

```text
test_size = 0.20
random_state = 42
stratified by Churn_binary
```

Development uses:

```text
data/processed/train.csv
```

Final evaluation will use:

```text
data/processed/test.csv
```

## Documentation structure

Recommended structure:

```text
docs/knowledge_notes/
    00_documentation_workflow.md
    01_model_inventory_and_roadmap.md
    current_project_status_and_next_actions.md
    current_notebook_documentation_audit.md

    context_history/
        telco_churn_chat_handoff_context_2.md
        telco_churn_chat_handoff_context_3.md
        telco_churn_chat_handoff_context_4.md

    methodology/
        evaluation_foundations.md
        cross_validation_and_model_selection.md
        statistical_uncertainty_and_tests.md
        final_model_comparison_plan.md
        hyperparameter_tuning.md

    models/
        05_linear_classification_and_logistic_regression.md
        06_knn.md
        07_naive_bayes.md
        08_decision_trees.md
```

File roles:

```text
00_documentation_workflow.md:
    stable documentation rules

01_model_inventory_and_roadmap.md:
    strategic modelling roadmap

current_project_status_and_next_actions.md:
    tactical current status and next actions

current_notebook_documentation_audit.md:
    audit snapshot

context_history:
    handoff files for new chats

methodology notes:
    deep reusable evaluation/tuning/statistical theory

model notes:
    deep model-family theory

notebooks:
    executable workflow

reports/latex:
    polished report
```

## Completed sections

Completed and committed through the current checkpoint:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_statistical_evaluation_methodology
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
06_k_nearest_neighbours
07_naive_bayes
08_decision_trees
```

## Section 04 summary

Role:

```text
preprocessing, evaluation framework, and simple baselines
```

Important results:

```text
majority/prior baselines:
    accuracy about 0.735 but recall 0.000

EDA-inspired rule:
    recall about 0.907
    precision about 0.396
    many false positives
```

Main lesson:

```text
accuracy alone is misleading
learned models should beat dummy baselines and improve precision/specificity relative to the broad EDA rule
```

## Section 05 summary

Role:

```text
linear classification and logistic regression
```

Main results:

```text
Logistic L1 C=1:
    PR-AUC about 0.659
    ROC-AUC about 0.846

Logistic L2 C=1:
    PR-AUC about 0.658
    ROC-AUC about 0.846

Class-weighted L2:
    recall about 0.801
    lower precision and many more false positives
```

Main interpretation:

```text
L1 and L2 logistic regression are effectively tied
L2 C=1 is used as a stable interpretable benchmark
class weighting shifts the precision-recall tradeoff toward recall
the exact best C should not be overinterpreted
```

## Section 06 summary

Role:

```text
k-nearest neighbours as local, non-parametric, distance-based learning
```

Selected development configuration:

```text
k = 101
uniform weights
Manhattan distance
```

Main results:

```text
ROC-AUC about 0.836
PR-AUC about 0.628
```

Main interpretation:

```text
small k values are noisy
larger smoother neighbourhoods perform better
kNN improves substantially over default k=5
logistic regression remains stronger by ranking metrics
```

## Section 07 summary

Role:

```text
Naive Bayes and generative classification
```

Important source-code addition:

```text
HybridGaussianBernoulliNB in src/telco_churn/models.py
```

Selected Naive Bayes model:

```text
Hybrid Gaussian-BernoulliNB alpha=1
```

Main results:

```text
Accuracy about 0.727
Balanced accuracy about 0.753
Precision about 0.491
Recall about 0.809
Specificity about 0.697
F1 about 0.611
ROC-AUC about 0.822
PR-AUC about 0.615
TP = 1209
FN = 286
FP = 1253
TN = 2886
```

Interpretation:

```text
hybrid NB is theoretically cleaner than full GaussianNB
categorical features contain much of the NB signal
Naive Bayes has high recall but moderate precision
conditional independence remains a limitation
```

## Section 08 summary

Role:

```text
single decision trees as nonlinear, rule-based classifiers and foundation for later tree ensembles
```

Important files:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
reports/figures/decision_tree_*.png
reports/tables/decision_tree_*.csv
```

Topics covered:

```text
recursive partitioning
split nodes and terminal leaves
leaf class proportions
Gini impurity
entropy and information gain
ranking with decision trees
stepwise tied score structure
pre-pruning
cost-complexity pruning
validation discipline for pruning and hyperparameter selection
feature importance
top-level tree interpretation
```

Selected decision tree:

```text
criterion = gini
max_depth = 6
min_samples_split = 25
min_samples_leaf = 10
ccp_alpha = 0.0
```

Main results:

```text
Selected pre-pruned tree:
    accuracy about 0.789
    balanced accuracy about 0.701
    precision about 0.624
    recall about 0.514
    F1 about 0.564
    ROC-AUC about 0.824
    PR-AUC about 0.628
    TP = 769
    FN = 726
    FP = 463
    TN = 3676

Best cost-complexity-pruned tree:
    ROC-AUC about 0.822
    PR-AUC about 0.615

Default unrestricted tree:
    ROC-AUC about 0.648
    PR-AUC about 0.371
```

Interpretation:

```text
default unrestricted trees overfit strongly
regularization is essential for useful single-tree performance
pre-pruning gave the best single-tree result in the tried grids
cost-complexity pruning improved strongly over the default tree but did not beat the selected pre-pruned tree
the selected tree is useful and interpretable but does not overtake logistic regression
single trees motivate bagging, random forests, and boosting
```

## Major methodology decision that still applies

Section-level cross-validation results are development-stage estimates, not final performance claims.

Main ideas:

```text
reported metrics are finite-sample estimates
hyperparameter tuning creates selection optimism
repeated CV improves stability
nested CV evaluates tuning procedures
final test set is used once after all choices are fixed
threshold selection and calibration are model-selection decisions
```

## Current source modules

Important modules:

```text
src/telco_churn/config.py
src/telco_churn/data.py
src/telco_churn/preprocessing.py
src/telco_churn/models.py
src/telco_churn/evaluation.py
src/telco_churn/features.py
src/telco_churn/visualization.py
```

Important reusable model/evaluation utilities include:

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
EDAInspiredRuleClassifier
make_ridge_classifier
make_l2_logistic_regression_classifier
make_l1_logistic_regression_classifier
HybridGaussianBernoulliNB
make_hybrid_gaussian_bernoulli_nb_classifier
extract_linear_model_coefficients
save_regularization_metric_plot
save_coefficient_plot
save_threshold_tradeoff_plot
save_roc_curve_plot
save_precision_recall_curve_plot
```

Decision-tree helper code is currently notebook-local because it is section-specific rather than stable package functionality.

## Collaborative workflow for future model sections

For each model-family section, use this loop:

```text
1. Create/update the `.md` knowledge note first.
2. Create/update the notebook source `.py`.
3. User runs the `.py` locally.
4. User sends back the executed `.ipynb` plus generated tables/figures/files.
5. Assistant updates the `.py` and, when useful, the `.ipynb` interpretation using the actual outputs.
6. Assistant writes or revises the LaTeX report section.
7. User compiles/checks the report.
8. Fix issues, then commit.
```

Do not write final report claims before seeing the actual executed results. If the assistant does not have the current local version of a file, ask the user to upload it or provide exact copy-paste replacement text.

## Immediate next actions

Before continuing modelling in the new chat:

```text
1. Check GitHub for commits newer than the checkpoint listed above.
2. Confirm the working tree is clean locally.
3. Start section 09: bagging and random forests, following the collaborative workflow described above.
```

Next modelling section:

```text
09_bagging_and_random_forests
```
