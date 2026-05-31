# Telco Customer Churn Classification: Chat Handoff Context 4

## Purpose of this handoff

Use this file when continuing the Telco Customer Churn classification project in a new chat.

This handoff is intended to be standalone. It supersedes earlier handoff files for the current state, while earlier files remain useful for detailed background on raw data audit, cleaning, splitting, EDA, methodology cleanup, and early model-family sections.

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

Do not write report sections as if they are lecture notes. The source slides and course material should guide the work, but the report should read as a standalone technical report.

The project workflow is collaborative:

```text
Assistant reads project files and course slides.
Assistant creates the actual files.
User places files locally, runs notebooks/LaTeX, and pushes to Git.
Codex is used only for small mechanical tasks unless explicitly requested.
```

When updating code or files, be careful not to overwrite good current content. If the exact current file is not available, ask the user to upload it or provide exact copy-paste replacement text.

## Latest project state at time of this handoff update

Section 08, decision trees, was completed locally in the chat workflow.

Prepared files:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
```

The report was compiled locally after adding the decision-tree section to `reports/latex/main.tex`:

```latex
\newpage
\input{sections/08_decision_trees}
```

The compiled report shows the decision-tree section as report Section 9 with subsections from recursive partitioning through summary.

When starting a later chat from this handoff, first check GitHub for newer commits. This handoff records the completed local section 08 workflow, but the exact commit hash may need to be checked after the user pushes.

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

Completed through the section 08 local workflow:

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

Baselines:

```text
majority class
prior probability
stratified random
uniform random
EDA-inspired rule
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

Models:

```text
RidgeClassifier
L2 logistic regression
L1 logistic regression
class-weighted L2 logistic regression
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

Grid:

```text
n_neighbors = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
weights = ["uniform", "distance"]
p = [1, 2]
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

Models:

```text
GaussianNB numeric only
BernoulliNB categorical only
Hybrid Gaussian-BernoulliNB
GaussianNB full transformed
```

Important source-code addition:

```text
HybridGaussianBernoulliNB in src/telco_churn/models.py
```

Reason:

```text
The Telco feature space is mixed:
    numeric features should use Gaussian likelihoods
    one-hot categorical indicators should use Bernoulli likelihoods
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
logistic regression remains strongest so far by PR-AUC and ROC-AUC
```

## Section 08 summary

Role:

```text
decision trees as the first single-tree, nonlinear, rule-based classifier
foundation for later bagging, random forests, and boosting
```

Files created or updated:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
reports/latex/main.tex
```

Important theory included:

```text
recursive partitioning
split rules and terminal leaves
local leaf churn proportions
hard predictions from leaf proportions
ranking by leaf churn proportions
stepwise scores and ties
Gini impurity
entropy and information gain
weighted child impurity
impurity reduction
pre-pruning
cost-complexity pruning
decision stumps
validation discipline for pruning and hyperparameter tuning
```

Important validation discussion from this section:

```text
Pruning strength is a hyperparameter, just like max_depth or min_samples_leaf.
For this section, ccp_alpha was tuned by ordinary training-set cross-validation.
This is appropriate for development-stage model selection.
If a separate validation set is held out for higher-level model-family comparison, that same validation set should not also be used to tune pruning.
A stricter estimate of the full tune-and-select procedure would require nested validation.
```

Experiments:

```text
decision stump
default unrestricted decision tree
pre-pruned tree grid
cost-complexity pruning grid
selected pre-pruned tree
threshold diagnostics
ROC curve
precision-recall curve
truncated tree plot
impurity-based feature importance
```

Selected development tree:

```text
variant = pre-pruned grid
criterion = gini
max_depth = 6
min_samples_split = 25
min_samples_leaf = 10
ccp_alpha = 0
```

Selected tree results:

```text
Accuracy about 0.789
Balanced accuracy about 0.701
Precision about 0.624
Recall about 0.514
Specificity about 0.888
F1 about 0.564
ROC-AUC about 0.824
PR-AUC about 0.628
```

Comparison results:

```text
Selected pre-pruned tree:
    ROC-AUC about 0.824
    PR-AUC about 0.628
    TP = 769
    FN = 726
    FP = 463
    TN = 3676

Best cost-complexity-pruned tree:
    ROC-AUC about 0.822
    PR-AUC about 0.615
    TP = 807
    FN = 688
    FP = 495
    TN = 3644

Decision stump:
    ROC-AUC about 0.726
    PR-AUC about 0.413
    TP = 0
    FN = 1495
    FP = 0
    TN = 4139

Default unrestricted tree:
    ROC-AUC about 0.648
    PR-AUC about 0.371
    TP = 723
    FN = 772
    FP = 777
    TN = 3362
```

Main interpretation:

```text
The unrestricted tree overfits and performs poorly in ranking metrics.
Regularization is essential for single decision trees.
Pre-pruning with moderate depth and minimum leaf-size constraints gives the strongest single-tree result in the tried grid.
Cost-complexity pruning substantially improves over the unrestricted tree but does not beat the best pre-pruned tree here.
The decision stump illustrates that hard classification and ranking can differ: it predicts no churn at threshold 0.5 but still has useful ROC-AUC because the leaf scores rank customers somewhat.
The selected tree is interpretable and nonlinear but does not overtake logistic regression.
This motivates bagging and random forests, because averaging many trees can reduce single-tree variance and instability.
```

Selected tree interpretation:

```text
The first split is on Contract_Month-to-month.
Top impurity-based feature importances include:
    Contract_Month-to-month
    InternetService_Fiber optic
    tenure
    TotalCharges
    MonthlyCharges
```

Important caution:

```text
The selected tree was refitted on the full training set only for interpretation.
That refit is not used to estimate performance.
Impurity-based feature importance is not causal and can be affected by correlated predictors and feature structure.
```

## Major methodology decision still active

The project paused before decision trees to improve statistical evaluation methodology.

Reason:

```text
model sections compare tuned models and close hyperparameter settings
small cross-validated differences should not be overinterpreted
the report should explain true metric versus sample estimate, uncertainty,
CV, repeated CV, nested CV, tests, and final test evaluation discipline
```

Main ideas:

```text
reported metrics are finite-sample estimates
section-level CV results are development-stage estimates
hyperparameter tuning creates selection optimism
repeated CV improves stability
nested CV evaluates tuning procedures
final test set is used once after all choices are fixed
bootstrap CIs should be used for the single final model's test metrics
paired bootstrap differences should be used only before final model selection
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

Decision-tree section helpers were kept in the notebook source because they are section-specific and not stable reusable project abstractions.

## Local LaTeX setup

The user compiles locally with TinyTeX / TeX Live 2024.

`pdflatex` path:

```text
C:\Users\shaka\AppData\Roaming\TinyTeX\bin\windows\pdflatex.exe
```

`tlmgr.bat` path:

```text
C:\Users\shaka\AppData\Roaming\TinyTeX\bin\windows\tlmgr.bat
```

Repository configured:

```bash
/c/Users/shaka/AppData/Roaming/TinyTeX/bin/windows/tlmgr.bat option repository https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/tlnet-final
```

Installed packages include:

```text
enumitem
microtype
```

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

Before continuing modelling:

```text
1. Check whether the section 08 files were committed and pushed.
2. If not, commit/push the section 08 files.
3. Suggested commit message: Add decision tree modelling section.
4. Start section 09: bagging and random forests.
```

Section 09 should follow the same workflow:

```text
1. Read the relevant tree/ensemble slides and current project files.
2. Create the bagging/random-forest knowledge note.
3. Create the notebook source.
4. User runs locally and returns outputs.
5. Update interpretations from actual outputs.
6. Write the report section.
7. User compiles and checks.
```

## Next modelling stage

Next section:

```text
09_bagging_and_random_forests
```

Expected topics:

```text
bootstrap aggregation
single-tree variance
variance reduction by averaging
bagged decision trees
random forests
feature subsampling to decorrelate trees
out-of-bag intuition
forest-size effects
max_features
minimum leaf size
feature importance limitations
```

Expected experiments:

```text
bagged trees
random forest
small or moderate hyperparameter grid
comparison against selected single tree
threshold diagnostics
ROC and precision-recall curves
feature importance diagnostics
possibly out-of-bag score discussion if implemented
```
