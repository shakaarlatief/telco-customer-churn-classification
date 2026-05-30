# Telco Customer Churn Classification: Chat Handoff Context 3

## Purpose of this handoff

Use this file when continuing the Telco Customer Churn classification project in a new chat.

This handoff is intended to be standalone. It supersedes earlier handoff files for the current state, while earlier files remain useful for detailed background on raw data audit, cleaning, splitting, EDA, and early documentation decisions.

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
4d6a61991110dde65b8720197d3c18a2be982f3b
Enforce single final test model evaluation policy
```

This commit includes:

```text
documentation cleanup
four methodology knowledge notes
statistical evaluation methodology report section
revised report wording for sections 05, 06, and 07
strict single-final-test-model policy
compiled report PDF update, if locally generated and committed
```

When starting a later chat from this handoff, first check GitHub for newer commits. Treat this commit as the latest confirmed checkpoint only as of the time this handoff was written.

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

## Documentation structure after cleanup

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

Completed and committed through the latest confirmed checkpoint at the time this handoff was written:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
06_k_nearest_neighbours
07_naive_bayes
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

## Major methodology decision made before this handoff update

Before continuing to decision trees, the project paused to improve statistical evaluation methodology.

Reason:

```text
model sections compare tuned models and close hyperparameter settings
small cross-validated differences should not be overinterpreted
the report should explain true metric versus sample estimate, uncertainty,
CV, repeated CV, nested CV, tests, and final test evaluation discipline
```

Methodology knowledge notes committed:

```text
docs/knowledge_notes/methodology/evaluation_foundations.md
docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
docs/knowledge_notes/methodology/final_model_comparison_plan.md
```

Main ideas:

```text
reported metrics are finite-sample estimates
section-level CV results are development-stage estimates
hyperparameter tuning creates selection optimism
repeated CV improves stability
nested CV evaluates tuning procedures
final test set is used once after all choices are fixed
bootstrap CIs should be used for the single final model's test metrics, and paired bootstrap differences should be used only before final model selection
threshold selection and calibration are model-selection decisions
```

## Report methodology rewrite committed

Committed files:

```text
reports/latex/main.tex
reports/latex/sections/04_statistical_evaluation_methodology.tex
reports/latex/sections/05_linear_classification_and_logistic_regression.tex
reports/latex/sections/06_k_nearest_neighbours.tex
reports/latex/sections/07_naive_bayes.tex
```

Purpose:

```text
add statistical evaluation methodology before model sections
revise abstract
revise sections 05, 06, 07 to use statistically careful language
clarify that close differences are development evidence
clarify that final test performance is deferred
```

These report files are committed as of the latest confirmed checkpoint listed above.

## Documentation cleanup committed

Committed files:

```text
docs/knowledge_notes/00_documentation_workflow.md
docs/knowledge_notes/01_model_inventory_and_roadmap.md
docs/knowledge_notes/current_project_status_and_next_actions.md
docs/knowledge_notes/current_notebook_documentation_audit.md
docs/knowledge_notes/context_history/telco_churn_chat_handoff_context_3.md
```

Purpose:

```text
separate stable workflow rules from tactical next steps
make current_project_status_and_next_actions.md the live task file
make current_notebook_documentation_audit.md an audit snapshot
update roadmap to current completed state
create this standalone handoff for new chats
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

Installed packages:

```text
enumitem
microtype
```

## Immediate next actions

Before continuing modelling in the new chat:

```text
1. Check GitHub for commits newer than the checkpoint listed above.
2. Confirm the working tree is clean locally.
3. Start section 08: decision trees.
```


## Next modelling stage: decision trees

Expected decision tree knowledge note topics:

```text
recursive partitioning
feature-space regions
leaf prediction
Gini impurity
entropy
information gain / impurity reduction
tree depth
min samples split
min samples leaf
overfitting
pruning
cost-complexity pruning
decision stumps
interpretability
threshold behaviour
```

Expected notebook workflow:

```text
load training data only
use unscaled preprocessing where appropriate
evaluate a stump
evaluate a default tree
tune max_depth, min_samples_leaf, min_samples_split, criterion
possibly study cost-complexity pruning
save model comparison table
save confusion matrices
save threshold curve
save ROC and PR curves
compare against logistic regression, kNN, and Naive Bayes cautiously
```

## Language to use going forward

Use:

```text
development-stage cross-validated estimate
selected within the tried grid
representative candidate
strong evidence in the development workflow
small differences should be interpreted cautiously
final test evaluation is deferred
```

Avoid:

```text
definitively best
uniquely optimal
proves superiority
final performance
```

## Final project plan

After all model-family sections:

```text
1. Build a candidate shortlist.
2. Use repeated CV for stable tuning of serious candidates.
3. Consider nested CV if top model families are close.
4. Select final model and threshold using training data only.
5. Optionally calibrate probabilities with proper validation discipline.
6. Fit final model on full training data.
7. Evaluate once on untouched test data.
8. Report bootstrap confidence intervals.
9. Use paired bootstrap differences before final model selection, using validation or cross-validation outputs.
10. Add ablation and interpretability analysis if useful.
```


## Strict final test-set policy clarification

The final test set should be used for exactly one frozen final model.

Do not use the test set to compare multiple candidate models, additional candidate models, alternative thresholds, alternative calibration methods, or alternative preprocessing decisions. All model-family comparison, repeated CV, nested CV, statistical tests, paired bootstrap differences, McNemar-style comparisons, DeLong-style comparisons, threshold selection, calibration selection, and ablation decisions should happen before final test evaluation using training-only validation evidence.

After one final model is selected and frozen, evaluate that model once on the untouched test set. Bootstrap confidence intervals may be reported for that single final model's test metrics.
