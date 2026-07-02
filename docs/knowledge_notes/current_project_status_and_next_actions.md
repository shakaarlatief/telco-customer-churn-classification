# Current Project Status and Next Actions

## Purpose

This file is the live tactical status tracker for the Telco Customer Churn classification project.

Use it to answer:

```text
Where is the project now?
What completed work is available in the repository?
What is the immediate next modelling step?
What must happen before the held-out test set is used?
```

It is intentionally shorter and more operational than the model knowledge notes, the final-comparison methodology documents, and the LaTeX report.

## Latest confirmed project checkpoint

The individual model-family workflows are complete through:

```text
12_multilayer_perceptrons_and_neural_networks
```

The formal LaTeX report includes the MLP section as Section 13 and has been compiled locally at 126 pages. The report is an executed-workflow artifact, not a substitute for the remaining final-selection stage.

The final-comparison implementation is also complete through Phase 8B:

```text
Phase 1: protocol, deterministic repeated outer splits, run storage, resume safety
Phase 2: persistent two-stage Optuna HPO inside each outer training partition
Phase 3: complete core C01-C23 candidate registry
Phase 4: deterministic F0/F1/F2 feature-policy layer
Phase 5: candidate-specific feature-policy routing
Phase 6: candidate-specific S0/S1/S2 feature-selection routing
Phase 7: fold-safe I0-I4 imbalance primitives
Phase 8A: fit-time imbalance pipeline adapter
Phase 8B: candidate-specific imbalance routing inside Optuna and final pipelines
```

All associated smoke tests passed on the local Windows environment before the Phase 8B changes were committed and pushed.

## Current project state

Completed and committed workflow stages:

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
09_bagging_and_random_forests
10_boosting
11_support_vector_machines
12_multilayer_perceptrons_and_neural_networks
```

The held-out test set remains untouched.

```text
Clean modelling dataset: 7043 rows
Development training set: 5634 rows
Held-out test set:       1409 rows
Positive class:          Churn_binary = 1
Development churn rate:  approximately 26.54%
```

### Workflow-number convention

The identifiers in this file refer to repository workflow and notebook numbers, not necessarily LaTeX report section numbers.

```text
Workflow 12:
    multilayer-perceptron workflow identifier 12

LaTeX report:
    Multilayer Perceptrons and Neural Networks is Section 13 because the report also
    contains data-audit, EDA, and methodology sections before the model-family sequence.
```

## Governing evaluation policy

All current model-family results and all future candidate-comparison results remain development-stage estimates until a single final procedure has been frozen.

```text
Allowed before final test evaluation:
    training-only preprocessing and feature engineering;
    inner-CV hyperparameter optimization;
    repeated nested cross-validation;
    training-only feature-selection and imbalance-policy comparisons;
    calibration and threshold studies that do not use the held-out test set;
    candidate uncertainty and stability analysis.

Not allowed before final model freeze:
    selecting candidate families with the held-out test set;
    selecting feature policies, hyperparameters, resampling, calibration, or thresholds
    with the held-out test set;
    repeatedly checking test metrics while development continues.
```

The held-out test set is used exactly once after the following are frozen:

```text
candidate family or stack
feature policy
feature-selection policy
imbalance policy
model and HPO configuration
calibration policy, when selected
threshold or operating-point policy
```

## Completed model-family checkpoint summary

Representative development-stage results are retained for orientation only. They are not a final ranking and do not replace the dedicated final comparison.

```text
Regularized logistic regression:
    pooled OOF average precision about 0.658
    pooled OOF ROC-AUC about 0.846

k-nearest neighbours:
    pooled OOF average precision about 0.628
    pooled OOF ROC-AUC about 0.836

Hybrid Gaussian-Bernoulli Naive Bayes:
    pooled OOF average precision about 0.615
    pooled OOF ROC-AUC about 0.822

Regularized decision tree:
    pooled OOF average precision about 0.628
    pooled OOF ROC-AUC about 0.824

Bagging and random forest:
    pooled OOF average precision about 0.662 and 0.660, respectively
    pooled OOF ROC-AUC about 0.846 and 0.847, respectively

Boosted trees:
    CatBoost, GradientBoostingClassifier, and XGBoost fixed-grid mean average precision
    estimates are approximately 0.673, 0.672, and 0.672.

Linear and RBF SVM:
    mean-fold average precision about 0.659 for both representative candidates.

Multilayer perceptron:
    pooled OOF average precision about 0.654.
```

The consistent lesson is that the leading boosted, bagged, regularized-linear, SVM, and MLP procedures are sufficiently close that final selection must use the predeclared repeated nested-CV protocol rather than informal comparison of historical fixed-grid point estimates.

## Available final-comparison infrastructure

### Candidate library

The implemented core registry contains 17 candidate families from the documented C01-C23 universe:

```text
Ridge classifier
Regularized logistic regression
k-nearest neighbours
Hybrid Gaussian-Bernoulli Naive Bayes
Decision tree
Extra Trees
Bagging
Random forest
AdaBoost
GradientBoostingClassifier
HistGradientBoostingClassifier
XGBoost
LightGBM
CatBoost
Linear SVM
RBF SVM
Multilayer perceptron
```

Each procedure is constructed as an unfitted, fold-safe pipeline. Parallel estimators are constrained to one native worker so the outer execution layer owns task-level parallelism.

### Feature and selection policies

```text
F0_RAW:
    raw cleaned predictors

F1_DOMAIN_ENRICHED:
    predeclared target-free service aggregates, tenure summaries, selected interactions,
    and one categorical contract-by-payment interaction

F2_LINEAR_EXPANDED:
    controlled nonlinear and interaction basis available only to ridge and logistic
    regression procedures

S0_NONE:
    no feature selection

S1_VARIANCE_MUTUAL_INFO:
    variance filtering followed by mutual-information SelectKBest

S2_L1_LOGISTIC_SELECT_FROM_MODEL:
    L1-logistic embedded selection
```

Feature-policy and feature-selection choices are made only inside the relevant inner HPO loop.

### Imbalance policies

```text
I0_NONE:
    preserve the observed class distribution

I1_CLASS_WEIGHT_BALANCED:
    compute balanced sample weights from the active fitting target

I2_RANDOM_OVERSAMPLING:
    duplicate minority training observations after representation preprocessing

I3_RANDOM_UNDERSAMPLING:
    remove majority training observations after representation preprocessing

I4_SMOTENC:
    synthesize mixed raw numeric/categorical observations before one-hot encoding
```

The policies are mutually exclusive. I4 is intentionally available only with F0 raw features because synthetic F1/F2 derived columns could violate the deterministic relationships implied by the underlying raw customer profile.

### Reproducibility and HPO safety

The final-comparison implementation provides:

```text
protocol, data, and environment fingerprints
persistent Optuna RDB studies
trial continuation after interruption
atomic result artifacts
SQLite run/task coordination
deterministic repeated stratified split generation
strict mismatch protection when a protocol or candidate contract changes
outer-task process parallelism with native estimator threads limited to one
```

The frozen intended evaluation design remains:

```text
5 outer folds x 10 repeats
Stage A: persistent 3-fold Optuna exploration
Stage B: 5-fold confirmation of the top Stage-A configurations
primary ranking metric: average precision
```

No master repeated nested-CV comparison has been run yet.

## Immediate next actions

### 1. Resolve the remaining F2 design review before a master run

The F2 feature policy is the one material pre-run design issue still requiring explicit review. Before the final comparison is frozen, inspect and revise it if needed to ensure that:

```text
there is no duplicate tenure-squared construct inherited from F1 and added again in F2;
TotalCharges interactions are justified despite TotalCharges being a cumulative quantity
strongly related to tenure and MonthlyCharges;
any final F2 revision is reflected in the feature-policy contract, smoke test, candidate
procedure fingerprint, and methodology documentation.
```

This is a protocol-freeze task, not an opportunity to search feature ideas against final-comparison results. Any change creates a new feature-policy contract and must be completed before the master repeated nested-CV run.

### 2. Freeze the master-comparison configuration and run a pilot

Before launching the full 5 x 10 repeated nested-CV comparison:

```text
confirm candidate inclusion and search budgets;
confirm the final F0/F1/F2, S0/S1/S2, and I0-I4 compatibility contracts;
run a small but realistic end-to-end pilot using the persistent runner;
inspect stored artifacts, resume behavior, selected-configuration records, runtimes, and
failure diagnostics;
freeze the full comparison protocol revision.
```

### 3. Run the master training-only candidate comparison

The full run should compare complete candidate procedures, not bare estimators. It must persist outer-fold predictions, selected configurations, inner-search records, runtime information, warnings, and failures.

### 4. Perform post-ranking calibration and threshold work

Calibration and threshold policy are not selected from the held-out test set. After the ranking comparison identifies a defensible finalist set:

```text
compare calibration only where probability use is operationally relevant;
perform cross-fitted threshold or capacity/cost analysis;
keep margin-only SVM scores distinct from calibrated probabilities;
consider stacking only after constituent procedures and out-of-fold evidence are frozen.
```

### 5. Final development-only decision and one test evaluation

```text
select one procedure or justified stack using only development evidence;
rerun its frozen search on all 5,634 development rows;
fit the selected calibration and threshold policy using development data only;
write a complete final-pipeline manifest;
evaluate once on the untouched 1,409-row test set.
```

## Documentation and build state

```text
00_documentation_workflow.md:
    stable documentation architecture and collaboration rules

01_model_inventory_and_roadmap.md:
    strategic model-family and final-selection roadmap

current_project_status_and_next_actions.md:
    live tactical state and immediate actions

context_history/telco_churn_chat_handoff_context_6.md:
    standalone continuation snapshot for the current post-Phase-8B state
```

The LaTeX report compiles locally through TinyTeX. Cosmetic report cleanup remains deferred:

```text
long-title collision in the table of contents
label-width cleanup for Figures 58 and 59
```

Those presentation issues are separate from the current final-comparison implementation work.
