# Telco Customer Churn Classification: Chat Handoff Context 7

## Purpose

Use this file when continuing the Telco Customer Churn classification project in a new chat.

This handoff supersedes earlier handoff files for the current working state. Earlier handoffs remain useful for detailed history of the raw audit, splitting, EDA, individual model workflows, and earlier documentation decisions.

The current work is the final-comparison runner and monitoring system, especially the v5 monitoring pilot.

## Project identity

Project:

```text
Telco Customer Churn binary classification
```

Repository:

```text
shakaarlatief/telco-customer-churn-classification
```

Goal:

```text
Build a professional, portfolio-ready churn-classification project and use it as a
reusable technical reference for classification modelling, evaluation, implementation,
and reporting.
```

The project deliberately studies many relevant model families. The aim is not merely to find a high score. It is to create a rigorous reusable reference for preprocessing, feature engineering, evaluation, model comparison, imbalance handling, calibration, thresholds, and final test discipline.

## User preferences and working rules

The user prefers:

```text
professional, portfolio-ready work
deep technical explanations with mathematics when useful
standalone report prose that does not read like course notes
long report sections when detail is useful
deep reusable knowledge notes
notebooks that are educational but not giant textbooks
LaTeX for the formal report
no emojis in technical or professional responses
no em dashes
explicit description of meaningful changes
no silent deletion or shortening of good content
```

When changing code or documents:

```text
inspect the current file state first
preserve good existing content
state every meaningful change
validate through appropriate checks
do not use Codex for foundational project content
```

Do not directly write to GitHub, modify repository files through a connected tool, stage changes, create commits, or push commits unless the user explicitly authorizes that specific action. Authorization for one action does not create continuing permission for later writes.

The default workflow remains local-first and user-controlled:

```text
1. Prepare a patch or downloadable artifact in chat.
2. State every meaningful change and intended destination.
3. User reviews, applies locally, runs checks, and inspects the diff.
4. User stages, commits, and pushes by default.
```

## Dataset state

```text
Clean modelling dataset: 7043 observations
Development training set: 5634 observations
Held-out test set:       1409 observations
Positive class:          Churn_binary = 1
Development churn rate:  approximately 26.54%
```

Feature groups:

```text
Numeric:
    tenure
    MonthlyCharges
    TotalCharges

Categorical:
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

`customerID` is excluded from modelling as a unique identifier.

## Strict evaluation policy

The held-out test set has not been used for model selection, threshold selection, calibration selection, feature-policy selection, or model-family comparison.

All current model tables, pilot metrics, and future candidate-comparison metrics are development-stage evidence until one final procedure is frozen.

Use:

```text
selected within the tried development grid
representative strong candidate
development-stage cross-validated estimate
small differences should be interpreted cautiously
final test evaluation is deferred
```

Do not claim:

```text
definitively best
uniquely optimal
proven superior
final performance
```

Final test policy:

```text
1. Complete final procedure selection on development data only.
2. Freeze one complete end-to-end pipeline or justified stack.
3. Rerun its frozen search on the 5,634 development rows.
4. Fit all learned steps using development data only.
5. Evaluate once on the untouched 1,409-row test set.
6. Report uncertainty for that one final evaluation where feasible.
```

## Completed model-family workflows

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

The LaTeX report contains the completed MLP section as Section 13 and has been compiled locally at 126 pages.

## Key development-stage model context

These values orient the final-selection stage. They are not a final ranking.

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

Decision tree:
    pooled OOF average precision about 0.628
    pooled OOF ROC-AUC about 0.824

Bagging:
    pooled OOF average precision about 0.662
    pooled OOF ROC-AUC about 0.846

Random forest:
    pooled OOF average precision about 0.660
    pooled OOF ROC-AUC about 0.847

Boosted trees:
    leading fixed-grid candidates around 0.672 to 0.673 average precision
    XGBoost pooled OOF diagnostic average precision about 0.670

Linear and RBF SVM:
    mean-fold average precision about 0.659 for both representative candidates

Multilayer perceptron:
    pooled OOF average precision about 0.654
```

The leading boosted, bagged, regularized-linear, SVM, and MLP procedures are close enough that final selection must use the frozen repeated nested-CV protocol rather than historical fixed-grid point estimates.

## Final-comparison design

The project has reusable implementation through Phase 8B for training-only comparison of complete procedures.

### Candidate registry

The implemented core candidate registry contains 17 families:

```text
C01  Ridge classifier
C02  Regularized logistic regression
C06  k-nearest neighbours
C07  Hybrid Gaussian-Bernoulli Naive Bayes
C08  Decision tree
C09  Extra Trees
C10  Bagging
C11  Random forest
C13  AdaBoost
C15  GradientBoostingClassifier
C16  HistGradientBoostingClassifier
C17  XGBoost
C18  LightGBM
C19  CatBoost
C21  Linear SVM
C22  RBF SVM
C23  Multilayer perceptron
```

Every procedure remains an unfitted, fold-safe pipeline. Native parallel estimators are configured with one worker or one thread because the outer runner controls process-level parallelism.

### Target master protocol

The target architecture, pending final protocol freeze, is:

```text
5 outer folds x 10 repeats

Stage A:
    candidate-specific persistent Optuna exploration
    3-fold stratified inner CV

Stage B:
    5-fold confirmation of top Stage-A configurations

Primary metric:
    average precision
```

No master repeated nested-CV comparison has been performed yet.

## Feature, selection, and imbalance policies

### Feature policies

```text
F0_RAW:
    raw cleaned predictors

F1_DOMAIN_ENRICHED:
    target-free service aggregates, tenure summaries, selected interactions, and one
    categorical contract-by-payment interaction

F2_LINEAR_EXPANDED:
    controlled nonlinear and interaction basis, available only to ridge and logistic
    regression
```

Important current state: F2 has been pruned during final-comparison implementation and pilot work, but the formal protocol documentation still needs to be revised before the master run. Review potential duplicate tenure-squared constructs and the rationale for `TotalCharges` interactions, because `TotalCharges` is cumulative and strongly related to tenure and monthly charges.

### Feature selection

```text
S0_NONE:
    no feature selection

S1_VARIANCE_MUTUAL_INFO:
    variance filtering followed by mutual-information SelectKBest

S2_L1_LOGISTIC_SELECT_FROM_MODEL:
    embedded L1-logistic selection
```

Selection is restricted to families for which it is coherent. Tree and native categorical boosting procedures retain S0 only.

### Imbalance treatment

```text
I0_NONE:
    no explicit adjustment

I1_CLASS_WEIGHT_BALANCED:
    fold-local balanced sample weights from the active fitting target

I2_RANDOM_OVERSAMPLING:
    fit-time-only random oversampling after representation preprocessing

I3_RANDOM_UNDERSAMPLING:
    fit-time-only random undersampling after representation preprocessing

I4_SMOTENC:
    raw-only mixed-data synthetic oversampling before one-hot encoding
```

The policies are mutually exclusive. F1 and F2 are not compatible with I4 because synthetic derived features could be internally inconsistent with their raw inputs.

The Phase 8B compatibility matrix is:

```text
Ridge and logistic regression:
    I0, I1, I2, I3, I4 with F0
    I0, I1, I2, I3 with F1 or F2

Linear SVM, RBF SVM, and MLP:
    I0, I1, I2, I3, I4 with F0
    I0, I1, I2, I3 with F1

kNN and hybrid Gaussian-Bernoulli Naive Bayes:
    I0, I2, I3, I4 with F0
    I0, I2, I3 with F1

Decision tree, Extra Trees, bagging, random forest, AdaBoost,
GradientBoostingClassifier, HistGradientBoostingClassifier,
XGBoost, LightGBM, and CatBoost:
    I0, I1
```

## Final-comparison implementation state

Completed phases:

```text
Phase 1:
    protocol, deterministic repeated splits, SQLite coordination, atomic artifacts,
    fingerprint and resume foundations

Phase 2:
    persistent two-stage Optuna HPO with Windows-safe study-resource cleanup and actual
    serial/interrupted/resumed/two-worker smoke coverage

Phase 3:
    complete 17-family core candidate registry and single-threaded builder contract

Phase 4:
    deterministic F0/F1/F2 feature-policy transformer layer

Phase 5:
    candidate-specific feature-policy routing inside HPO and pipelines

Phase 6:
    S0/S1/S2 feature-selection routing inside HPO and pipelines

Phase 7:
    I0-I4 fold-safe imbalance primitives and standalone sampler smoke coverage

Phase 8A:
    fit-time imbalance pipeline adapter, including balanced sample weights, random
    sampling, raw-only SMOTENC, and CatBoost sample-weight compatibility

Phase 8B:
    candidate-specific imbalance routing, conditional Optuna parameters, static
    categorical-distribution safety, and procedure-contract persistence
```

Smoke tests passed after Phase 8B for persistent nested HPO, core registry, feature policies, feature selection, imbalance primitives, imbalance topology, and candidate-specific imbalance routing.

## Nested-CV interpretation for the pilot

One outer task is one candidate family on one outer fold, for example:

```text
C19_CATBOOST r00/f02
```

For each outer task:

```text
Outer-validation fold:
    held aside until the final task-level evaluation

Outer-training partition:
    used for Stage A, Stage B, and final refit
```

Stage A:

```text
12 valid configurations x 3 inner folds = 36 fits
```

`valid 6/12` means six complete successful Stage-A configurations, not six folds.

Stage B:

```text
top 3 Stage-A configurations x 3 inner folds in the pilot = 9 fits
```

Stage B uses a new deterministic internal split of the same outer-training data. It does not use the outer fold.

Final outer evaluation:

```text
1 fit of the selected configuration on all outer-training data
1 prediction pass on the untouched outer-validation fold
```

Approximate successful pilot task total:

```text
36 Stage-A fits + 9 Stage-B fits + 1 final outer fit = 46 fits
```

The pilot has 18 outer tasks:

```text
6 representative candidates x 3 outer folds x 1 repeat = 18 tasks
```

So the small pilot can still produce hundreds of actual model fits.

## Monitoring-pilot history

### v1

First operational full-development pilot runner with six representative candidates. It helped validate the end-to-end structure but left scheduler and monitoring limitations.

### v2

Added monitorable persistent execution, progress sidecars, task registry inspection, and clean pause behavior. The v2 run was deliberately paused to improve monitoring.

### v3 and v4

Added timestamped coordinator logs, JSONL event logs, colorized terminal dashboards, alternate-screen rendering, compact event viewers, progress bars, and clearer AP labels.

The v4 run identifier was:

```text
pilot_pruned_f2_v4_observable
```

The v4 pilot completed 17 of 18 outer tasks and exposed one monitoring-side failure:

```text
C01_RIDGE_CLASSIFIER r00/f02
failed during Stage B because progress JSON persistence raised
PermissionError: [WinError 5] Access denied during os.replace(...)
```

This was not a legitimate Ridge model failure. It was a Windows file-lock issue in progress-sidecar monitoring.

### v5

v5 adds the configuration-history layer and better dashboard readability.

Current v5 run identifier:

```text
pilot_pruned_f2_v5_history
```

v5 adds:

```text
normal dashboard parameters one per line
details mode with recent completed configuration history
logs/configuration_history.log
logs/configuration_history.jsonl
per-configuration start, finish, duration, AP, best AP, parameters, and fold history
history mode in scripts/monitor_final_comparison.sh
```

The known Windows progress-sidecar lock issue remains deliberately deferred and may still recur in v5.

## Monitoring commands

Standard live dashboard:

```bash
bash scripts/monitor_final_comparison.sh
```

Refresh every 5 seconds:

```bash
bash scripts/monitor_final_comparison.sh --interval 5
```

Details view:

```bash
bash scripts/monitor_final_comparison.sh details
```

Recent coordinator events:

```bash
bash scripts/monitor_final_comparison.sh events
```

Durable configuration history:

```bash
bash scripts/monitor_final_comparison.sh history
```

One task's history:

```bash
bash scripts/monitor_final_comparison.sh history   --task-key c19_catboost__r00__f02
```

The watched views use the terminal alternate screen. They are visually stable but do not provide normal scrollback.

For a static, scrollable dashboard snapshot:

```bash
python scripts/final_comparison_status.py   --run-id pilot_pruned_f2_v5_history   --details
```

Check current helper options before relying on a helper shortcut:

```bash
bash scripts/monitor_final_comparison.sh --help
```

## Important artifact locations

For v5:

```text
artifacts/final_comparison/pilot_pruned_f2_v5_history/
```

Important files:

```text
logs/coordinator.log:
    compact human-readable run timeline

logs/coordinator_events.jsonl:
    machine-readable coordinator event stream

logs/configuration_history.log:
    scrollable human-readable configuration timing, score, fold, and parameter history

logs/configuration_history.jsonl:
    structured configuration history for analysis

logs/tasks/:
    task-level human logs and JSONL logs

progress/:
    live task progress sidecars

status/latest_invocation.json:
    current invocation state and timing

task_registry.sqlite:
    authoritative task-state database, inspect read-only only

optuna_studies/:
    persistent task-local Optuna databases, inspect read-only only
```

## Current known risk

The Windows progress-sidecar `PermissionError` must be fixed before the master comparison.

Required reliability work:

```text
retry transient Windows PermissionError / sharing violations around atomic replace
make progress-sidecar persistence best effort after bounded retries
surface monitoring-degraded or stale-progress warnings instead of crashing workers
ensure reporter.close() does not mask successful model results or meaningful exceptions
add smoke tests for temporary and persistent PermissionError cases
improve failed-task rendering so root cause and log path are visible
```

Until this is fixed, a monitoring-side task failure must not be interpreted as candidate-model evidence.

## Immediate next actions

1. Let the v5 monitoring pilot finish or pause cleanly. Do not modify code while it is active.

2. Inspect whether v5 solves the practical usability issues:

```text
dashboard parameters are one per line
details mode is useful
configuration_history.log is readable in VS Code
history mode is useful globally and for one task
durations and fold histories are easy to inspect
```

3. Fix the progress-sidecar reliability issue before the true master run.

4. Resolve the F2/master-protocol documentation mismatch.

5. Freeze the master comparison revision.

6. Run the 5 x 10 repeated nested-CV comparison on development data only.

7. Analyze average precision, secondary metrics, practical equivalence, paired uncertainty, selected-hyperparameter stability, runtime, warnings, and failure rates.

8. Build calibration, threshold, and possible stacking workflows only after candidate ranking is understood.

9. Select one final procedure or justified stack, refit on all development rows, write a complete manifest, and evaluate once on the held-out test set.

## Report and documentation state

The LaTeX report compiles locally through TinyTeX and is currently 126 pages. The MLP section is Section 13.

Known deferred presentation cleanup:

```text
long-title collision in the table of contents
label-width cleanup for Figures 58 and 59
```

Documentation roles:

```text
00_documentation_workflow.md:
    stable rules and documentation architecture

01_model_inventory_and_roadmap.md:
    strategic model-family and final-selection roadmap

current_project_status_and_next_actions.md:
    live tactical state and immediate actions

context_history/telco_churn_chat_handoff_context_7.md:
    current standalone continuation snapshot
```

Before starting a new chat from this handoff, verify the actual local repository state:

```bash
git rev-parse HEAD
git status --short
```
