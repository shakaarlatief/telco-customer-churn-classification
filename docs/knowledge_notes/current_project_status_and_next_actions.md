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

The current implementation focus is the persistent nested-CV final-comparison runner and its monitoring system.

The last substantive final-comparison implementation checkpoint before the monitoring-pilot work was Phase 8B:

```text
Phase 1: protocol, deterministic repeated outer splits, run storage, resume safety
Phase 2: persistent two-stage Optuna HPO inside each outer training partition
Phase 3: implemented 17-family registry drawn from the documented C01-C23 universe
Phase 4: deterministic F0/F1/F2 feature-policy layer
Phase 5: candidate-specific feature-policy routing
Phase 6: candidate-specific S0/S1/S2 feature-selection routing
Phase 7: fold-safe I0-I4 imbalance primitives
Phase 8A: fit-time imbalance pipeline adapter
Phase 8B: candidate-specific imbalance routing inside Optuna and final pipelines
```

After Phase 8B, the project entered a monitoring-pilot phase for the final-comparison runner. The current v5 monitoring pilot uses the run identifier:

```text
pilot_pruned_f2_v5_history
```

This v5 run is an operational and usability pilot. It is not final model-selection evidence.

Before continuing from this file, verify the actual local Git state:

```bash
git rev-parse HEAD
git status --short
```

The most recent local monitoring code may be ahead of GitHub and may be uncommitted, depending on the user's last local commit.

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

## Governing evaluation policy

All current model-family results, all pilot results, and all future candidate-comparison results remain development-stage evidence until one final procedure is frozen.

Allowed before final test evaluation:

```text
training-only preprocessing and feature engineering
inner-CV hyperparameter optimization
repeated nested cross-validation
training-only feature-selection and imbalance-policy comparisons
calibration and threshold studies that do not use the held-out test set
candidate uncertainty and stability analysis
operational runner and monitoring pilots
```

Not allowed before final model freeze:

```text
selecting candidate families with the held-out test set
selecting feature policies, hyperparameters, resampling, calibration, or thresholds
with the held-out test set
using pilot AP values as final model-selection evidence
repeatedly checking test metrics while development continues
```

The held-out test set is used exactly once after the final procedure has been frozen.

## Available final-comparison infrastructure

### Candidate library

The implemented core registry contains 17 candidate families from the documented C01-C23 universe:

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

The F2 policy has been pruned during the pilot work, but the master protocol documentation still needs to be revised before any master comparison is frozen.

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

The policies are mutually exclusive. I4 is available only with F0 raw features because synthetic F1/F2 derived columns could violate deterministic relationships implied by the underlying raw customer profile.

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

The target master evaluation design, pending protocol freeze, remains:

```text
5 outer folds x 10 repeats
Stage A: persistent 3-fold Optuna exploration
Stage B: 5-fold confirmation of the top Stage-A configurations
primary ranking metric: average precision
```

No master repeated nested-CV comparison has been run yet.

## Monitoring-pilot state

### Pilot lineage

```text
v1:
    first full-development pilot runner with six representative candidates.

v2:
    monitorable persistent runner with progress sidecars, task registry inspection,
    and clean pause behavior.

v3 and v4:
    progressively improved dashboard, event viewer, colour output, alternate-screen
    watch mode, compact coordinator logs, and clearer Stage-A/B telemetry.

v5:
    configuration-history monitoring with one-parameter-per-line dashboard output,
    recent completed-configuration history, and durable configuration_history logs.
```

Current v5 run identifier:

```text
pilot_pruned_f2_v5_history
```

The v5 pilot scope is deliberately smaller than the master run:

```text
6 representative candidate families
3 outer folds x 1 repeat
3-fold Stage A
3-fold Stage B
12 valid Stage-A configurations per outer task
top 3 Stage-A configurations confirmed in Stage B
```

Approximate work per successful outer task:

```text
12 Stage-A configurations x 3 inner folds = 36 fits
3 Stage-B configurations x 3 inner folds = 9 fits
1 final fit on all outer-training data = 1 fit
total ≈ 46 fits per successful outer task
```

### Monitoring commands

Standard live dashboard:

```bash
bash scripts/monitor_final_comparison.sh
```

Details view:

```bash
bash scripts/monitor_final_comparison.sh details
```

Recent events:

```bash
bash scripts/monitor_final_comparison.sh events
```

Configuration history:

```bash
bash scripts/monitor_final_comparison.sh history
```

One task's configuration history:

```bash
bash scripts/monitor_final_comparison.sh history   --task-key c19_catboost__r00__f02
```

The watched dashboard uses the terminal's alternate screen. It does not keep ordinary scrollback. For a scrollable one-time snapshot:

```bash
python scripts/final_comparison_status.py   --run-id pilot_pruned_f2_v5_history   --details
```

Check whether the helper has a one-shot option in the local version:

```bash
bash scripts/monitor_final_comparison.sh --help
```

### Important v4 reliability finding

The v4 pilot completed 17 of 18 outer tasks and exposed one monitoring-side failure:

```text
C01_RIDGE_CLASSIFIER r00/f02
failed during Stage B because progress JSON persistence raised
PermissionError: [WinError 5] Access denied during os.replace(...)
```

This was not a legitimate Ridge-model quality failure. It was a Windows progress-sidecar file-lock problem.

Do not treat the failed v4 task as model-selection evidence.

## Immediate next actions

### 1. Let the v5 monitoring pilot finish or pause cleanly

Do not modify source code while the v5 pilot is active.

Use the dashboard, details view, events view, and configuration-history log to inspect whether v5 solves the practical usability issues:

```text
parameters are readable one per line
completed configurations are visible with AP and duration
configuration_history.log is useful in VS Code
history mode is useful for one task and recent global records
watch mode remains visually stable
```

### 2. Fix the progress-sidecar reliability issue before the master run

The sidecar progress JSON is for monitoring. It must never be able to fail an otherwise successful modelling task.

Required fix before the true master repeated nested-CV comparison:

```text
retry transient Windows PermissionError / sharing violations around atomic replace
make progress-sidecar persistence best effort after bounded retries
surface monitoring-degraded or stale-progress warnings instead of crashing workers
ensure reporter.close() does not mask successful model results or meaningful exceptions
add smoke tests for temporary and persistent PermissionError cases
```

### 3. Inspect v5 artifacts and update the protocol-freeze checklist

After v5 finishes or pauses:

```text
inspect task_registry.sqlite read-only
inspect logs/coordinator.log
inspect logs/configuration_history.log
inspect logs/configuration_history.jsonl
inspect task logs for failures or warnings
record runtime behaviour and whether history mode is sufficient
```

### 4. Resolve the remaining F2/master-protocol documentation mismatch

The implementation has a pruned F2 direction, but the formal protocol documents still need a deliberate v2/master-freeze update.

Before the master run:

```text
confirm final F0/F1/F2 contracts
confirm S0/S1/S2 compatibility
confirm I0-I4 compatibility
confirm search budgets and Stage-B top-k
confirm candidate inclusion
update the final-comparison protocol document
```

### 5. Run the master training-only candidate comparison only after reliability and protocol freeze

The full run should compare complete candidate procedures, not bare estimators. It must persist outer-fold predictions, selected configurations, inner-search records, runtime information, warnings, and failures.

### 6. Later steps after ranking

After the master comparison:

```text
analyze outer-fold AP, secondary metrics, runtime, failures, and selected-configuration stability
define a defensible finalist set
run calibration and threshold work using training-only cross-fitted evidence
consider stacking only after base procedures are frozen
select one final procedure or justified stack
rerun the frozen search on all 5,634 development rows
fit the selected final pipeline
evaluate once on the untouched 1,409-row test set
```

## Documentation and build state

```text
00_documentation_workflow.md:
    stable documentation architecture and collaboration rules

01_model_inventory_and_roadmap.md:
    strategic model-family and final-selection roadmap

current_project_status_and_next_actions.md:
    live tactical state and immediate actions

context_history/telco_churn_chat_handoff_context_7.md:
    standalone continuation snapshot for the current final-comparison monitoring-pilot state
```

The LaTeX report compiles locally through TinyTeX. Cosmetic report cleanup remains deferred:

```text
long-title collision in the table of contents
label-width cleanup for Figures 58 and 59
```

Those presentation issues are separate from the current final-comparison implementation work.
