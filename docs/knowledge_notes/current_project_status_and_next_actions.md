# Current Project Status and Next Actions

## Purpose

This file is the live tactical tracker for the Telco Customer Churn classification project. It records the latest confirmed implementation state, the immediate modelling gate, and the conditions that must be satisfied before the held-out test set can be used.

For candidate-family inclusion and admission status, use:

```text
docs/knowledge_notes/02_candidate_status_register.md
```

For the strategic modelling roadmap, use:

```text
docs/knowledge_notes/01_model_inventory_and_roadmap.md
```

For the original C01-C28 protocol universe and methodology design, use:

```text
docs/knowledge_notes/methodology/final_comparison_protocol_v1.md
```

## Latest confirmed checkpoint

### Remote baseline

The filesystem-resilience implementation is committed on remote `main` as:

```text
b174372f7b623595c4915116c477de9febc4ffd7
Harden final comparison filesystem persistence
```

That revision makes atomic artifact replacement resilient to transient Windows file-lock errors and ensures monitoring telemetry cannot terminate an otherwise valid modelling task.

### Operational pilot outcome

The completed operational pilot is:

```text
pilot_pruned_f2_v6_io_resilient
```

Its scope and outcome were:

```text
Candidates:
    C01 Ridge classifier
    C02 Regularized logistic regression
    C07 Hybrid Gaussian-Bernoulli Naive Bayes
    C08 Regularized decision tree
    C19 CatBoost
    C23 Dense multilayer perceptron

Outer evaluation:
    3 outer folds x 1 repeat

Inner HPO per outer task:
    Stage A: 12 valid configurations x 3 folds
    Stage B: top 3 Stage-A configurations x 3 folds

Operational result:
    18 submitted, 18 completed, 0 failed, 0 interrupted, 0 skipped
    all persisted result artifacts passed checksum validation
```

This v6 result resolves the earlier Windows filesystem-persistence blocker observed in v4 and left unresolved in v5. It is an operational and search-budget pilot only. Its AP values, runtime values, and sampled candidates are not master-selection evidence and must not be used to include or exclude candidate families.

### Local pre-master workflow state

The local working tree contains uncommitted pre-master workflow additions and the associated SQLite-cleanup correction:

```text
src/telco_churn/pre_master_workflows.py
scripts/run_final_comparison_admission_smoke.py
scripts/run_final_comparison_search_budget_calibration.py
scripts/audit_final_comparison_run.py
scripts/smoke_test_pre_master_workflows.py
```

The structural smoke test passed after the explicit Windows SQLite-handle cleanup correction:

```text
Pre-master workflow structural smoke test passed.
```

These local additions have not yet been committed to remote `main`. They must be reviewed against the updated candidate-status scope before any real pre-master experiment is launched.

## Project state

### Completed model-family workflows

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

The formal LaTeX report includes the MLP section as Section 13 and has been compiled locally at 126 pages.

### Dataset and final-test boundary

```text
Clean modelling dataset: 7043 rows
Development training set: 5634 rows
Held-out test set:       1409 rows
Positive class:          Churn_binary = 1
Development churn rate:  approximately 26.54%
```

The held-out test set remains untouched for model-family selection, preprocessing selection, feature-policy selection, hyperparameter search, calibration selection, threshold selection, stacking, and final candidate comparison. It is used exactly once after one complete final procedure is frozen.

## Candidate-universe state

```text
Documented protocol universe:
    C01 through C28

Currently implemented registry:
    17 candidate families

Candidates currently master-admitted:
    none

Master protocol:
    protocol v2 has not been frozen
```

The 17 implemented candidates are:

```text
C01  Ridge classifier
C02  Regularized logistic regression
C06  k-nearest neighbours
C07  Hybrid Gaussian-Bernoulli Naive Bayes
C08  Regularized decision tree
C09  Extra Trees
C10  Bagged decision trees
C11  Random forest
C13  AdaBoost
C15  GradientBoostingClassifier
C16  HistGradientBoostingClassifier
C17  XGBoost
C18  LightGBM
C19  CatBoost
C21  Linear SVM
C22  RBF-kernel SVM
C23  Dense multilayer perceptron
```

The following documented conventional core candidates remain unimplemented:

```text
C03  Spline logistic regression
C04  Shrinkage linear discriminant analysis
C05  Regularized quadratic discriminant analysis
C12  Balanced random forest
C14  RUSBoost or EasyEnsemble
C20  Explainable Boosting Machine
```

The following documented advanced or external candidates remain subject to explicit technical admission:

```text
C24  TabNet
C25  FT-Transformer
C26  TabM
C27  TabPFN
C28  AutoML tabular ensemble
```

The complete implementation and admission matrix, including the required checks, is maintained in `02_candidate_status_register.md`. No item in that register is silently excluded merely because it is not part of the current 17-model code registry.

## Current final-comparison infrastructure

The existing final-comparison implementation provides:

```text
- deterministic repeated stratified outer splits
- persistent SQLite task coordination and task-local Optuna studies
- two-stage inner HPO with Stage-A exploration and Stage-B confirmation
- atomic result artifacts, fingerprints, and resume protections
- fold-safe F0/F1/F2 feature-policy routing
- fold-safe S0/S1/S2 feature-selection routing
- fold-safe I0-I4 imbalance-policy routing
- candidate-specific unfitted pipeline builders
- outer-task process parallelism with native estimator threads constrained
- monitoring, progress telemetry, artifact auditing, and filesystem-resilience coverage
```

The feature-policy direction remains:

```text
F0_RAW:
    raw cleaned predictors

F1_DOMAIN_ENRICHED:
    predeclared target-free service aggregates, tenure summaries, and selected interactions

F2_LINEAR_EXPANDED:
    controlled regularized-linear nonlinear and interaction basis for Ridge and logistic
    regression only
```

F2 has been pruned during pilot work. Its final contract, together with search budgets and Stage-B confirmation depth, must be explicitly frozen in protocol v2 before the master comparison.

## Current experiment gate

Do not launch either real pre-master workflow yet.

The pre-master scripts were intentionally designed around the 17 currently implemented candidates. The project now has an explicit record that the documented C01-C28 universe is broader. The correct next phase is candidate-completeness and admission work, not another score-producing run.

The required order is:

```text
1. Preserve the current 17-model pre-master tooling as a validated operational baseline.

2. Implement and smoke-test the six missing conventional core candidates:
       C03, C04, C05, C12, C14, C20.

3. Conduct explicit admission reviews for C24-C28:
       installation,
       licence and model-weight terms where applicable,
       reproducible fit/predict,
       fold-safe preprocessing,
       checkpoint/resume,
       CPU or GPU scheduling,
       bounded runtime and search-budget feasibility.

4. Record every pass, deferral, or technical exclusion in the candidate-status register.

5. Update the all-candidate admission smoke so its scope covers every candidate admitted
   at that point, not merely the original 17 implemented candidates.

6. Run the real all-admitted-candidate admission smoke.

7. Run the representative search-budget calibration.

8. Freeze protocol v2:
       master candidate registry,
       candidate contracts,
       feature policies,
       feature-selection and imbalance compatibility,
       search spaces,
       trial budgets,
       Stage-B top-K,
       split rules,
       resource policy,
       audit and resume contract.

9. Only then launch the repeated nested-CV master comparison on development data.
```

## Immediate local checks

Before continuing local work after a remote documentation update, first inspect the local tree:

```bash
git status --short
git log --oneline -5
git fetch origin
git log --oneline HEAD..origin/main
```

The local pre-master additions are currently uncommitted. Do not overwrite them. Review the remote documentation update and integrate it with the local working tree deliberately before staging any project changes.

## Subsequent selection sequence

After protocol v2 is frozen and the master comparison is complete:

```text
1. Analyse repeated outer-fold average precision, secondary metrics, runtime,
   failures, warnings, and selected-configuration stability.

2. Define a defensible finalist set using training-only evidence and the
   predeclared practical-equivalence rule.

3. Run calibration and threshold work using training-only cross-fitted evidence.

4. Consider stacking only after base procedures and their out-of-fold evidence are frozen.

5. Freeze one final procedure or justified stack.

6. Rerun its frozen search on all 5,634 development rows.

7. Fit the complete final pipeline and manifest.

8. Evaluate once on the untouched 1,409-row test set.
```

## Documentation roles

```text
00_documentation_workflow.md:
    stable documentation architecture and collaboration rules

01_model_inventory_and_roadmap.md:
    strategic model-family and final-selection roadmap

02_candidate_status_register.md:
    documented, implemented, admission, exclusion, and master-freeze status by candidate

current_project_status_and_next_actions.md:
    live tactical state and immediate actions

context_history/:
    historical chat handoff snapshots

methodology/final_comparison_protocol_v1.md:
    original C01-C28 universe and version-1 final-comparison design
```

The next new handoff file should use this status tracker and the candidate-status register rather than treating the older v5 handoff as current.