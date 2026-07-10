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

For the current protocol-v2 base-comparison draft, use:

```text
docs/knowledge_notes/methodology/final_comparison_protocol_v2_draft.md
```

## Latest confirmed checkpoint

### Latest pushed coordination and scaffold checkpoint

The latest pushed checkpoint is:

```text
05c6bdd
Update status after protocol v2 scaffold
```

This commit records the frozen protocol-v2 base-comparison scaffold status. The frozen protocol-v2 declaration remains:

```text
freeze_state = frozen
is_frozen = true
```

The frozen protocol-v2 scaffold can dry-run the official base-comparison task plan, candidate budgets, and CatBoost runtime policy. Non-dry-run execution still requires an explicit confirmation flag. This freeze does not mean that the official base comparison has run, does not master-admit any candidate, and does not access the held-out test set.

The current local implementation adds a separate fast-completion protocol scaffold for quickly finishing the complete project pipeline. It is not a mutation of frozen protocol v2 and is not intended as strong model-comparison evidence.

The prior read-only final-comparison analysis checkpoint remains:

```text
441f331
Add read-only final-comparison analysis scaffold
```

That tooling can summarize completed task artifacts, metrics, runtimes, warnings, selected parameters, and development-data out-of-fold predictions after a future completed official run. It does not fit models, resume workflows, select winners, freeze protocol v2, master-admit candidates, or access the held-out test set.

The source revision recorded by the completed warning-clean admission-smoke workflow remains:

```text
ffd9a3bb25a1a813d2660ab1dbd15d307157dfc4
Restrict Linear SVM search to squared hinge
```

The latest pushed candidate-family implementation checkpoints are:

```text
3d0a371 Add TabM final-comparison candidate
8bb0b3c Add FT-Transformer final-comparison candidate
2f422a2 Add TabNet final-comparison candidate
d5d34cf Add explainable boosting machine candidate
```

Together, these commits make the implemented final-comparison registry cover C01 through C26. C27 TabPFN and C28 AutoGluon remain deferred.

The most recent remote commit that changed final-comparison runner filesystem behaviour remains:

```text
b174372f7b623595c4915116c477de9febc4ffd7
Harden final comparison filesystem persistence
```

That runner revision makes atomic artifact replacement resilient to transient Windows file-lock errors and ensures monitoring telemetry cannot terminate an otherwise valid modelling task. The C03-C26 candidate additions change candidate builders, search spaces, dependencies, routing, and smoke coverage, but they do not use the held-out test set and do not freeze protocol v2.

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

### Completed pre-master admission smoke

The completed implementation-admission smoke run is:

```text
admission_smoke_c26_warning_clean_v2
```

This was an implementation-admission validation only. It is not model-selection evidence, does not freeze protocol v2, does not master-admit any candidate, and does not use or reference the held-out test set.

This warning-clean run supersedes the earlier warning-producing `admission_smoke_c26_probe` checkpoint for operational readiness. The earlier run remains historical provenance for the C01-C26 admission-cleanup sequence.

Run configuration:

```text
C01-C26 implemented candidate universe
C27_TABPFN deferred
C28_AUTOGLUON deferred
development data only
5634 development rows
2 outer folds x 1 repeat
Stage A: 3 valid trials per outer task
Stage B: top 2 Stage-A configurations confirmed
search_profile="smoke"
max_workers=1
source revision recorded by workflow: ffd9a3bb25a1a813d2660ab1dbd15d307157dfc4
working_tree_clean=True
```

Final result:

```text
submitted: 52
completed: 52
failed: 0
interrupted: 0
pending: 0
running: 0
checksum-verified completed result artifacts: 52
integrity and task-level budget check passed
every registry task reached its registered Stage-A and Stage-B budget
git status was clean after completion
```

Runtime notes from the audit:

```text
total sum of per-task wall times: about 11m 49s
C19_CATBOOST was the slowest smoke candidate: around 2m 49s mean task wall time
C24_TABNET mean task wall time: around 21s
C25_FT_TRANSFORMER mean task wall time: around 19s
C26_TABM mean task wall time: around 14s
C21_LINEAR_SVM mean task wall time: around 4s
```

Warning-clean result:

```text
Stage-A trial warnings: none recorded
persisted selected-configuration and outer-task warnings: none recorded
C13 AdaBoost Optuna step warning is resolved
C21 Linear SVM convergence warning is resolved after restricting C21 search to squared_hinge
C24 TabNet known warning noise is resolved
```

### Paused representative calibration runtime evidence

The representative search-budget calibration run is:

```text
search_budget_calibration_v1_warning_clean
```

This run was paused cleanly during C19 CatBoost. It is runtime evidence only, not model-selection evidence, not candidate-ranking evidence, and not candidate-elimination evidence.

Current interpretation:

```text
C19_CATBOOST:
    known runtime bottleneck
    Stage-A trials took multiple minutes each during the paused calibration run
    requires an explicit protocol-v2 runtime and budget decision before any official
    base comparison

protocol implication:
    CatBoost remains implemented and admission-smoke-passed
    CatBoost must not be silently removed
    the frozen protocol-v2 scaffold encodes a runtime-limited C19 policy
    with profile="catboost_v2", Stage-A 8, and Stage-B top 2
    no model-selection evidence has been generated from this policy
```

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

Currently implemented final-comparison core registry:
    26 candidate families, C01 through C26

Candidates currently master-admitted:
    none

Master protocol:
    protocol v2 base comparison is frozen but has not run
```

The 26 implemented candidates are:

```text
C01  Ridge classifier
C02  Regularized logistic regression
C03  Spline logistic regression
C04  Shrinkage linear discriminant analysis
C05  Regularized quadratic discriminant analysis
C06  k-nearest neighbours
C07  Hybrid Gaussian-Bernoulli Naive Bayes
C08  Regularized decision tree
C09  Extra Trees
C10  Bagged decision trees
C11  Random forest
C12  Balanced random forest
C13  AdaBoost
C14  RUSBoost
C15  GradientBoostingClassifier
C16  HistGradientBoostingClassifier
C17  XGBoost
C18  LightGBM
C19  CatBoost
C20  Explainable Boosting Machine
C21  Linear SVM
C22  RBF-kernel SVM
C23  Dense multilayer perceptron
C24  TabNet
C25  FT-Transformer
C26  TabM
```

The C01-C26 pre-master admission smoke passed:

```text
run id:
    admission_smoke_c26_warning_clean_v2

completed registry tasks:
    52/52

failed, interrupted, pending, running:
    0/0/0/0

artifact integrity:
    52 checksum-verified completed result artifacts

budget integrity:
    every registry task reached its registered Stage-A and Stage-B budget

warning integrity:
    Stage-A trial warnings: none recorded
    persisted selected-configuration and outer-task warnings: none recorded
```

C20's current candidate contract is:

```text
package:
    interpret-core==0.7.8

estimator:
    interpret.glassbox.ExplainableBoostingClassifier

representation:
    native categorical string columns

feature policies:
    F0_RAW
    F1_DOMAIN_ENRICHED

excluded feature policy:
    F2_LINEAR_EXPANDED

feature selection:
    S0_NONE only

imbalance policies:
    I0_NONE
    I1_CLASS_WEIGHT_BALANCED

excluded imbalance policies:
    I2_RANDOM_OVERSAMPLING
    I3_RANDOM_UNDERSAMPLING
    I4_SMOTENC

resource policy:
    n_jobs=1
```

The following documented advanced or external candidates remain outside the implemented C01-C26 admission-smoke universe:

```text
C27  TabPFN
C28  AutoML tabular ensemble
```

Current advanced-candidate admission state:

```text
C24 TabNet:
    implemented; included in the completed pre-master admission smoke;
    not master-admitted

C25 FT-Transformer:
    implemented through the official rtdl_revisiting_models route;
    included in the completed pre-master admission smoke;
    not master-admitted

C26 TabM:
    implemented; included in the completed pre-master admission smoke;
    not master-admitted

C27 TabPFN:
    deferred because current CPU practicality and model-weight/licence constraints make it
    unsuitable for this project stage

C28 AutoGluon:
    deferred because the resolver would downgrade the numerical stack
```

The complete implementation and admission matrix, including the required checks, is maintained in `02_candidate_status_register.md`. No candidate is master-admitted merely because protocol v2 is frozen; master admission still requires the official base-comparison and downstream training-only selection process.

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

The read-only final-comparison analysis scaffold now also provides:

```text
- checksum-verified completed-task artifact loading
- candidate-level metric summaries
- runtime and warning summaries
- selected-parameter stability summaries
- development-data out-of-fold prediction export
- safe handling of partial runs without treating pending or interrupted tasks as completed
```

This scaffold is analysis tooling only. It does not select winners, freeze protocol v2, resume workflows, fit models, or inspect the held-out test set.

The executable protocol-v2 base-comparison scaffold now provides:

```text
- a JSON protocol declaration at protocols/final_comparison_protocol_v2_base.json
- explicit frozen state: freeze_state=frozen and is_frozen=true
- C01-C26 candidate universe with C27/C28 deferred
- 5 folds x 3 repeats development-only outer CV design
- candidate-specific cheap, medium, expensive, and CatBoost runtime-limited budget lanes
- C19 CatBoost profile="catboost_v2" as an explicit runtime-limited draft policy
- dry-run task-plan inspection for 390 official base-comparison tasks
- explicit confirmation required for non-dry-run official execution
```

This executable scaffold is now the frozen protocol-v2 base-comparison contract. It is not a completed official base comparison and has not generated model-selection evidence.

The separate fast-completion protocol scaffold provides:

```text
- protocol declaration at protocols/final_comparison_fast_completion_v1.json
- C01-C26 candidate universe with C27/C28 deferred
- 2 folds x 1 repeat development-only outer CV design
- Stage A: 2-fold inner CV with 2 trials per candidate task
- Stage B: 2-fold confirmation with top 1 configuration
- C19 CatBoost retained with profile="catboost_v2"
- 52 total outer tasks
- evidence_role=fast_completion_pipeline_evidence
- explicit warning that this is not the robust protocol-v2 benchmark
- non-dry-run confirmation required through --confirm-fast-completion-run
```

This fast-completion protocol is for finishing the portfolio project pipeline quickly. It must not be described as the robust frozen protocol-v2 benchmark.

### Implementation provenance and provisional master-design reference

The current infrastructure was built in the following reusable phases:

```text
Phase 1:
    protocol foundations, deterministic repeated outer splits, SQLite task coordination,
    atomic artifacts, fingerprints, and resume protections

Phase 2:
    persistent two-stage Optuna HPO and interrupted/resumed execution coverage

Phase 3:
    the current 17-family implemented candidate registry and single-native-worker contract

Phases 4 to 6:
    deterministic F0/F1/F2 feature policies and candidate-compatible S0/S1/S2 routing

Phases 7 to 8B:
    fold-safe I0-I4 imbalance primitives, fit-time sampling/weight handling,
    and candidate-specific imbalance routing
```

Until protocol v2 is frozen, the following is inherited from protocol v1 as a planning reference, not an already-approved master configuration:

```text
Outer evaluation:
    5 stratified folds x 10 repeats

Stage A:
    candidate-specific Optuna exploration using 3-fold inner CV

Stage B:
    confirmation of the strongest Stage-A configurations using 5-fold inner CV

Primary ranking metric:
    average precision
```

### Current policy reference

```text
F0_RAW:
    raw cleaned predictors

F1_DOMAIN_ENRICHED:
    predeclared target-free service aggregates, tenure summaries, selected interactions,
    and one categorical contract-by-payment interaction

F2_LINEAR_EXPANDED:
    controlled regularized-linear nonlinear and interaction basis for Ridge and logistic
    regression only

S0_NONE:
    no feature selection

S1_VARIANCE_MUTUAL_INFO:
    variance filtering followed by mutual-information SelectKBest

S2_L1_LOGISTIC_SELECT_FROM_MODEL:
    embedded L1-logistic feature selection

I0_NONE:
    preserve the observed fitting-fold class distribution

I1_CLASS_WEIGHT_BALANCED:
    fold-local balanced sample weighting

I2_RANDOM_OVERSAMPLING:
    fit-time-only random minority oversampling

I3_RANDOM_UNDERSAMPLING:
    fit-time-only random majority undersampling

I4_SMOTENC:
    raw-feature mixed-data synthetic oversampling before one-hot encoding
```

Feature selection is restricted to candidate families for which it has a coherent role. Tree and native-categorical boosting procedures retain no-selection variants as their primary route. I4 remains available only with F0 raw features because derived F1/F2 columns could become internally inconsistent after synthetic sampling.

The older Phase-8B imbalance compatibility reference is historical for the 17-candidate baseline:

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

F2 has been pruned during pilot work. Its final contract, together with search budgets and Stage-B confirmation depth, is now part of the frozen protocol-v2 base-comparison scaffold. C01-C26 have implementation and admission-smoke coverage, but none of them is master-admitted.

## Current experiment gate

Do not interpret admission-smoke output or fast-completion output as robust protocol-v2 model-selection evidence. Protocol v2 base comparison is frozen but currently too slow for the immediate completion path. No official base comparison has run and no candidate is master-admitted.

### Historical monitoring provenance and local operational inspection

The v1-v5 monitoring history, including the original v4 Windows progress-sidecar failure and the v5 observability work, is retained in:

```text
docs/knowledge_notes/context_history/telco_churn_chat_handoff_context_7.md
```

That handoff is historical provenance only. It must not be treated as the live next-action document because its v5 instructions predate the completed v6 filesystem-resilient pilot.

Use the following commands to inspect the current local environment before any later experiment:

```bash
git status --short
bash scripts/monitor_final_comparison.sh --help
python scripts/final_comparison_status.py --help
```

The tracked audit script can inspect a completed run without fitting models:

```bash
python scripts/audit_final_comparison_run.py --run-id <run_id>
```

The pre-master workflow files are now part of the pushed workflow checkpoint:

```text
scripts/audit_final_comparison_run.py
scripts/run_final_comparison_admission_smoke.py
scripts/run_final_comparison_search_budget_calibration.py
scripts/smoke_test_pre_master_workflows.py
src/telco_churn/pre_master_workflows.py
```

Those files produced the completed C01-C26 admission-smoke run described above. Search-budget calibration remains separate and representative; it is not full-universe admission and must not be treated as candidate elimination.

The required order is now:

```text
1. Keep C27 TabPFN and C28 AutoGluon deferred unless their package, licence,
   resource, and dependency constraints materially change.

2. Run the fast-completion protocol dry-run, then intentionally execute the
   fast-completion development-only workflow if the dry-run plan is acceptable.

3. Preserve the frozen protocol-v2 contract:
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

4. Treat any future change after official results exist as a new protocol version
   or supplemental analysis, not a silent mutation of protocol v2.
```

## Immediate local checks

Before continuing local work after a remote documentation update, first inspect the local tree:

```bash
git status --short
git log --oneline -5
git fetch origin
git log --oneline HEAD..origin/main
```

The exact local working tree is user-controlled. Preserve any uncommitted files, inspect `git status`, and do not infer that a reviewed local change set is already tracked merely because it is described in this status file.

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
