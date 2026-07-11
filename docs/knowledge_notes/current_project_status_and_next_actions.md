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

### New-chat handoff checkpoint

Repository:

```text
shakaarlatief/telco-customer-churn-classification
branch: main
```

The current local Git state at the time of this handoff was:

```text
HEAD:        6104dcf05495edbed76ea99318c5d9053f644929
origin/main: 6104dcf05495edbed76ea99318c5d9053f644929
short log:   6104dcf Add guarded final held-out evaluation workflow
status:      only artifacts/ is untracked
```

The guarded final held-out evaluation workflow is therefore not an uncommitted code change in this local state; it is present at `HEAD` and at the local `origin/main` tracking ref. The local `artifacts/` tree is intentionally untracked and must not be added to Git wholesale. Future work should decide deliberately which compact summaries, reports, tables, metadata, or selected manifests belong in version control; large run outputs, OOF prediction files, joblib files, and row-level outputs should remain excluded unless a deliberate artifact-publication policy is defined.

The latest chat-handoff snapshot is:

```text
docs/knowledge_notes/context_history/telco_churn_chat_handoff_context_9.md
```

It supersedes context 8 for current chat-continuation state while preserving contexts 1 through 8 as historical background.

The frozen protocol-v2 declaration remains:

```text
protocols/final_comparison_protocol_v2_base.json
freeze_state = frozen
is_frozen = true
```

The frozen robust protocol-v2 scaffold can dry-run the official base-comparison task plan, candidate budgets, and CatBoost runtime policy. It remains untouched and optional future work. It was prepared but not completed; it is not required before documentation/reporting can continue.

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

The held-out test set remains untouched. It has not been loaded, inspected, counted, fingerprinted, summarized, or scored. No final test metrics exist. The user explicitly chose not to run the final held-out evaluation yet. No model, threshold, weights, members, calibration decision, or feature definition may be changed using future test results.

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

The fast-completion run has completed:

```text
run id: fast_completion_v1
evidence role: fast_completion_pipeline_evidence
submitted: 52
completed: 52
failed: 0
interrupted: 0
paused: 0
held-out test: not loaded or referenced
```

This was deliberately small and fast: 2 outer folds x 1 repeat, 2 tuning trials per candidate, 2-fold inner CV, primary metric average precision. It is sufficient for pipeline completion and development decisions, but it is not robust protocol-v2 benchmark evidence.

Its completed summaries supported an automatic top-five development-data leading-candidate selection stored under `artifacts/final_selection/fast_completion_v1/`:

```text
1. C03_SPLINE_LOGISTIC_REGRESSION
2. C20_EXPLAINABLE_BOOSTING_MACHINE
3. C25_FT_TRANSFORMER
4. C01_RIDGE_CLASSIFIER
5. C18_LIGHTGBM
```

The repository contains a separate fast-finalization scaffold:

```text
- protocol declaration at protocols/fast_finalization_v1.json
- runner at scripts/run_fast_finalization.py
- synthetic structural smoke at scripts/smoke_test_fast_finalization.py
- default input: artifacts/final_selection/fast_completion_v1/leading_candidates.json
- default output root: artifacts/final_selection/fast_completion_v1/fast_finalization_v1
- tuning: 2 trials with 2-fold inner CV for selected leading candidates
- OOF evaluation: 2 folds x 1 repeat on development data only
- simple probability-level soft-voting checks where enough probability candidates exist
- evidence_role=fast_finalization_pipeline_evidence
```

Fast finalization has now selected the development-data final procedure for the fast completion path:

```text
selected procedure: top3_unweighted_soft_average
procedure type: ensemble
members:
    C03_SPLINE_LOGISTIC_REGRESSION
    C25_FT_TRANSFORMER
    C20_EXPLAINABLE_BOOSTING_MACHINE
aggregation: unweighted arithmetic mean of member probabilities
selected OOF average precision: about 0.669605382543
selected OOF F1 threshold: 0.39106601395524887
calibration method: none
calibration status: deferred_fast_completion
```

Individual fast-finalization OOF average precision values were:

```text
C03_SPLINE_LOGISTIC_REGRESSION: approximately 0.666354114311
C25_FT_TRANSFORMER:            approximately 0.665475534103
C20_EXPLAINABLE_BOOSTING_MACHINE: approximately 0.664879135290
C01_RIDGE_CLASSIFIER:          approximately 0.649499243122
C18_LIGHTGBM:                  approximately 0.616658334430
```

The selected ensemble exceeded the best individual by approximately 0.003251 AP, which exceeded the frozen 0.002 simplicity tolerance. Therefore the ensemble remained selected.

This remains fast development-data evidence. It must not be described as robust protocol-v2 evidence, official base-comparison evidence, or held-out-test evidence. The final procedure has now been frozen and refitted on all 5,634 development rows, and independent loading/prediction roundtrip validation passed. The repository contains a guarded one-time held-out-test readiness audit and evaluator, but the real held-out-test evaluation has not been run.

The final frozen procedure is:

```text
procedure: top3_unweighted_soft_average
member order:
    1. C03_SPLINE_LOGISTIC_REGRESSION
    2. C25_FT_TRANSFORMER
    3. C20_EXPLAINABLE_BOOSTING_MACHINE
weights:
    0.3333333333333333
    0.3333333333333333
    0.3333333333333333
aggregation: arithmetic mean of positive-class probabilities
threshold origin: development-data OOF F1 maximization
calibration method: none
calibration status: deferred_fast_completion
```

The final development refit is stored at:

```text
artifacts/final_selection/fast_completion_v1/final_development_refit_v1
```

Important files:

```text
fitted_final_pipeline.joblib
final_refit_manifest.json
model_environment.json
feature_schema.json
artifact_checksums.json
roundtrip_validation.json
```

Validation status:

```text
serialization completed
joblib reload completed
pre-save and post-load probabilities matched
pre-save and post-load predictions matched
independent manual loading succeeded with PYTHONPATH=src
model type: FrozenProbabilityVotingEnsemble
held-out data accessed: no
```

Standalone commands that load the serialized model must make `src` importable, for example:

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe ...
```

The serialized object exposes:

```text
member_ids
member_weights
decision_threshold
calibration_method
calibration_status
```

The guarded final held-out evaluator scaffold exists:

```text
src/telco_churn/final_evaluation.py
scripts/audit_final_test_readiness.py
scripts/evaluate_final_held_out_test.py
scripts/smoke_test_final_held_out_evaluation.py
```

Status:

```text
py_compile passed
smoke test passed
readiness audit reported READY
evaluator dry-run passed
dry-run did not load the held-out test set
real evaluator was not run
final evaluation output directory does not exist
no evaluation receipt exists
no final test metrics exist
```

The real evaluator requires the exact confirmation phrase:

```text
I_UNDERSTAND_THIS_CONSUMES_THE_FINAL_TEST_SET
```

Do not present running that command as the immediate next action. The user explicitly deferred held-out evaluation. The eventual output directory would be:

```text
artifacts/final_evaluation/fast_completion_v1/held_out_test_v1
```

That directory must remain absent until the user intentionally chooses to consume the final test set.

Resolved implementation bug to remember:

```text
Optuna best_trial.params contained only trial.suggest_* values.
Fixed executable fields such as C03 max_iter and class_weight were omitted.
Fast finalization now stores and retrieves the complete parameter mapping for each trial.
The fix was implemented generically rather than as candidate-specific repair logic.
Regression coverage was added.
```

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

For historical context, the original protocol-v1 planning reference was:

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

Do not interpret admission-smoke output, fast-completion output, leading-candidate selection, fast-finalization output, or the full-development refit as robust protocol-v2 model-selection evidence. Protocol v2 base comparison is frozen but currently too slow for the immediate completion path. The completed `fast_completion_v1` run has been used for read-only leading-candidate selection, fast finalization, and full-development refit for project completion. The held-out evaluator exists and is ready, but the user explicitly deferred running it. The current priority is documentation/reporting, not test evaluation. No official base comparison has run and no candidate is master-admitted.

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

The project should now be described as:

```text
model-development pipeline: complete
frozen final model: fitted and serialized
held-out evaluation: intentionally deferred
current phase: documentation/reporting
```

Recommended next work:

```text
1. Inspect and organize existing project documentation.
2. Build the final technical report from completed development evidence.
3. Write the mathematical and methodological explanations for preprocessing,
   model families, tuning, nested CV, class imbalance, feature selection,
   thresholds, calibration decisions, and ensembling.
4. Create development-result tables and figures.
5. Improve README and repository navigation.
6. Document limitations of the fast-completion evidence.
7. Decide which compact generated artifacts should be version-controlled.
8. Leave the held-out test untouched unless the user explicitly requests final evaluation.
9. Optionally return to robust protocol v2 later.
```

Do not describe held-out test evaluation as required before documentation work can continue.

The required guardrails are:

```text
1. Keep C27 TabPFN and C28 AutoGluon deferred unless their package, licence,
   resource, and dependency constraints materially change.

2. Treat the selected `top3_unweighted_soft_average` fast-finalization procedure as
   the fast-route development-data final procedure, while preserving its
   development-data-only and non-robust-evidence label.

3. Do not run the one-time held-out-test evaluator unless the user explicitly
   requests it and provides the required confirmation phrase.

4. Preserve the frozen protocol-v2 contract:
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

5. Treat any future change after official results exist as a new protocol version
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

## New-Chat Startup Checklist

For a new assistant session:

```text
1. Read:
       docs/knowledge_notes/current_project_status_and_next_actions.md
       docs/knowledge_notes/02_candidate_status_register.md
       docs/knowledge_notes/01_model_inventory_and_roadmap.md
2. Run git status --short.
3. Run git log --oneline -8.
4. Confirm whether guarded final-evaluation files remain committed and pushed.
5. Confirm artifacts/ remains untracked.
6. Do not run the held-out test evaluator.
7. Continue with documentation/reporting unless the user explicitly changes priorities.
```

Documentation wording rules:

```text
- distinguish implementation-admission evidence, runtime evidence, fast-completion
  development evidence, robust protocol-v2 evidence, and held-out test evidence;
- do not claim final test performance;
- do not claim the entire portfolio project is finished;
- do state that the modelling and final-refit pipeline is complete;
- do not call the fast-completion procedure a robust benchmark winner;
- keep descriptions technically precise and self-contained.
```

## Optional Robust-Protocol Sequence

If the project later returns to the frozen robust protocol-v2 base comparison instead of the fast-completion route:

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
