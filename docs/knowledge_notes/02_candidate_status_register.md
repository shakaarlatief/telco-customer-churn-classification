# Candidate Universe and Admission Status Register

## Purpose

This register makes the model-plan status explicit at one point in time. It distinguishes between:

```text
documented candidate universe:
    methods intentionally included in the project plan

implemented candidate:
    a runnable candidate family exists in the repository's current registry

pre-master admission candidate:
    an implemented candidate that still needs to pass the designated
    all-candidate operational smoke workflow

conditionally advanced candidate:
    a documented candidate that additionally requires package, licence,
    hardware, reproducibility, preprocessing, and resume checks

master-admitted candidate:
    a candidate explicitly frozen into the future protocol-v2 master comparison

excluded candidate:
    a documented candidate that failed admission for a recorded technical reason
```

This is a status and governance document. It does not itself tune a model, decide a winner, change the test-set policy, or freeze the master comparison. The held-out test set remains untouched.

## Authority and interpretation

The related documents have different roles:

```text
final_comparison_protocol_v1.md:
    defines the intended C01-C28 candidate universe and the original master-design plan

candidate registry source code:
    defines the procedures that are actually runnable in the current implementation

this register:
    records the gap between the documented universe and current implementation,
    together with each candidate's admission state

future final_comparison_protocol_v2.md:
    will freeze the actual master-comparison registry only after candidate admission,
    feature-policy review, search-budget calibration, and other pre-master checks
```

Therefore, a candidate being named here does not mean that it is already implemented, admitted to the master run, or expected to be competitive. Likewise, the absence of an implementation must not be interpreted as a silent removal from the original project plan.

## Current global state

```text
Documented candidate universe:
    C01 through C28

Currently implemented registry:
    26 candidate families, C01 through C26

Current master-admitted registry:
    none; protocol v2 base comparison is frozen but has not run

Held-out test use:
    none for model selection, tuning, feature-policy choice, calibration,
    threshold choice, or candidate comparison
```

The successful `pilot_pruned_f2_v6_io_resilient` run was an operational and HPO-budget pilot, not a master comparison. It evaluated six representative implemented candidates over three outer folds and one repeat. Its results must not be used to select or exclude model families.

The successful `admission_smoke_c26_warning_clean_v2` run was a warning-clean implementation-admission smoke for the C01-C26 implemented registry. It supersedes the earlier warning-producing `admission_smoke_c26_probe` checkpoint for operational readiness, while preserving that earlier run as historical provenance. It is not model-selection evidence, does not freeze protocol v2, does not master-admit any candidate, and does not use or reference the held-out test set.

The paused `search_budget_calibration_v1_warning_clean` run is runtime evidence only. It showed that C19 CatBoost is a protocol-level runtime bottleneck, but it must not be used to rank candidates, select candidates, eliminate candidates, or reinterpret admission status.

The latest pushed checkpoint records the frozen protocol-v2 base-comparison scaffold status:

```text
05c6bdd
Update status after protocol v2 scaffold
```

Its protocol declaration has now been intentionally frozen:

```text
freeze_state = frozen
is_frozen = true
```

No official base comparison has run, no candidate is master-admitted, and no model-selection evidence has been generated from the scaffold. A separate fast-completion protocol has completed as `fast_completion_v1` with 52/52 tasks completed, and its summaries have been used to derive a leading candidate set under `artifacts/final_selection/fast_completion_v1/`. Fast finalization selected `top3_unweighted_soft_average` from C03, C25, and C20 for the fast completion path. The next gate is frozen final-procedure spec review and then intentional full-development refit if approved, not held-out-test evaluation.

## Status definitions used below

```text
Implemented, admission pending:
    The current registry can construct the procedure, but it has not yet passed the
    full all-admitted-candidate pre-master operational smoke.

Implemented, admission smoke passed:
    The current registry can construct the procedure and it has passed the bounded
    all-implemented-candidate pre-master admission smoke, but it is not yet
    master-admitted.

Planned conventional core, not implemented:
    The procedure is part of the C01-C23 core plan and should be implemented with
    the same fold-safe candidate contract before a complete all-admitted-candidate smoke.

Conditional advanced admission pending:
    The procedure is part of C24-C28, but no implementation or master admission is
    assumed until all required technical checks have passed.

Not master-admitted:
    No candidate is in the frozen master comparison yet. This remains true even for
    currently implemented candidates and for the six families observed in the v6 pilot.
```

## C01-C23: core candidate universe

| ID | Candidate family | Current implementation status | Current admission status | Notes |
|---|---|---|---|---|
| C01 | Ridge classifier | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing fold-safe linear baseline. |
| C02 | Regularized logistic regression | Implemented | Warning-clean admission smoke passed; not master-admitted | Includes the existing regularization and feature-policy routing. |
| C03 | Spline logistic regression | Implemented | Warning-clean admission smoke passed; not master-admitted | Uses fold-safe spline-style nonlinear logistic modelling with bounded search. |
| C04 | Shrinkage linear discriminant analysis | Implemented | Warning-clean admission smoke passed; not master-admitted | Uses a shrinkage-LDA contract with compatible preprocessing and routing. |
| C05 | Regularized quadratic discriminant analysis | Implemented | Warning-clean admission smoke passed; not master-admitted | Uses a regularized-QDA contract for numerically stable class-conditional covariance estimation. |
| C06 | k-nearest neighbours | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing scaled one-hot local-learning procedure. |
| C07 | Hybrid Gaussian-Bernoulli Naive Bayes | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing mixed-likelihood generative procedure. |
| C08 | Regularized decision tree | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing pruned-tree procedure. |
| C09 | Extra Trees | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing randomized-tree ensemble. |
| C10 | Bagged decision trees | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing bagging procedure. |
| C11 | Random forest | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing random-forest procedure. |
| C12 | Balanced random forest | Implemented | Warning-clean admission smoke passed; not master-admitted | Distinct imbalance-aware random-forest procedure; routed as its own candidate rather than ordinary random forest. |
| C13 | AdaBoost | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing boosting procedure; smoke search-space step alignment resolved before calibration. |
| C14 | RUSBoost | Implemented | Warning-clean admission smoke passed; not master-admitted | Choice resolved in favor of RUSBoost as the bounded imbalance-aware boosting candidate. |
| C15 | GradientBoostingClassifier | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing scikit-learn gradient boosting procedure. |
| C16 | HistGradientBoostingClassifier | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing histogram-based gradient boosting procedure. |
| C17 | XGBoost | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing external boosting procedure. |
| C18 | LightGBM | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing external boosting procedure. |
| C19 | CatBoost | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing native-categorical boosting procedure; known runtime bottleneck. The frozen executable scaffold keeps C19 included with runtime-limited `catboost_v2`, Stage-A 8, and Stage-B top 2; this is not model-selection evidence. |
| C20 | Explainable Boosting Machine | Implemented | Warning-clean admission smoke passed; not master-admitted | Uses interpret-core, native categorical strings, F0/F1, S0 only, and weighted-only I0/I1 routing. |
| C21 | Linear SVM | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing margin-based linear procedure; convergence warning resolved by restricting future search suggestions to squared_hinge loss. |
| C22 | RBF-kernel SVM | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing nonlinear kernel procedure. |
| C23 | Dense multilayer perceptron | Implemented | Warning-clean admission smoke passed; not master-admitted | Existing dense neural-network procedure. |

## C24-C28: advanced and external candidate universe

These candidates are documented because they represent modern tabular-learning or benchmark directions. They do not enter the master comparison merely because they are named. Their admission is intentionally conditional.

| ID | Candidate family | Current implementation status | Current admission status | Additional concerns |
|---|---|---|---|---|
| C24 | TabNet | Implemented | Warning-clean admission smoke passed; not master-admitted | Passed bounded C01-C26 admission smoke; known TabNet best-weights and SciPy sparse warning noise resolved before calibration. |
| C25 | FT-Transformer | Implemented | Warning-clean admission smoke passed; not master-admitted | Implemented through rtdl_revisiting_models with CPU-bounded training and fold-safe categorical/numeric handling. |
| C26 | TabM | Implemented | Warning-clean admission smoke passed; not master-admitted | Implemented with CPU-bounded training, fold-safe categorical/numeric handling, and explicit ensemble-output probability handling. |
| C27 | TabPFN | Not implemented | Deferred; not master-admitted | Exact package/model-weight version, licence terms, hardware practicality, reproducible inference, and resource scheduling must be checked before any future admission attempt. |
| C28 | AutoML tabular ensemble | Not implemented | Deferred; not master-admitted | Deferred because dependency resolution would downgrade the numerical stack; any future evaluation must be one bounded, fully nested end-to-end procedure. |

## Required admission checks

### Conventional core candidates C01-C23

C01 through C23 are implemented in the final-comparison core registry and passed the warning-clean C01-C26 admission smoke. Before any implemented candidate can join a frozen protocol-v2 master comparison, it still needs:

```text
1. a reproducible candidate-builder implementation;
2. a candidate-specific fold-safe preprocessing and representation contract;
3. declared compatible feature, feature-selection, and imbalance policies;
4. a bounded HPO search space and deterministic seed contract;
5. fit/predict and persistence/resume smoke coverage;
6. inclusion in the generic status, audit, and admission-workflow infrastructure.
```

C14's documented imbalance-aware ensemble choice has been resolved in favor of RUSBoost. C20 has been added with a bounded EBM contract using `interpret-core==0.7.8`, native categorical string representation, F0/F1 feature policies, S0-only feature selection, and weighted-only I0/I1 imbalance routing.

### Advanced candidates: C24-C28

C24 through C26 are now implemented and passed the same bounded warning-clean admission smoke. C27 and C28 remain deferred. The documented admission gate for any future advanced-candidate change remains:

```text
1. installation check;
2. package licence and, where applicable, model-weight licence check;
3. reproducible fit-and-predict smoke test;
4. fold-safe preprocessing smoke test;
5. checkpoint and resume smoke test;
6. CPU or GPU resource-scheduling smoke test;
7. a bounded runtime and search-budget assessment;
8. a persisted technical-exclusion reason if any check fails.
```

For TabPFN, the package version, model-weight terms, and hardware guidance must be verified at the time of admission. This prevents the project from relying on outdated assumptions about an evolving external implementation.

## Completed pre-master admission smoke

The completed run was:

```text
run id:
    admission_smoke_c26_warning_clean_v2

purpose:
    implementation-admission validation only

scope:
    C01-C26 implemented candidate universe
    C27_TABPFN deferred
    C28_AUTOGLUON deferred
    development data only
    no held-out test use or reference

configuration:
    5634 development rows
    2 outer folds x 1 repeat
    Stage A: 3 valid trials per outer task
    Stage B: top 2 Stage-A configurations confirmed
    search_profile="smoke"
    max_workers=1
    source revision recorded by workflow: ffd9a3bb25a1a813d2660ab1dbd15d307157dfc4
    working_tree_clean=True

result:
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

warning-clean result:
    Stage-A trial warnings: none recorded
    persisted selected-configuration and outer-task warnings: none recorded
    C13 AdaBoost Optuna step warning is resolved
    C21 Linear SVM convergence warning is resolved after restricting C21 search to squared_hinge
    C24 TabNet known warning noise is resolved
```

Runtime notes from the audit:

```text
total sum of per-task wall times:
    about 11m 49s

slowest smoke candidate:
    C19_CATBOOST, around 2m 49s mean task wall time

advanced-candidate smoke means:
    C24_TABNET: around 21s
    C25_FT_TRANSFORMER: around 19s
    C26_TABM: around 14s

linear-SVM smoke mean:
    C21_LINEAR_SVM: around 4s
```

## Pre-master and master implications

C01-C26 are implemented and have passed bounded warning-clean pre-master admission smoke. This confirms implementation-admission mechanics only. It does not select a model, exclude a candidate, freeze protocol v2, or master-admit any candidate.

Implementation admission is complete for C01-C26, and protocol-v2 runtime and budget policy is frozen for the base comparison. The executable protocol-v2 scaffold remains unchanged as the stronger benchmark contract. The separate fast-completion protocol completed as `fast_completion_v1` with C01-C26, 2 outer folds x 1 repeat, 2 Stage-A trials, Stage-B top 1, and C19 retained on `catboost_v2`. Fast finalization selected a top-three unweighted soft-voting ensemble from C03, C25, and C20 for the fast completion path, and a guarded full-development refit scaffold now exists. No official base-comparison results have been generated or inspected yet.

The correct order is:

```text
1. Keep C27 TabPFN deferred because of CPU practicality and model-weight/licence constraints.

2. Keep C28 AutoGluon deferred because its resolver would downgrade the numerical stack.

3. Use the selected top-three fast-finalization ensemble as input to the frozen
   final-procedure specification and guarded full-development refit only if the
   fast completion path is intentionally approved.

4. The frozen protocol v2 includes the base-comparison registry, search budgets, top-K confirmation rule,
   feature contracts, imbalance contracts, and resource policy.

5. Start the repeated nested-CV master comparison only after that freeze.
```

No model is being selected or excluded by the admission-smoke result. It separates implementation completeness from a fair and reproducible master comparison.

## Update rules

Update this register whenever any of the following occurs:

```text
- a candidate implementation is added or materially revised;
- a candidate passes or fails an admission check;
- an external package, licence, or hardware constraint changes the feasible design;
- the all-candidate admission-smoke scope changes;
- protocol v2 freezes the actual master registry;
- a candidate is formally excluded with a recorded technical reason.
```

Do not use this table to retroactively interpret pilot AP values as candidate-selection evidence. Candidate inclusion, exclusion, and final selection remain training-only protocol decisions.
