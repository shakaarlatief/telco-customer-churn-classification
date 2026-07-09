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
    none; protocol v2 has not been frozen

Held-out test use:
    none for model selection, tuning, feature-policy choice, calibration,
    threshold choice, or candidate comparison
```

The successful `pilot_pruned_f2_v6_io_resilient` run was an operational and HPO-budget pilot, not a master comparison. It evaluated six representative implemented candidates over three outer folds and one repeat. Its results must not be used to select or exclude model families.

The successful `admission_smoke_c26_probe` run was an implementation-admission smoke for the C01-C26 implemented registry. It is not model-selection evidence, does not freeze protocol v2, does not master-admit any candidate, and does not use or reference the held-out test set.

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
| C01 | Ridge classifier | Implemented | Admission smoke passed; not master-admitted | Existing fold-safe linear baseline. |
| C02 | Regularized logistic regression | Implemented | Admission smoke passed; not master-admitted | Includes the existing regularization and feature-policy routing. |
| C03 | Spline logistic regression | Implemented | Admission smoke passed; not master-admitted | Uses fold-safe spline-style nonlinear logistic modelling with bounded search. |
| C04 | Shrinkage linear discriminant analysis | Implemented | Admission smoke passed; not master-admitted | Uses a shrinkage-LDA contract with compatible preprocessing and routing. |
| C05 | Regularized quadratic discriminant analysis | Implemented | Admission smoke passed; not master-admitted | Uses a regularized-QDA contract for numerically stable class-conditional covariance estimation. |
| C06 | k-nearest neighbours | Implemented | Admission smoke passed; not master-admitted | Existing scaled one-hot local-learning procedure. |
| C07 | Hybrid Gaussian-Bernoulli Naive Bayes | Implemented | Admission smoke passed; not master-admitted | Existing mixed-likelihood generative procedure. |
| C08 | Regularized decision tree | Implemented | Admission smoke passed; not master-admitted | Existing pruned-tree procedure. |
| C09 | Extra Trees | Implemented | Admission smoke passed; not master-admitted | Existing randomized-tree ensemble. |
| C10 | Bagged decision trees | Implemented | Admission smoke passed; not master-admitted | Existing bagging procedure. |
| C11 | Random forest | Implemented | Admission smoke passed; not master-admitted | Existing random-forest procedure. |
| C12 | Balanced random forest | Implemented | Admission smoke passed; not master-admitted | Distinct imbalance-aware random-forest procedure; routed as its own candidate rather than ordinary random forest. |
| C13 | AdaBoost | Implemented | Admission smoke passed; not master-admitted | Existing boosting procedure; step-alignment cleanup needed before protocol v2 freeze. |
| C14 | RUSBoost | Implemented | Admission smoke passed; not master-admitted | Choice resolved in favor of RUSBoost as the bounded imbalance-aware boosting candidate. |
| C15 | GradientBoostingClassifier | Implemented | Admission smoke passed; not master-admitted | Existing scikit-learn gradient boosting procedure. |
| C16 | HistGradientBoostingClassifier | Implemented | Admission smoke passed; not master-admitted | Existing histogram-based gradient boosting procedure. |
| C17 | XGBoost | Implemented | Admission smoke passed; not master-admitted | Existing external boosting procedure. |
| C18 | LightGBM | Implemented | Admission smoke passed; not master-admitted | Existing external boosting procedure. |
| C19 | CatBoost | Implemented | Admission smoke passed; not master-admitted | Existing native-categorical boosting procedure. |
| C20 | Explainable Boosting Machine | Implemented | Admission smoke passed; not master-admitted | Uses interpret-core, native categorical strings, F0/F1, S0 only, and weighted-only I0/I1 routing. |
| C21 | Linear SVM | Implemented | Admission smoke passed; not master-admitted | Existing margin-based linear procedure; convergence-warning cleanup needed before protocol v2 freeze. |
| C22 | RBF-kernel SVM | Implemented | Admission smoke passed; not master-admitted | Existing nonlinear kernel procedure. |
| C23 | Dense multilayer perceptron | Implemented | Admission smoke passed; not master-admitted | Existing dense neural-network procedure. |

## C24-C28: advanced and external candidate universe

These candidates are documented because they represent modern tabular-learning or benchmark directions. They do not enter the master comparison merely because they are named. Their admission is intentionally conditional.

| ID | Candidate family | Current implementation status | Current admission status | Additional concerns |
|---|---|---|---|---|
| C24 | TabNet | Implemented | Admission smoke passed; not master-admitted | Passed bounded C01-C26 admission smoke; repeated harmless-but-noisy best-weights warning and one SciPy sparse deprecation warning need review before protocol v2 freeze. |
| C25 | FT-Transformer | Implemented | Admission smoke passed; not master-admitted | Implemented through rtdl_revisiting_models with CPU-bounded training and fold-safe categorical/numeric handling. |
| C26 | TabM | Implemented | Admission smoke passed; not master-admitted | Implemented with CPU-bounded training, fold-safe categorical/numeric handling, and explicit ensemble-output probability handling. |
| C27 | TabPFN | Not implemented | Deferred; not master-admitted | Exact package/model-weight version, licence terms, hardware practicality, reproducible inference, and resource scheduling must be checked before any future admission attempt. |
| C28 | AutoML tabular ensemble | Not implemented | Deferred; not master-admitted | Deferred because dependency resolution would downgrade the numerical stack; any future evaluation must be one bounded, fully nested end-to-end procedure. |

## Required admission checks

### Conventional core candidates C01-C23

C01 through C23 are implemented in the final-comparison core registry and passed the C01-C26 admission smoke. Before any implemented candidate can join a frozen protocol-v2 master comparison, it still needs:

```text
1. a reproducible candidate-builder implementation;
2. a candidate-specific fold-safe preprocessing and representation contract;
3. declared compatible feature, feature-selection, and imbalance policies;
4. a bounded HPO search space and deterministic seed contract;
5. fit/predict and persistence/resume smoke coverage;
6. inclusion in the generic status, audit, and admission-workflow infrastructure;
7. cleanup of warning-producing search-space or solver settings where recorded.
```

C14's documented imbalance-aware ensemble choice has been resolved in favor of RUSBoost. C20 has been added with a bounded EBM contract using `interpret-core==0.7.8`, native categorical string representation, F0/F1 feature policies, S0-only feature selection, and weighted-only I0/I1 imbalance routing.

### Advanced candidates: C24-C28

C24 through C26 are now implemented and passed the same bounded admission smoke. C27 and C28 remain deferred. The documented admission gate for any future advanced-candidate change remains:

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
    admission_smoke_c26_probe

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
    max_workers=1 for the actual resumed chunks
    source revision recorded by workflow: 62e52dad585eb92709ab6a5748cd8dc82f8b9755
    working_tree_clean=True

result:
    registry tasks: 52
    completed: 52
    failed: 0
    interrupted: 0
    pending: 0
    checksum-verified completed result artifacts: 52
    integrity and task-level budget check passed
    every registry task reached its registered Stage-A and Stage-B budget
    git status was clean after completion
```

Runtime notes from the audit:

```text
total sum of per-task wall times:
    about 10m 17s

slowest smoke candidate:
    C19_CATBOOST, around 2m 33s mean task wall time

advanced-candidate smoke means:
    C24_TABNET: around 19s
    C25_FT_TRANSFORMER: around 17s
    C26_TABM: around 6s
```

Warnings and cleanup items to record before protocol v2 freeze:

```text
C21_LINEAR_SVM:
    Liblinear convergence warning recorded 3 times.
    Later cleanup: increase max_iter or adjust solver/search-space settings.

C24_TABNET:
    Repeated warning: "Best weights from best epoch are automatically used!"
    Treat as harmless but noisy unless further investigation suggests otherwise.
    One SciPy sparse deprecation warning was also recorded.

C13_ADABOOST:
    Terminal-only Optuna warning appeared earlier:
    distribution [25, 120] with step=25 is not divisible and is internally replaced
    by [25, 100].
    Later cleanup: adjust the upper bound to a step-aligned value before master
    protocol freeze.
```

## Pre-master and master implications

C01-C26 are implemented and have passed bounded pre-master admission smoke. This confirms implementation-admission mechanics only. It does not select a model, exclude a candidate, freeze protocol v2, or master-admit any candidate.

The correct order is:

```text
1. Keep C27 TabPFN deferred because of CPU practicality and model-weight/licence constraints.

2. Keep C28 AutoGluon deferred because its resolver would downgrade the numerical stack.

3. Clean up the warning-producing search-space/settings issues recorded by the
   C01-C26 admission smoke.

4. Treat search-budget calibration as separate and representative, not full-universe
   admission and not candidate elimination.

5. Freeze protocol v2, including the master registry, search budgets, top-K confirmation rule,
   feature contracts, imbalance contracts, and resource policy.

6. Start the repeated nested-CV master comparison only after that freeze.
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
