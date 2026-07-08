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
    23 candidate families, C01 through C23

Current master-admitted registry:
    none; protocol v2 has not been frozen

Held-out test use:
    none for model selection, tuning, feature-policy choice, calibration,
    threshold choice, or candidate comparison
```

The successful `pilot_pruned_f2_v6_io_resilient` run was an operational and HPO-budget pilot, not a master comparison. It evaluated six representative implemented candidates over three outer folds and one repeat. Its results must not be used to select or exclude model families.

## Status definitions used below

```text
Implemented, admission pending:
    The current registry can construct the procedure, but it has not yet passed the
    full all-admitted-candidate pre-master operational smoke.

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
| C01 | Ridge classifier | Implemented | Admission pending; not master-admitted | Existing fold-safe linear baseline. |
| C02 | Regularized logistic regression | Implemented | Admission pending; not master-admitted | Includes the existing regularization and feature-policy routing. |
| C03 | Spline logistic regression | Implemented | Admission pending; not master-admitted | Uses fold-safe spline-style nonlinear logistic modelling with bounded search. |
| C04 | Shrinkage linear discriminant analysis | Implemented | Admission pending; not master-admitted | Uses a shrinkage-LDA contract with compatible preprocessing and routing. |
| C05 | Regularized quadratic discriminant analysis | Implemented | Admission pending; not master-admitted | Uses a regularized-QDA contract for numerically stable class-conditional covariance estimation. |
| C06 | k-nearest neighbours | Implemented | Admission pending; not master-admitted | Existing scaled one-hot local-learning procedure. |
| C07 | Hybrid Gaussian-Bernoulli Naive Bayes | Implemented | Admission pending; not master-admitted | Existing mixed-likelihood generative procedure. |
| C08 | Regularized decision tree | Implemented | Admission pending; not master-admitted | Existing pruned-tree procedure. |
| C09 | Extra Trees | Implemented | Admission pending; not master-admitted | Existing randomized-tree ensemble. |
| C10 | Bagged decision trees | Implemented | Admission pending; not master-admitted | Existing bagging procedure. |
| C11 | Random forest | Implemented | Admission pending; not master-admitted | Existing random-forest procedure. |
| C12 | Balanced random forest | Implemented | Admission pending; not master-admitted | Distinct imbalance-aware random-forest procedure; routed as its own candidate rather than ordinary random forest. |
| C13 | AdaBoost | Implemented | Admission pending; not master-admitted | Existing boosting procedure. |
| C14 | RUSBoost | Implemented | Admission pending; not master-admitted | Choice resolved in favor of RUSBoost as the bounded imbalance-aware boosting candidate. |
| C15 | GradientBoostingClassifier | Implemented | Admission pending; not master-admitted | Existing scikit-learn gradient boosting procedure. |
| C16 | HistGradientBoostingClassifier | Implemented | Admission pending; not master-admitted | Existing histogram-based gradient boosting procedure. |
| C17 | XGBoost | Implemented | Admission pending; not master-admitted | Existing external boosting procedure. |
| C18 | LightGBM | Implemented | Admission pending; not master-admitted | Existing external boosting procedure. |
| C19 | CatBoost | Implemented | Admission pending; not master-admitted | Existing native-categorical boosting procedure. |
| C20 | Explainable Boosting Machine | Implemented | Admission pending; not master-admitted | Uses interpret-core, native categorical strings, F0/F1, S0 only, and weighted-only I0/I1 routing. |
| C21 | Linear SVM | Implemented | Admission pending; not master-admitted | Existing margin-based linear procedure. |
| C22 | RBF-kernel SVM | Implemented | Admission pending; not master-admitted | Existing nonlinear kernel procedure. |
| C23 | Dense multilayer perceptron | Implemented | Admission pending; not master-admitted | Existing dense neural-network procedure. |

## C24-C28: advanced and external candidate universe

These candidates are documented because they represent modern tabular-learning or benchmark directions. They do not enter the master comparison merely because they are named. Their admission is intentionally conditional.

| ID | Candidate family | Current implementation status | Current admission status | Additional concerns |
|---|---|---|---|---|
| C24 | TabNet | Not implemented | Conditional advanced admission pending | Package maturity, categorical representation, CPU/GPU scheduling, deterministic fit/predict, checkpoint/resume support. |
| C25 | FT-Transformer | Not implemented | Conditional advanced admission pending | Package choice, categorical/numeric tokenisation, CPU/GPU scheduling, deterministic fit/predict, checkpoint/resume support. |
| C26 | TabM | Not implemented | Conditional advanced admission pending | Reference implementation, architecture contract, resource budget, deterministic fit/predict, checkpoint/resume support. |
| C27 | TabPFN | Not implemented | Conditional advanced admission pending | Exact package/model-weight version, licence terms, hardware practicality, reproducible inference, and resource scheduling must be checked at admission time. |
| C28 | AutoML tabular ensemble | Not implemented | Conditional advanced admission pending | Must be evaluated as one bounded, fully nested end-to-end procedure. Its internal search, ensembling, and stacking are part of the procedure, not a shortcut around the protocol. |

## Required admission checks

### Conventional core candidates C01-C23

C01 through C23 are now implemented in the final-comparison core registry. Before any implemented candidate can join a frozen protocol-v2 master comparison, it still needs:

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

The documented admission gate is:

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

## Pre-master and master implications

The current local pre-master workflow additions were originally reviewed before the registry reached 23 implemented core candidates. They should be treated as a validated operational baseline, not as a final candidate-admission scope. The files remain intentionally untracked locally and must be reconciled with the 23-candidate registry and any later C24-C26 decisions before a real pre-master run.

The correct order is:

```text
1. Preserve the local pre-master workflow files and avoid accidental staging.

2. Treat C01-C23 as implemented but still admission pending and not master-admitted.

3. Decide whether to implement/admit C24-C26:
       C24 TabNet,
       C25 FT-Transformer through rtdl_revisiting_models,
       C26 TabM.

4. Keep C27 TabPFN deferred because of CPU practicality and model-weight/licence constraints.

5. Keep C28 AutoGluon deferred because its resolver would downgrade the numerical stack.

6. Update the all-candidate admission workflow to cover every candidate admitted at that point.

7. Run the actual all-admitted-candidate admission smoke.

8. Run representative search-budget calibration.

9. Freeze protocol v2, including the master registry, search budgets, top-K confirmation rule,
   feature contracts, imbalance contracts, and resource policy.

10. Start the repeated nested-CV master comparison only after that freeze.
```

No model is being excluded by this sequence. It separates implementation completeness from a fair and reproducible master comparison.

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
