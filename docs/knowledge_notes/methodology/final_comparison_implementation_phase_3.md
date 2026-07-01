# Final Comparison Implementation Phase 3: Core Candidate Registry

## Purpose

Phase 3 extends the persistent-HPO infrastructure from its initial mechanical smoke
subset to the project's complete **core** candidate library. This is still a
training-only implementation stage. It makes all previously implemented classical,
tree, bagging, boosting, support-vector-machine, and neural-network families
available to the resumable nested-CV system, but it does not launch the full
50-outer-evaluation master comparison.

The registry is deliberately staged. A candidate is only added once it has a complete
procedure definition:

```text
candidate identifier
+ score semantics
+ fold-internal input representation
+ smoke search space
+ full search space
+ fresh fold-safe pipeline factory
+ deterministic random-state policy
+ declared inner thread policy
```

This is stricter than merely listing estimator names. It prevents an experiment from
silently treating a historical notebook model, a different preprocessing route, and a
new HPO procedure as interchangeable evidence.

## Core registry after Phase 3

The reusable registry now contains these candidate procedures:

| Identifier | Candidate procedure | Continuous score type | Representation |
|---|---|---:|---|
| C01 | Ridge classifier | Margin | Scaled one-hot |
| C02 | Regularized logistic regression | Probability | Scaled one-hot |
| C06 | k-nearest neighbours | Probability | Scaled one-hot |
| C07 | Hybrid Gaussian-Bernoulli Naive Bayes | Probability | Numeric-first, unscaled one-hot |
| C08 | Decision tree | Probability | Unscaled one-hot |
| C09 | Extra Trees | Probability | Unscaled one-hot |
| C10 | Bagged trees | Probability | Unscaled one-hot |
| C11 | Random forest | Probability | Unscaled one-hot |
| C13 | AdaBoost | Probability | Dense unscaled one-hot |
| C15 | Classical gradient boosting | Probability | Dense unscaled one-hot |
| C16 | Histogram gradient boosting | Probability | Dense unscaled one-hot |
| C17 | XGBoost | Probability | Dense unscaled one-hot |
| C18 | LightGBM | Probability | Native categorical columns |
| C19 | CatBoost | Probability | Native categorical columns |
| C21 | Linear SVM | Margin | Scaled one-hot |
| C22 | RBF SVM | Margin | Scaled one-hot |
| C23 | Multilayer perceptron | Probability | Dense scaled one-hot |

The identifier sequence intentionally contains gaps. The protocol reserves the
remaining identifiers for later distinct procedures, including spline logistic
regression, discriminant analysis, balanced ensembles, resampling branches, EBM, and
advanced neural tabular models. Existing identifiers are not renumbered because they
are part of the persisted-task and study-contract identity.

## Why the final-comparison builders do not simply reuse every notebook factory

Historical notebook factories were designed for individual workflow exploration. Some
parallel tree and boosting factories therefore use `n_jobs=-1` or package-default
thread counts. That is appropriate for a one-model notebook but unsafe inside the
resumable comparison runner, which already parallelizes independent outer tasks.

Phase 3 provides final-comparison-specific constructors for the affected candidates.
They set `n_jobs=1` or `thread_count=1` inside each worker, while the coordinator owns
process-level parallelism. This prevents nested oversubscription such as multiple outer
workers each attempting to use all CPU cores.

The candidate-specific builders retain the same learning algorithms and
preprocessing philosophies as the corresponding exploratory workflows. They do not
reuse fitted notebook objects or historical validation scores.

## Search-space scope

Each core candidate now has two parameter profiles.

```text
smoke:
    intentionally small ranges used only for preflight validation

full:
    broader fixed ranges intended for the later frozen master-run configuration
```

The full profile is a candidate-level search space, not yet a complete experiment
budget. Phase 4 will freeze the per-candidate trial budgets, Stage-B confirmation
counts, runtime policy, and final master-run manifest before any long comparison is
started.

A full profile includes only hyperparameters that meaningfully change the candidate
procedure. Examples include regularization strength and penalty for linear models,
neighbour count and distance geometry for kNN, tree depth and leaf regularization for
tree methods, boosting complexity and shrinkage, and kernel width for RBF SVM.

## Preserved modelling distinctions

The registry records score semantics explicitly.

```text
Probability candidates:
    expose raw class-one probabilities. Their outer result can include raw Brier
    score and log loss, subject to later calibration analysis.

Margin candidates:
    expose a continuous decision score. They are valid for average precision and
    ROC-AUC, but raw margin values are not treated as probabilities. Probability
    metrics and calibration are deferred to the later calibrated-procedure stage.
```

In particular, linear and RBF SVMs deliberately retain decision scores rather than
silently enabling the expensive internal probability-fitting mode. A calibrated SVM
will be evaluated later as a separate end-to-end candidate procedure.

## Preflight smoke test

Run the new registry preflight after applying this phase:

```bash
python scripts/smoke_test_final_comparison_core_candidates.py
```

The test uses a deterministic 480-row stratified sample from `train.csv` only. For
every registered candidate it verifies:

```text
1. smoke-space suggestion produces JSON-safe parameters;
2. the pipeline can be cloned and fit on a training-only partition;
3. predicted labels have the expected binary shape;
4. continuous output matches the registry's probability or margin declaration;
5. declared inner worker-count parameters equal one.
```

It is not a performance comparison, it does not create a final comparison run
artifact, and it never reads the frozen held-out test set. Persistent study resume and
process-level parallel execution remain covered by the Phase-2 smoke test.

## Deliberately deferred procedures

Phase 3 does not yet add:

```text
feature-engineering alternatives
feature-selection alternatives
class weighting as an explicit cross-family policy branch
random over-sampling, under-sampling, SMOTENC, or balanced ensembles
spline logistic regression and discriminant-analysis procedures
EBM
TabNet, FT-Transformer, TabM, TabPFN, or AutoML candidates
calibration wrappers
threshold selection
stacking and blending
full repeated nested cross-validation
final fitting or held-out test evaluation
```

Those additions change the unit of comparison. They therefore require their own
fold-internal policy definitions and must be added to a new protocol revision or a
clearly recorded extension before the master run begins.
