# Final Comparison Implementation Phase 2: Persistent HPO and First Candidate Procedures

## Purpose

Phase 1 established generic run identity, deterministic outer splits, task state,
atomic result persistence, safe resume checks, and controlled process-level execution.

Phase 2 connects that infrastructure to real nested hyperparameter optimization. It is
still a smoke-scale implementation phase. It proves the end-to-end contract using a
small, training-only candidate subset before the complete final-comparison registry,
feature-policy alternatives, resampling policies, calibration workflow, and full
50-outer-evaluation protocol are enabled.

The phase has four outputs:

```text
1. persistent task-local Optuna studies;
2. an initial validated candidate registry;
3. a reusable Extra Trees procedure inside the candidate registry; and
4. an interruption-and-resume nested-HPO smoke test.
```

## Persistent-study contract

Each outer task owns exactly one Optuna SQLite database:

```text
artifacts/final_comparison/<run_id>/
    optuna_studies/
        <candidate_id>/
            r<repeat>_f<fold>.sqlite
            r<repeat>_f<fold>.sampler.pkl
            r<repeat>_f<fold>.pruner.pkl
            r<repeat>_f<fold>.stage_b_confirmation.json
```

This is deliberately different from multiple workers sharing one SQLite study. The
project parallelizes independent outer tasks, while each task's inner Optuna trial loop
is sequential. This avoids SQLite write contention and preserves a clear relationship
between one outer split, one search space, one study history, and one final selected
configuration.

The study contract binds:

```text
candidate identifier
outer task key
outer split hash
inner-CV scheme
search profile
primary metric
confirmation policy
```

A mismatched study is rejected rather than silently reused.

Optuna persists trial history in its RDB backend, but the sampler and pruner instances
require separate checkpointing for reproducible seeded resume behaviour. Phase 2 saves
both after each completed trial using atomic replacement.

## Two-stage inner selection

The phase implements the protocol's two-stage inner selection pattern:

```text
Stage A:
    persistent TPE exploration using a smaller inner CV splitter

Stage B:
    reevaluate the top Stage-A configurations under a confirmation splitter;
    select the highest Stage-B mean average precision configuration
```

The smoke test uses tiny trial budgets and two-fold inner CV only to test mechanics.
Those values are not full-run hyperparameter budgets.

## Initial implemented candidate registry

The reusable registry initially contains:

```text
C02_LOGISTIC_REGRESSION
C09_EXTRA_TREES
C21_LINEAR_SVM
C23_MULTILAYER_PERCEPTRON
```

The smoke test executes logistic regression, Extra Trees, and linear SVM. MLP is
registered and has an existing dedicated workflow smoke test, but is not included in
the initial nested-HPO smoke run to keep the infrastructure validation focused.

The registry is an implementation subset, not the final frozen comprehensive candidate
library. The complete protocol candidate registry will be implemented in controlled
batches after the generic HPO path has passed its smoke test.

## Extra Trees

Extra Trees is added as a distinct randomized tree ensemble rather than being treated as
a duplicate random forest. The final-comparison candidate registry owns the fold-safe
pipeline and exposes:

```text
number of estimators
split criterion
maximum depth
minimum split and leaf sizes
maximum features per split
bootstrap choice and sample fraction
class weighting
cost-complexity pruning
```

As with every outer-worker estimator, `n_jobs` is set to one. The outer experiment
runner owns process-level parallelism.

## Smoke-test interruption scenario

The smoke test intentionally creates one task that:

```text
1. completes one Stage-A Optuna trial and persists it;
2. raises a controlled interruption;
3. is marked failed in the outer task registry;
4. is retried through the normal resume path;
5. reopens the same study;
6. completes the remaining trial budget;
7. runs Stage-B confirmation;
8. persists one validated outer-task result.
```

The test asserts that completed tasks are skipped, the interrupted study reaches its
target total trial count rather than restarting at zero, all completed outer artifacts
pass SHA-256 validation, and process-level resume works with two outer workers.

## Dependencies

Phase 2 adds Optuna to `requirements.txt`.

Install or update the environment from the repository root before running the smoke
test:

```bash
pip install -r requirements.txt
```

## Commands

```bash
python scripts/smoke_test_final_comparison_optuna.py
```

The smoke test never reads the held-out test set.

## Non-actions

Phase 2 does not yet:

```text
run the full 5-fold x 10-repeat master comparison
freeze final production HPO budgets
enable feature engineering or feature selection candidates
enable resampling, SMOTENC, or imbalance ensembles
fit calibration models
select thresholds
run stacking or blending
load or evaluate the held-out test set
```

Those additions follow only after the generic persistent HPO and candidate-factory path
has been verified on the actual operating environment.
