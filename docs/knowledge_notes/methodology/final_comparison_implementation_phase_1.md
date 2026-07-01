# Final Comparison Implementation Phase 1: Durable Run Foundation

## Purpose

This document records the first implementation layer of the final-comparison system. It does **not** run the full repeated nested cross-validation protocol, tune models with Optuna, or evaluate the held-out test set.

Phase 1 creates the safety-critical infrastructure that must exist before a long model-comparison run can be trusted:

```text
immutable run protocol identity
development-data fingerprint
deterministic repeated outer split generation
SQLite task registry
atomic per-task result persistence
intentional pause and ordinary interruption recovery
safe resume checks
one controlled process-level parallelism layer
a real smoke test using train.csv only
```

The system is intentionally implemented before model-specific HPO. A sophisticated search cannot be considered robust if an interrupted laptop session loses its progress, silently changes folds, or mixes results from a different protocol revision.

---

## 1. New reusable modules

```text
src/telco_churn/experiment_protocol.py
src/telco_churn/experiment_splits.py
src/telco_churn/experiment_store.py
src/telco_churn/experiment_runner.py
```

### 1.1 `experiment_protocol.py`

Defines an immutable `ExperimentProtocol` with:

```text
protocol identifier and revision
candidate identifiers
primary metric
outer and inner validation structure
root random seed
JSON-compatible metadata
SHA-256 protocol fingerprint
```

The same module creates:

```text
development-data fingerprint:
    row content, row order, target labels, feature names, dtypes, and index

environment fingerprint:
    Python implementation and version
    platform information
    selected package versions
```

A resumed run compares the requested protocol fingerprint and development-data fingerprint against the original manifest. A mismatch raises an error instead of silently continuing.

### 1.2 `experiment_splits.py`

Creates deterministic repeated stratified outer splits. The full protocol will use:

```text
5 folds x 10 repeats = 50 outer evaluation tasks per candidate procedure
```

The smoke test uses two folds and two repeats only to make its checks fast.

Each split has:

```text
repeat index
fold index
training indices
validation indices
SHA-256 split hash
```

The split hash becomes part of every atomic task identity. This makes it impossible for a task result to be reused with a different validation partition accidentally.

### 1.3 `experiment_store.py`

Provides an `ExperimentStore` backed by:

```text
run_manifest.json:
    immutable protocol, dataset, and environment provenance

task_registry.sqlite:
    durable task status, attempts, timing, error text, and result references

results/<task_key>.json:
    atomic result payload for each completed task
```

Task states are:

```text
pending
running
completed
failed
interrupted
```

A result file is written atomically before the task state changes to `completed`. Consequently, a crash cannot leave a partially written result file marked as complete.

### 1.4 `experiment_runner.py`

Provides controlled task execution.

The coordinator process owns:

```text
task registration
task claiming
task state changes
SQLite writes
result artifact persistence
failure recording
terminal-level scheduling decisions
```

Worker processes own only:

```text
model fit
prediction
metric calculation
JSON-compatible result return
```

This is deliberate. Concurrent worker writes to SQLite can become fragile on Windows. A single coordinator writer is safer, while independent model-fitting tasks still run in parallel.

---

## 2. Parallelism policy in Phase 1

The runner permits process-level parallelism only at the outer-task layer:

```text
candidate procedure x outer repeat x outer fold
```

Inside each worker:

```text
OMP_NUM_THREADS = 1
MKL_NUM_THREADS = 1
OPENBLAS_NUM_THREADS = 1
BLIS_NUM_THREADS = 1
NUMEXPR_NUM_THREADS = 1
threadpoolctl limit = 1
```

The policy prevents nested oversubscription such as:

```text
8 outer processes
x 8 XGBoost threads
x 8 OpenMP / BLAS threads
= hundreds of competing CPU threads
```

The later HPO layer will run sequential Optuna trials inside each worker, while independent outer tasks are parallel. This keeps the parallelism topology easy to inspect and avoids the Jupyter and worker-cleanup problems seen in earlier notebook execution.

---

## 3. What the smoke test proves

The smoke test is:

```text
scripts/smoke_test_final_comparison_infrastructure.py
```

It uses a small fixed stratified subset of `train.csv` only. It builds two real pipelines from existing reusable factories:

```text
L2 logistic regression
random forest
```

It then verifies:

```text
protocol validation
data and environment fingerprints
repeated stratified split integrity
task registration
two completed tasks followed by an intentional pause
a deliberately claimed but unfinished task, simulating a hard interruption
resume recovery of that interrupted task
process-level parallel execution for the remaining tasks
skipping previously completed tasks
atomic artifact hashes for every completed result
blocking resume under a protocol mismatch
finite AP and ROC-AUC metrics for every persisted smoke task
```

The smoke test does not use the test set and does not make any model-selection claim.

Run it from the repository root:

```bash
python scripts/smoke_test_final_comparison_infrastructure.py
```

Expected final line:

```text
Final-comparison infrastructure smoke test passed.
```

---

## 4. What Phase 1 intentionally does not yet do

The following are deferred to subsequent implementation phases:

```text
candidate-specific Optuna objectives and persistent Optuna studies
inner tuning and repeated nested CV execution
candidate registry and compatibility matrix
Telco deterministic feature-engineering transformer
feature-selection policies
imbalance / SMOTENC pipelines
Extra Trees and other new estimator factories
live Rich terminal dashboard
coordinator lock and heartbeat-age policy for simultaneous coordinators
GPU resource pool
calibration, threshold selection, stacking, and final test evaluation
```

Deferring them does not reduce the final scope. It ensures that each later component is built on an interruption-safe foundation.

---

## 5. Resume semantics

The eventual full runner will expose commands such as:

```text
run:
    start a new identifier and write an immutable manifest

resume:
    reopen one existing identifier only after protocol and data checks

fork:
    create a new identifier when the protocol changes materially
```

In Phase 1, the store already enforces the essential rule:

```text
same run directory
+
different protocol fingerprint
=
unsafe resume error
```

This distinction is important. A changed candidate set, different split count, different metric, or different feature policy should create a new evidence stream rather than overwrite or blend with prior results.

---

## 6. Immediate next implementation phase

After Phase 1 passes on the user's actual Windows environment, Phase 2 adds:

```text
candidate registry with compatibility validation
Optuna persistent study adapter
generic inner-CV objective runner
task-specific seed derivation
candidate search-space definitions
Extra Trees as the first newly added model factory
a small end-to-end nested-HPO smoke test:
    logistic regression
    Extra Trees
    CatBoost
    linear SVM
    MLP
```

Only after that end-to-end smoke test passes will the project expand to the full core candidate library and advanced conditional candidates.
