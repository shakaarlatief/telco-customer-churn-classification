"""Interruption-safe infrastructure smoke test for final comparison.

The test executes a small but real training-only workload on a stratified subset of
``train.csv``. It validates the durable run manifest, development-data fingerprint,
deterministic repeated stratified splits, atomic task artifacts, intentional pause,
simulated interrupted-task recovery, safe resume, and controlled process-level
parallelism.

This is an infrastructure test, not a final comparison and not a performance estimate.
It never reads ``test.csv``. The temporary run directory is deleted after successful
verification.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.experiment_protocol import (  # noqa: E402
    ExperimentProtocol,
    make_dataframe_fingerprint,
    make_environment_fingerprint,
)
from telco_churn.experiment_runner import execute_registered_tasks  # noqa: E402
from telco_churn.experiment_splits import (  # noqa: E402
    make_repeated_stratified_outer_splits,
)
from telco_churn.experiment_store import (  # noqa: E402
    ExperimentStore,
    ExperimentTask,
    TASK_COMPLETED,
    TASK_INTERRUPTED,
    UnsafeResumeError,
)
from telco_churn.models import (  # noqa: E402
    make_l2_logistic_regression_pipeline,
    make_random_forest_pipeline,
)


SAMPLE_SIZE = 360


def _make_small_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a fixed stratified development-data subset without touching test data."""
    train_df = load_train_data()
    X, y = split_features_target(train_df)

    X_small, _, y_small, _ = train_test_split(
        X,
        y,
        train_size=min(SAMPLE_SIZE, len(y)),
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_small.reset_index(drop=True), y_small.reset_index(drop=True)


def _make_task_key(candidate_id: str, repeat_index: int, fold_index: int) -> str:
    """Return a filesystem-safe, deterministic task key."""
    return f"{candidate_id}__repeat_{repeat_index:02d}__fold_{fold_index:02d}"


def _smoke_worker(task: ExperimentTask) -> dict[str, Any]:
    """Fit one small real model task inside a worker process.

    The data path and split indices are passed through the serializable task payload.
    This mirrors the full runner's design: workers own model fitting, while the
    coordinator owns task-state transitions and artifact persistence.
    """
    data_path = Path(str(task.payload["data_path"]))
    frame = pd.read_csv(data_path)
    y = frame.pop("__target__").astype(int)

    train_indices = np.asarray(task.payload["train_indices"], dtype=int)
    validation_indices = np.asarray(task.payload["validation_indices"], dtype=int)

    X_train = frame.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_validation = frame.iloc[validation_indices]
    y_validation = y.iloc[validation_indices]

    if task.candidate_id == "smoke_logistic":
        estimator = make_l2_logistic_regression_pipeline(
            C=1.0,
            max_iter=1_000,
            random_state=RANDOM_STATE,
        )
    elif task.candidate_id == "smoke_random_forest":
        estimator = make_random_forest_pipeline(
            n_estimators=30,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            oob_score=False,
        )
    else:
        raise ValueError(f"Unexpected smoke candidate: {task.candidate_id}")

    estimator.fit(X_train, y_train)
    y_score = estimator.predict_proba(X_validation)[:, 1]

    return {
        "candidate_id": task.candidate_id,
        "repeat_index": task.repeat_index,
        "fold_index": task.fold_index,
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "average_precision": float(average_precision_score(y_validation, y_score)),
        "roc_auc": float(roc_auc_score(y_validation, y_score)),
        "mean_score": float(np.mean(y_score)),
    }


def _build_protocol() -> ExperimentProtocol:
    """Create a deliberately small protocol used only by the infrastructure smoke test."""
    return ExperimentProtocol(
        protocol_id="final_comparison_infrastructure_smoke",
        version="v1",
        candidate_ids=("smoke_logistic", "smoke_random_forest"),
        primary_metric="average_precision",
        outer_n_splits=2,
        outer_n_repeats=2,
        inner_n_splits=2,
        random_state=RANDOM_STATE,
        metadata={"purpose": "interruption-safe infrastructure smoke test"},
    )


def _build_tasks(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    protocol: ExperimentProtocol,
    data_path: Path,
) -> list[ExperimentTask]:
    """Build all candidate-by-outer-split atomic smoke tasks."""
    splits = make_repeated_stratified_outer_splits(
        y,
        n_splits=protocol.outer_n_splits,
        n_repeats=protocol.outer_n_repeats,
        random_state=protocol.random_state,
    )

    tasks: list[ExperimentTask] = []
    for candidate_id in protocol.candidate_ids:
        for split in splits:
            task_key = _make_task_key(
                candidate_id,
                split.repeat_index,
                split.fold_index,
            )
            tasks.append(
                ExperimentTask(
                    task_key=task_key,
                    candidate_id=candidate_id,
                    repeat_index=split.repeat_index,
                    fold_index=split.fold_index,
                    split_hash=split.split_hash,
                    payload={
                        "data_path": str(data_path),
                        "train_indices": split.train_indices.tolist(),
                        "validation_indices": split.validation_indices.tolist(),
                    },
                )
            )
    return tasks


def _assert_complete(store: ExperimentStore, expected_task_count: int) -> None:
    """Verify task completion, output integrity, and finite persisted metrics."""
    records = store.list_tasks()
    if len(records) != expected_task_count:
        raise AssertionError("Unexpected registered task count.")
    if any(record.status != TASK_COMPLETED for record in records):
        raise AssertionError("Not every task reached the completed state.")
    if any(record.attempts < 1 for record in records):
        raise AssertionError("Every completed task should have at least one attempt.")

    store.validate_completed_artifacts()

    for record in records:
        result_path = store.run_directory / str(record.result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["result"]
        for key in ("average_precision", "roc_auc", "mean_score"):
            if not np.isfinite(float(result[key])):
                raise AssertionError(f"Non-finite persisted metric {key}.")


def main() -> None:
    """Execute the full infrastructure smoke-test contract."""
    X, y = _make_small_training_data()
    protocol = _build_protocol()
    data_fingerprint = make_dataframe_fingerprint(X, y)
    environment_fingerprint = make_environment_fingerprint()

    with tempfile.TemporaryDirectory(prefix="telco-final-comparison-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        data_path = temp_root / "smoke_training_data.csv"
        smoke_frame = X.copy()
        smoke_frame["__target__"] = y.to_numpy()
        smoke_frame.to_csv(data_path, index=False)

        tasks = _build_tasks(
            X=X,
            y=y,
            protocol=protocol,
            data_path=data_path,
        )

        run_root = temp_root / "artifacts"
        run_id = "resume_contract_demo"

        print("Creating new smoke run...", flush=True)
        with ExperimentStore.create(
            artifacts_root=run_root,
            run_id=run_id,
            protocol_payload=protocol.to_dict(),
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint=data_fingerprint,
            environment_fingerprint=environment_fingerprint,
        ) as store:
            first_summary = execute_registered_tasks(
                store=store,
                tasks=tasks,
                worker=_smoke_worker,
                max_workers=1,
                stop_after_completed=2,
            )
            if first_summary["completed"] != 2 or first_summary["paused"] != 1:
                raise AssertionError("Intentional pause did not stop after two tasks.")

            completed_before_interruption = {
                record.task_key
                for record in store.list_tasks()
                if record.status == TASK_COMPLETED
            }
            if len(completed_before_interruption) != 2:
                raise AssertionError("Expected exactly two completed tasks before pause.")

            interrupted_task = next(
                record for record in store.list_tasks() if record.status != TASK_COMPLETED
            )
            if not store.claim_task(interrupted_task.task_key):
                raise AssertionError("Could not create a simulated interrupted task.")

        print("Reopening run after simulated interruption...", flush=True)
        with ExperimentStore.open_for_resume(
            artifacts_root=run_root,
            run_id=run_id,
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint_sha256=data_fingerprint["sha256"],
        ) as resumed_store:
            recovered = resumed_store.get_task(interrupted_task.task_key)
            if recovered.status != TASK_INTERRUPTED:
                raise AssertionError("Running task was not recovered as interrupted.")

            resumed_summary = execute_registered_tasks(
                store=resumed_store,
                tasks=tasks,
                worker=_smoke_worker,
                max_workers=2,
            )
            if resumed_summary["failed"] != 0:
                raise AssertionError("A resumed process-pool task failed unexpectedly.")
            _assert_complete(resumed_store, expected_task_count=len(tasks))

            for task_key in completed_before_interruption:
                record = resumed_store.get_task(task_key)
                if record.attempts != 1:
                    raise AssertionError(
                        "A completed task was recomputed instead of skipped during resume."
                    )

            second_resume_summary = execute_registered_tasks(
                store=resumed_store,
                tasks=tasks,
                worker=_smoke_worker,
                max_workers=1,
            )
            if second_resume_summary["submitted"] != 0:
                raise AssertionError(
                    "A fully completed run should submit no task during a later resume."
                )

        print("Checking protocol mismatch protection...", flush=True)
        incompatible_protocol = ExperimentProtocol(
            protocol_id=protocol.protocol_id,
            version="v2",
            candidate_ids=protocol.candidate_ids,
            primary_metric=protocol.primary_metric,
            outer_n_splits=protocol.outer_n_splits,
            outer_n_repeats=protocol.outer_n_repeats,
            inner_n_splits=protocol.inner_n_splits,
            random_state=protocol.random_state,
        )
        try:
            ExperimentStore.open_for_resume(
                artifacts_root=run_root,
                run_id=run_id,
                protocol_fingerprint=incompatible_protocol.fingerprint,
                data_fingerprint_sha256=data_fingerprint["sha256"],
            )
        except UnsafeResumeError:
            pass
        else:
            raise AssertionError("Protocol mismatch did not block unsafe resume.")

    print("Final-comparison infrastructure smoke test passed.")


if __name__ == "__main__":
    main()
