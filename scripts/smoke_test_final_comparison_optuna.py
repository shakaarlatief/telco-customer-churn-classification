"""End-to-end smoke test for persistent nested Optuna HPO.

This test intentionally uses a small stratified subset of ``train.csv`` only. It
exercises three real candidate families through the new process-level outer-task runner:

* regularized logistic regression;
* Extra Trees; and
* linear SVM.

One task is deliberately interrupted after one persisted Stage-A Optuna trial. The
coordinator records that task as failed. The test then reopens the same run and retries
only the failed task. The retried task must reuse its study database and complete the
remaining trial budget instead of restarting from trial zero.

The test is a durability and compatibility check, not a model-performance experiment.
It never loads the held-out test set.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    import optuna  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Run `pip install -r requirements.txt` "
        "from the repository root, then rerun this smoke test."
    ) from exc


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_EXTRA_TREES,
    CANDIDATE_LINEAR_SVM,
    CANDIDATE_LOGISTIC_REGRESSION,
    INITIAL_CANDIDATE_REGISTRY,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.experiment_protocol import (  # noqa: E402
    ExperimentProtocol,
    make_dataframe_fingerprint,
    make_environment_fingerprint,
)
from telco_churn.experiment_runner import execute_registered_tasks  # noqa: E402
from telco_churn.experiment_splits import (  # noqa: E402
    derive_seed,
    make_repeated_stratified_outer_splits,
)
from telco_churn.experiment_store import (  # noqa: E402
    ExperimentStore,
    ExperimentTask,
    TASK_COMPLETED,
    TASK_FAILED,
)
from telco_churn.experiment_tasks import run_nested_hpo_outer_task  # noqa: E402


SAMPLE_SIZE = 360
SMOKE_CANDIDATES = (
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_EXTRA_TREES,
    CANDIDATE_LINEAR_SVM,
)
OUTER_N_SPLITS = 2
OUTER_N_REPEATS = 1
INNER_N_SPLITS = 2
STAGE_A_N_TRIALS = 3
CONFIRMATION_TOP_K = 2


def make_sample_positions(y) -> np.ndarray:
    """Return deterministic full-training row positions for the smoke subset."""
    all_positions = np.arange(len(y), dtype=np.int64)
    sample_positions, _ = train_test_split(
        all_positions,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return np.sort(np.asarray(sample_positions, dtype=np.int64))


def make_tasks(
    *,
    run_directory: Path,
    y_sample,
    sample_positions: np.ndarray,
    protocol_fingerprint: str,
) -> list[ExperimentTask]:
    """Create deterministic real nested-HPO task descriptions."""
    outer_splits = make_repeated_stratified_outer_splits(
        y_sample,
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    tasks: list[ExperimentTask] = []
    for candidate_id in SMOKE_CANDIDATES:
        for split in outer_splits:
            task_seed = derive_seed(
                RANDOM_STATE,
                "phase2_smoke",
                candidate_id,
                split.repeat_index,
                split.fold_index,
            )
            task_key = (
                f"{candidate_id.lower()}__r{split.repeat_index:02d}"
                f"__f{split.fold_index:02d}"
            )
            study_path = (
                run_directory
                / "optuna_studies"
                / candidate_id.lower()
                / f"r{split.repeat_index:02d}_f{split.fold_index:02d}.sqlite"
            )
            payload = {
                "task_kind": "nested_hpo_outer_v1",
                "protocol_fingerprint": protocol_fingerprint,
                "sample_positions": [int(value) for value in sample_positions],
                "outer_train_indices": [
                    int(value) for value in split.train_indices
                ],
                "outer_validation_indices": [
                    int(value) for value in split.validation_indices
                ],
                "stage_a_n_splits": INNER_N_SPLITS,
                "stage_b_n_splits": INNER_N_SPLITS,
                "stage_a_n_trials": STAGE_A_N_TRIALS,
                "confirmation_top_k": CONFIRMATION_TOP_K,
                "search_profile": "smoke",
                "study_database_path": str(study_path),
                "study_name": task_key,
                "task_seed": int(task_seed),
            }

            # The first candidate task is deliberately interrupted once after one
            # durable HPO trial. A retry below must preserve that trial and finish
            # the remaining trial budget.
            if (
                candidate_id == CANDIDATE_LOGISTIC_REGRESSION
                and split.repeat_index == 0
                and split.fold_index == 0
            ):
                payload["simulate_hpo_interrupt_marker_path"] = str(
                    run_directory / "checkpoints" / "phase2_interrupt_once.marker"
                )

            tasks.append(
                ExperimentTask(
                    task_key=task_key,
                    candidate_id=candidate_id,
                    repeat_index=split.repeat_index,
                    fold_index=split.fold_index,
                    split_hash=split.split_hash,
                    payload=payload,
                )
            )
    return tasks


def assert_completed_hpo_results(store: ExperimentStore) -> None:
    """Validate durable result, study, and confirmation artifacts."""
    records = store.list_tasks()
    if len(records) != len(SMOKE_CANDIDATES) * OUTER_N_SPLITS:
        raise AssertionError("Unexpected number of registered nested-HPO tasks.")
    if any(record.status != TASK_COMPLETED for record in records):
        raise AssertionError("All smoke tasks must be completed after retry.")

    store.validate_completed_artifacts()

    result_files = sorted(store.results_directory.glob("*.json"))
    if len(result_files) != len(records):
        raise AssertionError("Each completed task must have one atomic result artifact.")

    for record in records:
        result_path = store.run_directory / str(record.result_path)
        payload = __import__("json").loads(result_path.read_text(encoding="utf-8"))
        result = payload["result"]
        if result["candidate_id"] not in SMOKE_CANDIDATES:
            raise AssertionError("Unexpected candidate result was persisted.")
        if result["metrics"]["average_precision"] is None:
            raise AssertionError("Outer average precision was not persisted.")
        if result["inner_search"]["stage_a_completed_trials"] != STAGE_A_N_TRIALS:
            raise AssertionError(
                "Persistent Optuna study did not reach the intended total trial budget."
            )
        if not result["inner_search"]["stage_b_records"]:
            raise AssertionError("Stage-B confirmation records are missing.")

    study_databases = sorted(store.run_directory.glob("optuna_studies/**/*.sqlite"))
    sampler_checkpoints = sorted(
        store.run_directory.glob("optuna_studies/**/*.sampler.pkl")
    )
    pruner_checkpoints = sorted(
        store.run_directory.glob("optuna_studies/**/*.pruner.pkl")
    )
    confirmation_files = sorted(
        store.run_directory.glob("optuna_studies/**/*.stage_b_confirmation.json")
    )
    expected_count = len(records)
    if not (
        len(study_databases)
        == len(sampler_checkpoints)
        == len(pruner_checkpoints)
        == len(confirmation_files)
        == expected_count
    ):
        raise AssertionError(
            "Persistent study databases and all checkpoint sidecars are required."
        )


def _failed_task_diagnostics(store: ExperimentStore) -> str:
    """Return persisted task failures so a smoke-test assertion remains actionable."""
    details: list[str] = []
    for record in store.list_tasks():
        if record.status != TASK_FAILED:
            continue
        details.append(
            "\n".join(
                [
                    f"task_key={record.task_key}",
                    f"attempts={record.attempts}",
                    record.error_text or "<no persisted error text>",
                ]
            )
        )
    return "\n\n".join(details) if details else "<no failed task records>"


def _close_store_if_open(store: ExperimentStore | None) -> None:
    """Close one smoke-test store without hiding the original failure."""
    if store is None:
        return
    try:
        store.close()
    except Exception:
        pass


def main() -> None:
    validate_candidate_registry(INITIAL_CANDIDATE_REGISTRY)

    train_df = load_train_data()
    X_all, y_all = split_features_target(train_df)
    sample_positions = make_sample_positions(y_all)
    X_sample = X_all.iloc[sample_positions].reset_index(drop=True)
    y_sample = y_all.iloc[sample_positions].reset_index(drop=True)

    protocol = ExperimentProtocol(
        protocol_id="telco_final_comparison_phase2_smoke",
        version="v1",
        candidate_ids=SMOKE_CANDIDATES,
        primary_metric="average_precision",
        outer_n_splits=OUTER_N_SPLITS,
        outer_n_repeats=OUTER_N_REPEATS,
        inner_n_splits=INNER_N_SPLITS,
        random_state=RANDOM_STATE,
        metadata={
            "purpose": "persistent nested Optuna smoke test",
            "search_profile": "smoke",
            "stage_a_n_trials": STAGE_A_N_TRIALS,
            "confirmation_top_k": CONFIRMATION_TOP_K,
        },
    )
    data_fingerprint = make_dataframe_fingerprint(X_sample, y_sample)
    environment_fingerprint = make_environment_fingerprint(
        package_names=(
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "optuna",
            "threadpoolctl",
        )
    )

    temporary_root = Path(tempfile.mkdtemp(prefix="telco-final-comparison-phase2-"))
    completed_successfully = False

    try:
        artifacts_root = temporary_root / "artifacts"
        run_id = "phase2_nested_hpo_smoke"

        print("Creating persistent nested-HPO smoke run...", flush=True)
        store: ExperimentStore | None = ExperimentStore.create(
            artifacts_root=artifacts_root,
            run_id=run_id,
            protocol_payload=protocol.to_dict(),
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint=data_fingerprint,
            environment_fingerprint=environment_fingerprint,
        )
        tasks = make_tasks(
            run_directory=store.run_directory,
            y_sample=y_sample,
            sample_positions=sample_positions,
            protocol_fingerprint=protocol.fingerprint,
        )

        print("Running first pass with one deliberate HPO interruption...", flush=True)
        try:
            first_summary = execute_registered_tasks(
                store=store,
                tasks=tasks,
                worker=run_nested_hpo_outer_task,
                max_workers=1,
            )
            if first_summary["failed"] != 1:
                raise AssertionError(
                    "The first pass must record exactly one simulated HPO interruption. "
                    f"Summary: {first_summary}.\n\nPersisted failures:\n"
                    f"{_failed_task_diagnostics(store)}"
                )
            first_status = store.task_summary()
            if first_status[TASK_FAILED] != 1:
                raise AssertionError(
                    "The interrupted task must remain recorded as failed. "
                    f"Task state summary: {first_status}."
                )
        finally:
            _close_store_if_open(store)
            store = None

        print("Reopening run and retrying only the interrupted task...", flush=True)
        store = ExperimentStore.open_for_resume(
            artifacts_root=artifacts_root,
            run_id=run_id,
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint_sha256=data_fingerprint["sha256"],
        )
        try:
            resumed_summary = execute_registered_tasks(
                store=store,
                tasks=tasks,
                worker=run_nested_hpo_outer_task,
                max_workers=2,
                retry_failed=True,
            )
            if resumed_summary["completed"] != 1:
                raise AssertionError(
                    "Resume should complete only the previously interrupted HPO task. "
                    f"Summary: {resumed_summary}."
                )
            if resumed_summary["skipped"] != len(tasks) - 1:
                raise AssertionError(
                    "Resume must skip every already completed task. "
                    f"Summary: {resumed_summary}."
                )

            assert_completed_hpo_results(store)
        finally:
            _close_store_if_open(store)
            store = None

        # Run a second independent smoke-scale experiment with two workers from the
        # beginning. The interruption/retry scenario above proves durable resumption,
        # while this pass proves that multiple ordinary outer tasks can execute under
        # the controlled Windows process-pool configuration.
        parallel_run_id = "phase2_parallel_nested_hpo_smoke"
        print("Running independent two-worker outer-task smoke pass...", flush=True)
        parallel_store: ExperimentStore | None = ExperimentStore.create(
            artifacts_root=artifacts_root,
            run_id=parallel_run_id,
            protocol_payload=protocol.to_dict(),
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint=data_fingerprint,
            environment_fingerprint=environment_fingerprint,
        )
        parallel_tasks = make_tasks(
            run_directory=parallel_store.run_directory,
            y_sample=y_sample,
            sample_positions=sample_positions,
            protocol_fingerprint=protocol.fingerprint,
        )
        for task in parallel_tasks:
            task.payload.pop("simulate_hpo_interrupt_marker_path", None)

        try:
            parallel_summary = execute_registered_tasks(
                store=parallel_store,
                tasks=parallel_tasks,
                worker=run_nested_hpo_outer_task,
                max_workers=2,
            )
            if parallel_summary["completed"] != len(parallel_tasks):
                raise AssertionError(
                    "Two-worker smoke pass did not complete every outer task. "
                    f"Summary: {parallel_summary}.\n\nPersisted failures:\n"
                    f"{_failed_task_diagnostics(parallel_store)}"
                )
            if parallel_summary["failed"] != 0:
                raise AssertionError(
                    "Two-worker smoke pass must not record task failures. "
                    f"Summary: {parallel_summary}.\n\nPersisted failures:\n"
                    f"{_failed_task_diagnostics(parallel_store)}"
                )
            assert_completed_hpo_results(parallel_store)
        finally:
            _close_store_if_open(parallel_store)
            parallel_store = None

        completed_successfully = True
    finally:
        if completed_successfully:
            try:
                shutil.rmtree(temporary_root)
            except OSError as exc:
                print(
                    "Smoke test passed, but temporary artifacts could not be removed. "
                    f"They were retained at: {temporary_root}\n{type(exc).__name__}: {exc}",
                    flush=True,
                )
        else:
            print(
                "Smoke test failed. Diagnostic artifacts were retained at: "
                f"{temporary_root}",
                flush=True,
            )

    print("Persistent nested Optuna HPO smoke test passed.")


if __name__ == "__main__":
    main()
