"""Integration smoke test for final-comparison monitoring and clean pause recovery.

No Telco model is fitted and the held-out test data is never read. The fixture verifies
four operational properties in disposable directories:

1. task progress sidecars refresh periodically during a long worker operation;
2. the status reader inspects Optuna SQLite state through read-only queries;
3. a clean stop control interrupts an active monitored task without marking it failed;
4. the same interrupted task and pending work complete on a compatible resume, including
   the process-pool scheduler path used by the actual pilot.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telco_churn.experiment_progress import (  # noqa: E402
    GracefulStopRequested,
    TaskProgressReporter,
    clear_graceful_stop_request,
    collect_run_status,
    progress_path,
    request_graceful_stop,
)
from telco_churn.experiment_protocol import (  # noqa: E402
    ExperimentProtocol,
    make_dataframe_fingerprint,
    make_environment_fingerprint,
)
from telco_churn.experiment_runner import execute_monitored_registered_tasks  # noqa: E402
from telco_churn.experiment_store import (  # noqa: E402
    ExperimentStore,
    ExperimentTask,
    TASK_COMPLETED,
    TASK_INTERRUPTED,
    TASK_PENDING,
)


def monitoring_smoke_worker(task: ExperimentTask) -> Mapping[str, Any]:
    """Run a picklable synthetic worker through the real monitored scheduler path."""
    run_directory = Path(str(task.payload["run_directory"]))
    reporter = TaskProgressReporter(
        run_directory=run_directory,
        task_key=task.task_key,
        candidate_id=task.candidate_id,
        repeat_index=task.repeat_index,
        fold_index=task.fold_index,
        heartbeat_interval_seconds=float(task.payload.get("heartbeat_interval_seconds", 0.05)),
    )
    reporter.start(stage="stage_a", message="Synthetic monitorable work.")
    try:
        deadline = time.monotonic() + float(task.payload.get("work_seconds", 0.25))
        while time.monotonic() < deadline:
            if reporter.stop_requested():
                raise GracefulStopRequested("Synthetic worker observed the clean stop request.")
            time.sleep(0.01)
        reporter.close(final_stage="completed", message="Synthetic worker completed.")
        return {"schema_version": "monitoring_scheduler_smoke_v1", "task_key": task.task_key}
    except GracefulStopRequested as exc:
        reporter.close(final_stage="interrupted", message=str(exc))
        raise
    except BaseException as exc:
        reporter.close(final_stage="failed", message=f"{type(exc).__name__}: {exc}")
        raise


def _make_protocol() -> ExperimentProtocol:
    """Return a tiny protocol used only to construct disposable store fixtures."""
    return ExperimentProtocol(
        protocol_id="telco_final_monitoring_smoke",
        version="v2",
        candidate_ids=("MONITORING_SMOKE",),
        primary_metric="average_precision",
        outer_n_splits=2,
        outer_n_repeats=1,
        inner_n_splits=2,
        random_state=123,
        metadata={"purpose": "monitoring and interruption integration smoke test"},
    )


def _create_store(root: Path, run_id: str) -> ExperimentStore:
    """Create one disposable durable store without loading project data."""
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0, 1], name="Churn_binary")
    protocol = _make_protocol()
    return ExperimentStore.create(
        artifacts_root=root / "artifacts",
        run_id=run_id,
        protocol_payload=protocol.to_dict(),
        protocol_fingerprint=protocol.fingerprint,
        data_fingerprint=make_dataframe_fingerprint(X, y),
        environment_fingerprint=make_environment_fingerprint(
            package_names=("numpy", "pandas", "optuna")
        ),
    )


def _make_task(store: ExperimentStore, index: int, *, study_path: Path | None = None) -> ExperimentTask:
    """Create one synthetic task with the progress contract used by the pilot worker."""
    task_key = f"monitoring_smoke__r00__f{index:02d}"
    payload: dict[str, Any] = {
        "task_kind": "monitoring_smoke",
        "run_directory": str(store.run_directory),
        "heartbeat_interval_seconds": 0.05,
        "work_seconds": 0.30,
    }
    if study_path is not None:
        payload["study_database_path"] = str(study_path)
        payload["study_name"] = task_key
    return ExperimentTask(
        task_key=task_key,
        candidate_id="MONITORING_SMOKE",
        repeat_index=0,
        fold_index=index,
        split_hash=f"monitoring-smoke-split-{index}",
        payload=payload,
    )


def _assert_periodic_sidecar_heartbeat(store: ExperimentStore) -> None:
    """Verify the worker-side reporter refreshes a sidecar without stage transitions."""
    reporter = TaskProgressReporter(
        run_directory=store.run_directory,
        task_key="direct_reporter",
        candidate_id="MONITORING_SMOKE",
        repeat_index=0,
        fold_index=0,
        heartbeat_interval_seconds=0.04,
    )
    reporter.start(stage="stage_a", message="Direct heartbeat fixture.")
    path = reporter.path
    first_payload = json.loads(path.read_text(encoding="utf-8"))
    time.sleep(0.12)
    second_payload = json.loads(path.read_text(encoding="utf-8"))
    reporter.close(final_stage="completed", message="Direct heartbeat fixture completed.")
    if first_payload["updated_at"] == second_payload["updated_at"]:
        raise AssertionError("Progress reporter did not refresh its periodic sidecar heartbeat.")


def _assert_read_only_optuna_status(store: ExperimentStore, task: ExperimentTask) -> None:
    """Create one study and verify the status query sees it without loading RDBStorage."""
    import optuna
    from sqlalchemy.engine import URL

    study_path = Path(str(task.payload["study_database_path"]))
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=str(task.payload["study_name"]),
        storage=str(URL.create(drivername="sqlite", database=str(study_path.resolve()))),
        direction="maximize",
    )
    study.optimize(lambda trial: 0.5, n_trials=1)
    snapshot = collect_run_status(store.run_directory)
    observed = next(item for item in snapshot.tasks if item.task_key == task.task_key)
    if observed.study.complete_trials != 1:
        raise AssertionError("Read-only status did not observe the completed Optuna trial.")
    if observed.study.best_average_precision != 0.5:
        raise AssertionError("Read-only status did not observe the Optuna objective value.")


def _assert_serial_pause_and_resume(store: ExperimentStore, tasks: list[ExperimentTask]) -> None:
    """Exercise the real monitored scheduler's stop-control and resume states."""
    def request_stop_later() -> None:
        time.sleep(0.10)
        request_graceful_stop(store.run_directory, reason="Synthetic monitored pause.")

    stop_thread = threading.Thread(target=request_stop_later, daemon=True)
    stop_thread.start()
    first_summary = execute_monitored_registered_tasks(
        store=store,
        tasks=tasks,
        worker=monitoring_smoke_worker,
        max_workers=1,
        stop_control_run_directory=store.run_directory,
        progress_directory=store.run_directory / "progress",
        poll_interval_seconds=0.02,
    )
    stop_thread.join(timeout=1.0)
    first_states = store.task_summary()
    if first_summary["interrupted"] != 1 or first_summary["paused"] != 1:
        raise AssertionError(f"Unexpected monitored pause summary: {first_summary}")
    if first_states.get(TASK_INTERRUPTED) != 1 or first_states.get(TASK_PENDING) != 1:
        raise AssertionError(f"Unexpected durable states after clean pause: {first_states}")
    interrupted_path = progress_path(store.run_directory, tasks[0].task_key)
    interrupted_payload = json.loads(interrupted_path.read_text(encoding="utf-8"))
    if interrupted_payload.get("stage") != "interrupted":
        raise AssertionError("Interrupted monitored task did not write a terminal progress sidecar.")

    clear_graceful_stop_request(store.run_directory)
    resumed_summary = execute_monitored_registered_tasks(
        store=store,
        tasks=tasks,
        worker=monitoring_smoke_worker,
        max_workers=1,
        stop_control_run_directory=store.run_directory,
        progress_directory=store.run_directory / "progress",
        poll_interval_seconds=0.02,
    )
    if resumed_summary["completed"] != 2:
        raise AssertionError(f"Compatible resume did not complete both unfinished tasks: {resumed_summary}")
    if any(record.status != TASK_COMPLETED for record in store.list_tasks()):
        raise AssertionError("All serial monitored smoke tasks must be completed after resume.")
    store.validate_completed_artifacts()


def _assert_parallel_scheduler(store: ExperimentStore, tasks: list[ExperimentTask]) -> None:
    """Exercise the process-pool scheduler path used by the real pilot."""
    summary = execute_monitored_registered_tasks(
        store=store,
        tasks=tasks,
        worker=monitoring_smoke_worker,
        max_workers=2,
        stop_control_run_directory=store.run_directory,
        progress_directory=store.run_directory / "progress",
        poll_interval_seconds=0.02,
    )
    if summary["completed"] != len(tasks) or summary["failed"] != 0:
        raise AssertionError(f"Parallel monitored scheduler did not complete every task: {summary}")
    if any(record.status != TASK_COMPLETED for record in store.list_tasks()):
        raise AssertionError("Parallel monitored smoke tasks were not all marked completed.")
    store.validate_completed_artifacts()


def main() -> None:
    """Run all disposable monitoring integration checks."""
    try:
        import numpy  # noqa: F401
        import optuna  # noqa: F401
        import pandas  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Monitoring smoke test requires the locked project environment.") from exc

    temporary_root = Path(tempfile.mkdtemp(prefix="telco-final-monitoring-"))
    serial_store: ExperimentStore | None = None
    parallel_store: ExperimentStore | None = None
    try:
        serial_store = _create_store(temporary_root, "monitoring_serial_smoke")
        study_path = serial_store.run_directory / "optuna_studies" / "monitoring" / "r00_f00.sqlite"
        serial_tasks = [
            _make_task(serial_store, 0, study_path=study_path),
            _make_task(serial_store, 1),
        ]
        serial_store.register_tasks(serial_tasks)
        _assert_periodic_sidecar_heartbeat(serial_store)
        _assert_read_only_optuna_status(serial_store, serial_tasks[0])
        _assert_serial_pause_and_resume(serial_store, serial_tasks)
        serial_store.close()
        serial_store = None

        parallel_store = _create_store(temporary_root, "monitoring_parallel_smoke")
        parallel_tasks = [_make_task(parallel_store, index) for index in range(3)]
        _assert_parallel_scheduler(parallel_store, parallel_tasks)
        parallel_store.close()
        parallel_store = None

        print("Final-comparison monitoring integration smoke test passed.")
    finally:
        if serial_store is not None:
            serial_store.close()
        if parallel_store is not None:
            parallel_store.close()
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
