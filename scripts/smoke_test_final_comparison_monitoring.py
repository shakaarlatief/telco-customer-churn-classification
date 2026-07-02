"""Integration smoke test for final-comparison monitoring, event logs, and clean pause.

The test uses disposable synthetic tasks only. It never loads Telco data, fits a Telco model,
or accesses the held-out test set. It validates the real monitored scheduler path,
progress sidecars, append-only task and coordinator logs, read-only status inspection,
clean pause recovery, and process-pool worker-event forwarding.
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
    RunEventLogger,
    TaskProgressReporter,
    clear_graceful_stop_request,
    collect_run_status,
    progress_path,
    render_run_status,
    request_graceful_stop,
    task_event_jsonl_path,
)
from telco_churn.hpo import release_persistent_study_resources  # noqa: E402
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
    """Run a top-level synthetic worker through the real process-pool scheduler."""
    run_directory = Path(str(task.payload["run_directory"]))
    reporter = TaskProgressReporter(
        run_directory=run_directory,
        task_key=task.task_key,
        candidate_id=task.candidate_id,
        repeat_index=task.repeat_index,
        fold_index=task.fold_index,
        heartbeat_interval_seconds=float(task.payload.get("heartbeat_interval_seconds", 0.04)),
    )
    reporter.start(
        stage="stage_a",
        message="Synthetic Stage-A task started.",
        target_completed_trials=2,
    )
    try:
        reporter.update(
            stage="stage_a",
            message="Synthetic trial one started.",
            event_name="stage_a_trial_started",
            current_trial_number=1,
            target_completed_trials=2,
            completed_trials=0,
            current_trial_parameters={"alpha": 0.1, "depth": 3},
        )
        deadline = time.monotonic() + float(task.payload.get("work_seconds", 0.30))
        emitted_fold = False
        while time.monotonic() < deadline:
            if reporter.stop_requested():
                raise GracefulStopRequested("Synthetic worker observed the clean stop request.")
            if not emitted_fold:
                reporter.update(
                    stage="stage_a",
                    message="Synthetic trial one completed inner fold one.",
                    current_trial_number=1,
                    target_completed_trials=2,
                    completed_trials=0,
                    inner_fold_index=1,
                    inner_fold_total=2,
                    completed_inner_folds=1,
                    fold_average_precision=0.5,
                    partial_mean_average_precision=0.5,
                )
                emitted_fold = True
            time.sleep(0.01)
        reporter.update(
            stage="stage_a",
            message="Synthetic trial one completed.",
            event_name="stage_a_trial_terminal",
            current_trial_number=1,
            target_completed_trials=2,
            completed_trials=1,
            last_trial_average_precision=0.5,
            best_stage_a_average_precision=0.5,
        )
        reporter.close(
            final_stage="completed",
            message="Synthetic worker completed.",
        )
        return {"schema_version": "monitoring_scheduler_smoke_v2", "task_key": task.task_key}
    except GracefulStopRequested as exc:
        reporter.close(final_stage="interrupted", message=str(exc))
        raise
    except BaseException as exc:
        reporter.close(final_stage="failed", message=f"{type(exc).__name__}: {exc}")
        raise


def _make_protocol() -> ExperimentProtocol:
    """Create a disposable protocol without loading project data."""
    return ExperimentProtocol(
        protocol_id="telco_final_monitoring_smoke",
        version="v3",
        candidate_ids=("MONITORING_SMOKE",),
        primary_metric="average_precision",
        outer_n_splits=2,
        outer_n_repeats=1,
        inner_n_splits=2,
        random_state=123,
        metadata={"purpose": "monitoring, event, and interruption integration smoke test"},
    )


def _create_store(root: Path, run_id: str) -> ExperimentStore:
    """Create one disposable durable store with a tiny synthetic data fingerprint."""
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


def _make_task(store: ExperimentStore, index: int) -> ExperimentTask:
    """Create one synthetic outer task with the same telemetry contract as the pilot."""
    task_key = f"monitoring_smoke__r00__f{index:02d}"
    return ExperimentTask(
        task_key=task_key,
        candidate_id="MONITORING_SMOKE",
        repeat_index=0,
        fold_index=index,
        split_hash=f"monitoring-smoke-split-{index}",
        payload={
            "task_kind": "monitoring_smoke",
            "run_directory": str(store.run_directory),
            "heartbeat_interval_seconds": 0.04,
            "work_seconds": 0.30,
        },
    )


def _assert_reporter_heartbeat_and_events(store: ExperimentStore) -> None:
    """Verify periodic sidecar liveness and durable worker event emission."""
    reporter = TaskProgressReporter(
        run_directory=store.run_directory,
        task_key="direct_reporter",
        candidate_id="MONITORING_SMOKE",
        repeat_index=0,
        fold_index=0,
        heartbeat_interval_seconds=0.04,
    )
    reporter.start(stage="stage_a", message="Direct heartbeat fixture.", target_completed_trials=2)
    first = json.loads(reporter.path.read_text(encoding="utf-8"))
    time.sleep(0.12)
    second = json.loads(reporter.path.read_text(encoding="utf-8"))
    reporter.update(
        stage="stage_a",
        message="Direct trial started.",
        event_name="stage_a_trial_started",
        current_trial_number=1,
        target_completed_trials=2,
        current_trial_parameters={"alpha": 0.1},
        partial_mean_average_precision=None,
        completed_inner_folds=0,
    )
    reporter.update(
        stage="stage_a",
        message="Direct fold telemetry only.",
        event_name="stage_a_fold_completed",
        current_trial_number=1,
        target_completed_trials=2,
        completed_inner_folds=1,
        inner_fold_index=1,
        inner_fold_total=2,
        partial_mean_average_precision=0.5,
    )
    reporter.close(final_stage="completed", message="Direct heartbeat fixture completed.")
    if first["updated_at"] == second["updated_at"]:
        raise AssertionError("Progress reporter did not refresh its periodic sidecar heartbeat.")
    event_path = task_event_jsonl_path(store.run_directory, "direct_reporter")
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    if not any(event["event"] == "stage_a_trial_started" for event in events):
        raise AssertionError("Task reporter did not persist the detailed trial-start event.")
    if any(event["event"] == "stage_a_fold_completed" for event in events):
        raise AssertionError("Fold telemetry must remain in the live sidecar, not the durable task log.")


def _assert_serial_pause_and_resume(store: ExperimentStore, tasks: list[ExperimentTask]) -> None:
    """Exercise clean pause and compatible interrupted-task resume."""
    def request_stop_later() -> None:
        time.sleep(0.10)
        request_graceful_stop(store.run_directory, reason="Synthetic monitored pause.")

    threading.Thread(target=request_stop_later, daemon=True).start()
    first_summary = execute_monitored_registered_tasks(
        store=store,
        tasks=tasks,
        worker=monitoring_smoke_worker,
        max_workers=1,
        stop_control_run_directory=store.run_directory,
        progress_directory=store.run_directory / "progress",
        worker_event_directory=store.run_directory / "logs" / "tasks",
        poll_interval_seconds=0.02,
    )
    first_states = store.task_summary()
    if first_summary["interrupted"] != 1 or first_summary["paused"] != 1:
        raise AssertionError(f"Unexpected monitored pause summary: {first_summary}")
    if first_states.get(TASK_INTERRUPTED) != 1 or first_states.get(TASK_PENDING) != 1:
        raise AssertionError(f"Unexpected states after clean pause: {first_states}")

    interrupted_payload = json.loads(
        progress_path(store.run_directory, tasks[0].task_key).read_text(encoding="utf-8")
    )
    if interrupted_payload.get("stage") != "interrupted":
        raise AssertionError("Interrupted task did not persist terminal progress telemetry.")

    clear_graceful_stop_request(store.run_directory)
    resumed_summary = execute_monitored_registered_tasks(
        store=store,
        tasks=tasks,
        worker=monitoring_smoke_worker,
        max_workers=1,
        stop_control_run_directory=store.run_directory,
        progress_directory=store.run_directory / "progress",
        worker_event_directory=store.run_directory / "logs" / "tasks",
        poll_interval_seconds=0.02,
    )
    if resumed_summary["completed"] != 2:
        raise AssertionError(f"Compatible resume did not complete unfinished work: {resumed_summary}")
    if any(record.status != TASK_COMPLETED for record in store.list_tasks()):
        raise AssertionError("All serial smoke tasks must be completed after resume.")
    store.validate_completed_artifacts()


def _assert_parallel_event_forwarding(store: ExperimentStore, tasks: list[ExperimentTask]) -> None:
    """Exercise the pilot's two-worker path and coordinator forwarding of worker events."""
    observed_events: list[str] = []

    def callback(event_name: str, task: ExperimentTask | None, details: Mapping[str, Any]) -> None:
        if event_name == "worker_event":
            worker_event = details.get("worker_event", {})
            if isinstance(worker_event, Mapping):
                observed_events.append(str(worker_event.get("event")))

    summary = execute_monitored_registered_tasks(
        store=store,
        tasks=tasks,
        worker=monitoring_smoke_worker,
        max_workers=2,
        event_callback=callback,
        stop_control_run_directory=store.run_directory,
        progress_directory=store.run_directory / "progress",
        worker_event_directory=store.run_directory / "logs" / "tasks",
        poll_interval_seconds=0.02,
    )
    if summary["completed"] != len(tasks) or summary["failed"] != 0:
        raise AssertionError(f"Parallel monitored scheduler did not complete all tasks: {summary}")
    if "stage_a_trial_started" not in observed_events:
        raise AssertionError("Coordinator did not forward worker trial events during parallel work.")
    if any(record.status != TASK_COMPLETED for record in store.list_tasks()):
        raise AssertionError("Parallel monitored tasks were not all completed.")
    store.validate_completed_artifacts()


def _assert_read_only_optuna_study_inspection(store: ExperimentStore) -> None:
    """Verify status reads a real Optuna study without mutating its SQLite file."""
    try:
        import optuna
        from sqlalchemy.engine import URL
    except ImportError as exc:
        raise AssertionError("The monitoring smoke test requires Optuna's SQLite dependency.") from exc

    study_path = store.run_directory / "optuna_studies" / "read_only" / "status.sqlite"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study_name = "read_only_dashboard_study"
    storage = optuna.storages.RDBStorage(
        url=str(URL.create(drivername="sqlite", database=str(study_path.resolve())))
    )
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="maximize",
    )
    study.optimize(lambda trial: 0.625, n_trials=1)
    release_persistent_study_resources(study)

    task = ExperimentTask(
        task_key="read_only_optuna_status",
        candidate_id="MONITORING_SMOKE",
        repeat_index=0,
        fold_index=99,
        split_hash="read-only-optuna-status",
        payload={
            "study_database_path": str(study_path),
            "study_name": study_name,
        },
    )
    store.register_tasks([task])
    before = study_path.read_bytes()
    snapshot = collect_run_status(store.run_directory)
    after = study_path.read_bytes()
    if before != after:
        raise AssertionError("Read-only Optuna status inspection modified the study database.")

    matching = [item for item in snapshot.tasks if item.task_key == task.task_key]
    if len(matching) != 1:
        raise AssertionError("Read-only Optuna status task was not present in the snapshot.")
    observed = matching[0].study
    if not observed.present or observed.complete_trials != 1:
        raise AssertionError("Read-only status did not observe the completed Optuna trial.")
    if (
        observed.best_average_precision is None
        or abs(float(observed.best_average_precision) - 0.625) > 1e-12
    ):
        raise AssertionError("Read-only status did not expose the Optuna objective value.")


def _assert_read_only_dashboard_and_run_log(store: ExperimentStore) -> None:
    """Verify bounded event-tail reads and durable latest-invocation metadata."""
    logger = RunEventLogger(store.run_directory)
    logger.emit(
        "run_started",
        message="Synthetic monitoring invocation started.",
        details={"worker_capacity": 2, "task_total": 2},
    )
    # Force the run-start event outside the 24-record dashboard tail. The invocation
    # sidecar must retain start time and worker capacity independently of log length.
    for index in range(30):
        logger.emit(
            "task_completed",
            message=f"Synthetic durable event {index + 1}.",
            details={"task_position": index + 1, "task_total": 30},
        )
    logger.emit(
        "run_completed",
        message="Synthetic monitoring invocation completed.",
        details={"completed": 2},
    )

    # A completed invocation must show its bounded run duration rather than continue to
    # grow with wall-clock time after completion. With no active tasks, the complete
    # dashboard must therefore remain identical across a later refresh.
    snapshot = collect_run_status(store.run_directory)
    rendered = render_run_status(snapshot)
    time.sleep(1.1)
    rendered_after_wait = render_run_status(collect_run_status(store.run_directory))
    if rendered != rendered_after_wait:
        raise AssertionError("Completed invocation duration continued to grow after completion.")
    if snapshot.current_invocation_started_at is None:
        raise AssertionError("Invocation sidecar did not retain the invocation start time.")
    if snapshot.current_invocation_finished_at is None:
        raise AssertionError("Invocation sidecar did not retain the terminal timestamp.")
    if snapshot.current_invocation_state != "completed":
        raise AssertionError("Invocation sidecar did not retain the completed lifecycle state.")
    if snapshot.worker_capacity != 2:
        raise AssertionError("Invocation sidecar did not retain configured worker capacity.")
    if not snapshot.latest_event or snapshot.latest_event.get("event") != "run_completed":
        raise AssertionError("Tail-only coordinator-log reading did not retain the latest event.")
    if "Outer tasks:" not in rendered or "[" not in rendered:
        raise AssertionError("Read-only dashboard did not render the run-level progress bar.")
    if "Invocation: completed" not in rendered or "elapsed:" not in rendered:
        raise AssertionError("Completed invocation lifecycle metadata was not rendered.")
    if not (store.run_directory / "logs" / "coordinator.log").exists():
        raise AssertionError("Human-readable coordinator event log is missing.")
    if not (store.run_directory / "logs" / "coordinator_events.jsonl").exists():
        raise AssertionError("Structured coordinator event log is missing.")

def main() -> None:
    """Run all disposable monitoring and event-log integration checks."""
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Monitoring smoke test requires the locked project environment.") from exc

    temporary_root = Path(tempfile.mkdtemp(prefix="telco-final-monitoring-v3-"))
    serial_store: ExperimentStore | None = None
    parallel_store: ExperimentStore | None = None
    try:
        serial_store = _create_store(temporary_root, "monitoring_serial_smoke")
        serial_tasks = [_make_task(serial_store, 0), _make_task(serial_store, 1)]
        serial_store.register_tasks(serial_tasks)
        _assert_reporter_heartbeat_and_events(serial_store)
        _assert_serial_pause_and_resume(serial_store, serial_tasks)
        _assert_read_only_optuna_study_inspection(serial_store)
        _assert_read_only_dashboard_and_run_log(serial_store)
        serial_store.close()
        serial_store = None

        parallel_store = _create_store(temporary_root, "monitoring_parallel_smoke")
        parallel_tasks = [_make_task(parallel_store, index) for index in range(3)]
        _assert_parallel_event_forwarding(parallel_store, parallel_tasks)
        parallel_store.close()
        parallel_store = None

        print("Final-comparison monitoring and event-log integration smoke test passed.")
    finally:
        if serial_store is not None:
            serial_store.close()
        if parallel_store is not None:
            parallel_store.close()
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
