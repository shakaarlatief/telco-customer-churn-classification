"""Controlled task execution for resumable final-comparison experiments.

The runner owns one parallelism layer only: independent outer tasks may run in
separate Python processes. Each worker is configured to use one native numerical
thread so a worker pool cannot accidentally multiply into thousands of OpenMP, BLAS,
joblib, LightGBM, XGBoost, CatBoost, or tree-estimator threads.

Workers never write directly to the SQLite task registry. They return a JSON-compatible
result to the coordinator, which atomically persists the artifact and then marks the
task complete. This single-writer architecture is deliberately conservative and avoids
SQLite write contention on Windows while retaining process-level parallel model fitting.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, as_completed, wait
import os
from pathlib import Path
import signal
import time
import traceback
from typing import Any, Callable, Iterable, Mapping

from threadpoolctl import threadpool_limits

from telco_churn.experiment_progress import GracefulStopRequested, request_graceful_stop
from telco_churn.experiment_store import (
    ExperimentStore,
    ExperimentTask,
    TASK_COMPLETED,
)


TaskWorker = Callable[[ExperimentTask], Mapping[str, Any]]

_THREADPOOL_LIMITER = None


def configure_worker_threads(worker_threads: int = 1) -> None:
    """Limit one worker to a known number of native numerical threads.

    The function is safe to use as a ``ProcessPoolExecutor`` initializer. Environment
    variables are set before worker task code imports model libraries. The persistent
    ``threadpool_limits`` controller is retained in a module global so its limits
    remain active for the worker lifetime.
    """
    if worker_threads < 1:
        raise ValueError("worker_threads must be at least one.")

    for environment_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[environment_name] = str(worker_threads)

    global _THREADPOOL_LIMITER
    _THREADPOOL_LIMITER = threadpool_limits(limits=worker_threads)


def _run_worker(task: ExperimentTask, worker: TaskWorker) -> Mapping[str, Any]:
    """Execute one top-level picklable task worker."""
    return worker(task)


def execute_registered_tasks(
    *,
    store: ExperimentStore,
    tasks: Iterable[ExperimentTask],
    worker: TaskWorker,
    max_workers: int = 1,
    retry_failed: bool = False,
    stop_after_completed: int | None = None,
) -> dict[str, int]:
    """Execute pending tasks and persist results through the coordinator.

    Parameters
    ----------
    store:
        Durable experiment store. Its task registry is the authoritative source of
        completion state.
    tasks:
        Serializable task descriptions. Completed tasks are registered but skipped.
    worker:
        A top-level picklable function accepting one :class:`ExperimentTask` and
        returning a JSON-compatible result dictionary.
    max_workers:
        Number of process-level workers. Set to one for deterministic serial
        debugging; use more than one only at the outer-task level.
    retry_failed:
        Whether tasks previously marked failed may be claimed again.
    stop_after_completed:
        Testing and graceful-stop hook. In serial mode, stop after this number of
        newly completed tasks, leaving all remaining tasks pending for resume.

    Returns
    -------
    dict
        Counts of submitted, completed, skipped, failed, and intentionally paused
        tasks for this invocation.

    Notes
    -----
    The function does not schedule nested joblib or Optuna workers. Candidate-specific
    task workers must keep their own estimator-level thread count at one when
    ``max_workers`` exceeds one.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least one.")
    if stop_after_completed is not None and stop_after_completed < 1:
        raise ValueError("stop_after_completed must be positive when provided.")

    task_list = list(tasks)
    store.register_tasks(task_list)

    summary = {
        "submitted": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "paused": 0,
    }

    if max_workers == 1:
        configure_worker_threads(worker_threads=1)
        for task in task_list:
            record = store.get_task(task.task_key)
            if record.status == TASK_COMPLETED:
                summary["skipped"] += 1
                continue
            if not store.claim_task(task.task_key, retry_failed=retry_failed):
                summary["skipped"] += 1
                continue

            summary["submitted"] += 1
            try:
                result = worker(task)
                store.complete_task(task.task_key, result)
                summary["completed"] += 1
            except KeyboardInterrupt:
                raise
            except BaseException:
                store.fail_task(task.task_key, traceback.format_exc())
                summary["failed"] += 1

            if (
                stop_after_completed is not None
                and summary["completed"] >= stop_after_completed
            ):
                summary["paused"] = 1
                break

        return summary

    claimed_tasks: list[ExperimentTask] = []
    for task in task_list:
        record = store.get_task(task.task_key)
        if record.status == TASK_COMPLETED:
            summary["skipped"] += 1
            continue
        if store.claim_task(task.task_key, retry_failed=retry_failed):
            claimed_tasks.append(task)
            summary["submitted"] += 1
        else:
            summary["skipped"] += 1

    if not claimed_tasks:
        return summary

    futures = {}
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=configure_worker_threads,
            initargs=(1,),
        ) as executor:
            for task in claimed_tasks:
                futures[executor.submit(_run_worker, task, worker)] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    store.complete_task(task.task_key, result)
                    summary["completed"] += 1
                except BaseException:
                    store.fail_task(task.task_key, traceback.format_exc())
                    summary["failed"] += 1
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        raise

    return summary

TaskEventCallback = Callable[[str, ExperimentTask | None, Mapping[str, Any]], None]


def _emit_monitored_event(
    callback: TaskEventCallback | None,
    event_name: str,
    task: ExperimentTask | None,
    **details: Any,
) -> None:
    """Emit a coordinator-side monitoring event without risking experiment progress."""
    if callback is None:
        return
    try:
        callback(event_name, task, dict(details))
    except Exception:
        return


def _terminate_monitored_worker_processes(executor: ProcessPoolExecutor) -> None:
    """Best-effort emergency termination used only after a second Ctrl+C request."""
    processes = getattr(executor, "_processes", {})
    for process in processes.values():
        if process.is_alive():
            process.terminate()
    for process in processes.values():
        process.join(timeout=2.0)


def _refresh_monitored_progress_heartbeats(
    *,
    store: ExperimentStore,
    active_tasks: Mapping[Future, tuple[ExperimentTask, float]],
    progress_directory: Path | None,
    observed_mtimes: dict[str, int],
) -> None:
    """Mirror fresh worker progress sidecars into coordinator-owned task heartbeats."""
    if progress_directory is None:
        return
    for task, _ in active_tasks.values():
        sidecar = Path(progress_directory) / f"{task.task_key}.json"
        try:
            mtime_ns = sidecar.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if observed_mtimes.get(task.task_key) == mtime_ns:
            continue
        try:
            store.heartbeat(task.task_key)
        except Exception:
            continue
        observed_mtimes[task.task_key] = mtime_ns


def _configure_monitored_worker() -> None:
    """Configure a spawned worker and reserve Ctrl+C handling for the coordinator."""
    configure_worker_threads(worker_threads=1)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass


def execute_monitored_registered_tasks(
    *,
    store: ExperimentStore,
    tasks: Iterable[ExperimentTask],
    worker: TaskWorker,
    max_workers: int = 1,
    retry_failed: bool = False,
    stop_after_completed: int | None = None,
    event_callback: TaskEventCallback | None = None,
    stop_control_run_directory: Path | None = None,
    progress_directory: Path | None = None,
    poll_interval_seconds: float = 1.0,
    terminal_update_interval_seconds: float = 60.0,
) -> dict[str, int]:
    """Execute tasks with live monitoring and clean interruption semantics.

    This monitored scheduler claims a task only when it is actually submitted to an
    available worker. A ``running`` record therefore means active execution rather than
    a queued future. The legacy scheduler remains available for historical smoke tests;
    long-running pilots and later master runs should use this monitored variant.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least one.")
    if stop_after_completed is not None and stop_after_completed < 1:
        raise ValueError("stop_after_completed must be positive when provided.")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive.")
    if terminal_update_interval_seconds <= 0:
        raise ValueError("terminal_update_interval_seconds must be positive.")

    task_list = list(tasks)
    store.register_tasks(task_list)
    summary = {
        "submitted": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "interrupted": 0,
        "paused": 0,
    }

    if max_workers == 1:
        configure_worker_threads(worker_threads=1)
        for task in task_list:
            record = store.get_task(task.task_key)
            if record.status == TASK_COMPLETED:
                summary["skipped"] += 1
                continue
            if not store.claim_task(task.task_key, retry_failed=retry_failed):
                summary["skipped"] += 1
                continue

            summary["submitted"] += 1
            started = time.perf_counter()
            _emit_monitored_event(event_callback, "task_started", task, active_tasks=1)
            try:
                result = worker(task)
                store.complete_task(task.task_key, result)
                summary["completed"] += 1
                _emit_monitored_event(
                    event_callback,
                    "task_completed",
                    task,
                    elapsed_seconds=time.perf_counter() - started,
                    completed=summary["completed"],
                )
            except GracefulStopRequested as exc:
                store.interrupt_task(task.task_key, str(exc))
                summary["interrupted"] += 1
                summary["paused"] = 1
                _emit_monitored_event(event_callback, "task_interrupted", task, reason=str(exc))
                break
            except KeyboardInterrupt:
                store.interrupt_task(
                    task.task_key,
                    "Serial coordinator interrupted by user before task completion.",
                )
                summary["interrupted"] += 1
                summary["paused"] = 1
                if stop_control_run_directory is not None:
                    request_graceful_stop(
                        stop_control_run_directory,
                        reason="User requested a serial clean pause with Ctrl+C.",
                    )
                _emit_monitored_event(event_callback, "graceful_stop_requested", task)
                break
            except BaseException:
                store.fail_task(task.task_key, traceback.format_exc())
                summary["failed"] += 1
                _emit_monitored_event(
                    event_callback,
                    "task_failed",
                    task,
                    elapsed_seconds=time.perf_counter() - started,
                )

            if stop_after_completed is not None and summary["completed"] >= stop_after_completed:
                summary["paused"] = 1
                _emit_monitored_event(event_callback, "intentional_pause", task)
                break
        return summary

    task_iterator = iter(task_list)
    active: dict[Future, tuple[ExperimentTask, float]] = {}
    observed_progress_mtimes: dict[str, int] = {}
    stop_requested = False
    iterator_exhausted = False
    last_terminal_update = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_configure_monitored_worker)

    def submit_available_tasks() -> None:
        nonlocal iterator_exhausted
        while not stop_requested and not iterator_exhausted and len(active) < max_workers:
            try:
                task = next(task_iterator)
            except StopIteration:
                iterator_exhausted = True
                return
            record = store.get_task(task.task_key)
            if record.status == TASK_COMPLETED:
                summary["skipped"] += 1
                continue
            if not store.claim_task(task.task_key, retry_failed=retry_failed):
                summary["skipped"] += 1
                continue
            future = executor.submit(_run_worker, task, worker)
            active[future] = (task, time.perf_counter())
            summary["submitted"] += 1
            _emit_monitored_event(
                event_callback,
                "task_started",
                task,
                active_tasks=len(active),
                worker_capacity=max_workers,
            )

    try:
        submit_available_tasks()
        while active:
            try:
                completed_futures, _ = wait(
                    active,
                    timeout=poll_interval_seconds,
                    return_when=FIRST_COMPLETED,
                )
            except KeyboardInterrupt:
                if not stop_requested:
                    stop_requested = True
                    summary["paused"] = 1
                    if stop_control_run_directory is not None:
                        request_graceful_stop(
                            stop_control_run_directory,
                            reason="User requested a clean pause with Ctrl+C.",
                        )
                    _emit_monitored_event(
                        event_callback,
                        "graceful_stop_requested",
                        None,
                        active_tasks=len(active),
                        message=(
                            "No new outer tasks will start. Active tasks will stop at their "
                            "next safe persistent boundary. Press Ctrl+C again only for an "
                            "emergency hard stop."
                        ),
                    )
                    continue

                _emit_monitored_event(
                    event_callback,
                    "hard_stop_requested",
                    None,
                    active_tasks=len(active),
                )
                for future, (task, _) in active.items():
                    future.cancel()
                    store.interrupt_task(
                        task.task_key,
                        "Emergency second Ctrl+C terminated the active worker process.",
                    )
                    summary["interrupted"] += 1
                _terminate_monitored_worker_processes(executor)
                executor.shutdown(wait=False, cancel_futures=True)
                active.clear()
                summary["paused"] = 1
                return summary

            _refresh_monitored_progress_heartbeats(
                store=store,
                active_tasks=active,
                progress_directory=progress_directory,
                observed_mtimes=observed_progress_mtimes,
            )

            now = time.perf_counter()
            if now - last_terminal_update >= terminal_update_interval_seconds and active:
                _emit_monitored_event(
                    event_callback,
                    "active_snapshot",
                    None,
                    active_task_keys=[task.task_key for task, _ in active.values()],
                    elapsed_seconds={
                        task.task_key: now - started
                        for task, started in active.values()
                    },
                )
                last_terminal_update = now

            for future in completed_futures:
                task, started = active.pop(future)
                elapsed_seconds = time.perf_counter() - started
                try:
                    result = future.result()
                    store.complete_task(task.task_key, result)
                    summary["completed"] += 1
                    _emit_monitored_event(
                        event_callback,
                        "task_completed",
                        task,
                        elapsed_seconds=elapsed_seconds,
                        completed=summary["completed"],
                    )
                except GracefulStopRequested as exc:
                    store.interrupt_task(task.task_key, str(exc))
                    summary["interrupted"] += 1
                    summary["paused"] = 1
                    stop_requested = True
                    _emit_monitored_event(
                        event_callback,
                        "task_interrupted",
                        task,
                        elapsed_seconds=elapsed_seconds,
                        reason=str(exc),
                    )
                except BaseException:
                    store.fail_task(task.task_key, traceback.format_exc())
                    summary["failed"] += 1
                    _emit_monitored_event(
                        event_callback,
                        "task_failed",
                        task,
                        elapsed_seconds=elapsed_seconds,
                    )

            if not stop_requested:
                submit_available_tasks()

        return summary
    finally:
        if active:
            for _, (task, _) in active.items():
                try:
                    store.interrupt_task(
                        task.task_key,
                        "Coordinator exited before the active task completed.",
                    )
                except Exception:
                    continue
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
