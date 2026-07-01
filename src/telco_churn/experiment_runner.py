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

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import traceback
from typing import Any, Callable, Iterable, Mapping

from threadpoolctl import threadpool_limits

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
