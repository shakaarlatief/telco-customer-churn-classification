"""Operational progress, stop-control, and strictly read-only status helpers.

Long nested-CV experiments are intentionally run outside notebooks. This module keeps
operational state observable without allowing worker processes to write directly to the
central SQLite task registry. Each worker writes an atomic progress sidecar at task and
stage boundaries, then refreshes that sidecar periodically while a long fitting operation
is in progress. The coordinator remains the sole writer of task-state transitions and
result artifacts.

The status view combines three durable sources:

* the run-level SQLite task registry for authoritative task states;
* worker progress sidecars for current stage and liveness; and
* task-local Optuna SQLite files for Stage-A trial counts and best inner AP.

Status inspection opens both the registry and Optuna study databases in SQLite read-only
mode. It never resumes a run, changes task states, updates Optuna heartbeats, or creates
any artifact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import threading
from typing import Any, Mapping


class GracefulStopRequested(RuntimeError):
    """Signal that a worker reached a durable boundary after a clean pause request.

    The coordinator records this outcome as ``interrupted`` rather than ``failed``. A
    later compatible resume may therefore claim the outer task without an explicit
    failed-task retry switch, while the task-local Optuna study retains every completed
    Stage-A trial.
    """


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with explicit offset information."""
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one small JSON control or progress artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def progress_path(run_directory: Path, task_key: str) -> Path:
    """Return the durable progress-sidecar path for one outer task."""
    return Path(run_directory) / "progress" / f"{task_key}.json"


def stop_request_path(run_directory: Path) -> Path:
    """Return the run-level control file used for a clean user-requested pause."""
    return Path(run_directory) / "control" / "stop_requested.json"


def request_graceful_stop(run_directory: Path, *, reason: str) -> Path:
    """Persist one run-level clean-stop request and return its artifact path.

    A pre-existing request is retained so the first interruption's reason and timestamp
    cannot be overwritten by a subsequent terminal signal before workers observe it.
    """
    path = stop_request_path(run_directory)
    if path.exists():
        return path
    _atomic_write_json(
        path,
        {
            "schema_version": "final_comparison_stop_request_v1",
            "requested_at": _utc_now(),
            "reason": str(reason),
            "hostname": socket.gethostname(),
            "process_id": int(os.getpid()),
        },
    )
    return path


def clear_graceful_stop_request(run_directory: Path) -> None:
    """Remove a prior stop request immediately before a deliberate new invocation."""
    stop_request_path(run_directory).unlink(missing_ok=True)


def read_graceful_stop_request(run_directory: Path) -> dict[str, Any] | None:
    """Read one stop request without changing it, returning ``None`` when absent."""
    path = stop_request_path(run_directory)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable_control_file": str(path)}
    return dict(payload) if isinstance(payload, Mapping) else {"invalid_control_file": str(path)}


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp defensively for elapsed-time rendering."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_duration(seconds: float | None) -> str:
    """Render a non-negative duration compactly for a terminal status table."""
    if seconds is None:
        return "-"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


class TaskProgressReporter:
    """Write a synchronized liveness record for one worker-owned outer task.

    The worker never writes to the coordinator-owned task registry. It writes a separate
    atomic JSON sidecar instead. In addition to explicit stage and trial updates, a daemon
    heartbeat thread refreshes the sidecar while the worker is inside a lengthy model fit
    or CV evaluation. This prevents a healthy CatBoost, MLP, or other long running trial
    from appearing frozen merely because no Python-level stage boundary has occurred.
    """

    def __init__(
        self,
        *,
        run_directory: Path,
        task_key: str,
        candidate_id: str,
        repeat_index: int,
        fold_index: int,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive.")
        self.run_directory = Path(run_directory)
        self.task_key = str(task_key)
        self.candidate_id = str(candidate_id)
        self.repeat_index = int(repeat_index)
        self.fold_index = int(fold_index)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._stage = "initializing"
        self._message: str | None = None
        self._extra: dict[str, Any] = {}
        self._started_at = _utc_now()
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def path(self) -> Path:
        """Return this task's progress-sidecar path."""
        return progress_path(self.run_directory, self.task_key)

    def _payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": "final_comparison_task_progress_v1",
                "updated_at": _utc_now(),
                "started_at": self._started_at,
                "hostname": socket.gethostname(),
                "process_id": int(os.getpid()),
                "task_key": self.task_key,
                "candidate_id": self.candidate_id,
                "outer_repeat_index": self.repeat_index,
                "outer_fold_index": self.fold_index,
                "stage": self._stage,
                "message": self._message,
                "details": dict(self._extra),
            }

    def _write(self) -> None:
        _atomic_write_json(self.path, self._payload())

    def _heartbeat_loop(self) -> None:
        """Refresh the sidecar until normal completion or process termination."""
        while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
            try:
                self._write()
            except Exception:
                # Liveness telemetry must never terminate a model-fitting worker. The next
                # scheduled interval may succeed after a transient filesystem contention.
                continue

    def _start_heartbeat(self) -> None:
        with self._lock:
            if self._heartbeat_thread is not None:
                return
            thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"telco-progress-{self.task_key}",
                daemon=True,
            )
            self._heartbeat_thread = thread
            thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        with self._lock:
            thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.heartbeat_interval_seconds + 1.0))

    def start(self, *, stage: str = "initializing", message: str | None = None) -> None:
        """Create the initial sidecar and begin periodic worker liveness updates."""
        self.update(stage=stage, message=message)
        self._start_heartbeat()

    def update(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        **details: Any,
    ) -> None:
        """Persist a stage transition or supplementary task-progress details."""
        with self._lock:
            if stage is not None:
                self._stage = str(stage)
            if message is not None:
                self._message = str(message)
            if details:
                self._extra.update(details)
        self._write()

    def stop_requested(self) -> bool:
        """Return whether the coordinator has requested a clean run-level pause."""
        return stop_request_path(self.run_directory).exists()

    def close(self, *, final_stage: str, message: str | None = None) -> None:
        """Stop periodic liveness updates and persist one terminal worker state."""
        self._stop_heartbeat()
        self.update(stage=final_stage, message=message)


@dataclass(frozen=True)
class StudyProgress:
    """Read-only current summary of one task-local Optuna study."""

    present: bool
    complete_trials: int | None
    failed_trials: int | None
    pruned_trials: int | None
    running_trials: int | None
    best_average_precision: float | None
    latest_trial_at: str | None
    stage_b_present: bool
    error: str | None = None


@dataclass(frozen=True)
class TaskStatusSnapshot:
    """Read-only status row suitable for terminal rendering."""

    task_key: str
    candidate_id: str
    repeat_index: int
    fold_index: int
    status: str
    attempts: int
    started_at: str | None
    heartbeat_at: str | None
    completed_at: str | None
    error_text: str | None
    progress: Mapping[str, Any] | None
    study: StudyProgress

    def elapsed_seconds(self, *, now: datetime | None = None) -> float | None:
        start = _parse_timestamp(self.started_at)
        if start is None:
            return None
        end = _parse_timestamp(self.completed_at) or now or datetime.now(UTC)
        return max(0.0, (end - start).total_seconds())


@dataclass(frozen=True)
class RunStatusSnapshot:
    """Full read-only status snapshot for one experiment run directory."""

    run_directory: Path
    run_id: str
    created_at: str | None
    purpose: str | None
    status_counts: Mapping[str, int]
    tasks: tuple[TaskStatusSnapshot, ...]
    stop_request: Mapping[str, Any] | None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite file with the operating system-enforced read-only URI mode."""
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=3.0)


def _read_optuna_study_progress(
    *,
    study_path: Path | None,
    study_name: str | None,
) -> StudyProgress:
    """Inspect an Optuna SQLite study through read-only SQL queries.

    The project locks Optuna to a known database schema. Querying its durable tables
    directly avoids constructing a write-capable ``RDBStorage`` merely to display status.
    This is particularly important while a live worker owns the study connection.
    """
    if study_path is None or study_name is None or not study_path.exists():
        return StudyProgress(False, None, None, None, None, None, None, False)

    stage_b_present = study_path.with_suffix(".stage_b_confirmation.json").exists()
    try:
        with _open_sqlite_read_only(study_path) as connection:
            study_row = connection.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (str(study_name),),
            ).fetchone()
            if study_row is None:
                raise LookupError(f"No Optuna study named {study_name!r}.")
            study_id = int(study_row[0])
            rows = connection.execute(
                """
                SELECT
                    trials.state,
                    trials.datetime_start,
                    trials.datetime_complete,
                    trial_values.value,
                    trial_heartbeats.heartbeat
                FROM trials
                LEFT JOIN trial_values
                    ON trials.trial_id = trial_values.trial_id
                    AND trial_values.objective = 0
                LEFT JOIN trial_heartbeats
                    ON trials.trial_id = trial_heartbeats.trial_id
                WHERE trials.study_id = ?
                ORDER BY trials.number
                """,
                (study_id,),
            ).fetchall()

        counts: Counter[str] = Counter()
        completed_values: list[float] = []
        timestamps: list[str] = []
        for state, started_at, completed_at, value, heartbeat_at in rows:
            normalized_state = str(state).lower()
            counts[normalized_state] += 1
            if normalized_state == "complete" and value is not None:
                completed_values.append(float(value))
            for timestamp in (started_at, completed_at, heartbeat_at):
                if timestamp is not None:
                    timestamps.append(str(timestamp))

        return StudyProgress(
            True,
            int(counts.get("complete", 0)),
            int(counts.get("fail", 0)),
            int(counts.get("pruned", 0)),
            int(counts.get("running", 0)),
            max(completed_values, default=None),
            max(timestamps, default=None),
            stage_b_present,
        )
    except Exception as exc:
        return StudyProgress(
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            stage_b_present,
            f"{type(exc).__name__}: {exc}",
        )


def collect_run_status(run_directory: Path) -> RunStatusSnapshot:
    """Collect a completely read-only snapshot for a created experiment run."""
    run_directory = Path(run_directory)
    manifest_path = run_directory / "run_manifest.json"
    database_path = run_directory / "task_registry.sqlite"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    if not database_path.exists():
        raise FileNotFoundError(f"Task registry does not exist: {database_path}")

    manifest = _read_json(manifest_path) or {}
    protocol = manifest.get("protocol", {})
    metadata = protocol.get("metadata", {}) if isinstance(protocol, Mapping) else {}
    with _open_sqlite_read_only(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                task_key, candidate_id, repeat_index, fold_index, status, attempts,
                started_at, heartbeat_at, completed_at, error_text, payload_json
            FROM tasks
            ORDER BY candidate_id, repeat_index, fold_index, task_key
            """
        ).fetchall()

    task_snapshots: list[TaskStatusSnapshot] = []
    for row in rows:
        payload = json.loads(row[10])
        task_payload = payload.get("payload", {}) if isinstance(payload, Mapping) else {}
        study_path_value = task_payload.get("study_database_path")
        study_path = Path(str(study_path_value)) if study_path_value else None
        study = _read_optuna_study_progress(
            study_path=study_path,
            study_name=task_payload.get("study_name"),
        )
        task_snapshots.append(
            TaskStatusSnapshot(
                task_key=str(row[0]),
                candidate_id=str(row[1]),
                repeat_index=int(row[2]),
                fold_index=int(row[3]),
                status=str(row[4]),
                attempts=int(row[5]),
                started_at=row[6],
                heartbeat_at=row[7],
                completed_at=row[8],
                error_text=row[9],
                progress=_read_json(progress_path(run_directory, str(row[0]))),
                study=study,
            )
        )

    counts = Counter(task.status for task in task_snapshots)
    return RunStatusSnapshot(
        run_directory=run_directory,
        run_id=str(manifest.get("run_id", run_directory.name)),
        created_at=manifest.get("created_at"),
        purpose=metadata.get("purpose") if isinstance(metadata, Mapping) else None,
        status_counts=dict(sorted(counts.items())),
        tasks=tuple(task_snapshots),
        stop_request=read_graceful_stop_request(run_directory),
    )


def _truncate(value: str | None, width: int) -> str:
    if not value:
        return "-"
    if len(value) <= width:
        return value
    return f"{value[: max(1, width - 1)]}…"


def render_run_status(
    snapshot: RunStatusSnapshot,
    *,
    include_completed: bool = False,
    failures_only: bool = False,
) -> str:
    """Render a stable plain-text status table without ranking candidate procedures."""
    counts = ", ".join(f"{status}={count}" for status, count in snapshot.status_counts.items())
    lines = [
        f"Run: {snapshot.run_id}",
        f"Directory: {snapshot.run_directory}",
        f"Task states: {counts or 'no registered tasks'}",
    ]
    if snapshot.stop_request is not None:
        lines.append(
            "Control: clean stop requested"
            + (
                f" at {snapshot.stop_request.get('requested_at')}"
                if snapshot.stop_request.get("requested_at") else ""
            )
        )
    if snapshot.purpose:
        lines.append(f"Purpose: {snapshot.purpose}")

    selected: list[TaskStatusSnapshot] = []
    for task in snapshot.tasks:
        if failures_only and task.status not in {"failed", "interrupted"}:
            continue
        if not include_completed and task.status == "completed":
            continue
        selected.append(task)

    if not selected:
        lines.append("No matching task rows.")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "task                              state        stage        trials                 best AP  elapsed   heartbeat",
            "-" * 123,
        ]
    )
    now = datetime.now(UTC)
    for task in selected:
        if task.progress:
            progress_stage = str(task.progress.get("stage", "-"))
        elif task.study.present:
            progress_stage = "stage_b" if task.study.stage_b_present else "stage_a"
        else:
            progress_stage = "awaiting_worker" if task.status == "running" else "-"

        if task.study.present and task.study.complete_trials is not None:
            trial_parts = [f"{task.study.complete_trials} ok"]
            if task.study.failed_trials:
                trial_parts.append(f"{task.study.failed_trials} fail")
            if task.study.pruned_trials:
                trial_parts.append(f"{task.study.pruned_trials} pruned")
            if task.study.running_trials:
                trial_parts.append(f"{task.study.running_trials} running")
            trials = "/".join(trial_parts)
        elif task.study.error:
            trials = "study error"
        else:
            trials = "-"

        best = "-" if task.study.best_average_precision is None else f"{task.study.best_average_precision:.4f}"
        heartbeat_at = None
        if task.progress:
            heartbeat_at = task.progress.get("updated_at")
        heartbeat_at = heartbeat_at or task.heartbeat_at
        heartbeat_time = _parse_timestamp(str(heartbeat_at) if heartbeat_at else None)
        heartbeat = "-" if heartbeat_time is None else f"{format_duration((now - heartbeat_time).total_seconds())} ago"
        label = f"{task.candidate_id} r{task.repeat_index:02d}f{task.fold_index:02d}"
        lines.append(
            f"{_truncate(label, 33):33} "
            f"{_truncate(task.status, 12):12} "
            f"{_truncate(progress_stage, 12):12} "
            f"{_truncate(trials, 21):21} "
            f"{best:>7}  "
            f"{_truncate(format_duration(task.elapsed_seconds(now=now)), 8):8} "
            f"{_truncate(heartbeat, 15):15}"
        )
        if task.study.error:
            lines.append(f"  study: {_truncate(task.study.error, 220)}")
        if task.status in {"failed", "interrupted"} and task.error_text:
            first_line = task.error_text.strip().splitlines()[0]
            lines.append(f"  error: {_truncate(first_line, 220)}")
    return "\n".join(lines)
