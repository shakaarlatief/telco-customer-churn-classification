"""Operational telemetry, event logs, stop control, and read-only monitoring.

The final-comparison workflow can run for a long time because each durable outer task
contains an inner two-stage hyperparameter selection procedure. This module separates
three responsibilities:

* the coordinator-owned SQLite registry remains authoritative for task state;
* worker processes write atomic progress sidecars and task-local event logs; and
* a read-only monitor combines those durable artifacts into a compact dashboard.

Worker processes never mutate the central registry. They only write progress telemetry
and their own append-only task event files. The coordinator is the sole writer of
registry transitions, completed-result artifacts, and the combined coordinator event log.
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
import statistics
import tempfile
import threading
from typing import Any, Mapping, Sequence


class GracefulStopRequested(RuntimeError):
    """Signal that a worker observed a clean pause request at a durable boundary."""


_LOW_LEVEL_FOLD_EVENT_NAMES = frozenset(
    {
        "stage_a_fold_started",
        "stage_a_fold_completed",
        "stage_b_fold_started",
        "stage_b_fold_completed",
    }
)


def is_low_level_fold_event(event_name: str | None) -> bool:
    """Return whether an event is live fold telemetry rather than a durable audit event.

    Fold boundaries are useful in the active dashboard because they show that a lengthy
    cross-validation trial remains alive and how much of the current trial is complete.
    They are intentionally not retained in the append-only task or coordinator logs:
    a master comparison would otherwise produce many low-value log records that repeat
    information already preserved in the current progress sidecar.
    """
    return str(event_name) in _LOW_LEVEL_FOLD_EVENT_NAMES


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _local_timestamp() -> str:
    """Return a concise timezone-aware local timestamp for human-readable event logs."""
    timestamp = datetime.now().astimezone()
    zone = timestamp.tzname() or timestamp.strftime("%z")
    return f"{timestamp:%Y-%m-%d %H:%M:%S} {zone}"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON artifact in the target directory."""
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


def _append_text_line(path: Path, line: str) -> None:
    """Append one already-formatted line and synchronize it to the local filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line.rstrip("\n"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one structured event record to an append-only JSON Lines artifact."""
    _append_text_line(
        path,
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str),
    )


def progress_path(run_directory: Path, task_key: str) -> Path:
    """Return the atomic worker-progress sidecar for one outer task."""
    return Path(run_directory) / "progress" / f"{task_key}.json"


def stop_request_path(run_directory: Path) -> Path:
    """Return the run-level clean-stop control artifact."""
    return Path(run_directory) / "control" / "stop_requested.json"


def coordinator_event_jsonl_path(run_directory: Path) -> Path:
    """Return the coordinator-owned structured event log."""
    return Path(run_directory) / "logs" / "coordinator_events.jsonl"


def coordinator_log_path(run_directory: Path) -> Path:
    """Return the coordinator-owned human-readable event log."""
    return Path(run_directory) / "logs" / "coordinator.log"


def task_event_jsonl_path(run_directory: Path, task_key: str) -> Path:
    """Return one worker-owned structured task-event log."""
    return Path(run_directory) / "logs" / "tasks" / f"{task_key}.jsonl"


def task_log_path(run_directory: Path, task_key: str) -> Path:
    """Return one worker-owned human-readable task-event log."""
    return Path(run_directory) / "logs" / "tasks" / f"{task_key}.log"


def invocation_status_path(run_directory: Path) -> Path:
    """Return the small atomic sidecar for the latest runner invocation."""
    return Path(run_directory) / "status" / "latest_invocation.json"


def request_graceful_stop(run_directory: Path, *, reason: str) -> Path:
    """Persist the first clean-stop request without overwriting its timestamp or reason."""
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
    """Clear an earlier pause request immediately before a deliberate new invocation."""
    stop_request_path(run_directory).unlink(missing_ok=True)


def read_graceful_stop_request(run_directory: Path) -> dict[str, Any] | None:
    """Read the current stop request without changing experiment state."""
    path = stop_request_path(run_directory)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable_control_file": str(path)}
    return dict(payload) if isinstance(payload, Mapping) else {"invalid_control_file": str(path)}


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp defensively."""
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def format_duration(seconds: float | None) -> str:
    """Render a duration in a compact human-readable form."""
    if seconds is None:
        return "-"
    rounded = max(0, int(round(float(seconds))))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds_part:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds_part:02d}s"
    return f"{seconds_part:d}s"


def render_progress_bar(
    completed: int | float,
    total: int | float,
    *,
    width: int = 24,
) -> str:
    """Return an ASCII-safe progress bar with a deterministic bounded fill level."""
    if width < 4:
        raise ValueError("width must be at least four.")
    if total <= 0:
        return "[" + "-" * width + "]"
    fraction = min(1.0, max(0.0, float(completed) / float(total)))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_scalar(value: Any, *, max_length: int = 28) -> str:
    """Render one parameter value compactly without losing simple numeric precision."""
    if isinstance(value, float):
        rendered = f"{value:.6g}"
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ", ".join(_format_scalar(item, max_length=12) for item in value) + "]"
    elif isinstance(value, Mapping):
        rendered = "{" + ", ".join(
            f"{key}={_format_scalar(item, max_length=12)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ) + "}"
    else:
        rendered = str(value)
    return rendered if len(rendered) <= max_length else rendered[: max_length - 1] + "…"


def format_parameter_summary(
    parameters: Mapping[str, Any] | None,
    *,
    max_items: int = 6,
    max_value_length: int = 28,
) -> str:
    """Render a stable compact hyperparameter summary for status rows and event logs."""
    if not parameters:
        return "-"
    ordered = sorted((str(key), value) for key, value in parameters.items())
    rendered = [
        f"{key}={_format_scalar(value, max_length=max_value_length)}"
        for key, value in ordered[:max_items]
    ]
    hidden = len(ordered) - len(rendered)
    if hidden > 0:
        rendered.append(f"+{hidden} more")
    return ", ".join(rendered)


def _make_human_event_line(record: Mapping[str, Any]) -> str:
    """Render a concise timestamped event line from one structured event record."""
    timestamp = str(record.get("occurred_at_local") or _local_timestamp())
    event = str(record.get("event") or "event").upper().replace("_", " ")
    task_key = record.get("task_key")
    stage = record.get("stage")
    fragments = [f"[{timestamp}]", event]
    if task_key:
        fragments.append(str(task_key))
    if stage:
        fragments.append(f"stage={stage}")
    message = record.get("message")
    if message:
        fragments.append(str(message))
    details = record.get("details")
    if isinstance(details, Mapping):
        if "parameters" in details and isinstance(details["parameters"], Mapping):
            fragments.append(
                "parameters="
                + format_parameter_summary(details["parameters"], max_items=5)
            )
        display_names = {
            "current_trial_number": "optuna_trial_id",
            "target_completed_trials": "valid_target",
        }
        for key in (
            "task_position",
            "task_total",
            "completed_trials",
            "target_completed_trials",
            "current_trial_number",
            "partial_mean_average_precision",
            "last_trial_average_precision",
            "best_stage_a_average_precision",
            "confirmation_position",
            "confirmation_total",
            "elapsed_seconds",
        ):
            if key in details and details[key] is not None:
                value = details[key]
                if key == "elapsed_seconds":
                    value = format_duration(float(value))
                elif "precision" in key:
                    value = f"{float(value):.4f}"
                fragments.append(f"{display_names.get(key, key)}={value}")
    return " | ".join(fragments)


_INVOCATION_EVENT_STATES = {
    "run_started": "running",
    "run_resumed": "running",
    "run_paused": "paused",
    "run_failed": "failed",
    "run_completed": "completed",
}


def _update_latest_invocation_status(
    run_directory: Path,
    record: Mapping[str, Any],
) -> None:
    """Persist latest-invocation metadata independently of the append-only event tail.

    The dashboard needs a stable invocation start time and worker capacity even after a
    long run has written more events than the small event-log tail retained for display.
    Only run-level lifecycle events update this artifact. The coordinator remains its
    sole writer because ``RunEventLogger`` is coordinator-owned.
    """
    event_name = str(record.get("event") or "")
    state = _INVOCATION_EVENT_STATES.get(event_name)
    if state is None:
        return

    path = invocation_status_path(run_directory)
    existing = _read_json(path) or {}
    details = record.get("details")
    details_map = details if isinstance(details, Mapping) else {}
    occurred_at = str(record.get("occurred_at_utc") or _utc_now())
    worker_capacity = details_map.get("worker_capacity")

    if event_name in {"run_started", "run_resumed"}:
        payload: dict[str, Any] = {
            "schema_version": "final_comparison_latest_invocation_v1",
            "run_id": str(details_map.get("run_id") or run_directory.name),
            "invocation_state": state,
            "invocation_started_at": occurred_at,
            "invocation_finished_at": None,
            "worker_capacity": worker_capacity,
            "last_event": event_name,
            "last_event_at": occurred_at,
            "message": str(record.get("message") or ""),
        }
    else:
        payload = dict(existing)
        started_at = payload.get("invocation_started_at")
        if not started_at:
            # A terminal lifecycle record without a prior local start is unusual, but this
            # fallback makes the sidecar self-consistent rather than leaving elapsed time
            # undefined after an abrupt monitoring restart.
            started_at = occurred_at
        payload.update(
            {
                "schema_version": "final_comparison_latest_invocation_v1",
                "run_id": str(payload.get("run_id") or run_directory.name),
                "invocation_state": state,
                "invocation_started_at": str(started_at),
                "invocation_finished_at": occurred_at,
                "worker_capacity": (
                    worker_capacity
                    if worker_capacity is not None
                    else payload.get("worker_capacity")
                ),
                "last_event": event_name,
                "last_event_at": occurred_at,
                "message": str(record.get("message") or ""),
            }
        )
    _atomic_write_json(path, payload)


class RunEventLogger:
    """Coordinator-owned append-only log writer for one experiment run.

    The coordinator is the only writer to the combined log. Worker processes instead
    write task-local logs, which the coordinator can tail and forward into this combined
    chronological audit trail.
    """

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory)

    def emit(
        self,
        event: str,
        *,
        message: str,
        task: Any | None = None,
        details: Mapping[str, Any] | None = None,
        source: str = "coordinator",
    ) -> dict[str, Any]:
        """Append one structured and human-readable event and return its record."""
        record: dict[str, Any] = {
            "schema_version": "final_comparison_event_v1",
            "event": str(event),
            "source": str(source),
            "occurred_at_utc": _utc_now(),
            "occurred_at_local": _local_timestamp(),
            "message": str(message),
            "details": dict(details or {}),
        }
        if task is not None:
            record.update(
                {
                    "task_key": str(task.task_key),
                    "candidate_id": str(task.candidate_id),
                    "outer_repeat_index": int(task.repeat_index),
                    "outer_fold_index": int(task.fold_index),
                }
            )
        try:
            _update_latest_invocation_status(self.run_directory, record)
        except OSError:
            # Monitoring sidecars must not compromise model execution if a transient local
            # filesystem problem prevents a best-effort status refresh.
            pass
        _append_jsonl(coordinator_event_jsonl_path(self.run_directory), record)
        _append_text_line(coordinator_log_path(self.run_directory), _make_human_event_line(record))
        return record


class TaskProgressReporter:
    """Worker-owned atomic progress and task-event reporter.

    A reporter refreshes its progress sidecar periodically while a long model fit is
    active, which makes liveness observable without relying on a stage transition. It
    records meaningful transitions in task-local JSONL and text logs, but it deliberately
    does not write an event for each heartbeat refresh.
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
        self._details: dict[str, Any] = {}
        self._started_at = _utc_now()
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def path(self) -> Path:
        """Return this worker's atomic progress-sidecar path."""
        return progress_path(self.run_directory, self.task_key)

    def _progress_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": "final_comparison_task_progress_v2",
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
                "details": dict(self._details),
            }

    def _write_progress(self) -> None:
        _atomic_write_json(self.path, self._progress_payload())

    def _heartbeat_loop(self) -> None:
        """Refresh only liveness telemetry until this task closes."""
        while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
            try:
                self._write_progress()
            except Exception:
                # Monitoring must never terminate a model fitting operation because of a
                # transient filesystem issue. A later heartbeat can recover naturally.
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

    def emit_event(
        self,
        event: str,
        *,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one meaningful task-local event without modifying central task state."""
        with self._lock:
            stage = self._stage
        record: dict[str, Any] = {
            "schema_version": "final_comparison_task_event_v1",
            "event": str(event),
            "source": "worker",
            "occurred_at_utc": _utc_now(),
            "occurred_at_local": _local_timestamp(),
            "task_key": self.task_key,
            "candidate_id": self.candidate_id,
            "outer_repeat_index": self.repeat_index,
            "outer_fold_index": self.fold_index,
            "stage": stage,
            "message": str(message),
            "details": dict(details or {}),
        }
        _append_jsonl(task_event_jsonl_path(self.run_directory, self.task_key), record)
        _append_text_line(task_log_path(self.run_directory, self.task_key), _make_human_event_line(record))
        return record

    def start(
        self,
        *,
        stage: str = "initializing",
        message: str | None = None,
        **details: Any,
    ) -> None:
        """Create the sidecar, emit a start event, and begin periodic liveness updates."""
        self.update(stage=stage, message=message, **details)
        self.emit_event(
            "task_worker_started",
            message=message or "Worker started the outer task.",
            details=details,
        )
        self._start_heartbeat()

    def update(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        event_name: str | None = None,
        **details: Any,
    ) -> None:
        """Persist a progress update and optionally append one meaningful task event."""
        with self._lock:
            if stage is not None:
                self._stage = str(stage)
            if message is not None:
                self._message = str(message)
            if details:
                self._details.update(details)
        self._write_progress()
        if event_name is not None and not is_low_level_fold_event(event_name):
            self.emit_event(
                event_name,
                message=message or str(event_name).replace("_", " "),
                details=details,
            )

    def stop_requested(self) -> bool:
        """Return whether the coordinator has persisted a clean-stop request."""
        return stop_request_path(self.run_directory).exists()

    def close(
        self,
        *,
        final_stage: str,
        message: str | None = None,
        **details: Any,
    ) -> None:
        """Persist terminal worker telemetry and stop background heartbeat refreshes."""
        self._stop_heartbeat()
        self.update(stage=final_stage, message=message, **details)
        self.emit_event(
            f"task_{final_stage}",
            message=message or f"Task reached terminal stage {final_stage}.",
            details=details,
        )


@dataclass(frozen=True)
class StudyProgress:
    """Read-only summary of one task-local Optuna study."""

    present: bool
    complete_trials: int | None
    failed_trials: int | None
    pruned_trials: int | None
    running_trials: int | None
    best_average_precision: float | None
    latest_trial_at: str | None
    stage_b_present: bool
    stage_b_completed: int | None
    stage_b_total: int | None
    error: str | None = None


@dataclass(frozen=True)
class TaskStatusSnapshot:
    """Read-only summary of one outer task and its current monitoring artifacts."""

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

    def elapsed_seconds(self, *, now: datetime) -> float | None:
        """Return observed task elapsed time without treating downtime as active work."""
        started = _parse_timestamp(self.started_at)
        if started is None:
            return None
        if self.status == "running":
            ended = now
        else:
            ended = _parse_timestamp(self.completed_at) or _parse_timestamp(self.heartbeat_at)
            if ended is None and self.progress:
                ended = _parse_timestamp(str(self.progress.get("updated_at")))
        if ended is None:
            return None
        return max(0.0, (ended - started).total_seconds())


@dataclass(frozen=True)
class RunStatusSnapshot:
    """Read-only run-level snapshot used by the terminal status command."""

    run_directory: Path
    run_id: str
    created_at: str | None
    purpose: str | None
    status_counts: Mapping[str, int]
    tasks: Sequence[TaskStatusSnapshot]
    stop_request: Mapping[str, Any] | None
    current_invocation_started_at: str | None
    current_invocation_finished_at: str | None
    current_invocation_state: str | None
    worker_capacity: int | None
    latest_event: Mapping[str, Any] | None


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON artifact defensively without changing it."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    """Open an SQLite database in operating-system-enforced read-only URI mode."""
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=3.0)


def _normalise_optuna_state(value: Any) -> str:
    """Map Optuna SQLite state representations to stable lower-case names."""
    if isinstance(value, int):
        return {
            0: "running",
            1: "complete",
            2: "pruned",
            3: "fail",
            4: "waiting",
        }.get(int(value), str(value).lower())
    rendered = str(value).strip().lower()
    if rendered.startswith("trialstate."):
        rendered = rendered.split(".", 1)[1]
    return rendered


def _read_stage_b_progress(study_path: Path) -> tuple[bool, int | None, int | None]:
    """Read durable Stage-B confirmation progress without opening a writable study."""
    path = study_path.with_suffix(".stage_b_confirmation.json")
    payload = _read_json(path)
    if payload is None:
        return False, None, None
    records = payload.get("records", [])
    completed = len(records) if isinstance(records, list) else None
    total = payload.get("confirmation_total")
    if total is None:
        total = payload.get("total_configurations")
    if total is None and completed is not None:
        total = completed
    try:
        total_value = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_value = None
    return True, completed, total_value


def _read_optuna_study_progress(
    *,
    study_path: Path | None,
    study_name: str | None,
) -> StudyProgress:
    """Inspect one task-local Optuna database through read-only SQL queries."""
    if study_path is None or study_name is None or not study_path.exists():
        return StudyProgress(False, None, None, None, None, None, None, False, None, None)

    stage_b_present, stage_b_completed, stage_b_total = _read_stage_b_progress(study_path)
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
            stage_b_completed,
            stage_b_total,
            f"{type(exc).__name__}: {exc}",
        )

    counts: Counter[str] = Counter()
    completed_values: list[float] = []
    timestamps: list[str] = []
    for state, started_at, completed_at, value, heartbeat_at in rows:
        normalized = _normalise_optuna_state(state)
        counts[normalized] += 1
        if normalized == "complete" and value is not None:
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
        stage_b_completed,
        stage_b_total,
    )


def _read_event_records(
    run_directory: Path,
    *,
    limit: int = 24,
    max_bytes: int = 65_536,
) -> list[dict[str, Any]]:
    """Read a bounded tail of coordinator JSONL without scanning the whole log.

    A live dashboard only needs recent context. Reading a fixed byte tail keeps refresh
    cost bounded even when an eventual master comparison produces many trial-level audit
    records. When the read begins mid-line, the incomplete first record is discarded.
    """
    path = coordinator_event_jsonl_path(run_directory)
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max(1, int(max_bytes)))
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []

    if start > 0:
        newline = data.find(b"\n")
        if newline < 0:
            return []
        data = data[newline + 1 :]

    records: list[dict[str, Any]] = []
    for raw_line in data.splitlines()[-max(1, int(limit)) :]:
        try:
            payload = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            records.append(dict(payload))
    return records


def collect_run_status(run_directory: Path) -> RunStatusSnapshot:
    """Collect a completely read-only snapshot of one created experiment run."""
    run_directory = Path(run_directory)
    manifest_path = run_directory / "run_manifest.json"
    registry_path = run_directory / "task_registry.sqlite"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Task registry does not exist: {registry_path}")

    manifest = _read_json(manifest_path) or {}
    protocol = manifest.get("protocol", {})
    metadata = protocol.get("metadata", {}) if isinstance(protocol, Mapping) else {}
    with _open_sqlite_read_only(registry_path) as connection:
        rows = connection.execute(
            """
            SELECT
                task_key, candidate_id, repeat_index, fold_index, status, attempts,
                started_at, heartbeat_at, completed_at, error_text, payload_json
            FROM tasks
            ORDER BY candidate_id, repeat_index, fold_index, task_key
            """
        ).fetchall()

    tasks: list[TaskStatusSnapshot] = []
    for row in rows:
        try:
            registered_payload = json.loads(row[10])
        except (TypeError, json.JSONDecodeError):
            registered_payload = {}
        task_payload = (
            registered_payload.get("payload", {})
            if isinstance(registered_payload, Mapping)
            else {}
        )
        study_value = task_payload.get("study_database_path")
        study_path = Path(str(study_value)) if study_value else None
        tasks.append(
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
                study=_read_optuna_study_progress(
                    study_path=study_path,
                    study_name=task_payload.get("study_name"),
                ),
            )
        )

    events = _read_event_records(run_directory)
    invocation = _read_json(invocation_status_path(run_directory)) or {}

    # Runs created before v3.2 have no invocation sidecar. Keep their dashboard readable
    # by using the old tail-based inference as a backward-compatible fallback.
    if not invocation:
        invocation_events = [
            event
            for event in events
            if event.get("event") in {"run_started", "run_resumed"}
        ]
        current_invocation = invocation_events[-1] if invocation_events else None
        if current_invocation:
            details = current_invocation.get("details")
            details_map = details if isinstance(details, Mapping) else {}
            invocation = {
                "invocation_started_at": current_invocation.get("occurred_at_utc"),
                "invocation_finished_at": None,
                "invocation_state": "running",
                "worker_capacity": details_map.get("worker_capacity"),
            }

    raw_capacity = invocation.get("worker_capacity")
    try:
        capacity = int(raw_capacity) if raw_capacity is not None else None
    except (TypeError, ValueError):
        capacity = None

    counts = Counter(task.status for task in tasks)
    return RunStatusSnapshot(
        run_directory=run_directory,
        run_id=str(manifest.get("run_id", run_directory.name)),
        created_at=manifest.get("created_at"),
        purpose=metadata.get("purpose") if isinstance(metadata, Mapping) else None,
        status_counts=dict(sorted(counts.items())),
        tasks=tuple(tasks),
        stop_request=read_graceful_stop_request(run_directory),
        current_invocation_started_at=(
            str(invocation.get("invocation_started_at"))
            if invocation.get("invocation_started_at")
            else None
        ),
        current_invocation_finished_at=(
            str(invocation.get("invocation_finished_at"))
            if invocation.get("invocation_finished_at")
            else None
        ),
        current_invocation_state=(
            str(invocation.get("invocation_state"))
            if invocation.get("invocation_state")
            else None
        ),
        worker_capacity=capacity,
        latest_event=events[-1] if events else None,
    )


def _task_progress_details(task: TaskStatusSnapshot) -> Mapping[str, Any]:
    """Return the progress-detail mapping or an empty mapping."""
    if not task.progress:
        return {}
    details = task.progress.get("details", {})
    return details if isinstance(details, Mapping) else {}


def _task_stage(task: TaskStatusSnapshot) -> str:
    """Return the most informative current stage available for a task."""
    if task.progress and task.progress.get("stage"):
        return str(task.progress["stage"])
    if task.study.stage_b_present:
        return "stage_b"
    if task.study.present:
        return "stage_a"
    return "pending" if task.status == "pending" else task.status


def _task_heartbeat_seconds(task: TaskStatusSnapshot, now: datetime) -> float | None:
    """Return age of the freshest worker or coordinator liveness timestamp."""
    candidates: list[datetime] = []
    if task.progress:
        parsed = _parse_timestamp(str(task.progress.get("updated_at")))
        if parsed is not None:
            candidates.append(parsed)
    parsed_registry = _parse_timestamp(task.heartbeat_at)
    if parsed_registry is not None:
        candidates.append(parsed_registry)
    if not candidates:
        return None
    freshest = max(candidates)
    return max(0.0, (now - freshest).total_seconds())


def _task_metric_summary(task: TaskStatusSnapshot) -> tuple[str, str, str]:
    """Return distinct current-partial, last-completed, and best Stage-A AP labels."""
    details = _task_progress_details(task)
    partial = details.get("partial_mean_average_precision")
    last = details.get("last_trial_average_precision")
    if last is None:
        last = details.get("last_completed_trial_average_precision")
    best = details.get("best_stage_a_average_precision")
    if best is None:
        best = task.study.best_average_precision
    partial_text = "-" if partial is None else f"{float(partial):.4f}"
    last_text = "-" if last is None else f"{float(last):.4f}"
    best_text = "-" if best is None else f"{float(best):.4f}"
    return partial_text, last_text, best_text


def _task_progress_label(task: TaskStatusSnapshot) -> str:
    """Return a readable stage-specific trial or confirmation-progress label."""
    stage = _task_stage(task)
    details = _task_progress_details(task)
    if stage == "stage_a":
        completed = details.get("completed_trials", task.study.complete_trials or 0)
        target = details.get("target_completed_trials")
        if target is None:
            target = "?"
        current_trial = details.get("current_trial_number")
        fold_index = details.get("inner_fold_index")
        fold_total = details.get("inner_fold_total")
        suffix = ""
        if current_trial is not None:
            suffix = f" | Optuna ID {current_trial}"
            if fold_index is not None and fold_total is not None:
                suffix += f", fold {fold_index}/{fold_total}"
        try:
            bar = render_progress_bar(int(completed), int(target), width=14)
        except (TypeError, ValueError):
            bar = "[" + "-" * 14 + "]"
        return f"Stage A {bar} valid {completed}/{target}{suffix}"
    if stage == "stage_b":
        completed = details.get("confirmation_completed", task.study.stage_b_completed or 0)
        total = details.get("confirmation_total", task.study.stage_b_total or "?")
        position = details.get("confirmation_position")
        fold_index = details.get("inner_fold_index")
        fold_total = details.get("inner_fold_total")
        suffix = ""
        if position is not None:
            suffix = f" | config {position}/{total}"
            if fold_index is not None and fold_total is not None:
                suffix += f", fold {fold_index}/{fold_total}"
        try:
            bar = render_progress_bar(int(completed), int(total), width=14)
        except (TypeError, ValueError):
            bar = "[" + "-" * 14 + "]"
        return f"Stage B {bar} {completed}/{total}{suffix}"
    return stage.replace("_", " ").title()


def _format_event_compact(event: Mapping[str, Any] | None) -> str:
    """Render the latest combined coordinator event for dashboard context."""
    if not event:
        return "-"
    timestamp = str(event.get("occurred_at_local") or event.get("occurred_at_utc") or "-")
    message = str(event.get("message") or event.get("event") or "-")
    task_key = event.get("task_key")
    prefix = f"{task_key}: " if task_key else ""
    return f"[{timestamp}] {prefix}{message}"


def _estimate_remaining_seconds(
    snapshot: RunStatusSnapshot,
    *,
    now: datetime,
) -> float | None:
    """Estimate remaining wall time only when every unfinished family has observations.

    Candidate procedures can have radically different per-task runtimes. A global average
    would be misleading while CatBoost or MLP work remains after cheap linear tasks. This
    conservative estimate appears only after at least one completed outer task exists for
    every candidate family that still has unfinished work.
    """
    by_candidate: dict[str, list[float]] = {}
    unfinished: list[TaskStatusSnapshot] = []
    for task in snapshot.tasks:
        if task.status == "completed":
            elapsed = task.elapsed_seconds(now=now)
            if elapsed is not None and elapsed > 0:
                by_candidate.setdefault(task.candidate_id, []).append(elapsed)
        else:
            unfinished.append(task)
    if not unfinished:
        return 0.0
    candidate_medians: dict[str, float] = {
        candidate: float(statistics.median(values))
        for candidate, values in by_candidate.items()
        if values
    }
    remaining_candidates = {task.candidate_id for task in unfinished}
    if not remaining_candidates.issubset(candidate_medians):
        return None
    serial_work = sum(candidate_medians[task.candidate_id] for task in unfinished)
    capacity = snapshot.worker_capacity or max(1, snapshot.status_counts.get("running", 0))
    return serial_work / max(1, capacity)


def _truncate(value: str, width: int) -> str:
    """Truncate a display value without splitting the terminal table."""
    if len(value) <= width:
        return value
    return value[: max(1, width - 1)] + "…"


def render_run_status(
    snapshot: RunStatusSnapshot,
    *,
    include_completed: bool = False,
    failures_only: bool = False,
    details: bool = False,
) -> str:
    """Render a compact plain-text dashboard from persisted read-only artifacts."""
    now = datetime.now(UTC)
    total = len(snapshot.tasks)
    completed = snapshot.status_counts.get("completed", 0)
    running = snapshot.status_counts.get("running", 0)
    pending = snapshot.status_counts.get("pending", 0)
    interrupted = snapshot.status_counts.get("interrupted", 0)
    failed = snapshot.status_counts.get("failed", 0)
    capacity = snapshot.worker_capacity
    percent = 100.0 * completed / total if total else 0.0
    remaining_eta = _estimate_remaining_seconds(snapshot, now=now)
    invocation_start = _parse_timestamp(snapshot.current_invocation_started_at)
    invocation_finished = _parse_timestamp(snapshot.current_invocation_finished_at)
    invocation_end = invocation_finished or now
    invocation_elapsed = (
        format_duration((invocation_end - invocation_start).total_seconds())
        if invocation_start is not None
        else "-"
    )

    lines = [
        "FINAL COMPARISON MONITOR",
        f"Run: {snapshot.run_id}",
        f"Purpose: {snapshot.purpose or '-'}",
        (
            f"Run created: {snapshot.created_at or '-'} | "
            f"Invocation: {snapshot.current_invocation_state or '-'} | "
            f"started: {snapshot.current_invocation_started_at or '-'} | "
            f"elapsed: {invocation_elapsed}"
        ),
        (
            f"Outer tasks: {render_progress_bar(completed, total, width=30)} "
            f"{completed}/{total} ({percent:.1f}%)"
        ),
        (
            f"States: completed={completed} | active={running}"
            + (f"/{capacity}" if capacity else "")
            + f" | pending={pending} | interrupted={interrupted} | failed={failed}"
        ),
        (
            "ETA: "
            + (
                f"observed candidate-median estimate {format_duration(remaining_eta)}"
                if remaining_eta is not None
                else "not shown until every unfinished candidate family has a completed outer task"
            )
        ),
        "Live AP values are operational diagnostics only and are not model-selection evidence.",
        f"Latest event: {_format_event_compact(snapshot.latest_event)}",
    ]
    if snapshot.stop_request is not None:
        request_time = snapshot.stop_request.get("requested_at")
        reason = snapshot.stop_request.get("reason")
        lines.append(
            "Control: clean pause requested"
            + (f" at {request_time}" if request_time else "")
            + (f" | {reason}" if reason else "")
        )

    selected: list[TaskStatusSnapshot] = []
    for task in snapshot.tasks:
        if failures_only and task.status not in {"failed", "interrupted"}:
            continue
        if not include_completed and task.status == "completed":
            continue
        selected.append(task)

    active_tasks = [task for task in selected if task.status == "running"]
    other_tasks = [task for task in selected if task.status != "running"]

    if active_tasks:
        lines.extend(["", "ACTIVE TASKS"])
        for task in active_tasks:
            details_map = _task_progress_details(task)
            stage = _task_stage(task)
            partial_ap, last_ap, best_ap = _task_metric_summary(task)
            heartbeat = _task_heartbeat_seconds(task, now)
            label = f"{task.candidate_id} r{task.repeat_index:02d}f{task.fold_index:02d}"
            lines.append(
                f"{label} | {_task_progress_label(task)} | elapsed="
                f"{format_duration(task.elapsed_seconds(now=now))} | heartbeat="
                f"{format_duration(heartbeat)} ago"
            )
            lines.append(
                "  stage="
                f"{stage} | current partial AP={partial_ap} | "
                f"last completed AP={last_ap} | best Stage-A AP={best_ap}"
            )
            if stage in {"outer_fit", "outer_prediction"}:
                parameters = details_map.get("selected_parameters")
            else:
                parameters = details_map.get("current_trial_parameters")
                if not isinstance(parameters, Mapping):
                    parameters = details_map.get("selected_parameters")
            if isinstance(parameters, Mapping):
                lines.append(
                    "  parameters: "
                    + format_parameter_summary(parameters, max_items=7, max_value_length=36)
                )
            if details:
                message = task.progress.get("message") if task.progress else None
                if message:
                    lines.append(f"  message: {message}")
                if isinstance(parameters, Mapping):
                    for key, value in sorted(parameters.items(), key=lambda pair: str(pair[0])):
                        lines.append(f"    {key}: {_format_scalar(value, max_length=120)}")

    if other_tasks:
        lines.extend(["", "OTHER OUTSTANDING TASKS"])
        header = "task                              state        stage              progress"
        lines.extend([header, "-" * len(header)])
        for task in other_tasks:
            label = f"{task.candidate_id} r{task.repeat_index:02d}f{task.fold_index:02d}"
            lines.append(
                f"{_truncate(label, 33):33} "
                f"{_truncate(task.status, 12):12} "
                f"{_truncate(_task_stage(task), 18):18} "
                f"{_truncate(_task_progress_label(task), 52):52}"
            )
            if task.status in {"failed", "interrupted"} and task.error_text:
                lines.append(f"  reason: {task.error_text.strip().splitlines()[0]}")
            if task.study.error:
                lines.append(f"  study: {task.study.error}")

    if not selected:
        lines.extend(["", "No matching task rows."])

    return "\n".join(lines)


def render_task_details(snapshot: RunStatusSnapshot, task_key: str) -> str:
    """Render full current telemetry for one task without altering the experiment."""
    matching = [task for task in snapshot.tasks if task.task_key == task_key]
    if not matching:
        available = ", ".join(task.task_key for task in snapshot.tasks)
        raise KeyError(f"Unknown task key {task_key!r}. Available task keys: {available}")
    task = matching[0]
    now = datetime.now(UTC)
    lines = [
        f"TASK DETAIL: {task.task_key}",
        f"Candidate: {task.candidate_id}",
        f"Outer split: repeat={task.repeat_index}, fold={task.fold_index}",
        f"State: {task.status} | attempts={task.attempts}",
        f"Stage: {_task_stage(task)}",
        f"Elapsed: {format_duration(task.elapsed_seconds(now=now))}",
        f"Progress: {_task_progress_label(task)}",
    ]
    partial_ap, last_ap, best_ap = _task_metric_summary(task)
    lines.append(
        "Inner AP: "
        f"current partial={partial_ap}, last completed={last_ap}, best Stage-A={best_ap}"
    )
    if task.progress:
        lines.append(f"Message: {task.progress.get('message') or '-'}")
        details = _task_progress_details(task)
        if details:
            lines.append("Current telemetry:")
            for key, value in sorted(details.items(), key=lambda pair: str(pair[0])):
                if key in {"current_trial_parameters", "selected_parameters"} and isinstance(value, Mapping):
                    lines.append(f"  {key}:")
                    for parameter, parameter_value in sorted(value.items(), key=lambda pair: str(pair[0])):
                        lines.append(f"    {parameter}: {_format_scalar(parameter_value, max_length=160)}")
                else:
                    lines.append(f"  {key}: {_format_scalar(value, max_length=160)}")
    if task.error_text:
        lines.append("Task error or interruption reason:")
        lines.extend(f"  {line}" for line in task.error_text.rstrip().splitlines())
    if task.study.error:
        lines.append(f"Study inspection error: {task.study.error}")
    return "\n".join(lines)
