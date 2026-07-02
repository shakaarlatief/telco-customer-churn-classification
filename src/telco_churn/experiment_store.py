"""Durable artifact storage and resume safety for final comparison experiments.

A long repeated-nested-CV experiment must survive ordinary failures: a deliberate
interrupt, laptop sleep, a worker crash, an exhausted package-level trial, or an
operating-system restart. This module provides a small transactional store with the
following contract:

1. A run directory owns an immutable protocol and development-data fingerprint.
2. Each atomic task has a unique key and a state in SQLite.
3. A task becomes complete only after its result payload has been written atomically.
4. A resumed run refuses to mix a different protocol or dataset into the same
   result directory.
5. Results are stored per task, so a completed outer-fold task is never recomputed
   merely because another task was interrupted.

The store is intentionally generic. It does not know what a candidate model is or how
a metric is calculated. That separation makes the resumability layer reusable for
classification, regression, calibration, stacking, or other computational workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping
from uuid import uuid4


class UnsafeResumeError(RuntimeError):
    """Raised when an existing run is incompatible with a requested resume."""


class TaskStateError(RuntimeError):
    """Raised when an invalid task-state transition is attempted."""


TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_INTERRUPTED = "interrupted"

_VALID_TASK_STATUSES = {
    TASK_PENDING,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INTERRUPTED,
}


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON used for task-result persistence."""
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _sha256_file(path: Path) -> str:
    """Return SHA-256 for an already-written artifact."""
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON file in the destination directory.

    The temporary file is created in the same directory as the destination so
    ``os.replace`` is an atomic rename on a single filesystem. Readers therefore
    observe either the prior complete file or the new complete file, never a
    partially written JSON document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file_handle:
            file_handle.write(_canonical_json(payload))
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ExperimentTask:
    """Serializable description of one atomic comparison task."""

    task_key: str
    candidate_id: str
    repeat_index: int
    fold_index: int
    split_hash: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible task description."""
        return {
            "task_key": self.task_key,
            "candidate_id": self.candidate_id,
            "repeat_index": int(self.repeat_index),
            "fold_index": int(self.fold_index),
            "split_hash": self.split_hash,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class TaskRecord:
    """Snapshot of one durable task row."""

    task_key: str
    candidate_id: str
    repeat_index: int
    fold_index: int
    split_hash: str
    status: str
    attempts: int
    started_at: str | None
    heartbeat_at: str | None
    completed_at: str | None
    error_text: str | None
    result_path: str | None
    result_sha256: str | None


class ExperimentStore:
    """Run-scoped task store backed by SQLite and atomic JSON artifacts."""

    MANIFEST_FILENAME = "run_manifest.json"
    DATABASE_FILENAME = "task_registry.sqlite"

    def __init__(self, run_directory: Path):
        self.run_directory = Path(run_directory)
        self.results_directory = self.run_directory / "results"
        self.logs_directory = self.run_directory / "logs"
        self.database_path = self.run_directory / self.DATABASE_FILENAME
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    @classmethod
    def create(
        cls,
        *,
        artifacts_root: Path,
        run_id: str,
        protocol_payload: Mapping[str, Any],
        protocol_fingerprint: str,
        data_fingerprint: Mapping[str, Any],
        environment_fingerprint: Mapping[str, Any],
    ) -> "ExperimentStore":
        """Create a new immutable run directory and initialize its manifest."""
        if not run_id.strip():
            raise ValueError("run_id must not be empty.")

        run_directory = Path(artifacts_root) / run_id
        if run_directory.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_directory}. "
                "Use open_for_resume for an existing run."
            )

        run_directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id,
            "created_at": _utc_now(),
            "protocol": dict(protocol_payload),
            "protocol_fingerprint": protocol_fingerprint,
            "data_fingerprint": dict(data_fingerprint),
            "environment_fingerprint": dict(environment_fingerprint),
        }
        atomic_write_json(run_directory / cls.MANIFEST_FILENAME, manifest)
        return cls(run_directory)

    @classmethod
    def open_for_resume(
        cls,
        *,
        artifacts_root: Path,
        run_id: str,
        protocol_fingerprint: str,
        data_fingerprint_sha256: str,
    ) -> "ExperimentStore":
        """Open an existing run only after strict protocol and data checks."""
        run_directory = Path(artifacts_root) / run_id
        manifest_path = run_directory / cls.MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_protocol = manifest.get("protocol_fingerprint")
        stored_data = manifest.get("data_fingerprint", {}).get("sha256")

        if stored_protocol != protocol_fingerprint:
            raise UnsafeResumeError(
                "Protocol fingerprint mismatch. Create a new run instead of mixing "
                "results from two different experimental protocols."
            )
        if stored_data != data_fingerprint_sha256:
            raise UnsafeResumeError(
                "Development-data fingerprint mismatch. Create a new run instead of "
                "mixing results from two different development datasets."
            )

        store = cls(run_directory)
        store.recover_interrupted_tasks()
        return store

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        """Create durable task tables when they do not already exist."""
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_key TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                repeat_index INTEGER NOT NULL,
                fold_index INTEGER NOT NULL,
                split_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                heartbeat_at TEXT,
                completed_at TEXT,
                error_text TEXT,
                result_path TEXT,
                result_sha256 TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON tasks(status)
            """
        )

    def register_tasks(self, tasks: Iterable[ExperimentTask]) -> None:
        """Insert tasks that do not already exist.

        A pre-existing task must match the same immutable candidate, split, and
        payload identity. This prevents accidental reuse of a task key for a
        different computation.
        """
        for task in tasks:
            payload_json = _canonical_json(task.to_dict())
            existing = self._connection.execute(
                """
                SELECT candidate_id, repeat_index, fold_index, split_hash, payload_json
                FROM tasks
                WHERE task_key = ?
                """,
                (task.task_key,),
            ).fetchone()

            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO tasks (
                        task_key,
                        candidate_id,
                        repeat_index,
                        fold_index,
                        split_hash,
                        payload_json,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.task_key,
                        task.candidate_id,
                        int(task.repeat_index),
                        int(task.fold_index),
                        task.split_hash,
                        payload_json,
                        TASK_PENDING,
                    ),
                )
                continue

            expected = (
                task.candidate_id,
                int(task.repeat_index),
                int(task.fold_index),
                task.split_hash,
                payload_json,
            )
            observed = (
                existing["candidate_id"],
                existing["repeat_index"],
                existing["fold_index"],
                existing["split_hash"],
                existing["payload_json"],
            )
            if expected != observed:
                raise TaskStateError(
                    f"Task key {task.task_key!r} already exists with different metadata."
                )

    def claim_task(self, task_key: str, *, retry_failed: bool = False) -> bool:
        """Atomically claim a pending or interrupted task for execution.

        The method returns ``False`` when the task is already complete, currently
        running, or failed without an explicit retry policy.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT status FROM tasks WHERE task_key = ?",
                (task_key,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task key: {task_key}")

            eligible_statuses = {TASK_PENDING, TASK_INTERRUPTED}
            if retry_failed:
                eligible_statuses.add(TASK_FAILED)

            if row["status"] not in eligible_statuses:
                self._connection.execute("COMMIT")
                return False

            now = _utc_now()
            self._connection.execute(
                """
                UPDATE tasks
                SET status = ?,
                    attempts = attempts + 1,
                    started_at = ?,
                    heartbeat_at = ?,
                    completed_at = NULL,
                    error_text = NULL
                WHERE task_key = ?
                """,
                (TASK_RUNNING, now, now, task_key),
            )
            self._connection.execute("COMMIT")
            return True
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    def heartbeat(self, task_key: str) -> None:
        """Record a liveness timestamp for a currently running task."""
        result = self._connection.execute(
            """
            UPDATE tasks
            SET heartbeat_at = ?
            WHERE task_key = ? AND status = ?
            """,
            (_utc_now(), task_key, TASK_RUNNING),
        )
        if result.rowcount != 1:
            raise TaskStateError(
                f"Cannot heartbeat task {task_key!r}; it is not in running state."
            )

    def complete_task(self, task_key: str, result: Mapping[str, Any]) -> Path:
        """Persist a task result atomically and mark the task complete.

        Result JSON is written before the database status transition. If a process
        crashes between the file write and the transition, the task remains
        resumable. The later integrity check may find an orphan result artifact,
        but it will never mistake a partially written file for a completed task.
        """
        result_payload = {
            "task_key": task_key,
            "completed_at": _utc_now(),
            "result": dict(result),
        }
        result_path = self.results_directory / f"{task_key}.json"
        atomic_write_json(result_path, result_payload)
        result_sha256 = _sha256_file(result_path)

        result_update = self._connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                completed_at = ?,
                heartbeat_at = ?,
                result_path = ?,
                result_sha256 = ?
            WHERE task_key = ? AND status = ?
            """,
            (
                TASK_COMPLETED,
                _utc_now(),
                _utc_now(),
                str(result_path.relative_to(self.run_directory)),
                result_sha256,
                task_key,
                TASK_RUNNING,
            ),
        )
        if result_update.rowcount != 1:
            raise TaskStateError(
                f"Cannot complete task {task_key!r}; it is not in running state."
            )
        return result_path

    def fail_task(self, task_key: str, error_text: str) -> None:
        """Persist an execution failure without discarding the task history."""
        result = self._connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                error_text = ?,
                heartbeat_at = ?
            WHERE task_key = ? AND status = ?
            """,
            (TASK_FAILED, error_text, _utc_now(), task_key, TASK_RUNNING),
        )
        if result.rowcount != 1:
            raise TaskStateError(
                f"Cannot fail task {task_key!r}; it is not in running state."
            )

    def interrupt_task(self, task_key: str, error_text: str) -> None:
        """Mark an active task interrupted without discarding durable study progress.

        An interrupted state differs from ``failed``: it indicates a controlled stop or
        coordinator recovery path. A compatible future resume may claim it without the
        explicit ``retry_failed`` switch, whereas an actual failed task remains visible
        until a user deliberately requests a retry.
        """
        result = self._connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                error_text = ?,
                heartbeat_at = ?
            WHERE task_key = ? AND status = ?
            """,
            (TASK_INTERRUPTED, error_text, _utc_now(), task_key, TASK_RUNNING),
        )
        if result.rowcount != 1:
            raise TaskStateError(
                f"Cannot interrupt task {task_key!r}; it is not in running state."
            )

    def recover_interrupted_tasks(self) -> int:
        """Mark tasks left running by an earlier coordinator as resumable.

        The store assumes that this method is called only after the earlier
        coordinator has stopped. The full runner will later add a coordinator lock
        and heartbeat-aware stale-task policy. For the initial local project
        workflow, an existing run is resumed by one user-controlled coordinator.
        """
        result = self._connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                error_text = COALESCE(
                    error_text,
                    'Task marked interrupted during resume recovery.'
                )
            WHERE status = ?
            """,
            (TASK_INTERRUPTED, TASK_RUNNING),
        )
        return int(result.rowcount)

    def get_task(self, task_key: str) -> TaskRecord:
        """Return one task row as a typed immutable record."""
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_key = ?",
            (task_key,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown task key: {task_key}")
        return self._record_from_row(row)

    def list_tasks(self) -> list[TaskRecord]:
        """Return all tasks in deterministic registration order."""
        rows = self._connection.execute(
            """
            SELECT *
            FROM tasks
            ORDER BY candidate_id, repeat_index, fold_index, task_key
            """
        ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def task_summary(self) -> dict[str, int]:
        """Return task counts grouped by durable state."""
        summary = {status: 0 for status in sorted(_VALID_TASK_STATUSES)}
        rows = self._connection.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        ).fetchall()
        for row in rows:
            summary[row["status"]] = int(row["count"])
        return summary

    def validate_completed_artifacts(self) -> None:
        """Verify every completed task has an intact result artifact."""
        for record in self.list_tasks():
            if record.status != TASK_COMPLETED:
                continue
            if record.result_path is None or record.result_sha256 is None:
                raise TaskStateError(
                    f"Completed task {record.task_key!r} is missing result metadata."
                )
            result_path = self.run_directory / record.result_path
            if not result_path.exists():
                raise TaskStateError(
                    f"Completed task {record.task_key!r} is missing {result_path}."
                )
            if _sha256_file(result_path) != record.result_sha256:
                raise TaskStateError(
                    f"Completed task {record.task_key!r} has a corrupted result artifact."
                )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> TaskRecord:
        """Convert one SQLite row into a public immutable task snapshot."""
        return TaskRecord(
            task_key=str(row["task_key"]),
            candidate_id=str(row["candidate_id"]),
            repeat_index=int(row["repeat_index"]),
            fold_index=int(row["fold_index"]),
            split_hash=str(row["split_hash"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
            completed_at=row["completed_at"],
            error_text=row["error_text"],
            result_path=row["result_path"],
            result_sha256=row["result_sha256"],
        )
