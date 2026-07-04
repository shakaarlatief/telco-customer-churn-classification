"""Focused smoke test for resilient final-comparison artifact writes.

This test does not load modelling data and does not fit estimators. It deterministically
simulates temporary and persistent ``PermissionError`` failures around atomic replacement
so Windows file-lock behavior can be validated without depending on an antivirus scanner,
editor, or filesystem race to reproduce naturally.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import json
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Callable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import telco_churn.atomic_io as atomic_io  # noqa: E402
from telco_churn.experiment_progress import (  # noqa: E402
    TaskProgressReporter,
    summarize_task_failure_reason,
)
from telco_churn.experiment_store import atomic_write_json  # noqa: E402
from telco_churn.hpo import _atomic_json as hpo_atomic_json  # noqa: E402
from telco_churn.hpo import _atomic_pickle as hpo_atomic_pickle  # noqa: E402


def _simulated_access_denied() -> PermissionError:
    """Return one portable stand-in for a temporary Windows sharing/access failure."""
    return PermissionError(errno.EACCES, "Simulated temporary file lock.")


@contextmanager
def _patched_replace(
    replacement: Callable[[Path, Path], None],
    *,
    retry_count: int = 2,
) -> Iterator[None]:
    """Patch the shared replace primitive with zero-delay retries for a deterministic test."""
    original_replace = atomic_io.os.replace
    original_delays = atomic_io.DEFAULT_REPLACE_RETRY_DELAYS_SECONDS
    atomic_io.os.replace = replacement  # type: ignore[assignment]
    atomic_io.DEFAULT_REPLACE_RETRY_DELAYS_SECONDS = (0.0,) * int(retry_count)
    try:
        yield
    finally:
        atomic_io.os.replace = original_replace  # type: ignore[assignment]
        atomic_io.DEFAULT_REPLACE_RETRY_DELAYS_SECONDS = original_delays


def _fail_then_delegate(failures: int):
    """Return a replacement callable that raises lock errors before delegating normally."""
    original_replace = atomic_io.os.replace
    state = {"calls": 0}

    def replacement(source: Path, destination: Path) -> None:
        state["calls"] += 1
        if state["calls"] <= failures:
            raise _simulated_access_denied()
        original_replace(source, destination)

    return replacement, state


def _always_fail(state: dict[str, int]):
    """Return a replacement callable that consistently simulates a locked destination."""
    def replacement(source: Path, destination: Path) -> None:
        state["calls"] += 1
        raise _simulated_access_denied()

    return replacement


def _assert_transient_retries(root: Path) -> None:
    """Verify all essential JSON/pickle writers survive temporary replacement locks."""
    writers = (
        ("hpo JSON", lambda path: hpo_atomic_json(path, {"writer": "hpo"})),
        ("store JSON", lambda path: atomic_write_json(path, {"writer": "store"})),
    )
    for label, writer in writers:
        destination = root / f"{label.replace(' ', '_')}.json"
        replacement, state = _fail_then_delegate(2)
        with _patched_replace(replacement, retry_count=2):
            writer(destination)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload["writer"] not in {"hpo", "store"}:
            raise AssertionError(f"{label} did not persist its expected JSON payload.")
        if state["calls"] != 3:
            raise AssertionError(f"{label} did not perform the expected retry sequence.")

    destination = root / "hpo_pickle.pkl"
    replacement, state = _fail_then_delegate(2)
    with _patched_replace(replacement, retry_count=2):
        hpo_atomic_pickle(destination, {"writer": "pickle"})
    with destination.open("rb") as handle:
        payload = pickle.load(handle)
    if payload != {"writer": "pickle"}:
        raise AssertionError("HPO pickle checkpoint did not survive temporary replacement locks.")
    if state["calls"] != 3:
        raise AssertionError("HPO pickle checkpoint did not perform the expected retries.")


def _assert_persistent_failure_preserves_prior_artifact(root: Path) -> None:
    """Verify an essential durable checkpoint fails clearly without corrupting prior JSON."""
    destination = root / "persistent_failure.json"
    destination.write_text('{"prior": true}\n', encoding="utf-8")
    state = {"calls": 0}
    with _patched_replace(_always_fail(state), retry_count=2):
        try:
            hpo_atomic_json(destination, {"replacement": "must not become visible"})
        except PermissionError:
            pass
        else:
            raise AssertionError("Persistent replacement failure unexpectedly succeeded.")

    if json.loads(destination.read_text(encoding="utf-8")) != {"prior": True}:
        raise AssertionError("Persistent failure corrupted the pre-existing checkpoint.")
    if state["calls"] != 3:
        raise AssertionError("Persistent failure did not exhaust the bounded retry sequence.")
    if list(root.glob(".persistent_failure.json.*.tmp")):
        raise AssertionError("Persistent replacement failure left a temporary JSON artifact behind.")


def _assert_progress_telemetry_is_best_effort(root: Path) -> None:
    """Verify persistent telemetry lock failures cannot terminate a task worker."""
    reporter = TaskProgressReporter(
        run_directory=root / "telemetry_run",
        task_key="c01_ridge_classifier__r00__f00",
        candidate_id="C01_RIDGE_CLASSIFIER",
        repeat_index=0,
        fold_index=0,
        heartbeat_interval_seconds=3600,
    )
    state = {"calls": 0}
    with _patched_replace(_always_fail(state), retry_count=2):
        reporter.start(stage="initializing", message="Testing best-effort telemetry.")
        reporter.update(stage="stage_a", message="Telemetry write is intentionally blocked.")
        reporter.close(final_stage="completed", message="Task result remains independent of telemetry.")

    # A later normal write must recover without recreating the reporter or the task.
    reporter.update(stage="completed", message="Telemetry write recovered.")
    progress_path = reporter.path
    if not progress_path.exists():
        raise AssertionError("A later successful telemetry refresh did not recover the sidecar.")
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if payload.get("stage") != "completed":
        raise AssertionError("Recovered telemetry sidecar does not contain the latest stage.")
    if state["calls"] != 9:
        raise AssertionError("Best-effort telemetry did not exhaust bounded retries per write.")


def _assert_failure_summary_extracts_root_cause() -> None:
    """Verify the dashboard suppresses process-pool wrapper text in favor of the real error."""
    error_text = "\n".join(
        [
            "concurrent.futures.process._RemoteTraceback:",
            '"""',
            "Traceback (most recent call last):",
            "  File 'worker.py', line 1, in run",
            "PermissionError: [WinError 5] Access is denied: 'temporary' -> 'destination'",
            '"""',
            "The above exception was the direct cause of the following exception:",
            "PermissionError: [WinError None] Access is denied: 'temporary' -> 'destination'",
        ]
    )
    summary = summarize_task_failure_reason(error_text)
    if not summary.startswith("PermissionError:"):
        raise AssertionError("Dashboard failure summary retained the process-pool wrapper.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="telco-final-comparison-io-") as temporary_directory:
        root = Path(temporary_directory)
        _assert_transient_retries(root)
        _assert_persistent_failure_preserves_prior_artifact(root)
        _assert_progress_telemetry_is_best_effort(root)
    _assert_failure_summary_extracts_root_cause()
    print("Final-comparison filesystem-resilience smoke test passed.")


if __name__ == "__main__":
    main()