"""Run a bounded persistent nested-CV pilot for the final comparison system.

This executable is an operational validation step between component-level smoke tests
and the eventual master comparison. It uses all 5,634 development rows but deliberately
uses a limited, representative candidate subset, three outer folds, one outer repeat,
three-fold inner stages, and twelve Stage-A trials per outer task. It is therefore not a
final model-ranking experiment and must not be used to select a production procedure.

The pilot exists to validate, under a meaningful workload, that the complete machinery
works together after the pruned F2 representation was frozen:

* deterministic full-development outer splits;
* persistent task-local Optuna studies and Stage-B confirmation;
* candidate-specific feature, selection, and imbalance routing;
* the F2 regularized-linear branch;
* sparse, dense, numeric-first, and native-categorical representations;
* atomic per-task artifacts, SQLite task coordination, and safe resume behaviour.

The candidate subset is selected for implementation-path coverage, not for historical
predictive ranking:

``C01_RIDGE_CLASSIFIER``
    Regularized linear margin model with F0/F1/F2, feature-selection alternatives, and
    the broadest compatible imbalance-policy family.

``C02_LOGISTIC_REGRESSION``
    Regularized probabilistic linear model with F0/F1/F2 and penalty-dependent tuning.

``C07_HYBRID_NAIVE_BAYES``
    Custom numeric-first Gaussian-Bernoulli representation.

``C08_DECISION_TREE``
    Standard unscaled one-hot tree route with tree-specific imbalance compatibility.

``C19_CATBOOST``
    Native categorical-string route and the clone-safe sample-weight wrapper.

``C23_MULTILAYER_PERCEPTRON``
    Dense scaled representation with optional fold-local feature selection and
    resampling.

The script reads only ``data/processed/train.csv`` through the shared project loader.
It never imports, reads, or evaluates the held-out test set. Completed pilot artifacts
remain under ``artifacts/final_comparison/<run_id>/`` and can be resumed only when their
protocol and full-development data fingerprints still match.

Typical use
-----------
Create and run the pilot with two outer worker processes:

``python scripts/run_final_comparison_pilot.py --max-workers 2``

To intentionally validate a graceful full-data resume path, stop after one completed
serial task, then reopen the identical run with two workers:

``python scripts/run_final_comparison_pilot.py --max-workers 1 --stop-after-completed 1``
``python scripts/run_final_comparison_pilot.py --resume --max-workers 2``

If a task fails, inspect its persisted error text in the run directory and retry only
failed tasks after resolving the cause:

``python scripts/run_final_comparison_pilot.py --resume --max-workers 2 --retry-failed``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    import optuna  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Run `pip install -r requirements.txt` from the "
        "repository root, then rerun this pilot."
    ) from exc


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_CATBOOST,
    CANDIDATE_DECISION_TREE,
    CANDIDATE_HYBRID_NAIVE_BAYES,
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_MLP,
    CANDIDATE_RIDGE_CLASSIFIER,
    validate_candidate_registry,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.experiment_protocol import (  # noqa: E402
    ExperimentProtocol,
    make_dataframe_fingerprint,
    make_environment_fingerprint,
)
from telco_churn.experiment_runner import execute_monitored_registered_tasks  # noqa: E402
from telco_churn.experiment_progress import (  # noqa: E402
    clear_graceful_stop_request,
    format_duration,
)
from telco_churn.experiment_splits import (  # noqa: E402
    derive_seed,
    make_repeated_stratified_outer_splits,
)
from telco_churn.experiment_store import ExperimentStore, ExperimentTask  # noqa: E402
from telco_churn.experiment_tasks import run_nested_hpo_outer_task  # noqa: E402


PILOT_CANDIDATE_IDS: tuple[str, ...] = (
    CANDIDATE_RIDGE_CLASSIFIER,
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_HYBRID_NAIVE_BAYES,
    CANDIDATE_DECISION_TREE,
    CANDIDATE_CATBOOST,
    CANDIDATE_MLP,
)

OUTER_N_SPLITS = 3
OUTER_N_REPEATS = 1
STAGE_A_N_SPLITS = 3
STAGE_B_N_SPLITS = 3
STAGE_A_N_TRIALS = 12
CONFIRMATION_TOP_K = 3
SEARCH_PROFILE = "full"
DEFAULT_RUN_ID = "pilot_pruned_f2_v2_monitorable"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "final_comparison"


class PilotConfigurationError(ValueError):
    """Raised when a requested pilot invocation is internally inconsistent."""


def _parse_arguments() -> argparse.Namespace:
    """Parse explicit run-identity, resume, and worker-control arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the bounded full-development persistent nested-CV pilot. "
            "This is an operational validation run, not the master comparison."
        )
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "Run-directory name below artifacts/final_comparison. The default is a "
            "versioned identifier for the pruned F2 pilot."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Open an existing compatible pilot run instead of creating a new one.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Allow previously failed tasks to be claimed again during this invocation.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of process-level outer-task workers. Defaults to 2.",
    )
    parser.add_argument(
        "--stop-after-completed",
        type=int,
        default=None,
        help=(
            "Gracefully stop after this many newly completed tasks. This is available "
            "only with --max-workers 1 and is useful for exercising resume behaviour."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the protocol and display the deterministic task plan without writing artifacts.",
    )
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject unsafe or ambiguous pilot-invocation combinations before any write."""
    if not arguments.run_id.strip():
        raise PilotConfigurationError("--run-id must not be empty.")
    if arguments.max_workers < 1:
        raise PilotConfigurationError("--max-workers must be at least one.")
    if arguments.stop_after_completed is not None:
        if arguments.stop_after_completed < 1:
            raise PilotConfigurationError("--stop-after-completed must be positive.")
        if arguments.max_workers != 1:
            raise PilotConfigurationError(
                "--stop-after-completed requires --max-workers 1 because the runner "
                "supports deterministic graceful pausing only in serial mode."
            )
    if arguments.retry_failed and not arguments.resume:
        raise PilotConfigurationError("--retry-failed requires --resume.")


def _make_tasks(
    *,
    run_directory: Path,
    y,
    full_training_positions: np.ndarray,
    protocol_fingerprint: str,
) -> list[ExperimentTask]:
    """Create deterministic full-development outer tasks for the fixed pilot contract."""
    outer_splits = make_repeated_stratified_outer_splits(
        y,
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    tasks: list[ExperimentTask] = []
    for candidate_id in PILOT_CANDIDATE_IDS:
        for split in outer_splits:
            task_seed = derive_seed(
                RANDOM_STATE,
                "pruned_f2_pilot_v2",
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
                "run_directory": str(run_directory),
                "protocol_fingerprint": protocol_fingerprint,
                "sample_positions": [int(value) for value in full_training_positions],
                "outer_train_indices": [int(value) for value in split.train_indices],
                "outer_validation_indices": [
                    int(value) for value in split.validation_indices
                ],
                "stage_a_n_splits": STAGE_A_N_SPLITS,
                "stage_b_n_splits": STAGE_B_N_SPLITS,
                "stage_a_n_trials": STAGE_A_N_TRIALS,
                "confirmation_top_k": CONFIRMATION_TOP_K,
                "search_profile": SEARCH_PROFILE,
                "study_database_path": str(study_path),
                "study_name": task_key,
                "task_seed": int(task_seed),
            }
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


def _source_provenance() -> dict[str, object]:
    """Return the checked-out repository revision and whether source files are clean.

    A persistent run should be associated with an exact source revision. The shell
    commands are intentionally advisory rather than a replacement for the protocol,
    data, environment, and candidate-contract fingerprints already enforced by the
    experiment store. A normal executable pilot refuses a dirty worktree, whereas a
    dry run remains available before the runner itself has been committed.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return {"git_revision": None, "working_tree_clean": None}

    git_revision = revision.stdout.strip() if revision.returncode == 0 else None
    working_tree_clean = (
        status.stdout.strip() == "" if status.returncode == 0 else None
    )
    return {
        "git_revision": git_revision or None,
        "working_tree_clean": working_tree_clean,
    }


def _make_protocol(source_provenance: dict[str, object]) -> ExperimentProtocol:
    """Return the immutable, explicitly non-master pilot protocol."""
    return ExperimentProtocol(
        protocol_id="telco_final_comparison_pilot_pruned_f2",
        version="v2",
        candidate_ids=PILOT_CANDIDATE_IDS,
        primary_metric="average_precision",
        outer_n_splits=OUTER_N_SPLITS,
        outer_n_repeats=OUTER_N_REPEATS,
        inner_n_splits=STAGE_A_N_SPLITS,
        random_state=RANDOM_STATE,
        metadata={
            "purpose": (
                "monitorable operational persistent nested-CV pilot after the pruned F2 policy; "
                "not a final ranking or procedure-selection run"
            ),
            "development_data_scope": "all rows in data/processed/train.csv only",
            "candidate_subset_role": "implementation-path coverage, not finalist selection",
            "search_profile": SEARCH_PROFILE,
            "stage_a_n_splits": STAGE_A_N_SPLITS,
            "stage_b_n_splits": STAGE_B_N_SPLITS,
            "stage_a_n_trials": STAGE_A_N_TRIALS,
            "confirmation_top_k": CONFIRMATION_TOP_K,
            "feature_policy_contract": "F2 pruned after target-free structural audit",
            "monitoring_contract": "task progress sidecars, read-only status command, and clean pause control",
            "held_out_test_set_policy": "not loaded or referenced",
            "source_provenance": dict(source_provenance),
        },
    )


def _make_environment_fingerprint() -> dict[str, object]:
    """Record packages whose versions can affect this pilot's fitted procedures."""
    return make_environment_fingerprint(
        package_names=(
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "optuna",
            "imbalanced-learn",
            "xgboost",
            "lightgbm",
            "catboost",
            "threadpoolctl",
        )
    )


def _print_plan(
    *,
    run_id: str,
    n_rows: int,
    protocol: ExperimentProtocol,
    tasks: list[ExperimentTask],
    dry_run: bool,
) -> None:
    """Print operational facts without reporting any candidate-performance metrics."""
    action = "Dry-run validation" if dry_run else "Persistent pilot execution"
    print(f"{action}: {run_id}", flush=True)
    print("Purpose: operational validation only, not final procedure selection.", flush=True)
    print(
        "Development data only: "
        f"{n_rows} rows, {OUTER_N_SPLITS} outer folds x {OUTER_N_REPEATS} repeat, "
        f"{STAGE_A_N_SPLITS}-fold Stage A, {STAGE_B_N_SPLITS}-fold Stage B.",
        flush=True,
    )
    print(
        f"Candidate coverage: {len(PILOT_CANDIDATE_IDS)} families, "
        f"{len(tasks)} outer tasks, {STAGE_A_N_TRIALS} Stage-A trials per task, "
        f"top {CONFIRMATION_TOP_K} Stage-A configurations confirmed in Stage B.",
        flush=True,
    )
    print("Candidate IDs:", flush=True)
    for candidate_id in protocol.candidate_ids:
        print(f"  {candidate_id}", flush=True)
    if not dry_run:
        print(
            "Monitor from a second terminal with: "
            f"python scripts/final_comparison_status.py --run-id {run_id} --watch",
            flush=True,
        )


def _pilot_event(
    event_name: str,
    task,
    details: dict[str, object],
) -> None:
    """Print concise coordinator-owned operational events without ranking candidates."""
    timestamp = __import__("datetime").datetime.now().strftime("%H:%M:%S")
    task_label = (
        f"{task.candidate_id} r{task.repeat_index:02d}f{task.fold_index:02d}"
        if task is not None
        else None
    )
    if event_name == "task_started":
        print(
            f"[{timestamp}] START {task_label} "
            f"(active={details.get('active_tasks')}/{details.get('worker_capacity', 1)})",
            flush=True,
        )
    elif event_name == "task_completed":
        print(
            f"[{timestamp}] COMPLETE {task_label} "
            f"elapsed={format_duration(details.get('elapsed_seconds'))}",
            flush=True,
        )
    elif event_name == "task_failed":
        print(
            f"[{timestamp}] FAILED {task_label} "
            f"elapsed={format_duration(details.get('elapsed_seconds'))}",
            flush=True,
        )
    elif event_name == "task_interrupted":
        print(
            f"[{timestamp}] INTERRUPTED {task_label}: {details.get('reason', '-')}",
            flush=True,
        )
    elif event_name == "active_snapshot":
        elapsed_by_task = details.get("elapsed_seconds", {})
        active_items = []
        for task_key in details.get("active_task_keys", []):
            elapsed = None
            if isinstance(elapsed_by_task, dict):
                elapsed = elapsed_by_task.get(task_key)
            active_items.append(f"{task_key} ({format_duration(elapsed)})")
        print(f"[{timestamp}] ACTIVE {', '.join(active_items) or '-'}", flush=True)
    elif event_name == "graceful_stop_requested":
        print(f"[{timestamp}] PAUSE REQUESTED: {details.get('message', '')}", flush=True)
    elif event_name == "hard_stop_requested":
        print(f"[{timestamp}] EMERGENCY STOP REQUESTED", flush=True)
    elif event_name == "intentional_pause":
        print(f"[{timestamp}] INTENTIONAL PAUSE AFTER COMPLETED TASK", flush=True)

def _open_store(
    *,
    artifacts_root: Path,
    run_id: str,
    resume: bool,
    protocol: ExperimentProtocol,
    data_fingerprint: dict[str, object],
    environment_fingerprint: dict[str, object],
) -> ExperimentStore:
    """Create one new run or reopen exactly one strictly compatible prior run."""
    run_directory = artifacts_root / run_id
    if resume:
        if not run_directory.exists():
            raise FileNotFoundError(
                f"Cannot resume because the pilot run does not exist: {run_directory}"
            )
        return ExperimentStore.open_for_resume(
            artifacts_root=artifacts_root,
            run_id=run_id,
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint_sha256=str(data_fingerprint["sha256"]),
        )

    if run_directory.exists():
        raise FileExistsError(
            f"Pilot run already exists: {run_directory}. Re-run with --resume to continue "
            "that exact compatible run, or choose a new --run-id."
        )
    return ExperimentStore.create(
        artifacts_root=artifacts_root,
        run_id=run_id,
        protocol_payload=protocol.to_dict(),
        protocol_fingerprint=protocol.fingerprint,
        data_fingerprint=data_fingerprint,
        environment_fingerprint=environment_fingerprint,
    )


def main() -> None:
    """Create or resume the bounded full-development persistent pilot."""
    arguments = _parse_arguments()
    _validate_arguments(arguments)
    validate_candidate_registry()

    train_df = load_train_data()
    X_all, y_all = split_features_target(train_df)
    full_training_positions = np.arange(len(X_all), dtype=np.int64)
    source_provenance = _source_provenance()
    if not arguments.dry_run and source_provenance["working_tree_clean"] is False:
        raise PilotConfigurationError(
            "Refusing to create or resume an executable pilot from a dirty worktree. "
            "Commit or stash intentional source changes first so the run manifest can "
            "record one checked-out revision. Use --dry-run before committing to inspect "
            "the deterministic plan without creating artifacts."
        )

    protocol = _make_protocol(source_provenance)
    data_fingerprint = make_dataframe_fingerprint(X_all, y_all)
    environment_fingerprint = _make_environment_fingerprint()

    planned_tasks = _make_tasks(
        run_directory=ARTIFACTS_ROOT / arguments.run_id,
        y=y_all,
        full_training_positions=full_training_positions,
        protocol_fingerprint=protocol.fingerprint,
    )
    _print_plan(
        run_id=arguments.run_id,
        n_rows=len(X_all),
        protocol=protocol,
        tasks=planned_tasks,
        dry_run=bool(arguments.dry_run),
    )
    print(
        "Source provenance: "
        f"revision={source_provenance['git_revision']}, "
        f"working_tree_clean={source_provenance['working_tree_clean']}",
        flush=True,
    )

    if arguments.dry_run:
        print("Dry-run completed without creating artifact directories or fitting models.")
        return

    store = _open_store(
        artifacts_root=ARTIFACTS_ROOT,
        run_id=arguments.run_id,
        resume=bool(arguments.resume),
        protocol=protocol,
        data_fingerprint=data_fingerprint,
        environment_fingerprint=environment_fingerprint,
    )
    try:
        clear_graceful_stop_request(store.run_directory)
        tasks = _make_tasks(
            run_directory=store.run_directory,
            y=y_all,
            full_training_positions=full_training_positions,
            protocol_fingerprint=protocol.fingerprint,
        )
        summary = execute_monitored_registered_tasks(
            store=store,
            tasks=tasks,
            worker=run_nested_hpo_outer_task,
            max_workers=arguments.max_workers,
            retry_failed=bool(arguments.retry_failed),
            stop_after_completed=arguments.stop_after_completed,
            event_callback=_pilot_event,
            stop_control_run_directory=store.run_directory,
            progress_directory=store.run_directory / "progress",
        )
        task_status = store.task_summary()
        store.validate_completed_artifacts()

        print("Execution summary:", flush=True)
        for key in ("submitted", "completed", "skipped", "failed", "interrupted", "paused"):
            print(f"  {key}: {summary[key]}", flush=True)
        print("Persistent task states:", flush=True)
        for status, count in sorted(task_status.items()):
            print(f"  {status}: {count}", flush=True)
        print(f"Run directory: {store.run_directory}", flush=True)

        if summary["failed"]:
            raise SystemExit(
                "One or more pilot tasks failed. Resolve the persisted task error and rerun "
                "with --resume --retry-failed; completed tasks will be skipped."
            )
        if summary["paused"]:
            print(
                "Pilot paused. Inspect it with scripts/final_comparison_status.py, then "
                "resume the same compatible run with --resume when ready.",
                flush=True,
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
