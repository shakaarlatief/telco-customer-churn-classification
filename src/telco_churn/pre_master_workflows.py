"""Reusable execution machinery for explicitly non-master final-comparison runs.

The project uses persistent nested cross-validation for several distinct purposes. A
bounded admission smoke run verifies that every currently implemented candidate can
complete the actual resumable execution path. A separate search-budget calibration run
measures how much exploration is computationally realistic before a master selection
protocol is frozen. Neither run is a model-selection experiment.

This module centralizes the common mechanics needed by those pre-master workflows:

* deterministic full-development outer splits and task identifiers;
* strict data, protocol, environment, and source-provenance fingerprints;
* one coordinator-owned event stream with worker-owned progress telemetry;
* clean pause, resume, and failed-task retry behaviour;
* refusal to create an executable run from an uncommitted source tree.

The module deliberately does not modify the established pilot runner. The v6 pilot is a
completed historical validation artifact, so preserving its executable source unchanged
keeps its provenance easy to audit. New workflows use this shared implementation from
creation onward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from telco_churn.candidates import INITIAL_CANDIDATE_REGISTRY, validate_candidate_registry
from telco_churn.config import RANDOM_STATE
from telco_churn.data import load_train_data, split_features_target
from telco_churn.experiment_progress import clear_graceful_stop_request
from telco_churn.experiment_protocol import (
    ExperimentProtocol,
    make_dataframe_fingerprint,
    make_environment_fingerprint,
)
from telco_churn.experiment_runner import execute_monitored_registered_tasks
from telco_churn.experiment_splits import derive_seed, make_repeated_stratified_outer_splits
from telco_churn.experiment_store import ExperimentStore, ExperimentTask
from telco_churn.experiment_tasks import run_nested_hpo_outer_task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "final_comparison"
IMPLEMENTED_CANDIDATE_UNIVERSE = "C01-C26"
DEFERRED_CANDIDATE_IDS = ("C27_TABPFN", "C28_AUTOGLUON")
MASTER_ADMISSION_STATE = "none"
PROTOCOL_V2_STATE = "not frozen"


class PreMasterWorkflowConfigurationError(ValueError):
    """Raised when an explicit pre-master workflow specification is inconsistent.

    The error is raised before development data are loaded, task artifacts are created,
    or worker processes are started. This makes a protocol-definition mistake distinct
    from a model-fitting failure recorded inside a durable task artifact.
    """


@dataclass(frozen=True)
class PreMasterWorkflowSpec:
    """Immutable declaration of one non-selection persistent nested-CV workflow.

    Parameters
    ----------
    workflow_id:
        Stable immutable workflow identity recorded in the protocol manifest and used as
        part of task-seed derivation. It must change when the workflow's scientific
        purpose changes.

    protocol_id, protocol_version:
        Explicit immutable protocol identity persisted in the run manifest. A different
        candidate set, split plan, search budget, or confirmation policy must use a new
        identifier or version rather than reusing an existing run directory.

    purpose:
        A concise statement of what operational question the run is permitted to answer.
        Every supplied specification in this module is intentionally non-selection.

    candidate_set_role:
        Describes why this candidate subset is present. It is stored in manifest metadata
        so a later report cannot silently reinterpret a convenience subset as finalists.

    stage_a_n_splits, stage_b_n_splits:
        Inner cross-validation structure for exploration and independent confirmation.
        The top-level protocol's ``inner_n_splits`` field stores the Stage-A value, while
        both values are retained in immutable metadata and in each task payload.

    seed_namespace:
        Immutable namespace used in deterministic task-seed derivation. It is persisted
        in protocol metadata, so an accidental seed-policy change cannot reuse an
        existing run directory under the same protocol fingerprint.
    """

    workflow_id: str
    protocol_id: str
    protocol_version: str
    default_run_id: str
    purpose: str
    candidate_set_role: str
    candidate_ids: tuple[str, ...]
    outer_n_splits: int
    outer_n_repeats: int
    stage_a_n_splits: int
    stage_b_n_splits: int
    stage_a_n_trials: int
    confirmation_top_k: int
    search_profile: str
    feature_policy_contract: str
    seed_namespace: str
    default_max_workers: int = 2

    def __post_init__(self) -> None:
        """Validate fixed workflow ingredients before any execution side effect."""
        for name, value in (
            ("workflow_id", self.workflow_id),
            ("protocol_id", self.protocol_id),
            ("protocol_version", self.protocol_version),
            ("default_run_id", self.default_run_id),
            ("purpose", self.purpose),
            ("candidate_set_role", self.candidate_set_role),
            ("search_profile", self.search_profile),
            ("feature_policy_contract", self.feature_policy_contract),
            ("seed_namespace", self.seed_namespace),
        ):
            if not str(value).strip():
                raise PreMasterWorkflowConfigurationError(f"{name} must not be empty.")

        if not self.candidate_ids:
            raise PreMasterWorkflowConfigurationError("candidate_ids must not be empty.")
        if any(not str(candidate_id).strip() for candidate_id in self.candidate_ids):
            raise PreMasterWorkflowConfigurationError(
                "candidate_ids cannot contain empty values."
            )
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise PreMasterWorkflowConfigurationError("candidate_ids must be unique.")

        for name, value, minimum in (
            ("outer_n_splits", self.outer_n_splits, 2),
            ("outer_n_repeats", self.outer_n_repeats, 1),
            ("stage_a_n_splits", self.stage_a_n_splits, 2),
            ("stage_b_n_splits", self.stage_b_n_splits, 2),
            ("stage_a_n_trials", self.stage_a_n_trials, 1),
            ("confirmation_top_k", self.confirmation_top_k, 1),
            ("default_max_workers", self.default_max_workers, 1),
        ):
            if int(value) < minimum:
                raise PreMasterWorkflowConfigurationError(
                    f"{name} must be at least {minimum}."
                )

        if int(self.confirmation_top_k) > int(self.stage_a_n_trials):
            raise PreMasterWorkflowConfigurationError(
                "confirmation_top_k cannot exceed stage_a_n_trials because Stage B can only "
                "confirm configurations that Stage A completed."
            )
        if self.search_profile not in {"smoke", "full"}:
            raise PreMasterWorkflowConfigurationError(
                "search_profile must be either 'smoke' or 'full'."
            )


@dataclass(frozen=True)
class WorkflowArguments:
    """Command-line controls that do not alter a workflow's immutable protocol.

    ``run_id`` is intentionally not part of the protocol. It names one physical artifact
    directory, whereas the stored protocol fingerprint determines whether a directory can
    be safely resumed. ``max_workers`` affects operational scheduling only and therefore
    remains outside the scientific procedure definition.
    """

    run_id: str
    resume: bool
    retry_failed: bool
    max_workers: int
    stop_after_completed: int | None
    dry_run: bool


def parse_workflow_arguments(
    spec: PreMasterWorkflowSpec,
    argv: Sequence[str] | None = None,
) -> WorkflowArguments:
    """Parse generic execution controls for one immutable pre-master specification."""
    parser = argparse.ArgumentParser(
        description=(
            "Create or resume a bounded, explicitly non-selection persistent nested-CV "
            "workflow. The command reads development data only and never evaluates the "
            "held-out test set."
        )
    )
    parser.add_argument(
        "--run-id",
        default=spec.default_run_id,
        help=(
            "Run-directory name below artifacts/final_comparison. Defaults to "
            f"{spec.default_run_id}."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Open an existing fingerprint-compatible run instead of creating a new one.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Allow previously failed tasks to be claimed again. Requires --resume.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=spec.default_max_workers,
        help=(
            "Number of process-level outer-task workers. Candidate estimators remain "
            f"single-threaded inside workers. Defaults to {spec.default_max_workers}."
        ),
    )
    parser.add_argument(
        "--stop-after-completed",
        type=int,
        default=None,
        help=(
            "Gracefully pause after this many newly completed tasks. This deterministic "
            "resume hook is available only with --max-workers 1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the exact protocol and print the deterministic task plan without "
            "creating artifact directories or fitting models."
        ),
    )
    namespace = parser.parse_args(argv)
    return WorkflowArguments(
        run_id=str(namespace.run_id),
        resume=bool(namespace.resume),
        retry_failed=bool(namespace.retry_failed),
        max_workers=int(namespace.max_workers),
        stop_after_completed=(
            None
            if namespace.stop_after_completed is None
            else int(namespace.stop_after_completed)
        ),
        dry_run=bool(namespace.dry_run),
    )


def validate_workflow_arguments(arguments: WorkflowArguments) -> None:
    """Reject unsafe combinations before a run is created or resumed."""
    if not arguments.run_id.strip():
        raise PreMasterWorkflowConfigurationError("--run-id must not be empty.")
    if arguments.max_workers < 1:
        raise PreMasterWorkflowConfigurationError("--max-workers must be at least one.")
    if arguments.retry_failed and not arguments.resume:
        raise PreMasterWorkflowConfigurationError("--retry-failed requires --resume.")
    if arguments.stop_after_completed is not None:
        if arguments.stop_after_completed < 1:
            raise PreMasterWorkflowConfigurationError(
                "--stop-after-completed must be positive."
            )
        if arguments.max_workers != 1:
            raise PreMasterWorkflowConfigurationError(
                "--stop-after-completed requires --max-workers 1 because a parallel pool "
                "can finish more than one task before the coordinator observes the pause "
                "boundary."
            )


def _known_candidate_ids() -> set[str]:
    """Return the current registry identifiers after structural validation."""
    validate_candidate_registry()
    return {definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY}


def validate_spec_candidate_ids(spec: PreMasterWorkflowSpec) -> None:
    """Verify that a workflow refers only to registered candidate procedures."""
    deferred = sorted(set(spec.candidate_ids) & set(DEFERRED_CANDIDATE_IDS))
    if deferred:
        raise PreMasterWorkflowConfigurationError(
            "Workflow specification includes deferred candidates that are not implemented "
            f"or admitted to pre-master runs: {deferred}."
        )
    unknown = sorted(set(spec.candidate_ids) - _known_candidate_ids())
    if unknown:
        raise PreMasterWorkflowConfigurationError(
            "Workflow specification contains candidate identifiers absent from the current "
            f"registry: {unknown}."
        )


def make_workflow_tasks(
    *,
    spec: PreMasterWorkflowSpec,
    run_directory: Path,
    y,
    full_training_positions: np.ndarray,
    protocol_fingerprint: str,
) -> list[ExperimentTask]:
    """Create deterministic, training-only outer tasks for one fixed specification.

    The same full-development row positions are included in every task payload. Each task
    then receives only its outer-training partition while the worker performs all feature
    construction, selection, imbalance treatment, Stage-A tuning, Stage-B confirmation,
    and final outer fit internally. The held-out test data are neither loaded nor named.
    """
    outer_splits = make_repeated_stratified_outer_splits(
        y,
        n_splits=spec.outer_n_splits,
        n_repeats=spec.outer_n_repeats,
        random_state=RANDOM_STATE,
    )
    tasks: list[ExperimentTask] = []
    for candidate_id in spec.candidate_ids:
        for split in outer_splits:
            task_seed = derive_seed(
                RANDOM_STATE,
                spec.seed_namespace,
                candidate_id,
                split.repeat_index,
                split.fold_index,
            )
            task_key = (
                f"{candidate_id.lower()}__r{split.repeat_index:02d}"
                f"__f{split.fold_index:02d}"
            )
            study_path = (
                Path(run_directory)
                / "optuna_studies"
                / candidate_id.lower()
                / f"r{split.repeat_index:02d}_f{split.fold_index:02d}.sqlite"
            )
            payload = {
                "task_kind": "nested_hpo_outer_v1",
                "run_directory": str(run_directory),
                "protocol_fingerprint": str(protocol_fingerprint),
                "sample_positions": [int(value) for value in full_training_positions],
                "outer_train_indices": [int(value) for value in split.train_indices],
                "outer_validation_indices": [
                    int(value) for value in split.validation_indices
                ],
                "stage_a_n_splits": int(spec.stage_a_n_splits),
                "stage_b_n_splits": int(spec.stage_b_n_splits),
                "stage_a_n_trials": int(spec.stage_a_n_trials),
                "confirmation_top_k": int(spec.confirmation_top_k),
                "search_profile": str(spec.search_profile),
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


def source_provenance(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Return checked-out source identity and working-tree cleanliness.

    The protocol, data, candidate contracts, and environment fingerprints carry the
    scientific reproducibility contract. Git provenance is complementary operational
    evidence: executable runs refuse a dirty worktree so the persisted run manifest can
    refer to one immutable source revision rather than an uncommitted local mixture.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return {"git_revision": None, "working_tree_clean": None}

    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_clean": (
            status.stdout.strip() == "" if status.returncode == 0 else None
        ),
    }


def make_workflow_protocol(
    spec: PreMasterWorkflowSpec,
    provenance: Mapping[str, object],
) -> ExperimentProtocol:
    """Build the immutable manifest protocol for one non-selection workflow."""
    return ExperimentProtocol(
        protocol_id=spec.protocol_id,
        version=spec.protocol_version,
        candidate_ids=spec.candidate_ids,
        primary_metric="average_precision",
        outer_n_splits=spec.outer_n_splits,
        outer_n_repeats=spec.outer_n_repeats,
        inner_n_splits=spec.stage_a_n_splits,
        random_state=RANDOM_STATE,
        metadata={
            "workflow_id": spec.workflow_id,
            "seed_namespace": spec.seed_namespace,
            "purpose": spec.purpose,
            "workflow_role": "pre-master operational and calibration evidence only",
            "candidate_set_role": spec.candidate_set_role,
            "implemented_candidate_universe": IMPLEMENTED_CANDIDATE_UNIVERSE,
            "deferred_candidate_ids": list(DEFERRED_CANDIDATE_IDS),
            "master_admission_state": MASTER_ADMISSION_STATE,
            "protocol_v2_state": PROTOCOL_V2_STATE,
            "development_data_scope": "all rows in data/processed/train.csv only",
            "search_profile": spec.search_profile,
            "stage_a_n_splits": spec.stage_a_n_splits,
            "stage_b_n_splits": spec.stage_b_n_splits,
            "stage_a_n_trials": spec.stage_a_n_trials,
            "confirmation_top_k": spec.confirmation_top_k,
            "feature_policy_contract": spec.feature_policy_contract,
            "monitoring_contract": (
                "atomic task progress, task and coordinator event logs, read-only "
                "dashboard, clean pause control, incremental Stage-B checkpoints, and "
                "bounded atomic-replacement retries for transient local filesystem locks"
            ),
            "held_out_test_set_policy": "not loaded or referenced",
            "source_provenance": dict(provenance),
        },
    )


def make_workflow_environment_fingerprint() -> dict[str, object]:
    """Record package versions that can affect any current core candidate route."""
    return make_environment_fingerprint(
        package_names=(
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "optuna",
            "imbalanced-learn",
            "interpret-core",
            "xgboost",
            "lightgbm",
            "catboost",
            "pytorch-tabnet",
            "rtdl_revisiting_models",
            "tabm",
            "torch",
            "rtdl_num_embeddings",
            "threadpoolctl",
        )
    )


def print_workflow_plan(
    *,
    spec: PreMasterWorkflowSpec,
    arguments: WorkflowArguments,
    n_rows: int,
    protocol: ExperimentProtocol,
    tasks: Sequence[ExperimentTask],
) -> None:
    """Print immutable execution facts without presenting performance as evidence."""
    action = "Dry-run validation" if arguments.dry_run else "Persistent workflow execution"
    print(f"{action}: {arguments.run_id}", flush=True)
    print(f"Purpose: {spec.purpose}", flush=True)
    print(
        "Development data only: "
        f"{n_rows} rows, {spec.outer_n_splits} outer folds x {spec.outer_n_repeats} repeat, "
        f"{spec.stage_a_n_splits}-fold Stage A, {spec.stage_b_n_splits}-fold Stage B.",
        flush=True,
    )
    print(
        f"Candidate coverage: {len(spec.candidate_ids)} families, {len(tasks)} outer tasks, "
        f"{spec.stage_a_n_trials} valid Stage-A trials per task, top "
        f"{spec.confirmation_top_k} Stage-A configurations confirmed in Stage B.",
        flush=True,
    )
    print(f"Search profile: {spec.search_profile}", flush=True)
    print("Candidate IDs:", flush=True)
    for candidate_id in protocol.candidate_ids:
        print(f"  {candidate_id}", flush=True)
    if not arguments.dry_run:
        print(
            "Monitor from a second terminal with: "
            f"bash scripts/monitor_final_comparison.sh details --run-id {arguments.run_id}",
            flush=True,
        )


def make_workflow_event_callback(run_directory: Path):
    """Create the sole coordinator-owned logger and terminal event renderer.

    Workers produce their own task-local event records. The coordinator tails those
    records and emits a single combined event history. This preserves a clear writer
    ownership model and avoids concurrent appends to one shared file on Windows.
    """
    from telco_churn.experiment_progress import RunEventLogger, format_terminal_event_line

    logger = RunEventLogger(run_directory)

    def callback(event_name: str, task, details: Mapping[str, Any]) -> None:
        if event_name == "worker_event":
            worker_event = details.get("worker_event", {})
            if not isinstance(worker_event, Mapping):
                return
            worker_details = worker_event.get("details", {})
            if not isinstance(worker_details, Mapping):
                worker_details = {}
            record = logger.emit(
                str(worker_event.get("event", "worker_event")),
                message=str(worker_event.get("message") or "Worker event."),
                task=task,
                details=dict(worker_details),
                source="worker",
            )
        else:
            fallback_messages = {
                "task_started": "Outer task started.",
                "task_completed": "Outer task completed.",
                "task_failed": "Outer task failed.",
                "task_interrupted": "Outer task interrupted.",
                "graceful_stop_requested": "Clean pause requested.",
                "hard_stop_requested": "Emergency stop requested.",
                "active_snapshot": "Active-worker snapshot.",
                "intentional_pause": "Intentional pause after completed task.",
                "run_started": "Run started.",
                "run_resumed": "Run resumed.",
                "run_paused": "Run paused cleanly.",
                "run_failed": "Run finished with failures.",
                "run_completed": "Run completed.",
            }
            record = logger.emit(
                event_name,
                message=fallback_messages.get(
                    event_name,
                    str(event_name).replace("_", " ").capitalize() + ".",
                ),
                task=task,
                details=dict(details),
            )
        print(format_terminal_event_line(record, color=True), flush=True)

    return callback


def open_workflow_store(
    *,
    artifacts_root: Path,
    run_id: str,
    resume: bool,
    protocol: ExperimentProtocol,
    data_fingerprint: Mapping[str, object],
    environment_fingerprint: Mapping[str, object],
) -> ExperimentStore:
    """Create a new immutable run or reopen exactly one compatible prior run."""
    run_directory = Path(artifacts_root) / run_id
    if resume:
        if not run_directory.exists():
            raise FileNotFoundError(
                f"Cannot resume because the workflow run does not exist: {run_directory}"
            )
        return ExperimentStore.open_for_resume(
            artifacts_root=Path(artifacts_root),
            run_id=run_id,
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint_sha256=str(data_fingerprint["sha256"]),
        )
    if run_directory.exists():
        raise FileExistsError(
            f"Workflow run already exists: {run_directory}. Re-run with --resume to continue "
            "that exact compatible run, or choose a new --run-id."
        )
    return ExperimentStore.create(
        artifacts_root=Path(artifacts_root),
        run_id=run_id,
        protocol_payload=protocol.to_dict(),
        protocol_fingerprint=protocol.fingerprint,
        data_fingerprint=dict(data_fingerprint),
        environment_fingerprint=dict(environment_fingerprint),
    )


def _print_execution_summary(summary: Mapping[str, int], task_status: Mapping[str, int], run_directory: Path) -> None:
    """Print coordinator and durable registry state without inferring model quality."""
    print("Execution summary:", flush=True)
    for key in ("submitted", "completed", "skipped", "failed", "interrupted", "paused"):
        print(f"  {key}: {int(summary.get(key, 0))}", flush=True)
    print("Persistent task states:", flush=True)
    for status, count in sorted(task_status.items()):
        print(f"  {status}: {count}", flush=True)
    print(f"Run directory: {run_directory}", flush=True)


def run_pre_master_workflow(
    spec: PreMasterWorkflowSpec,
    argv: Sequence[str] | None = None,
) -> None:
    """Execute or inspect one fixed pre-master workflow.

    A normal invocation loads only development data, refuses an uncommitted worktree,
    writes one immutable run manifest, and delegates all candidate fitting to the existing
    monitored nested-HPO worker. A dry run performs task-plan, registry, split, and
    fingerprint validation without creating an artifact directory.
    """
    arguments = parse_workflow_arguments(spec, argv)
    validate_workflow_arguments(arguments)
    validate_spec_candidate_ids(spec)

    train_df = load_train_data()
    X_all, y_all = split_features_target(train_df)
    full_training_positions = np.arange(len(X_all), dtype=np.int64)
    provenance = source_provenance()
    if not arguments.dry_run and provenance["working_tree_clean"] is False:
        raise PreMasterWorkflowConfigurationError(
            "Refusing to create or resume an executable pre-master workflow from a dirty "
            "worktree. Commit or stash intentional source changes first so the manifest "
            "records one checked-out revision. Use --dry-run before committing to inspect "
            "the deterministic plan without creating artifacts."
        )

    protocol = make_workflow_protocol(spec, provenance)
    data_fingerprint = make_dataframe_fingerprint(X_all, y_all)
    environment_fingerprint = make_workflow_environment_fingerprint()
    planned_tasks = make_workflow_tasks(
        spec=spec,
        run_directory=DEFAULT_ARTIFACTS_ROOT / arguments.run_id,
        y=y_all,
        full_training_positions=full_training_positions,
        protocol_fingerprint=protocol.fingerprint,
    )
    print_workflow_plan(
        spec=spec,
        arguments=arguments,
        n_rows=len(X_all),
        protocol=protocol,
        tasks=planned_tasks,
    )
    print(
        "Source provenance: "
        f"revision={provenance['git_revision']}, "
        f"working_tree_clean={provenance['working_tree_clean']}",
        flush=True,
    )
    if arguments.dry_run:
        print("Dry-run completed without creating artifact directories or fitting models.")
        return

    store = open_workflow_store(
        artifacts_root=DEFAULT_ARTIFACTS_ROOT,
        run_id=arguments.run_id,
        resume=arguments.resume,
        protocol=protocol,
        data_fingerprint=data_fingerprint,
        environment_fingerprint=environment_fingerprint,
    )
    try:
        clear_graceful_stop_request(store.run_directory)
        tasks = make_workflow_tasks(
            spec=spec,
            run_directory=store.run_directory,
            y=y_all,
            full_training_positions=full_training_positions,
            protocol_fingerprint=protocol.fingerprint,
        )
        event_callback = make_workflow_event_callback(store.run_directory)
        event_callback(
            "run_resumed" if arguments.resume else "run_started",
            None,
            {
                "run_id": arguments.run_id,
                "task_total": len(tasks),
                "worker_capacity": arguments.max_workers,
                "workflow_id": spec.workflow_id,
            },
        )
        summary = execute_monitored_registered_tasks(
            store=store,
            tasks=tasks,
            worker=run_nested_hpo_outer_task,
            max_workers=arguments.max_workers,
            retry_failed=arguments.retry_failed,
            stop_after_completed=arguments.stop_after_completed,
            event_callback=event_callback,
            stop_control_run_directory=store.run_directory,
            progress_directory=store.run_directory / "progress",
            worker_event_directory=store.run_directory / "logs" / "tasks",
        )
        task_status = store.task_summary()
        store.validate_completed_artifacts()
        _print_execution_summary(summary, task_status, store.run_directory)

        persistent_failed = int(task_status.get("failed", 0))
        if summary["failed"] or persistent_failed:
            event_callback(
                "run_failed",
                None,
                {
                    "failed": persistent_failed,
                    "new_failed": summary["failed"],
                    "completed": summary["completed"],
                    "interrupted": summary["interrupted"],
                },
            )
            raise SystemExit(
                "One or more workflow tasks failed. Inspect the persisted task error, resolve "
                "the cause, then rerun the same compatible run with --resume --retry-failed. "
                "Completed tasks will be skipped."
            )

        if summary["paused"]:
            task_state = store.task_summary()
            event_callback(
                "run_paused",
                None,
                {
                    "completed": task_state.get("completed", 0),
                    "interrupted": task_state.get("interrupted", 0),
                    "pending": task_state.get("pending", 0),
                },
            )
            print(
                "Workflow paused. Inspect it with scripts/final_comparison_status.py, then "
                "resume the same compatible run with --resume when ready.",
                flush=True,
            )
        else:
            event_callback("run_completed", None, {"completed": summary["completed"]})
    finally:
        store.close()
