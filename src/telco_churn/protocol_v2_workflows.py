"""Executable scaffolds for final-comparison protocol declarations.

This module prepares reviewed final-comparison workflow declarations without launching
them by default. Checked-in JSON declarations are the reviewable source of truth; the
Python layer validates them against the implemented registry and can produce dry-run
task plans. Non-dry-run execution requires an explicit runner confirmation flag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from telco_churn.candidates import (
    CANDIDATE_CATBOOST,
    INITIAL_CANDIDATE_REGISTRY,
    SEARCH_PROFILE_CATBOOST_V2,
    SEARCH_PROFILE_FULL,
    validate_candidate_registry,
)
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
from telco_churn.pre_master_workflows import make_workflow_event_callback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "final_comparison_protocol_v2_base.json"
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "final_comparison"
DEFERRED_CANDIDATE_IDS = ("C27_TABPFN", "C28_AUTOGLUON")
EXPECTED_FREEZE_STATE_DRAFT = "draft_pending_review"
FROZEN_FREEZE_STATES = frozenset({"frozen", "frozen_v2"})
OFFICIAL_BASE_EVIDENCE_ROLE = "official_base_comparison_candidate_protocol"
FAST_COMPLETION_EVIDENCE_ROLE = "fast_completion_pipeline_evidence"
SUPPORTED_EVIDENCE_ROLES = frozenset(
    {OFFICIAL_BASE_EVIDENCE_ROLE, FAST_COMPLETION_EVIDENCE_ROLE}
)


class ProtocolV2WorkflowConfigurationError(ValueError):
    """Raised before data loading, artifact creation, or model fitting."""


@dataclass(frozen=True)
class CandidateBudgetSpec:
    """Candidate-specific search budget for the protocol-v2 draft."""

    candidate_id: str
    budget_lane: str
    stage_a_n_trials: int
    confirmation_top_k: int
    search_profile: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateBudgetSpec":
        """Create and validate one budget row from JSON."""
        try:
            spec = cls(
                candidate_id=str(payload["candidate_id"]),
                budget_lane=str(payload["budget_lane"]),
                stage_a_n_trials=int(payload["stage_a_n_trials"]),
                confirmation_top_k=int(payload["confirmation_top_k"]),
                search_profile=str(payload["search_profile"]),
            )
        except KeyError as exc:
            raise ProtocolV2WorkflowConfigurationError(
                f"Candidate budget row is missing required field {exc.args[0]!r}."
            ) from exc
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate one row without consulting other candidates."""
        if not self.candidate_id.strip():
            raise ProtocolV2WorkflowConfigurationError("candidate_id must not be empty.")
        if not self.budget_lane.strip():
            raise ProtocolV2WorkflowConfigurationError("budget_lane must not be empty.")
        if self.stage_a_n_trials < 1:
            raise ProtocolV2WorkflowConfigurationError(
                f"{self.candidate_id} stage_a_n_trials must be positive."
            )
        if self.confirmation_top_k < 1:
            raise ProtocolV2WorkflowConfigurationError(
                f"{self.candidate_id} confirmation_top_k must be positive."
            )
        if self.confirmation_top_k > self.stage_a_n_trials:
            raise ProtocolV2WorkflowConfigurationError(
                f"{self.candidate_id} confirmation_top_k cannot exceed Stage-A trials."
            )
        if self.search_profile not in {SEARCH_PROFILE_FULL, SEARCH_PROFILE_CATBOOST_V2}:
            raise ProtocolV2WorkflowConfigurationError(
                f"{self.candidate_id} has unsupported protocol-v2 search profile "
                f"{self.search_profile!r}."
            )
        if self.search_profile == SEARCH_PROFILE_CATBOOST_V2 and self.candidate_id != CANDIDATE_CATBOOST:
            raise ProtocolV2WorkflowConfigurationError(
                f"Only {CANDIDATE_CATBOOST} may use {SEARCH_PROFILE_CATBOOST_V2}."
            )
        if self.candidate_id == CANDIDATE_CATBOOST and self.search_profile != SEARCH_PROFILE_CATBOOST_V2:
            raise ProtocolV2WorkflowConfigurationError(
                f"{CANDIDATE_CATBOOST} must use the runtime-limited "
                f"{SEARCH_PROFILE_CATBOOST_V2} profile."
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible budget metadata."""
        return {
            "candidate_id": self.candidate_id,
            "budget_lane": self.budget_lane,
            "stage_a_n_trials": int(self.stage_a_n_trials),
            "confirmation_top_k": int(self.confirmation_top_k),
            "search_profile": self.search_profile,
        }


@dataclass(frozen=True)
class ProtocolV2BaseWorkflowSpec:
    """Validated executable view of the protocol-v2 base-comparison draft."""

    protocol_path: Path
    raw_protocol: Mapping[str, Any]
    protocol_id: str
    protocol_version: str
    freeze_state: str
    is_frozen: bool
    evidence_role: str
    candidate_ids: tuple[str, ...]
    deferred_candidate_ids: tuple[str, ...]
    outer_n_splits: int
    outer_n_repeats: int
    stage_a_n_splits: int
    stage_b_n_splits: int
    primary_metric: str
    candidate_budgets: tuple[CandidateBudgetSpec, ...]

    @property
    def budget_by_candidate(self) -> dict[str, CandidateBudgetSpec]:
        """Return candidate-specific budgets keyed by candidate ID."""
        return {budget.candidate_id: budget for budget in self.candidate_budgets}

    @property
    def is_execution_frozen(self) -> bool:
        """Return whether non-dry-run execution is allowed in principle."""
        return self.is_frozen and self.freeze_state in FROZEN_FREEZE_STATES


@dataclass(frozen=True)
class ProtocolV2WorkflowArguments:
    """Operational CLI controls that do not change protocol identity."""

    run_id: str
    max_workers: int
    dry_run: bool
    resume: bool
    retry_failed: bool
    stop_after_completed: int | None
    confirm_official_base_comparison: bool


def _implemented_candidate_ids() -> tuple[str, ...]:
    """Return current registry IDs after structural validation."""
    validate_candidate_registry()
    return tuple(definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY)


def load_protocol_v2_base_spec(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> ProtocolV2BaseWorkflowSpec:
    """Load and validate the protocol-v2 base-comparison JSON declaration."""
    path = Path(protocol_path)
    try:
        raw_protocol = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolV2WorkflowConfigurationError(
            f"Protocol declaration does not exist: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProtocolV2WorkflowConfigurationError(
            f"Protocol declaration is not valid JSON: {path}"
        ) from exc

    if not isinstance(raw_protocol, Mapping):
        raise ProtocolV2WorkflowConfigurationError("Protocol declaration must be a JSON object.")

    try:
        protocol_id = str(raw_protocol["protocol_id"])
        protocol_version = str(raw_protocol["protocol_version"])
        freeze_state = str(raw_protocol["freeze_state"])
        is_frozen = bool(raw_protocol["is_frozen"])
        evidence_role = str(raw_protocol["evidence_role"])
        candidate_ids = tuple(str(value) for value in raw_protocol["candidate_universe"])
        outer_cv = raw_protocol["outer_cv"]
        stage_a_cv = raw_protocol["stage_a_inner_cv"]
        stage_b_cv = raw_protocol["stage_b_confirmation_cv"]
        primary_metric = str(raw_protocol["primary_metric"])
        budget_rows = raw_protocol["candidate_budgets"]
    except KeyError as exc:
        raise ProtocolV2WorkflowConfigurationError(
            f"Protocol declaration is missing required field {exc.args[0]!r}."
        ) from exc

    if not protocol_id.strip() or not protocol_version.strip():
        raise ProtocolV2WorkflowConfigurationError("Protocol ID and version must not be empty.")
    if freeze_state != EXPECTED_FREEZE_STATE_DRAFT and freeze_state not in FROZEN_FREEZE_STATES:
        raise ProtocolV2WorkflowConfigurationError(
            f"Unsupported freeze_state {freeze_state!r}."
        )
    if is_frozen and freeze_state not in FROZEN_FREEZE_STATES:
        raise ProtocolV2WorkflowConfigurationError(
            "is_frozen=true requires a frozen freeze_state."
        )
    if not is_frozen and freeze_state in FROZEN_FREEZE_STATES:
        raise ProtocolV2WorkflowConfigurationError(
            "A frozen freeze_state requires is_frozen=true."
        )
    if evidence_role not in SUPPORTED_EVIDENCE_ROLES:
        raise ProtocolV2WorkflowConfigurationError(
            "Protocol evidence_role must identify a supported final-comparison protocol."
        )

    implemented_ids = _implemented_candidate_ids()
    if candidate_ids != implemented_ids:
        raise ProtocolV2WorkflowConfigurationError(
            "Protocol candidate universe must exactly match the implemented registry "
            f"C01-C26. Expected {implemented_ids!r}, got {candidate_ids!r}."
        )

    deferred_payload = raw_protocol.get("deferred_candidates", ())
    deferred_candidate_ids = tuple(
        str(item.get("candidate_id")) if isinstance(item, Mapping) else str(item)
        for item in deferred_payload
    )
    if deferred_candidate_ids != DEFERRED_CANDIDATE_IDS:
        raise ProtocolV2WorkflowConfigurationError(
            f"Deferred candidates must be exactly {DEFERRED_CANDIDATE_IDS!r}."
        )
    if set(candidate_ids) & set(DEFERRED_CANDIDATE_IDS):
        raise ProtocolV2WorkflowConfigurationError(
            "Deferred C27/C28 candidates must be absent from the candidate universe."
        )

    held_out_policy = raw_protocol.get("held_out_test_policy", {})
    if not isinstance(held_out_policy, Mapping):
        raise ProtocolV2WorkflowConfigurationError("held_out_test_policy must be an object.")
    if held_out_policy.get("state") != "not_loaded_or_referenced":
        raise ProtocolV2WorkflowConfigurationError(
            "Protocol must explicitly state that held-out test data are not loaded or referenced."
        )

    budgets = tuple(CandidateBudgetSpec.from_mapping(row) for row in budget_rows)
    budget_ids = tuple(budget.candidate_id for budget in budgets)
    if set(budget_ids) != set(candidate_ids) or len(budget_ids) != len(candidate_ids):
        raise ProtocolV2WorkflowConfigurationError(
            "Candidate budget table must contain exactly one row per candidate."
        )

    catboost_policy = raw_protocol.get("catboost_runtime_policy", {})
    if not isinstance(catboost_policy, Mapping):
        raise ProtocolV2WorkflowConfigurationError("catboost_runtime_policy must be an object.")
    if catboost_policy.get("candidate_id") != CANDIDATE_CATBOOST:
        raise ProtocolV2WorkflowConfigurationError("CatBoost runtime policy must name C19.")
    if "runtime_evidence_only" not in str(catboost_policy.get("evidence_role", "")):
        raise ProtocolV2WorkflowConfigurationError(
            "CatBoost calibration evidence must remain labeled as runtime-only."
        )

    try:
        outer_n_splits = int(outer_cv["n_splits"])
        outer_n_repeats = int(outer_cv["n_repeats"])
        stage_a_n_splits = int(stage_a_cv["n_splits"])
        stage_b_n_splits = int(stage_b_cv["n_splits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolV2WorkflowConfigurationError(
            "Outer, Stage-A, and Stage-B CV declarations must contain integer n_splits/repeats."
        ) from exc
    for name, value, minimum in (
        ("outer_n_splits", outer_n_splits, 2),
        ("outer_n_repeats", outer_n_repeats, 1),
        ("stage_a_n_splits", stage_a_n_splits, 2),
        ("stage_b_n_splits", stage_b_n_splits, 2),
    ):
        if value < minimum:
            raise ProtocolV2WorkflowConfigurationError(
                f"{name} must be at least {minimum}."
            )
    if primary_metric != "average_precision":
        raise ProtocolV2WorkflowConfigurationError(
            "Protocol-v2 base comparison must use average_precision as primary metric."
        )

    return ProtocolV2BaseWorkflowSpec(
        protocol_path=path,
        raw_protocol=raw_protocol,
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        freeze_state=freeze_state,
        is_frozen=is_frozen,
        evidence_role=evidence_role,
        candidate_ids=candidate_ids,
        deferred_candidate_ids=deferred_candidate_ids,
        outer_n_splits=outer_n_splits,
        outer_n_repeats=outer_n_repeats,
        stage_a_n_splits=stage_a_n_splits,
        stage_b_n_splits=stage_b_n_splits,
        primary_metric=primary_metric,
        candidate_budgets=budgets,
    )


def parse_protocol_v2_workflow_arguments(
    argv: Sequence[str] | None = None,
) -> ProtocolV2WorkflowArguments:
    """Parse execution controls for the protocol-v2 base scaffold."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or run the protocol-v2 base-comparison scaffold. Dry-run mode "
            "creates no artifact directories and fits no models."
        )
    )
    parser.add_argument("--run-id", default="protocol_v2_base_comparison_v1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stop-after-completed", type=int, default=None)
    parser.add_argument(
        "--confirm-official-base-comparison",
        action="store_true",
        help=(
            "Required for any non-dry-run execution after the protocol declaration is "
            "explicitly frozen."
        ),
    )
    namespace = parser.parse_args(argv)
    arguments = ProtocolV2WorkflowArguments(
        run_id=str(namespace.run_id),
        max_workers=int(namespace.max_workers),
        dry_run=bool(namespace.dry_run),
        resume=bool(namespace.resume),
        retry_failed=bool(namespace.retry_failed),
        stop_after_completed=(
            None
            if namespace.stop_after_completed is None
            else int(namespace.stop_after_completed)
        ),
        confirm_official_base_comparison=bool(
            namespace.confirm_official_base_comparison
        ),
    )
    validate_protocol_v2_workflow_arguments(arguments)
    return arguments


def validate_protocol_v2_workflow_arguments(arguments: ProtocolV2WorkflowArguments) -> None:
    """Reject unsafe operational controls."""
    if not arguments.run_id.strip():
        raise ProtocolV2WorkflowConfigurationError("--run-id must not be empty.")
    if arguments.max_workers < 1:
        raise ProtocolV2WorkflowConfigurationError("--max-workers must be at least one.")
    if arguments.retry_failed and not arguments.resume:
        raise ProtocolV2WorkflowConfigurationError("--retry-failed requires --resume.")
    if arguments.stop_after_completed is not None:
        if arguments.stop_after_completed < 1:
            raise ProtocolV2WorkflowConfigurationError(
                "--stop-after-completed must be positive."
            )
        if arguments.max_workers != 1:
            raise ProtocolV2WorkflowConfigurationError(
                "--stop-after-completed requires --max-workers 1."
            )


def source_provenance(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Return checked-out source identity and working-tree cleanliness."""
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


def make_protocol_v2_experiment_protocol(
    spec: ProtocolV2BaseWorkflowSpec,
    provenance: Mapping[str, object],
) -> ExperimentProtocol:
    """Build immutable manifest metadata for an official base-comparison run."""
    workflow_role = (
        "fast-completion pipeline scaffold"
        if spec.evidence_role == FAST_COMPLETION_EVIDENCE_ROLE
        else "official protocol-v2 base-comparison scaffold"
    )
    return ExperimentProtocol(
        protocol_id=spec.protocol_id,
        version=spec.protocol_version,
        candidate_ids=spec.candidate_ids,
        primary_metric=spec.primary_metric,
        outer_n_splits=spec.outer_n_splits,
        outer_n_repeats=spec.outer_n_repeats,
        inner_n_splits=spec.stage_a_n_splits,
        random_state=RANDOM_STATE,
        metadata={
            "protocol_declaration_path": str(
                spec.protocol_path.relative_to(PROJECT_ROOT)
            ),
            "freeze_state": spec.freeze_state,
            "is_frozen": spec.is_frozen,
            "evidence_role": spec.evidence_role,
            "workflow_role": workflow_role,
            "development_data_scope": "all rows in data/processed/train.csv only",
            "held_out_test_set_policy": "not loaded or referenced",
            "fast_completion_warning": spec.raw_protocol.get(
                "fast_completion_warning",
                "",
            ),
            "deferred_candidate_ids": list(spec.deferred_candidate_ids),
            "admission_smoke_evidence_role": "non-selection implementation evidence only",
            "search_budget_calibration_evidence_role": (
                "non-selection runtime evidence only"
            ),
            "stage_a_n_splits": spec.stage_a_n_splits,
            "stage_b_n_splits": spec.stage_b_n_splits,
            "candidate_budget_table": [
                budget.to_dict() for budget in spec.candidate_budgets
            ],
            "catboost_runtime_policy": dict(
                spec.raw_protocol.get("catboost_runtime_policy", {})
            ),
            "downstream_stages": list(spec.raw_protocol.get("downstream_stages", ())),
            "source_provenance": dict(provenance),
        },
    )


def make_protocol_v2_tasks(
    *,
    spec: ProtocolV2BaseWorkflowSpec,
    run_directory: Path,
    y,
    full_training_positions: np.ndarray,
    protocol_fingerprint: str,
) -> list[ExperimentTask]:
    """Create deterministic per-candidate budgeted outer tasks."""
    outer_splits = make_repeated_stratified_outer_splits(
        y,
        n_splits=spec.outer_n_splits,
        n_repeats=spec.outer_n_repeats,
        random_state=RANDOM_STATE,
    )
    budgets = spec.budget_by_candidate
    tasks: list[ExperimentTask] = []
    for candidate_id in spec.candidate_ids:
        budget = budgets[candidate_id]
        for split in outer_splits:
            task_seed = derive_seed(
                RANDOM_STATE,
                spec.protocol_id,
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
                "stage_a_n_trials": int(budget.stage_a_n_trials),
                "confirmation_top_k": int(budget.confirmation_top_k),
                "search_profile": str(budget.search_profile),
                "budget_lane": str(budget.budget_lane),
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


def make_protocol_v2_environment_fingerprint() -> dict[str, object]:
    """Record package versions that can affect official base-comparison routes."""
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


def summarize_budget_lanes(spec: ProtocolV2BaseWorkflowSpec) -> dict[str, int]:
    """Return candidate counts by budget lane."""
    lanes: dict[str, int] = {}
    for budget in spec.candidate_budgets:
        lanes[budget.budget_lane] = lanes.get(budget.budget_lane, 0) + 1
    return dict(sorted(lanes.items()))


def print_protocol_v2_plan(
    *,
    spec: ProtocolV2BaseWorkflowSpec,
    arguments: ProtocolV2WorkflowArguments,
    n_rows: int,
    tasks: Sequence[ExperimentTask],
) -> None:
    """Print the dry-run or execution plan without presenting model evidence."""
    print(f"Protocol declaration: {spec.protocol_path}", flush=True)
    print(f"Protocol ID: {spec.protocol_id}", flush=True)
    print(f"Protocol version: {spec.protocol_version}", flush=True)
    print(f"Evidence role: {spec.evidence_role}", flush=True)
    print(f"freeze_state: {spec.freeze_state}", flush=True)
    print(f"is_frozen: {spec.is_frozen}", flush=True)
    print("Development data only.", flush=True)
    print(
        f"Rows: {n_rows}; outer CV: {spec.outer_n_splits} folds x "
        f"{spec.outer_n_repeats} repeats; Stage A: {spec.stage_a_n_splits} folds; "
        f"Stage B: {spec.stage_b_n_splits} folds.",
        flush=True,
    )
    print(
        f"Candidate count: {len(spec.candidate_ids)}; total task count: {len(tasks)}.",
        flush=True,
    )
    print("Budget lanes:", flush=True)
    for lane, count in summarize_budget_lanes(spec).items():
        print(f"  {lane}: {count} candidates", flush=True)
    stage_a_trials = sorted({budget.stage_a_n_trials for budget in spec.candidate_budgets})
    top_k_values = sorted({budget.confirmation_top_k for budget in spec.candidate_budgets})
    print(f"Stage-A trials per task: {stage_a_trials}", flush=True)
    print(f"Stage-B top-K per task: {top_k_values}", flush=True)
    catboost_budget = spec.budget_by_candidate[CANDIDATE_CATBOOST]
    print(
        "CatBoost policy: "
        f"profile={catboost_budget.search_profile}, "
        f"Stage-A trials={catboost_budget.stage_a_n_trials}, "
        f"Stage-B top-K={catboost_budget.confirmation_top_k}; "
        "runtime-limited, not excluded.",
        flush=True,
    )
    print(f"Run ID: {arguments.run_id}", flush=True)
    if arguments.dry_run:
        print(
            "Dry-run completed without creating artifact directories or fitting models.",
            flush=True,
        )


def _open_store(
    *,
    artifacts_root: Path,
    run_id: str,
    resume: bool,
    protocol: ExperimentProtocol,
    data_fingerprint: Mapping[str, object],
    environment_fingerprint: Mapping[str, object],
) -> ExperimentStore:
    """Create or resume a protocol-compatible official run."""
    if resume:
        return ExperimentStore.open_for_resume(
            artifacts_root=artifacts_root,
            run_id=run_id,
            protocol_fingerprint=protocol.fingerprint,
            data_fingerprint_sha256=str(data_fingerprint["sha256"]),
        )
    return ExperimentStore.create(
        artifacts_root=artifacts_root,
        run_id=run_id,
        protocol_payload=protocol.to_dict(),
        protocol_fingerprint=protocol.fingerprint,
        data_fingerprint=dict(data_fingerprint),
        environment_fingerprint=dict(environment_fingerprint),
    )


def run_declared_final_comparison_workflow(
    *,
    spec: ProtocolV2BaseWorkflowSpec,
    arguments: ProtocolV2WorkflowArguments,
    confirmation_flag_name: str,
    confirmation_granted: bool,
) -> None:
    """Inspect or execute one validated final-comparison protocol declaration."""
    if not arguments.dry_run:
        if not spec.is_execution_frozen:
            raise ProtocolV2WorkflowConfigurationError(
                "Refusing final-comparison workflow execution because the protocol is "
                f"not frozen: freeze_state={spec.freeze_state!r}, "
                f"is_frozen={spec.is_frozen!r}. Use --dry-run for plan inspection."
            )
        if not confirmation_granted:
            raise ProtocolV2WorkflowConfigurationError(
                f"Non-dry-run final-comparison workflow requires {confirmation_flag_name}."
            )

    train_df = load_train_data()
    X_all, y_all = split_features_target(train_df)
    full_training_positions = np.arange(len(X_all), dtype=np.int64)
    provenance = source_provenance()
    if (
        not arguments.dry_run
        and provenance.get("working_tree_clean") is False
    ):
        raise ProtocolV2WorkflowConfigurationError(
            "Refusing official base-comparison execution from a dirty worktree."
        )

    protocol = make_protocol_v2_experiment_protocol(spec, provenance)
    tasks = make_protocol_v2_tasks(
        spec=spec,
        run_directory=DEFAULT_ARTIFACTS_ROOT / arguments.run_id,
        y=y_all,
        full_training_positions=full_training_positions,
        protocol_fingerprint=protocol.fingerprint,
    )
    print_protocol_v2_plan(
        spec=spec,
        arguments=arguments,
        n_rows=len(X_all),
        tasks=tasks,
    )
    if arguments.dry_run:
        return

    data_fingerprint = make_dataframe_fingerprint(X_all, y_all)
    environment_fingerprint = make_protocol_v2_environment_fingerprint()
    store = _open_store(
        artifacts_root=DEFAULT_ARTIFACTS_ROOT,
        run_id=arguments.run_id,
        resume=arguments.resume,
        protocol=protocol,
        data_fingerprint=data_fingerprint,
        environment_fingerprint=environment_fingerprint,
    )
    try:
        clear_graceful_stop_request(store.run_directory)
        event_callback = make_workflow_event_callback(store.run_directory)
        event_callback(
            "run_resumed" if arguments.resume else "run_started",
            None,
            {
                "run_id": arguments.run_id,
                "task_total": len(tasks),
                "worker_capacity": arguments.max_workers,
                "workflow_id": spec.protocol_id,
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
        store.validate_completed_artifacts()
        print("Execution summary:", flush=True)
        for key in ("submitted", "completed", "skipped", "failed", "interrupted", "paused"):
            print(f"  {key}: {int(summary.get(key, 0))}", flush=True)
        print(f"Run directory: {store.run_directory}", flush=True)
    finally:
        store.close()


def run_protocol_v2_base_workflow(argv: Sequence[str] | None = None) -> None:
    """Inspect or execute the official protocol-v2 base-comparison scaffold."""
    arguments = parse_protocol_v2_workflow_arguments(argv)
    spec = load_protocol_v2_base_spec()
    run_declared_final_comparison_workflow(
        spec=spec,
        arguments=arguments,
        confirmation_flag_name="--confirm-official-base-comparison",
        confirmation_granted=arguments.confirm_official_base_comparison,
    )
