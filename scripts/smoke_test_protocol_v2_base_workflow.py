"""Structural smoke test for the protocol-v2 base-comparison scaffold.

This test does not run experiments, fit models, resume workflows, or touch the held-out
test set. It validates the frozen declaration, the generated task plan, the CatBoost
runtime policy, and the explicit non-dry-run confirmation gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_CATBOOST,
    CANDIDATE_TABM,
    CANDIDATE_TABNET,
    CANDIDATE_FT_TRANSFORMER,
    INITIAL_CANDIDATE_REGISTRY,
    SEARCH_PROFILE_CATBOOST_V2,
    suggest_candidate_parameters,
)
from telco_churn.protocol_v2_workflows import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFERRED_CANDIDATE_IDS,
    ProtocolV2WorkflowConfigurationError,
    load_protocol_v2_base_spec,
    make_protocol_v2_experiment_protocol,
    make_protocol_v2_tasks,
    run_protocol_v2_base_workflow,
)


class CapturingTrial:
    """Minimal Optuna-compatible trial that records suggested ranges."""

    def __init__(self) -> None:
        self.int_ranges: dict[str, tuple[int, int, int]] = {}
        self.float_ranges: dict[str, tuple[float, float, bool]] = {}

    def suggest_categorical(self, name: str, choices):
        if not choices:
            raise AssertionError(f"No choices supplied for {name!r}.")
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int, **kwargs):
        step = int(kwargs.get("step", 1))
        self.int_ranges[name] = (int(low), int(high), step)
        return int(low)

    def suggest_float(self, name: str, low: float, high: float, **kwargs):
        log = bool(kwargs.get("log", False))
        self.float_ranges[name] = (float(low), float(high), log)
        return float(low)


def assert_protocol_declaration_and_tasks() -> None:
    """Verify JSON protocol metadata and generated task payloads."""
    spec = load_protocol_v2_base_spec()
    if spec.freeze_state != "frozen":
        raise AssertionError("Protocol-v2 declaration must be frozen.")
    if not spec.is_frozen:
        raise AssertionError("Protocol-v2 declaration must set is_frozen=true.")

    registry_ids = tuple(definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY)
    if spec.candidate_ids != registry_ids:
        raise AssertionError("Protocol candidate universe must exactly match C01-C26 registry.")
    if len(spec.candidate_ids) != 26:
        raise AssertionError("Protocol-v2 base comparison must declare 26 implemented candidates.")
    if spec.deferred_candidate_ids != DEFERRED_CANDIDATE_IDS:
        raise AssertionError("C27/C28 must be listed as deferred.")
    if set(spec.candidate_ids) & set(DEFERRED_CANDIDATE_IDS):
        raise AssertionError("Deferred C27/C28 candidates must be excluded.")
    for candidate_id in (CANDIDATE_TABNET, CANDIDATE_FT_TRANSFORMER, CANDIDATE_TABM):
        if candidate_id not in spec.candidate_ids:
            raise AssertionError(f"{candidate_id} must remain in the C01-C26 universe.")

    y = pd.Series(np.tile([0, 1], 60), name="Churn")
    positions = np.arange(len(y), dtype=np.int64)
    protocol = make_protocol_v2_experiment_protocol(
        spec,
        {"git_revision": "smoke", "working_tree_clean": True},
    )
    metadata = protocol.metadata
    if metadata.get("freeze_state") != "frozen":
        raise AssertionError("Protocol metadata must preserve the frozen state.")
    if metadata.get("is_frozen") is not True:
        raise AssertionError("Protocol metadata must preserve is_frozen=true.")
    if metadata.get("held_out_test_set_policy") != "not loaded or referenced":
        raise AssertionError("Protocol metadata must prohibit held-out test access.")
    if "non-selection" not in str(metadata.get("admission_smoke_evidence_role")):
        raise AssertionError("Admission smoke must remain non-selection evidence.")
    if "non-selection" not in str(metadata.get("search_budget_calibration_evidence_role")):
        raise AssertionError("Paused calibration must remain non-selection evidence.")

    tasks = make_protocol_v2_tasks(
        spec=spec,
        run_directory=DEFAULT_ARTIFACTS_ROOT / "protocol_v2_base_smoke_plan",
        y=y,
        full_training_positions=positions,
        protocol_fingerprint=protocol.fingerprint,
    )
    expected_task_count = 26 * 5 * 3
    if len(tasks) != expected_task_count:
        raise AssertionError(f"Expected {expected_task_count} tasks, received {len(tasks)}.")
    if len({task.task_key for task in tasks}) != len(tasks):
        raise AssertionError("Task keys must be unique.")

    budget_by_candidate = spec.budget_by_candidate
    for task in tasks:
        budget = budget_by_candidate[task.candidate_id]
        payload = task.payload
        if int(payload["stage_a_n_trials"]) != budget.stage_a_n_trials:
            raise AssertionError(f"Stage-A budget mismatch for {task.task_key}.")
        if int(payload["confirmation_top_k"]) != budget.confirmation_top_k:
            raise AssertionError(f"Stage-B top-K mismatch for {task.task_key}.")
        if payload["search_profile"] != budget.search_profile:
            raise AssertionError(f"Search profile mismatch for {task.task_key}.")
        if payload["search_profile"] == SEARCH_PROFILE_CATBOOST_V2:
            if task.candidate_id != CANDIDATE_CATBOOST:
                raise AssertionError("Only C19 may use catboost_v2.")
            if payload["stage_a_n_trials"] != 8 or payload["confirmation_top_k"] != 2:
                raise AssertionError("C19 catboost_v2 budget must be 8 trials/top 2.")

    non_catboost_profiles = {
        task.payload["search_profile"]
        for task in tasks
        if task.candidate_id != CANDIDATE_CATBOOST
    }
    if SEARCH_PROFILE_CATBOOST_V2 in non_catboost_profiles:
        raise AssertionError("No non-C19 task may use catboost_v2.")


def assert_catboost_v2_search_space() -> None:
    """Verify C19's runtime-limited search profile and profile rejection elsewhere."""
    trial = CapturingTrial()
    parameters = suggest_candidate_parameters(
        trial,
        candidate_id=CANDIDATE_CATBOOST,
        profile=SEARCH_PROFILE_CATBOOST_V2,
    )
    if trial.int_ranges.get("iterations") != (100, 600, 50):
        raise AssertionError("catboost_v2 iterations must be 100..600 step 50.")
    if trial.int_ranges.get("depth") != (3, 6, 1):
        raise AssertionError("catboost_v2 depth must be 3..6.")
    if trial.float_ranges.get("learning_rate") != (0.003, 0.2, True):
        raise AssertionError("catboost_v2 learning_rate must be log 0.003..0.2.")
    if trial.float_ranges.get("l2_leaf_reg") != (0.001, 100.0, True):
        raise AssertionError("catboost_v2 l2_leaf_reg must be log 1e-3..1e2.")
    if int(parameters["iterations"]) != 100:
        raise AssertionError("Deterministic catboost_v2 trial should choose 100 iterations.")

    try:
        suggest_candidate_parameters(
            CapturingTrial(),
            candidate_id="C01_RIDGE_CLASSIFIER",
            profile=SEARCH_PROFILE_CATBOOST_V2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("catboost_v2 must be rejected for non-C19 candidates.")


def assert_execution_guards_and_dry_run() -> None:
    """Verify non-dry-run confirmation is required and dry-run creates no artifacts."""
    blocked_run_id = "protocol_v2_base_smoke_blocked_non_dry_run"
    try:
        run_protocol_v2_base_workflow(["--run-id", blocked_run_id, "--max-workers", "1"])
    except ProtocolV2WorkflowConfigurationError as exc:
        if "--confirm-official-base-comparison" not in str(exc):
            raise AssertionError(f"Unexpected non-dry-run refusal: {exc}") from exc
    else:
        raise AssertionError("Non-dry-run execution must require explicit confirmation.")
    if (DEFAULT_ARTIFACTS_ROOT / blocked_run_id).exists():
        raise AssertionError("Blocked non-dry-run must not create artifact directories.")

    dry_run_id = f"protocol_v2_base_smoke_dry_run_{os.getpid()}"
    dry_run_directory = DEFAULT_ARTIFACTS_ROOT / dry_run_id
    if dry_run_directory.exists():
        raise AssertionError(f"Unexpected pre-existing dry-run directory: {dry_run_directory}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_final_comparison_protocol_v2_base.py"),
        "--dry-run",
        "--max-workers",
        "1",
        "--run-id",
        dry_run_id,
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Protocol-v2 dry-run command failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if "Candidate count: 26; total task count: 390." not in result.stdout:
        raise AssertionError("Dry-run output must report 26 candidates and 390 tasks.")
    if "freeze_state: frozen" not in result.stdout:
        raise AssertionError("Dry-run output must report frozen state.")
    if "is_frozen: True" not in result.stdout:
        raise AssertionError("Dry-run output must report is_frozen=True.")
    if dry_run_directory.exists():
        raise AssertionError("Dry-run must not create artifact directories.")


def assert_no_held_out_loader_imports() -> None:
    """Static guard against accidental held-out loading helpers in scaffold code."""
    checked_paths = (
        PROJECT_ROOT / "src" / "telco_churn" / "protocol_v2_workflows.py",
        PROJECT_ROOT / "scripts" / "run_final_comparison_protocol_v2_base.py",
    )
    forbidden = ("load_test_data", "split_test")
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                raise AssertionError(f"{path} must not import or call {token}.")


def main() -> None:
    """Run structural no-fit checks for the protocol-v2 scaffold."""
    assert_protocol_declaration_and_tasks()
    assert_catboost_v2_search_space()
    assert_execution_guards_and_dry_run()
    assert_no_held_out_loader_imports()
    print("Protocol-v2 base workflow structural smoke test passed.")


if __name__ == "__main__":
    main()
