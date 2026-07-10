"""Structural smoke test for the fast-completion final-comparison protocol.

This test does not run experiments, fit models, resume workflows, or touch the held-out
test set. It validates the fast protocol declaration, generated task plan, confirmation
gate, and dry-run side-effect boundary.
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
    INITIAL_CANDIDATE_REGISTRY,
    SEARCH_PROFILE_CATBOOST_V2,
)
from telco_churn.protocol_v2_workflows import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFERRED_CANDIDATE_IDS,
    ProtocolV2WorkflowConfigurationError,
    load_protocol_v2_base_spec,
    make_protocol_v2_experiment_protocol,
    make_protocol_v2_tasks,
)
from run_final_comparison_fast_completion import (  # noqa: E402
    FAST_PROTOCOL_PATH,
    main as run_fast_completion_main,
)


def assert_fast_protocol_tasks() -> None:
    """Verify fast protocol declaration and task payload contracts."""
    spec = load_protocol_v2_base_spec(FAST_PROTOCOL_PATH)
    if spec.protocol_id != "telco_final_comparison_fast_completion_v1":
        raise AssertionError("Fast protocol ID mismatch.")
    if spec.evidence_role != "fast_completion_pipeline_evidence":
        raise AssertionError("Fast protocol must declare fast-completion evidence role.")
    if spec.freeze_state != "frozen" or not spec.is_frozen:
        raise AssertionError("Fast-completion protocol must be frozen before execution.")

    registry_ids = tuple(definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY)
    if spec.candidate_ids != registry_ids:
        raise AssertionError("Fast protocol candidate universe must exactly match C01-C26.")
    if set(spec.candidate_ids) & set(DEFERRED_CANDIDATE_IDS):
        raise AssertionError("C27/C28 must be excluded from the fast candidate universe.")
    if spec.deferred_candidate_ids != DEFERRED_CANDIDATE_IDS:
        raise AssertionError("C27/C28 must be explicitly deferred.")
    if spec.outer_n_splits != 2 or spec.outer_n_repeats != 1:
        raise AssertionError("Fast protocol outer CV must be 2 folds x 1 repeat.")
    if spec.stage_a_n_splits != 2 or spec.stage_b_n_splits != 2:
        raise AssertionError("Fast protocol inner CV must be 2-fold Stage A and Stage B.")

    y = pd.Series(np.tile([0, 1], 40), name="Churn")
    positions = np.arange(len(y), dtype=np.int64)
    protocol = make_protocol_v2_experiment_protocol(
        spec,
        {"git_revision": "smoke", "working_tree_clean": True},
    )
    metadata = protocol.metadata
    if metadata.get("evidence_role") != "fast_completion_pipeline_evidence":
        raise AssertionError("Manifest metadata must preserve fast evidence role.")
    if metadata.get("held_out_test_set_policy") != "not loaded or referenced":
        raise AssertionError("Manifest metadata must prohibit held-out test access.")
    if "not the robust protocol-v2 benchmark" not in str(
        metadata.get("fast_completion_warning", "")
    ):
        raise AssertionError("Manifest metadata must preserve fast-completion warning.")

    tasks = make_protocol_v2_tasks(
        spec=spec,
        run_directory=DEFAULT_ARTIFACTS_ROOT / "fast_completion_smoke_plan",
        y=y,
        full_training_positions=positions,
        protocol_fingerprint=protocol.fingerprint,
    )
    if len(tasks) != 52:
        raise AssertionError(f"Expected 52 fast-completion tasks, received {len(tasks)}.")
    if len({task.task_key for task in tasks}) != len(tasks):
        raise AssertionError("Fast-completion task keys must be unique.")

    for task in tasks:
        payload = task.payload
        if int(payload["stage_a_n_trials"]) != 2:
            raise AssertionError(f"Unexpected Stage-A trials for {task.task_key}.")
        if int(payload["confirmation_top_k"]) != 1:
            raise AssertionError(f"Unexpected Stage-B top-K for {task.task_key}.")
        if task.candidate_id == CANDIDATE_CATBOOST:
            if payload["search_profile"] != SEARCH_PROFILE_CATBOOST_V2:
                raise AssertionError("C19 must use catboost_v2 in fast-completion.")
        elif payload["search_profile"] == SEARCH_PROFILE_CATBOOST_V2:
            raise AssertionError("Only C19 may use catboost_v2 in fast-completion.")


def assert_execution_guard_and_dry_run() -> None:
    """Verify fast non-dry-run confirmation and dry-run side-effect boundary."""
    blocked_run_id = "fast_completion_smoke_blocked_non_dry_run"
    try:
        run_fast_completion_main(["--run-id", blocked_run_id, "--max-workers", "1"])
    except ProtocolV2WorkflowConfigurationError as exc:
        if "--confirm-fast-completion-run" not in str(exc):
            raise AssertionError(f"Unexpected fast non-dry-run refusal: {exc}") from exc
    else:
        raise AssertionError("Fast non-dry-run execution must require explicit confirmation.")
    if (DEFAULT_ARTIFACTS_ROOT / blocked_run_id).exists():
        raise AssertionError("Blocked fast non-dry-run must not create artifacts.")

    dry_run_id = f"fast_completion_smoke_dry_run_{os.getpid()}"
    dry_run_directory = DEFAULT_ARTIFACTS_ROOT / dry_run_id
    if dry_run_directory.exists():
        raise AssertionError(f"Unexpected pre-existing dry-run directory: {dry_run_directory}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_final_comparison_fast_completion.py"),
        "--dry-run",
        "--max-workers",
        "2",
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
            "Fast-completion dry-run command failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    expected_fragments = (
        "Evidence role: fast_completion_pipeline_evidence",
        "Rows: 5634; outer CV: 2 folds x 1 repeats; Stage A: 2 folds; Stage B: 2 folds.",
        "Candidate count: 26; total task count: 52.",
        "Stage-A trials per task: [2]",
        "Stage-B top-K per task: [1]",
        "CatBoost policy: profile=catboost_v2, Stage-A trials=2, Stage-B top-K=1",
    )
    for fragment in expected_fragments:
        if fragment not in result.stdout:
            raise AssertionError(f"Dry-run output missing expected fragment: {fragment!r}")
    if dry_run_directory.exists():
        raise AssertionError("Fast dry-run must not create artifact directories.")


def assert_no_held_out_loader_imports() -> None:
    """Static guard against accidental held-out loading helpers."""
    checked_paths = (
        PROJECT_ROOT / "src" / "telco_churn" / "protocol_v2_workflows.py",
        PROJECT_ROOT / "scripts" / "run_final_comparison_fast_completion.py",
    )
    forbidden = ("load_" + "test_data", "split_" + "test")
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                raise AssertionError(f"{path} must not import or call {token}.")


def main() -> None:
    """Run structural no-fit checks for the fast-completion workflow."""
    assert_fast_protocol_tasks()
    assert_execution_guard_and_dry_run()
    assert_no_held_out_loader_imports()
    print("Fast-completion workflow structural smoke test passed.")


if __name__ == "__main__":
    main()
