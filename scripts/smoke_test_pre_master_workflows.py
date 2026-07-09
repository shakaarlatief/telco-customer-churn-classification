"""Fast structural smoke test for the pre-master workflow declarations.

The test never creates an experiment run, loads the held-out test set, or fits a model.
It verifies that the admission and calibration commands declare the intended candidate
coverage and budgets, and that their deterministic task plans are structurally valid on
a small synthetic label vector.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_CATBOOST,
    CANDIDATE_DECISION_TREE,
    CANDIDATE_FT_TRANSFORMER,
    CANDIDATE_HIST_GRADIENT_BOOSTING,
    CANDIDATE_LIGHTGBM,
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_MLP,
    CANDIDATE_RANDOM_FOREST,
    CANDIDATE_RBF_SVM,
    CANDIDATE_RIDGE_CLASSIFIER,
    CANDIDATE_TABM,
    CANDIDATE_TABNET,
    CANDIDATE_XGBOOST,
    INITIAL_CANDIDATE_REGISTRY,
    validate_candidate_registry,
)
from telco_churn.pre_master_workflows import (  # noqa: E402
    DEFERRED_CANDIDATE_IDS,
    IMPLEMENTED_CANDIDATE_UNIVERSE,
    MASTER_ADMISSION_STATE,
    PROTOCOL_V2_STATE,
    PreMasterWorkflowConfigurationError,
    make_workflow_protocol,
    make_workflow_tasks,
    parse_workflow_arguments,
    validate_spec_candidate_ids,
    validate_workflow_arguments,
)


def load_script_module(path: Path, module_name: str):
    """Import a command script without triggering its ``__main__`` execution block."""
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"Cannot create import specification for {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def assert_task_plan(spec, expected_candidates: int, expected_trials: int, expected_top_k: int) -> None:
    """Assert task counts, payload budgets, split disjointness, and deterministic identity."""
    validate_spec_candidate_ids(spec)
    y = pd.Series(np.tile([0, 1], 20), name="Churn")
    positions = np.arange(len(y), dtype=np.int64)
    protocol = make_workflow_protocol(
        spec,
        {"git_revision": "smoke", "working_tree_clean": True},
    )
    metadata = protocol.metadata
    if metadata.get("workflow_id") != spec.workflow_id:
        raise AssertionError("Workflow identity must be persisted in immutable protocol metadata.")
    if metadata.get("seed_namespace") != spec.seed_namespace:
        raise AssertionError("Seed namespace must be persisted in immutable protocol metadata.")
    if metadata.get("implemented_candidate_universe") != IMPLEMENTED_CANDIDATE_UNIVERSE:
        raise AssertionError("Protocol metadata must record the implemented candidate universe.")
    if tuple(metadata.get("deferred_candidate_ids", ())) != DEFERRED_CANDIDATE_IDS:
        raise AssertionError("Protocol metadata must record deferred C27/C28 candidates.")
    if metadata.get("master_admission_state") != MASTER_ADMISSION_STATE:
        raise AssertionError("Protocol metadata must record that no candidate is master-admitted.")
    if metadata.get("protocol_v2_state") != PROTOCOL_V2_STATE:
        raise AssertionError("Protocol metadata must record that protocol v2 is not frozen.")
    if metadata.get("held_out_test_set_policy") != "not loaded or referenced":
        raise AssertionError("Protocol metadata must prohibit held-out test use.")
    altered_seed_spec = replace(
        spec,
        seed_namespace=spec.seed_namespace + "_altered",
    )
    altered_seed_protocol = make_workflow_protocol(
        altered_seed_spec,
        {"git_revision": "smoke", "working_tree_clean": True},
    )
    if altered_seed_protocol.fingerprint == protocol.fingerprint:
        raise AssertionError("Changing the task-seed namespace must change the protocol fingerprint.")
    with tempfile.TemporaryDirectory(prefix="telco-pre-master-smoke-") as temporary_root:
        run_directory = Path(temporary_root) / "run"
        tasks = make_workflow_tasks(
            spec=spec,
            run_directory=run_directory,
            y=y,
            full_training_positions=positions,
            protocol_fingerprint=protocol.fingerprint,
        )
        duplicate = make_workflow_tasks(
            spec=spec,
            run_directory=run_directory,
            y=y,
            full_training_positions=positions,
            protocol_fingerprint=protocol.fingerprint,
        )

    expected_task_count = expected_candidates * spec.outer_n_splits * spec.outer_n_repeats
    if len(tasks) != expected_task_count:
        raise AssertionError(f"Expected {expected_task_count} tasks, received {len(tasks)}.")
    if [task.task_key for task in tasks] != [task.task_key for task in duplicate]:
        raise AssertionError("Task keys are not deterministic across identical task-plan calls.")
    if len({task.task_key for task in tasks}) != len(tasks):
        raise AssertionError("Task keys must be unique within a workflow.")

    for task in tasks:
        payload = task.payload
        if payload["stage_a_n_trials"] != expected_trials:
            raise AssertionError(f"Unexpected Stage-A budget for {task.task_key}.")
        if payload["confirmation_top_k"] != expected_top_k:
            raise AssertionError(f"Unexpected Stage-B top-K budget for {task.task_key}.")
        if payload["study_name"] != task.task_key:
            raise AssertionError(f"Study identity mismatch for {task.task_key}.")
        train = set(payload["outer_train_indices"])
        validation = set(payload["outer_validation_indices"])
        if train & validation:
            raise AssertionError(f"Outer train/validation overlap for {task.task_key}.")
        if train | validation != set(range(len(y))):
            raise AssertionError(f"Outer split does not cover all synthetic rows for {task.task_key}.")

    arguments = parse_workflow_arguments(spec, ["--run-id", "smoke_plan", "--dry-run"])
    validate_workflow_arguments(arguments)


def assert_audit_registry_result_identity() -> None:
    """Verify that the generic audit retains every identity field needed for a result check."""
    audit = load_script_module(
        PROJECT_ROOT / "scripts" / "audit_final_comparison_run.py",
        "final_comparison_audit_smoke",
    )
    task_key = "c01_ridge_classifier__r00__f00"
    candidate_id = "C01_RIDGE_CLASSIFIER"
    split_hash = "audit-smoke-split"
    result = {
        "candidate_id": candidate_id,
        "outer_repeat_index": 0,
        "outer_fold_index": 0,
        "split_hash": split_hash,
        "inner_search": {
            "stage_a_completed_trials": 3,
            "stage_b_records": [{}, {}],
        },
        "selected_parameters": {},
        "timing_seconds": {},
    }
    raw_result = json.dumps({"result": result}).encode("utf-8")
    registered_task = {
        "task_key": task_key,
        "candidate_id": candidate_id,
        "repeat_index": 0,
        "fold_index": 0,
        "split_hash": split_hash,
        "payload": {
            "stage_a_n_trials": 3,
            "confirmation_top_k": 2,
        },
    }

    with tempfile.TemporaryDirectory(prefix="telco-audit-identity-smoke-") as temporary_root:
        run_directory = Path(temporary_root)
        results_directory = run_directory / "results"
        results_directory.mkdir()
        result_relative_path = Path("results") / f"{task_key}.json"
        (run_directory / result_relative_path).write_bytes(raw_result)

        with closing(sqlite3.connect(run_directory / "task_registry.sqlite")) as connection:
            connection.execute(
                """
                CREATE TABLE tasks (
                    task_key TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    repeat_index INTEGER NOT NULL,
                    fold_index INTEGER NOT NULL,
                    split_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    started_at TEXT,
                    heartbeat_at TEXT,
                    completed_at TEXT,
                    error_text TEXT,
                    result_path TEXT,
                    result_sha256 TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT INTO tasks (
                    task_key, candidate_id, repeat_index, fold_index, split_hash, payload_json,
                    status, attempts, started_at, heartbeat_at, completed_at, error_text,
                    result_path, result_sha256
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_key,
                    candidate_id,
                    0,
                    0,
                    split_hash,
                    json.dumps(registered_task, sort_keys=True),
                    "completed",
                    1,
                    "2026-07-04T10:00:00+00:00",
                    None,
                    "2026-07-04T10:00:01+00:00",
                    None,
                    str(result_relative_path),
                    sha256(raw_result).hexdigest(),
                ),
            )
            connection.commit()

        rows = audit.load_registry_rows(run_directory)
        if len(rows) != 1 or rows[0].get("split_hash") != split_hash:
            raise AssertionError(
                "The audit registry query must retain split_hash for result identity checks."
            )
        entries, problems = audit.load_completed_results(run_directory, rows)
        if len(entries) != 1 or problems:
            raise AssertionError(
                "The audit must accept a checksum-verified result with matching registry identity."
            )


def main() -> None:
    """Run deterministic no-fit checks for both new workflow commands."""
    validate_candidate_registry()
    assert_audit_registry_result_identity()
    admission = load_script_module(
        PROJECT_ROOT / "scripts" / "run_final_comparison_admission_smoke.py",
        "admission_workflow_smoke",
    )
    calibration = load_script_module(
        PROJECT_ROOT / "scripts" / "run_final_comparison_search_budget_calibration.py",
        "calibration_workflow_smoke",
    )

    registry_ids = tuple(definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY)
    if admission.ADMISSION_CANDIDATE_IDS != registry_ids:
        raise AssertionError("Admission workflow must cover exactly the current implemented registry.")
    required_advanced_ids = {
        CANDIDATE_TABNET,
        CANDIDATE_FT_TRANSFORMER,
        CANDIDATE_TABM,
    }
    if not required_advanced_ids.issubset(set(admission.ADMISSION_CANDIDATE_IDS)):
        raise AssertionError("Admission workflow must include C24, C25, and C26.")
    for spec in (admission.WORKFLOW_SPEC, calibration.WORKFLOW_SPEC):
        if set(spec.candidate_ids) & set(DEFERRED_CANDIDATE_IDS):
            raise AssertionError("Deferred C27/C28 candidates must not appear in pre-master specs.")
    rejected_spec = replace(
        admission.WORKFLOW_SPEC,
        candidate_ids=admission.WORKFLOW_SPEC.candidate_ids + (DEFERRED_CANDIDATE_IDS[0],),
    )
    try:
        validate_spec_candidate_ids(rejected_spec)
    except PreMasterWorkflowConfigurationError:
        pass
    else:
        raise AssertionError("Deferred candidates must be rejected before workflow execution.")

    expected_calibration_ids = (
        CANDIDATE_RIDGE_CLASSIFIER,
        CANDIDATE_LOGISTIC_REGRESSION,
        CANDIDATE_DECISION_TREE,
        CANDIDATE_RANDOM_FOREST,
        CANDIDATE_HIST_GRADIENT_BOOSTING,
        CANDIDATE_XGBOOST,
        CANDIDATE_LIGHTGBM,
        CANDIDATE_CATBOOST,
        CANDIDATE_RBF_SVM,
        CANDIDATE_MLP,
    )
    if calibration.CALIBRATION_CANDIDATE_IDS != expected_calibration_ids:
        raise AssertionError("Calibration must remain the declared 10-candidate subset.")
    assert_task_plan(admission.WORKFLOW_SPEC, len(registry_ids), 3, 2)
    assert_task_plan(calibration.WORKFLOW_SPEC, 10, 36, 5)

    if calibration.WORKFLOW_SPEC.search_profile != "full":
        raise AssertionError("Search-budget calibration must use the full search profile.")
    if admission.WORKFLOW_SPEC.search_profile != "smoke":
        raise AssertionError("Admission smoke must use the bounded smoke search profile.")
    print("Pre-master workflow structural smoke test passed.")


if __name__ == "__main__":
    main()
