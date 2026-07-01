"""Task workers for the initial real nested-HPO smoke comparison.

The task worker is intentionally top-level and importable. Windows process pools use
spawn semantics, so worker functions must not be local notebook closures or nested
script functions.

This phase implements one small real nested-HPO task path. It proves the connection
between durable outer-task state, persistent Optuna studies, fold-safe candidate
pipelines, score extraction, and training-only outer validation before the full registry
is added.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from telco_churn.candidates import build_candidate_pipeline, get_candidate_definition
from telco_churn.data import load_train_data, split_features_target
from telco_churn.experiment_splits import derive_seed
from telco_churn.experiment_store import ExperimentTask
from telco_churn.hpo import (
    PersistentStudyConfig,
    extract_continuous_scores,
    make_study_contract_fingerprint,
    release_persistent_study_resources,
    run_stage_a_optuna_search,
    run_two_stage_optuna_search,
)


class TaskPayloadError(ValueError):
    """Raised when a final-comparison task has an invalid or incomplete payload."""


def _required_payload(task: ExperimentTask, name: str) -> Any:
    """Return one required payload value or raise an actionable task error."""
    if name not in task.payload:
        raise TaskPayloadError(
            f"Task {task.task_key!r} is missing required payload field {name!r}."
        )
    return task.payload[name]


def _load_task_data(sample_positions: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Load a deterministic training-only sample for an outer task worker."""
    train_df = load_train_data()
    X_all, y_all = split_features_target(train_df)

    positions = np.asarray(sample_positions, dtype=np.int64)
    if positions.ndim != 1 or positions.size == 0:
        raise TaskPayloadError("sample_positions must be a non-empty one-dimensional list.")
    if np.any(positions < 0) or np.any(positions >= len(X_all)):
        raise TaskPayloadError("sample_positions contains an out-of-range training row.")

    X = X_all.iloc[positions].reset_index(drop=True)
    y = y_all.iloc[positions].reset_index(drop=True)
    return X, y


def _probability_metrics_available(score_kind: str) -> bool:
    """Return whether raw scores are valid probabilities."""
    return score_kind == "probability"


def _outer_metrics(
    *,
    y_true: pd.Series,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    score_kind: str,
) -> dict[str, float | None]:
    """Compute ranking, probability, and default-threshold metrics."""
    metrics: dict[str, float | None] = {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": None,
        "log_loss": None,
    }

    if _probability_metrics_available(score_kind):
        metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
        metrics["log_loss"] = float(log_loss(y_true, y_score, labels=[0, 1]))

    return metrics


def _write_once_marker(marker_path: Path) -> bool:
    """Atomically create a marker and return whether this call created it."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_descriptor = marker_path.open("x", encoding="utf-8")
    except FileExistsError:
        return False

    with file_descriptor:
        file_descriptor.write("simulated HPO interruption consumed\n")
        file_descriptor.flush()
    return True


def run_nested_hpo_outer_task(task: ExperimentTask) -> Mapping[str, Any]:
    """Run one complete training-only outer nested-HPO task.

    The payload specifies only training-row positions and outer split positions. The
    worker loads ``train.csv`` through the standard project data loader and never
    imports, reads, or references the held-out test set.

    A phase-2 smoke-only interruption hook can stop a task after one persistent
    Stage-A Optuna trial. The outer task is then marked failed by the coordinator;
    a retry resumes the same study and completes the remaining trial budget.
    """
    task_kind = _required_payload(task, "task_kind")
    if task_kind != "nested_hpo_outer_v1":
        raise TaskPayloadError(
            f"Unsupported task_kind {task_kind!r}; expected 'nested_hpo_outer_v1'."
        )

    candidate_id = str(task.candidate_id)
    definition = get_candidate_definition(candidate_id)
    sample_positions = list(_required_payload(task, "sample_positions"))
    outer_train_indices = np.asarray(
        _required_payload(task, "outer_train_indices"),
        dtype=np.int64,
    )
    outer_validation_indices = np.asarray(
        _required_payload(task, "outer_validation_indices"),
        dtype=np.int64,
    )
    stage_a_n_splits = int(_required_payload(task, "stage_a_n_splits"))
    stage_b_n_splits = int(_required_payload(task, "stage_b_n_splits"))
    stage_a_n_trials = int(_required_payload(task, "stage_a_n_trials"))
    confirmation_top_k = int(_required_payload(task, "confirmation_top_k"))
    search_profile = str(_required_payload(task, "search_profile"))
    study_database_path = Path(str(_required_payload(task, "study_database_path")))
    study_name = str(_required_payload(task, "study_name"))
    task_seed = int(_required_payload(task, "task_seed"))

    X, y = _load_task_data(sample_positions)
    if np.intersect1d(outer_train_indices, outer_validation_indices).size:
        raise TaskPayloadError("Outer train and validation positions overlap.")
    if (
        np.any(outer_train_indices < 0)
        or np.any(outer_validation_indices < 0)
        or np.any(outer_train_indices >= len(X))
        or np.any(outer_validation_indices >= len(X))
    ):
        raise TaskPayloadError("Outer split indices are outside the sampled training data.")

    X_outer_train = X.iloc[outer_train_indices].reset_index(drop=True)
    y_outer_train = y.iloc[outer_train_indices].reset_index(drop=True)
    X_outer_validation = X.iloc[outer_validation_indices].reset_index(drop=True)
    y_outer_validation = y.iloc[outer_validation_indices].reset_index(drop=True)

    stage_a_seed = derive_seed(task_seed, "stage_a_cv")
    stage_b_seed = derive_seed(task_seed, "stage_b_cv")
    stage_a_cv = StratifiedKFold(
        n_splits=stage_a_n_splits,
        shuffle=True,
        random_state=stage_a_seed,
    )
    stage_b_cv = StratifiedKFold(
        n_splits=stage_b_n_splits,
        shuffle=True,
        random_state=stage_b_seed,
    )
    task_fingerprint = make_study_contract_fingerprint(
        candidate_id=candidate_id,
        task_key=task.task_key,
        split_hash=task.split_hash,
        primary_metric="average_precision",
        search_profile=search_profile,
        stage_a_n_splits=stage_a_n_splits,
        stage_b_n_splits=stage_b_n_splits,
        confirmation_top_k=confirmation_top_k,
    )
    study_config = PersistentStudyConfig(
        study_name=study_name,
        database_path=study_database_path,
        candidate_id=candidate_id,
        task_key=task.task_key,
        task_fingerprint=task_fingerprint,
        random_state=task_seed,
        n_startup_trials=min(10, stage_a_n_trials),
    )

    simulate_marker = task.payload.get("simulate_hpo_interrupt_marker_path")
    if simulate_marker is not None:
        marker_path = Path(str(simulate_marker))
        if _write_once_marker(marker_path):
            # One persisted Stage-A trial demonstrates that a retry continues an
            # existing RDB study rather than restarting at trial zero.
            interrupted_study = run_stage_a_optuna_search(
                config=study_config,
                X=X_outer_train,
                y=y_outer_train,
                stage_a_cv=stage_a_cv,
                n_trials_target=1,
                search_profile=search_profile,
            )
            release_persistent_study_resources(interrupted_study)
            raise RuntimeError(
                "Simulated phase-2 HPO interruption after one persisted Stage-A trial."
            )

    search_started = time.perf_counter()
    search_result = run_two_stage_optuna_search(
        config=study_config,
        X=X_outer_train,
        y=y_outer_train,
        stage_a_cv=stage_a_cv,
        stage_b_cv=stage_b_cv,
        n_trials_target=stage_a_n_trials,
        confirmation_top_k=confirmation_top_k,
        search_profile=search_profile,
    )
    search_seconds = time.perf_counter() - search_started

    final_seed = derive_seed(task_seed, "outer_final_fit")
    estimator = build_candidate_pipeline(
        candidate_id,
        search_result.selected_parameters,
        random_state=final_seed,
    )

    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fitted = clone(estimator).fit(X_outer_train, y_outer_train)
    fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    y_score, observed_score_kind = extract_continuous_scores(fitted, X_outer_validation)
    y_pred = np.asarray(fitted.predict(X_outer_validation), dtype=int)
    prediction_seconds = time.perf_counter() - prediction_started

    if observed_score_kind != definition.score_kind:
        raise RuntimeError(
            f"Candidate {candidate_id} declared {definition.score_kind!r} scores "
            f"but produced {observed_score_kind!r} scores."
        )
    if y_score.shape != (len(y_outer_validation),):
        raise RuntimeError("Outer task returned an invalid continuous-score shape.")
    if y_pred.shape != (len(y_outer_validation),):
        raise RuntimeError("Outer task returned an invalid prediction shape.")
    if not np.isfinite(y_score).all():
        raise RuntimeError("Outer task returned non-finite continuous scores.")

    metrics = _outer_metrics(
        y_true=y_outer_validation,
        y_score=y_score,
        y_pred=y_pred,
        score_kind=observed_score_kind,
    )
    original_validation_positions = np.asarray(sample_positions, dtype=np.int64)[
        outer_validation_indices
    ]

    return {
        "schema_version": "nested_hpo_outer_task_result_v1",
        "candidate_id": candidate_id,
        "candidate_display_name": definition.display_name,
        "score_kind": observed_score_kind,
        "outer_repeat_index": int(task.repeat_index),
        "outer_fold_index": int(task.fold_index),
        "split_hash": task.split_hash,
        "n_outer_train": int(len(y_outer_train)),
        "n_outer_validation": int(len(y_outer_validation)),
        "selected_parameters": dict(search_result.selected_parameters),
        "inner_search": search_result.to_dict(),
        "metrics": metrics,
        "timing_seconds": {
            "inner_search": float(search_seconds),
            "outer_fit": float(fit_seconds),
            "outer_prediction": float(prediction_seconds),
        },
        "warnings": [
            f"{warning.category.__name__}: {warning.message}"
            for warning in caught_warnings
        ],
        "outer_validation_predictions": {
            "training_row_positions": [
                int(position) for position in original_validation_positions
            ],
            "y_true": [int(value) for value in y_outer_validation.to_numpy()],
            "y_score": [float(value) for value in y_score],
            "y_pred": [int(value) for value in y_pred],
        },
    }
