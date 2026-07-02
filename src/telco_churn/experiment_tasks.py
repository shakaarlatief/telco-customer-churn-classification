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
from telco_churn.experiment_progress import GracefulStopRequested, TaskProgressReporter
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


def _make_progress_reporter(task: ExperimentTask) -> TaskProgressReporter | None:
    """Create optional worker progress monitoring for runs that provide a directory."""
    run_directory = task.payload.get("run_directory")
    if run_directory is None:
        return None
    return TaskProgressReporter(
        run_directory=Path(str(run_directory)),
        task_key=task.task_key,
        candidate_id=task.candidate_id,
        repeat_index=task.repeat_index,
        fold_index=task.fold_index,
    )

def run_nested_hpo_outer_task(task: ExperimentTask) -> Mapping[str, Any]:
    """Run one complete training-only outer nested-HPO task.

    The payload specifies only training-row positions and outer split positions. The
    worker loads ``train.csv`` through the standard project data loader and never
    imports, reads, or references the held-out test set. A clean user stop is observed
    only at persistent safe boundaries, never converted into a model-family failure.
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
    reporter = _make_progress_reporter(task)

    if reporter is not None:
        reporter.start(
            stage="initializing",
            message="Validating task payload and loading development-only data.",
        )

    try:
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
                if reporter is not None:
                    reporter.update(
                        stage="stage_a",
                        message="Executing controlled smoke interruption after one durable trial.",
                        target_completed_trials=1,
                    )
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

        def should_stop() -> bool:
            return reporter.stop_requested() if reporter is not None else False

        def stage_callback(stage: str, details: Mapping[str, Any]) -> None:
            if reporter is None:
                return
            payload = dict(details)
            event_name = str(payload.pop("event_name", f"{stage}_updated"))
            messages = {
                "stage_a_started": "Starting persistent Stage-A Optuna exploration.",
                "stage_a_trial_started": "Stage A started a new candidate configuration.",
                "stage_a_fold_started": "Stage A is fitting an inner cross-validation fold.",
                "stage_a_fold_completed": "Stage A completed an inner cross-validation fold.",
                "stage_a_trial_terminal": "Stage A completed a persistent trial.",
                "stage_b_started": "Starting Stage-B confirmation of leading configurations.",
                "stage_b_configuration_started": "Stage B started confirming one configuration.",
                "stage_b_fold_started": "Stage B is fitting an inner cross-validation fold.",
                "stage_b_fold_completed": "Stage B completed an inner cross-validation fold.",
                "stage_b_configuration_completed": "Stage B completed one configuration confirmation.",
                "stage_b_selected_configuration": "Stage B selected the confirmed configuration.",
            }
            if event_name in {"stage_a_started", "stage_a_trial_started"}:
                # Partial fold metrics belong to exactly one active configuration. Clearing
                # them before a new Stage-A configuration prevents a prior trial's partial
                # AP from being displayed while the new trial has not completed a fold yet.
                payload.update(
                    {
                        "partial_mean_average_precision": None,
                        "fold_average_precision": None,
                        "completed_inner_folds": 0,
                        "inner_fold_index": None,
                        "inner_fold_total": None,
                    }
                )
            elif event_name in {"stage_b_started", "stage_b_configuration_started"}:
                payload.update(
                    {
                        "partial_mean_average_precision": None,
                        "fold_average_precision": None,
                        "completed_inner_folds": 0,
                        "inner_fold_index": None,
                        "inner_fold_total": None,
                    }
                )
            elif event_name == "stage_b_selected_configuration":
                # Confirmation is now complete. Clear the last exploratory configuration
                # so outer fitting can display only the durable Stage-B-selected parameters.
                payload.update(
                    {
                        "current_trial_parameters": None,
                        "current_trial_number": None,
                        "partial_mean_average_precision": None,
                        "fold_average_precision": None,
                        "completed_inner_folds": 0,
                        "inner_fold_index": None,
                        "inner_fold_total": None,
                    }
                )

            message = messages.get(event_name, f"{stage.replace('_', ' ').title()} updated.")
            if event_name in {
                "stage_a_fold_started",
                "stage_a_fold_completed",
                "stage_b_fold_started",
                "stage_b_fold_completed",
            }:
                # Fold events update the atomic sidecar for the live dashboard only. They
                # intentionally do not become append-only task or coordinator log entries.
                reporter.update(stage=stage, message=message, **payload)
            else:
                reporter.update(
                    stage=stage,
                    message=message,
                    event_name=event_name,
                    **payload,
                )

        if reporter is not None:
            reporter.update(
                stage="stage_a",
                message="Starting persistent Stage-A Optuna exploration.",
                event_name="stage_a_search_started",
                target_completed_trials=stage_a_n_trials,
                confirmation_top_k=confirmation_top_k,
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
            should_stop=should_stop if reporter is not None else None,
            stage_callback=stage_callback if reporter is not None else None,
        )
        search_seconds = time.perf_counter() - search_started

        if should_stop():
            raise GracefulStopRequested(
                "Clean stop requested after inner selection and before the outer refit."
            )
        if reporter is not None:
            reporter.update(
                stage="outer_fit",
                message="Fitting the Stage-B-selected configuration on the outer-training partition.",
                event_name="outer_fit_started",
                selected_parameters=dict(search_result.selected_parameters),
            )
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

        if should_stop():
            raise GracefulStopRequested(
                "Clean stop requested after the outer refit and before validation prediction."
            )
        if reporter is not None:
            reporter.update(
                stage="outer_prediction",
                message="Scoring the untouched outer-validation partition.",
                event_name="outer_prediction_started",
                selected_parameters=dict(search_result.selected_parameters),
            )
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

        result = {
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
        if reporter is not None:
            reporter.close(
                final_stage="completed",
                message="Outer task completed and is returning its result to the coordinator.",
                selected_parameters=dict(search_result.selected_parameters),
                outer_average_precision=float(metrics["average_precision"]),
            )
        return result
    except GracefulStopRequested as exc:
        if reporter is not None:
            reporter.close(final_stage="interrupted", message=str(exc))
        raise
    except BaseException as exc:
        if reporter is not None:
            reporter.close(
                final_stage="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
        raise
