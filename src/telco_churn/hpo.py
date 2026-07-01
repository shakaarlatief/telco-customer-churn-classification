"""Persistent Optuna hyperparameter optimization for nested-CV tasks.

Each outer-fold task owns one Optuna SQLite study. The design intentionally does not
run concurrent workers against the same study database: outer tasks are parallelized
by the project runner, while the inner study for one task runs sequentially. This avoids
SQLite write contention and keeps one study's resume history self-contained.

The module persists both the Optuna RDB study and the sampler/pruner objects. Optuna's
RDB storage preserves trials, but not sampler or pruner instance state. Persisting those
objects after each finished trial makes an interrupted, seeded TPE search reproducible
across normal resume operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import tempfile
import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from telco_churn.candidates import (
    CandidateRegistryError,
    build_candidate_pipeline,
    candidate_procedure_contract_fingerprint,
    suggest_candidate_parameters,
)
from telco_churn.experiment_splits import derive_seed


class OptunaUnavailableError(ImportError):
    """Raised when persistent HPO is requested without the Optuna dependency."""


class StudyCompatibilityError(RuntimeError):
    """Raised when an existing persistent study belongs to another task contract."""


class SearchExecutionError(RuntimeError):
    """Raised when an HPO stage cannot produce a valid selected configuration."""


def _require_optuna():
    """Import Optuna lazily so non-HPO project workflows remain usable."""
    try:
        import optuna
    except ImportError as exc:
        raise OptunaUnavailableError(
            "Optuna is required for persistent final-comparison HPO. "
            "Install the repository requirements before running this workflow."
        ) from exc
    return optuna


def _optuna_experimental_warning_class(optuna):
    """Return Optuna's explicit experimental-warning class when available."""
    return getattr(optuna.exceptions, "ExperimentalWarning", Warning)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON-compatible experiment metadata deterministically."""
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    """Return a SHA-256 fingerprint for a JSON-compatible study contract."""
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_pickle(path: Path, value: Any) -> None:
    """Atomically persist a Python object beside its SQLite study database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            pickle.dump(value, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist small stage-confirmation artifacts.

    The temporary file is opened for writing, flushed, and synchronized before the
    atomic replacement. On Windows, ``os.fsync`` on a read-only file descriptor can
    raise ``OSError: [Errno 9] Bad file descriptor``. Synchronizing the active
    write descriptor therefore matters for both durability and platform
    compatibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file_handle:
            temporary_path = Path(file_handle.name)
            file_handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            )
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class PersistentStudyConfig:
    """Immutable configuration for one task-local Optuna study."""

    study_name: str
    database_path: Path
    candidate_id: str
    task_key: str
    task_fingerprint: str
    random_state: int
    n_startup_trials: int = 10
    heartbeat_interval_seconds: int = 30
    heartbeat_grace_period_seconds: int = 180

    def __post_init__(self) -> None:
        if not self.study_name.strip():
            raise ValueError("study_name must not be empty.")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty.")
        if not self.task_key.strip():
            raise ValueError("task_key must not be empty.")
        if not self.task_fingerprint.strip():
            raise ValueError("task_fingerprint must not be empty.")
        if self.n_startup_trials < 0:
            raise ValueError("n_startup_trials must be non-negative.")
        if self.heartbeat_interval_seconds < 1:
            raise ValueError("heartbeat_interval_seconds must be positive.")
        if self.heartbeat_grace_period_seconds < self.heartbeat_interval_seconds:
            raise ValueError(
                "heartbeat_grace_period_seconds must not be smaller than heartbeat interval."
            )

    @property
    def sampler_path(self) -> Path:
        """Return the checkpoint path for the seeded TPE sampler."""
        return self.database_path.with_suffix(".sampler.pkl")

    @property
    def pruner_path(self) -> Path:
        """Return the checkpoint path for the study pruner."""

        return self.database_path.with_suffix(".pruner.pkl")

    @property
    def contract(self) -> dict[str, Any]:
        """Return immutable metadata that must match on a study resume."""
        return {
            "schema_version": "persistent_optuna_study_v1",
            "candidate_id": self.candidate_id,
            "task_key": self.task_key,
            "task_fingerprint": self.task_fingerprint,
        }


@dataclass(frozen=True)
class SearchResult:
    """Result of a completed two-stage inner search."""

    candidate_id: str
    selected_parameters: Mapping[str, Any]
    selected_stage_b_average_precision: float
    stage_a_completed_trials: int
    stage_a_best_average_precision: float
    stage_b_records: tuple[Mapping[str, Any], ...]
    study_database_path: str
    study_name: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible persistent representation."""
        return {
            "candidate_id": self.candidate_id,
            "selected_parameters": dict(self.selected_parameters),
            "selected_stage_b_average_precision": float(
                self.selected_stage_b_average_precision
            ),
            "stage_a_completed_trials": int(self.stage_a_completed_trials),
            "stage_a_best_average_precision": float(self.stage_a_best_average_precision),
            "stage_b_records": [dict(record) for record in self.stage_b_records],
            "study_database_path": self.study_database_path,
            "study_name": self.study_name,
        }


def make_study_contract_fingerprint(
    *,
    candidate_id: str,
    task_key: str,
    split_hash: str,
    primary_metric: str,
    search_profile: str,
    stage_a_n_splits: int,
    stage_b_n_splits: int,
    confirmation_top_k: int,
) -> str:
    """Fingerprint all ingredients that define one inner HPO objective.

    The candidate routing fingerprint binds feature-policy and feature-selection
    compatibility to persistent Optuna resume safety.  A study created under an older
    candidate-procedure contract is rejected instead of silently reusing trials that
    searched a different representation or selector universe.
    """
    return _sha256_payload(
        {
            "schema_version": "inner_hpo_objective_v2",
            "candidate_id": candidate_id,
            "candidate_procedure_contract_fingerprint": (
                candidate_procedure_contract_fingerprint(candidate_id)
            ),
            "task_key": task_key,
            "split_hash": split_hash,
            "primary_metric": primary_metric,
            "search_profile": search_profile,
            "stage_a_n_splits": int(stage_a_n_splits),
            "stage_b_n_splits": int(stage_b_n_splits),
            "confirmation_top_k": int(confirmation_top_k),
        }
    )


def _sqlite_url(database_path: Path) -> str:
    """Build a SQLAlchemy-safe SQLite URL for POSIX and Windows paths."""
    try:
        from sqlalchemy.engine import URL
    except ImportError as exc:
        raise OptunaUnavailableError(
            "Optuna's SQLAlchemy dependency is unavailable. Reinstall Optuna."
        ) from exc

    return str(URL.create(drivername="sqlite", database=str(database_path.resolve())))


def _load_or_create_sampler(config: PersistentStudyConfig):
    """Restore a persisted sampler or create the task's deterministic TPE sampler."""
    optuna = _require_optuna()
    if config.sampler_path.exists():
        with config.sampler_path.open("rb") as file_handle:
            sampler = pickle.load(file_handle)
        return sampler

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=_optuna_experimental_warning_class(optuna),
        )
        return optuna.samplers.TPESampler(
            seed=int(config.random_state),
            n_startup_trials=int(config.n_startup_trials),
            multivariate=True,
            group=True,
        )


def _load_or_create_pruner(config: PersistentStudyConfig):
    """Restore a persisted pruner or create a transparent no-pruning policy."""
    optuna = _require_optuna()
    if config.pruner_path.exists():
        with config.pruner_path.open("rb") as file_handle:
            pruner = pickle.load(file_handle)
        return pruner

    # The initial general objective reports a final fold-mean only. Pruning before
    # all folds would optimize noisy partial estimates. Candidate-specific iterative
    # pruning is added later only where a model exposes meaningful intermediate values.
    return optuna.pruners.NopPruner()


class _PersistStudyComponents:
    """Optuna callback that checkpoints sampler and pruner after every trial."""

    def __init__(self, config: PersistentStudyConfig):
        self.config = config

    def __call__(self, study, trial) -> None:
        """Persist state immediately after Optuna commits a trial result."""
        _atomic_pickle(self.config.sampler_path, study.sampler)
        _atomic_pickle(self.config.pruner_path, study.pruner)


def create_or_resume_study(config: PersistentStudyConfig):
    """Create or reopen one persistent task-local study after contract validation."""
    optuna = _require_optuna()
    config.database_path.parent.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=_optuna_experimental_warning_class(optuna),
        )
        storage = optuna.storages.RDBStorage(
            url=_sqlite_url(config.database_path),
            engine_kwargs={"connect_args": {"timeout": 60}},
            heartbeat_interval=int(config.heartbeat_interval_seconds),
            grace_period=int(config.heartbeat_grace_period_seconds),
        )
    study = optuna.create_study(
        storage=storage,
        study_name=config.study_name,
        direction="maximize",
        load_if_exists=True,
        sampler=_load_or_create_sampler(config),
        pruner=_load_or_create_pruner(config),
    )

    for key, expected_value in config.contract.items():
        observed_value = study.user_attrs.get(key)
        if observed_value is not None and observed_value != expected_value:
            raise StudyCompatibilityError(
                f"Persistent Optuna study {config.study_name!r} has incompatible "
                f"{key}: observed {observed_value!r}, expected {expected_value!r}."
            )
        if observed_value is None:
            study.set_user_attr(key, expected_value)

    # Mark heartbeat-stale trials from an interrupted process as failed before we
    # count remaining work. The RDB backend also performs this just before asking
    # for a new trial, but doing it explicitly makes the resume state visible.
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=_optuna_experimental_warning_class(optuna),
        )
        optuna.storages.fail_stale_trials(study)
    return study


def release_persistent_study_resources(study: Any) -> None:
    """Release task-local Optuna RDB resources after a study is no longer needed.

    Optuna's SQLite-backed storage maintains a thread-local SQLAlchemy session. On
    Windows, leaving that session open can keep the study database locked after a
    task finishes, preventing a later cleanup or a fresh smoke run from removing the
    temporary run directory. Every task-local study in this project owns its own
    database, so disposing its engine after the task is complete is safe.

    ``Study`` does not expose a public storage accessor. This cleanup helper therefore
    carefully accesses the internal storage reference and supports both a direct
    ``RDBStorage`` and an Optuna wrapper storage that exposes the underlying backend.
    It is called only after all operations on the study have finished.
    """
    storage = getattr(study, "_storage", None)
    candidate_storages = [storage, getattr(storage, "_backend", None)]
    visited: set[int] = set()

    for candidate_storage in candidate_storages:
        if candidate_storage is None or id(candidate_storage) in visited:
            continue
        visited.add(id(candidate_storage))

        remove_session = getattr(candidate_storage, "remove_session", None)
        if callable(remove_session):
            remove_session()

        engine = getattr(candidate_storage, "engine", None)
        if engine is None:
            engine = getattr(candidate_storage, "_engine", None)
        dispose = getattr(engine, "dispose", None)
        if callable(dispose):
            dispose()


def _positive_class_index(classes: Sequence[Any]) -> int:
    """Return the position of class label one in an estimator class vector."""
    classes_array = np.asarray(classes)
    matching = np.flatnonzero(classes_array == 1)
    if matching.size != 1:
        raise ValueError(
            "Binary churn estimators must expose exactly one class label equal to 1."
        )
    return int(matching[0])


def extract_continuous_scores(estimator, X) -> tuple[np.ndarray, str]:
    """Return class-one probabilities or decision scores from a fitted estimator."""
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("predict_proba must return an n-by-2 binary probability matrix.")
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            raise ValueError("Probability-producing estimator does not expose classes_.")
        return probabilities[:, _positive_class_index(classes)], "probability"

    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X), dtype=float)
        if decision.ndim == 1:
            return decision, "margin"
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            raise ValueError("Margin-producing estimator does not expose classes_.")
        return decision[:, _positive_class_index(classes)], "margin"

    raise TypeError(
        "Candidate estimator must expose predict_proba or decision_function for "
        "ranking evaluation."
    )


def evaluate_candidate_cv(
    *,
    candidate_id: str,
    parameters: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
) -> dict[str, Any]:
    """Evaluate one configuration by fold-internal fitting and average precision."""
    fold_scores: list[float] = []
    warning_messages: list[str] = []

    for fold_index, (train_indices, validation_indices) in enumerate(cv.split(X, y)):
        fold_seed = derive_seed(random_state, candidate_id, "inner", fold_index)
        estimator = build_candidate_pipeline(
            candidate_id,
            parameters,
            random_state=fold_seed,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            fitted = clone(estimator).fit(
                X.iloc[train_indices],
                y.iloc[train_indices],
            )
        warning_messages.extend(
            f"{warning.category.__name__}: {warning.message}"
            for warning in caught_warnings
        )

        score, _ = extract_continuous_scores(fitted, X.iloc[validation_indices])
        fold_score = average_precision_score(y.iloc[validation_indices], score)
        fold_scores.append(float(fold_score))

    score_array = np.asarray(fold_scores, dtype=float)
    if not np.isfinite(score_array).all():
        raise SearchExecutionError("Inner cross-validation produced non-finite AP.")

    return {
        "mean_average_precision": float(score_array.mean()),
        "std_average_precision": float(score_array.std(ddof=0)),
        "fold_average_precision": [float(value) for value in score_array],
        "warning_messages": warning_messages,
    }


def _terminal_trial_count(study) -> int:
    """Count trials with a terminal Optuna state."""
    optuna = _require_optuna()
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    }
    return sum(trial.state in terminal_states for trial in study.trials)


def run_stage_a_optuna_search(
    *,
    config: PersistentStudyConfig,
    X: pd.DataFrame,
    y: pd.Series,
    stage_a_cv: StratifiedKFold,
    n_trials_target: int,
    search_profile: str,
) -> Any:
    """Run or resume Stage-A TPE exploration until a total trial target is reached.

    The returned study remains open after a successful call because Stage B needs its
    trial history. The caller must invoke :func:`release_persistent_study_resources`
    once it has finished all reads and writes for that task. Exceptions occurring
    during Stage A release task-local SQLite resources here before propagating.
    """
    if n_trials_target < 1:
        raise ValueError("n_trials_target must be positive.")

    optuna = _require_optuna()
    study = create_or_resume_study(config)

    try:
        remaining_trials = max(0, int(n_trials_target) - _terminal_trial_count(study))

        if remaining_trials == 0:
            return study

        def objective(trial) -> float:
            parameters = suggest_candidate_parameters(
                trial,
                candidate_id=config.candidate_id,
                profile=search_profile,
            )
            trial.set_user_attr("resolved_parameters", dict(parameters))
            try:
                result = evaluate_candidate_cv(
                    candidate_id=config.candidate_id,
                    parameters=parameters,
                    X=X,
                    y=y,
                    cv=stage_a_cv,
                    random_state=derive_seed(
                        config.random_state,
                        "trial",
                        trial.number,
                    ),
                )
            except BaseException as exc:
                trial.set_user_attr("failure_type", type(exc).__name__)
                trial.set_user_attr("failure_message", str(exc))
                raise

            trial.set_user_attr(
                "fold_average_precision",
                result["fold_average_precision"],
            )
            trial.set_user_attr(
                "std_average_precision",
                result["std_average_precision"],
            )
            trial.set_user_attr(
                "warning_messages",
                result["warning_messages"],
            )
            return float(result["mean_average_precision"])

        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[_PersistStudyComponents(config)],
            gc_after_trial=True,
        )

        # Persist even when all work was already complete in a prior invocation. This
        # covers first-study creation where no callback was triggered because no new
        # trial had to run after a normal resume.
        _atomic_pickle(config.sampler_path, study.sampler)
        _atomic_pickle(config.pruner_path, study.pruner)
        return study
    except BaseException:
        release_persistent_study_resources(study)
        raise

def _stage_b_confirmation_path(config: PersistentStudyConfig) -> Path:
    """Return a durable file for the Stage-B confirmation result."""
    return config.database_path.with_suffix(".stage_b_confirmation.json")


def _top_completed_trials(study, top_k: int):
    """Return the best completed trials ordered by AP and then trial number."""
    optuna = _require_optuna()
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise SearchExecutionError("Stage A completed no valid Optuna trials.")
    completed.sort(key=lambda trial: (-float(trial.value), int(trial.number)))
    return completed[: min(top_k, len(completed))]


def _resolved_parameters_from_trial(trial) -> dict[str, Any]:
    """Return the actual pipeline configuration evaluated by one Optuna trial.

    Conditional search spaces can contain suggestion-only control parameters. For
    example, Extra Trees may expose a bootstrap-specific policy that resolves into a
    final ``class_weight`` value. Stage-B confirmation must refit the evaluated
    resolved configuration, not merely replay raw trial suggestion fields.
    """
    resolved = trial.user_attrs.get("resolved_parameters")
    if resolved is None:
        return dict(trial.params)
    if not isinstance(resolved, Mapping):
        raise SearchExecutionError(
            f"Trial {trial.number} has non-mapping resolved_parameters metadata."
        )
    return dict(resolved)


def _confirmation_fingerprint(
    *,
    config: PersistentStudyConfig,
    top_trials: Iterable[Any],
    stage_b_n_splits: int,
    stage_b_seed: int,
) -> str:
    """Fingerprint the exact Stage-B candidates and split policy."""
    return _sha256_payload(
        {
            "schema_version": "stage_b_confirmation_v1",
            "task_fingerprint": config.task_fingerprint,
            "trial_numbers": [int(trial.number) for trial in top_trials],
            "trial_parameters": [
                _resolved_parameters_from_trial(trial) for trial in top_trials
            ],
            "stage_b_n_splits": int(stage_b_n_splits),
            "stage_b_seed": int(stage_b_seed),
        }
    )


def run_two_stage_optuna_search(
    *,
    config: PersistentStudyConfig,
    X: pd.DataFrame,
    y: pd.Series,
    stage_a_cv: StratifiedKFold,
    stage_b_cv: StratifiedKFold,
    n_trials_target: int,
    confirmation_top_k: int,
    search_profile: str,
) -> SearchResult:
    """Run persistent Stage A and durable Stage-B configuration confirmation."""
    if confirmation_top_k < 1:
        raise ValueError("confirmation_top_k must be positive.")

    study = run_stage_a_optuna_search(
        config=config,
        X=X,
        y=y,
        stage_a_cv=stage_a_cv,
        n_trials_target=n_trials_target,
        search_profile=search_profile,
    )
    try:
        top_trials = _top_completed_trials(study, confirmation_top_k)
        stage_b_seed = int(getattr(stage_b_cv, "random_state", config.random_state))
        fingerprint = _confirmation_fingerprint(
            config=config,
            top_trials=top_trials,
            stage_b_n_splits=stage_b_cv.n_splits,
            stage_b_seed=stage_b_seed,
        )
        confirmation_path = _stage_b_confirmation_path(config)

        if confirmation_path.exists():
            persisted = json.loads(confirmation_path.read_text(encoding="utf-8"))
            if persisted.get("fingerprint") != fingerprint:
                raise StudyCompatibilityError(
                    "Existing Stage-B confirmation does not match the current top-trial "
                    "or split contract. Create a new run rather than overwriting it."
                )
            records = tuple(dict(record) for record in persisted["records"])
        else:
            records_list: list[dict[str, Any]] = []
            for trial in top_trials:
                result = evaluate_candidate_cv(
                    candidate_id=config.candidate_id,
                    parameters=_resolved_parameters_from_trial(trial),
                    X=X,
                    y=y,
                    cv=stage_b_cv,
                    random_state=derive_seed(
                        config.random_state,
                        "stage_b",
                        trial.number,
                    ),
                )
                records_list.append(
                    {
                        "stage_a_trial_number": int(trial.number),
                        "parameters": _resolved_parameters_from_trial(trial),
                        "stage_a_average_precision": float(trial.value),
                        "stage_b_average_precision": float(
                            result["mean_average_precision"]
                        ),
                        "stage_b_std_average_precision": float(
                            result["std_average_precision"]
                        ),
                        "stage_b_fold_average_precision": list(
                            result["fold_average_precision"]
                        ),
                        "stage_b_warning_messages": list(result["warning_messages"]),
                    }
                )

            records_list.sort(
                key=lambda record: (
                    -float(record["stage_b_average_precision"]),
                    -float(record["stage_a_average_precision"]),
                    int(record["stage_a_trial_number"]),
                )
            )
            _atomic_json(
                confirmation_path,
                {
                    "fingerprint": fingerprint,
                    "records": records_list,
                },
            )
            records = tuple(records_list)

        if not records:
            raise SearchExecutionError("Stage B produced no confirmation records.")

        selected = records[0]
        return SearchResult(
            candidate_id=config.candidate_id,
            selected_parameters=dict(selected["parameters"]),
            selected_stage_b_average_precision=float(
                selected["stage_b_average_precision"]
            ),
            stage_a_completed_trials=int(_terminal_trial_count(study)),
            stage_a_best_average_precision=float(study.best_value),
            stage_b_records=records,
            study_database_path=str(config.database_path),
            study_name=config.study_name,
        )
    finally:
        release_persistent_study_resources(study)
