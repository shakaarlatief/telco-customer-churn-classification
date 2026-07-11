"""Fast development-data finalization for selected leading candidates.

This script is intentionally separate from the frozen protocol-v2 benchmark. It can
inspect a leading-candidate JSON in dry-run mode, or, with an explicit confirmation
flag, tune the selected candidates on development data with tiny budgets, produce
leakage-safe out-of-fold predictions, evaluate simple probability averages, and write
a transparent final-procedure selection under ``artifacts/final_selection``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_CATBOOST,
    INITIAL_CANDIDATE_REGISTRY,
    SEARCH_PROFILE_CATBOOST_V2,
    SEARCH_PROFILE_FULL,
    build_candidate_pipeline,
    get_candidate_definition,
    suggest_candidate_parameters,
)
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.hpo import extract_continuous_scores  # noqa: E402


DEFAULT_SOURCE_RUN_ID = "fast_completion_v1"
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "fast_finalization_v1.json"
DEFAULT_TOP_K = 5
PRIMARY_METRIC = "average_precision"
FINALIZATION_EVIDENCE_ROLE = "fast_finalization_pipeline_evidence"
SOURCE_EVIDENCE_ROLE = "fast_completion_pipeline_evidence"
REQUIRED_LEADING_KEYS = frozenset({"candidate_id", "candidate_display_name", "rank"})
METRIC_FIELDNAMES = [
    "procedure_id",
    "procedure_type",
    "candidate_id",
    "candidate_display_name",
    "score_kind",
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "best_f1_threshold",
    "best_f1",
    "n_oof_rows",
]


class FastFinalizationError(ValueError):
    """Raised before finalization can proceed safely."""


def _json_default(value: Any) -> Any:
    """JSON serializer for NumPy scalar values used in candidate parameters."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a project dependency.
        np = None
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    """Load the fast-finalization protocol declaration."""
    if not path.exists():
        raise FastFinalizationError(f"Protocol declaration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("evidence_role") != FINALIZATION_EVIDENCE_ROLE:
        raise FastFinalizationError("Fast-finalization protocol has unexpected evidence_role.")
    if payload.get("held_out_test_policy") != "not_loaded_or_referenced":
        raise FastFinalizationError("Fast-finalization protocol must forbid held-out access.")
    return payload


def default_leading_candidates_file(source_run_id: str) -> Path:
    """Return the default leading-candidate JSON path for one source run."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / "leading_candidates.json"
    )


def default_output_dir(source_run_id: str) -> Path:
    """Return the default derived finalization output directory."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / "fast_finalization_v1"
    )


def load_leading_candidates(path: Path, *, top_k: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load selected leading candidates with schema checks."""
    if not path.exists():
        raise FastFinalizationError(f"Leading-candidate JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("source_evidence_role") != SOURCE_EVIDENCE_ROLE:
        raise FastFinalizationError(
            "Leading-candidate JSON must come from fast-completion pipeline evidence."
        )
    selected = payload.get("selected_candidates")
    if not isinstance(selected, list) or not selected:
        raise FastFinalizationError("Leading-candidate JSON has no selected_candidates rows.")
    if top_k is not None:
        if top_k < 1:
            raise FastFinalizationError("--top-k must be positive.")
        if top_k > len(selected):
            raise FastFinalizationError(
                f"--top-k={top_k} exceeds selected-candidate rows ({len(selected)})."
            )
        selected = selected[:top_k]

    implemented = {definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY}
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        if not isinstance(row, Mapping):
            raise FastFinalizationError("Each selected candidate row must be an object.")
        missing = sorted(REQUIRED_LEADING_KEYS - set(row))
        if missing:
            raise FastFinalizationError(f"Selected candidate row is missing {missing}.")
        candidate_id = str(row["candidate_id"])
        if candidate_id not in implemented:
            raise FastFinalizationError(f"Unknown selected candidate: {candidate_id!r}")
        if candidate_id in seen:
            raise FastFinalizationError(f"Duplicate selected candidate: {candidate_id!r}")
        seen.add(candidate_id)
        definition = get_candidate_definition(candidate_id)
        cleaned.append(
            {
                "rank": int(row["rank"]),
                "candidate_id": candidate_id,
                "candidate_display_name": str(
                    row.get("candidate_display_name") or definition.display_name
                ),
                "source_metric_mean": float(row.get("mean", math.nan)),
            }
        )
    return cleaned, metadata


def search_profile_for(candidate_id: str) -> str:
    """Return the fast-finalization search profile for one candidate."""
    return SEARCH_PROFILE_CATBOOST_V2 if candidate_id == CANDIDATE_CATBOOST else SEARCH_PROFILE_FULL


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Run or inspect fast development-data finalization for selected leading "
            "candidates. Dry-run mode creates no artifacts and fits no models."
        )
    )
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--leading-candidates-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-fast-finalization", action="store_true")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


def _assert_output_dir_is_derived(path: Path) -> None:
    """Refuse writes under immutable final-comparison run artifacts."""
    resolved = path.resolve()
    immutable_root = (PROJECT_ROOT / "artifacts" / "final_comparison").resolve()
    if resolved == immutable_root or immutable_root in resolved.parents:
        raise FastFinalizationError(
            "Finalization outputs must not be written under immutable run artifacts."
        )


def _assert_no_forbidden_loader_references() -> None:
    """Static guard against importing known final-evaluation loaders in this script."""
    source = Path(__file__).read_text(encoding="utf-8")
    for token in ("load_" + "test_data", "split_" + "test"):
        if token in source:
            raise FastFinalizationError(f"Forbidden loader reference found: {token}")


def _finite_array(values: Sequence[float]) -> list[float]:
    """Return finite floats from a numeric sequence."""
    return [float(value) for value in values if math.isfinite(float(value))]


def _binary_predictions(scores: Sequence[float], score_kind: str, threshold: float | None = None) -> list[int]:
    """Convert scores to binary predictions using probability or margin defaults."""
    if threshold is None:
        threshold = 0.5 if score_kind == "probability" else 0.0
    return [1 if float(score) >= float(threshold) else 0 for score in scores]


def compute_metrics(y_true: Sequence[int], scores: Sequence[float], *, score_kind: str) -> dict[str, Any]:
    """Compute ranking, calibration, and threshold metrics for OOF scores."""
    import numpy as np
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

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise FastFinalizationError("Metric inputs must be equal-length one-dimensional arrays.")
    if y.shape[0] == 0:
        raise FastFinalizationError("Cannot score empty OOF predictions.")
    if not np.all(np.isfinite(s)):
        raise FastFinalizationError("OOF scores must be finite.")

    predictions = np.asarray(_binary_predictions(s, score_kind), dtype=int)
    result: dict[str, Any] = {
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": "",
        "log_loss": "",
        "brier_score": "",
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "best_f1_threshold": "",
        "best_f1": "",
        "n_oof_rows": int(y.shape[0]),
    }
    if len(set(y.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(y, s))
    if score_kind == "probability":
        clipped = np.clip(s, 1e-15, 1 - 1e-15)
        result["log_loss"] = float(log_loss(y, clipped, labels=[0, 1]))
        result["brier_score"] = float(brier_score_loss(y, clipped))

    thresholds = sorted(set(float(value) for value in s))
    if thresholds:
        best_threshold = thresholds[0]
        best_f1 = -1.0
        for threshold in thresholds:
            threshold_predictions = np.asarray(_binary_predictions(s, score_kind, threshold), dtype=int)
            value = float(f1_score(y, threshold_predictions, zero_division=0))
            if value > best_f1:
                best_f1 = value
                best_threshold = threshold
        result["best_f1_threshold"] = float(best_threshold)
        result["best_f1"] = float(best_f1)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows with a stable CSV schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def tune_candidate(
    *,
    candidate_id: str,
    X: Any,
    y: Any,
    n_trials: int,
    cv_folds: int,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    """Tune one candidate with a tiny Optuna budget on development data."""
    import numpy as np
    import optuna
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    profile = search_profile_for(candidate_id)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(random_state))
    y_array = np.asarray(y, dtype=int)

    def objective(trial: Any) -> float:
        parameters = suggest_candidate_parameters(
            trial,
            candidate_id=candidate_id,
            profile=profile,
        )
        scores = np.empty(shape=y_array.shape[0], dtype=float)
        for train_index, validation_index in cv.split(X, y_array):
            estimator = build_candidate_pipeline(
                candidate_id,
                parameters,
                random_state=int(random_state),
            )
            fitted = clone(estimator).fit(X.iloc[train_index], y_array[train_index])
            fold_scores, _score_kind = extract_continuous_scores(
                fitted,
                X.iloc[validation_index],
            )
            scores[validation_index] = fold_scores
        return float(average_precision_score(y_array, scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(random_state)),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return dict(study.best_trial.params), float(study.best_value)


def collect_oof_predictions(
    *,
    candidate_id: str,
    display_name: str,
    parameters: Mapping[str, Any],
    X: Any,
    y: Any,
    cv_folds: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], str]:
    """Fit one selected candidate in fold-safe OOF mode and return row scores."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(random_state))
    y_array = np.asarray(y, dtype=int)
    rows: list[dict[str, Any]] = []
    observed_score_kinds: set[str] = set()
    for fold_index, (train_index, validation_index) in enumerate(cv.split(X, y_array), start=1):
        estimator = build_candidate_pipeline(
            candidate_id,
            parameters,
            random_state=int(random_state),
        )
        fitted = clone(estimator).fit(X.iloc[train_index], y_array[train_index])
        fold_scores, score_kind = extract_continuous_scores(fitted, X.iloc[validation_index])
        observed_score_kinds.add(score_kind)
        predictions = _binary_predictions(fold_scores, score_kind)
        for position, score, target, prediction in zip(
            validation_index,
            fold_scores,
            y_array[validation_index],
            predictions,
        ):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_display_name": display_name,
                    "fold": int(fold_index),
                    "row_position": int(position),
                    "row_index": str(X.index[int(position)]),
                    "target": int(target),
                    "score": float(score),
                    "probability": float(score) if score_kind == "probability" else "",
                    "prediction": int(prediction),
                    "score_kind": score_kind,
                }
            )
    if len(observed_score_kinds) != 1:
        raise FastFinalizationError(
            f"{candidate_id} produced inconsistent score kinds: {sorted(observed_score_kinds)}"
        )
    rows.sort(key=lambda row: int(row["row_position"]))
    return rows, observed_score_kinds.pop()


def build_ensemble_outputs(
    *,
    candidate_metrics: Sequence[Mapping[str, Any]],
    candidate_oof_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate simple soft-voting ensembles from probability OOF predictions."""
    from collections import defaultdict
    import numpy as np

    probability_candidates = [
        row
        for row in sorted(
            candidate_metrics,
            key=lambda item: (-float(item["average_precision"]), str(item["candidate_id"])),
        )
        if row.get("score_kind") == "probability"
    ]
    oof_by_candidate: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in candidate_oof_rows:
        if row.get("score_kind") != "probability":
            continue
        oof_by_candidate[str(row["candidate_id"])][int(row["row_position"])] = row

    ensemble_metrics: list[dict[str, Any]] = []
    ensemble_oof_rows: list[dict[str, Any]] = []
    for spec in protocol.get("ensembles", []):
        minimum = int(spec["minimum_probability_candidates"])
        member_count = int(spec["member_count"])
        if len(probability_candidates) < minimum:
            continue
        members = probability_candidates[:member_count]
        member_ids = [str(member["candidate_id"]) for member in members]
        common_positions = sorted(
            set.intersection(*(set(oof_by_candidate[candidate_id]) for candidate_id in member_ids))
        )
        if not common_positions:
            continue
        weights = np.ones(shape=len(member_ids), dtype=float)
        if spec["weighting"] == "oof_average_precision":
            weights = np.asarray([float(member["average_precision"]) for member in members], dtype=float)
            if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
                weights = np.ones(shape=len(member_ids), dtype=float)
        weights = weights / float(weights.sum())

        rows: list[dict[str, Any]] = []
        for position in common_positions:
            base_rows = [oof_by_candidate[candidate_id][position] for candidate_id in member_ids]
            target_values = {int(base_row["target"]) for base_row in base_rows}
            if len(target_values) != 1:
                raise FastFinalizationError(
                    f"OOF target mismatch for ensemble row position {position}."
                )
            probabilities = np.asarray([float(base_row["probability"]) for base_row in base_rows])
            probability = float(np.dot(weights, probabilities))
            rows.append(
                {
                    "ensemble_id": str(spec["ensemble_id"]),
                    "member_candidate_ids": "|".join(member_ids),
                    "row_position": int(position),
                    "row_index": str(base_rows[0]["row_index"]),
                    "target": int(base_rows[0]["target"]),
                    "probability": probability,
                    "prediction": int(probability >= 0.5),
                    "score_kind": "probability",
                }
            )
        metrics = compute_metrics(
            [int(row["target"]) for row in rows],
            [float(row["probability"]) for row in rows],
            score_kind="probability",
        )
        ensemble_metrics.append(
            {
                "procedure_id": str(spec["ensemble_id"]),
                "procedure_type": "ensemble",
                "candidate_id": "",
                "candidate_display_name": str(spec["ensemble_id"]),
                "score_kind": "probability",
                **metrics,
            }
        )
        ensemble_oof_rows.extend(rows)
    return ensemble_metrics, ensemble_oof_rows


def select_final_procedure(
    candidate_metrics: Sequence[Mapping[str, Any]],
    ensemble_metrics: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Select the highest-AP procedure with the simpler-model tolerance rule."""
    all_metrics = [dict(row) for row in candidate_metrics] + [dict(row) for row in ensemble_metrics]
    if not all_metrics:
        raise FastFinalizationError("No candidate or ensemble metrics are available.")
    ranked = sorted(
        all_metrics,
        key=lambda row: (-float(row["average_precision"]), str(row["procedure_type"]), str(row["procedure_id"])),
    )
    best = ranked[0]
    best_individuals = [
        row for row in ranked if row.get("procedure_type") == "individual"
    ]
    selected = best
    tie_break_applied = False
    if best.get("procedure_type") == "ensemble" and best_individuals:
        best_individual = best_individuals[0]
        if float(best["average_precision"]) - float(best_individual["average_precision"]) <= tolerance:
            selected = best_individual
            tie_break_applied = True
    return {
        "selected_procedure": selected,
        "ranking": ranked,
        "selection_rule": (
            "Select the procedure with highest OOF average precision; if an ensemble "
            f"wins by no more than {tolerance:g} AP over the best individual, prefer "
            "the simpler individual procedure."
        ),
        "simplicity_tolerance": float(tolerance),
        "tie_break_applied": tie_break_applied,
        "warning": (
            "This final procedure is selected from fast development-data evidence only. "
            "It is not a robust protocol-v2 benchmark winner and is not final evaluation evidence."
        ),
        "held_out_test_policy": "not_loaded_or_referenced",
    }


def run_finalization(
    *,
    source_run_id: str,
    leading_candidates_file: Path,
    output_dir: Path,
    top_k: int,
    random_state: int,
) -> dict[str, Any]:
    """Execute fast finalization and write derived outputs."""
    _assert_no_forbidden_loader_references()
    _assert_output_dir_is_derived(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FastFinalizationError(f"Output directory already exists and is not empty: {output_dir}")

    protocol = load_protocol()
    selected, source_metadata = load_leading_candidates(leading_candidates_file, top_k=top_k)

    train_df = load_train_data()
    X, y = split_features_target(train_df)

    tuning = protocol["tuning"]
    oof = protocol["oof_evaluation"]
    tuned_configs: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    candidate_oof_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        candidate_id = candidate["candidate_id"]
        candidate_random_state = int(random_state) + index
        parameters, tuning_score = tune_candidate(
            candidate_id=candidate_id,
            X=X,
            y=y,
            n_trials=int(tuning["stage_a_trials"]),
            cv_folds=int(tuning["inner_cv_folds"]),
            random_state=candidate_random_state,
        )
        oof_rows, score_kind = collect_oof_predictions(
            candidate_id=candidate_id,
            display_name=candidate["candidate_display_name"],
            parameters=parameters,
            X=X,
            y=y,
            cv_folds=int(oof["outer_cv_folds"]),
            random_state=candidate_random_state,
        )
        metrics = compute_metrics(
            [int(row["target"]) for row in oof_rows],
            [float(row["score"]) for row in oof_rows],
            score_kind=score_kind,
        )
        candidate_metrics.append(
            {
                "procedure_id": candidate_id,
                "procedure_type": "individual",
                "candidate_id": candidate_id,
                "candidate_display_name": candidate["candidate_display_name"],
                "score_kind": score_kind,
                **metrics,
            }
        )
        tuned_configs.append(
            {
                "candidate_id": candidate_id,
                "candidate_display_name": candidate["candidate_display_name"],
                "search_profile": search_profile_for(candidate_id),
                "tuning_average_precision": tuning_score,
                "parameters": parameters,
            }
        )
        candidate_oof_rows.extend(oof_rows)

    ensemble_metrics, ensemble_oof_rows = build_ensemble_outputs(
        candidate_metrics=candidate_metrics,
        candidate_oof_rows=candidate_oof_rows,
        protocol=protocol,
    )
    selection = select_final_procedure(
        candidate_metrics,
        ensemble_metrics,
        tolerance=float(protocol["selection_rule"]["simplicity_tolerance"]),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "evidence_role": FINALIZATION_EVIDENCE_ROLE,
        "source_run_id": source_run_id,
        "source_evidence_role": source_metadata.get("source_evidence_role"),
        "leading_candidates_file": str(leading_candidates_file),
        "selected_candidate_ids": [candidate["candidate_id"] for candidate in selected],
        "development_rows": int(len(X)),
        "tuning": tuning,
        "oof_evaluation": oof,
        "held_out_test_policy": "not_loaded_or_referenced",
        "warning": protocol["warning"],
    }
    (output_dir / "finalization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tuned_candidate_configs.json").write_text(
        json.dumps(tuned_configs, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "tuned_candidate_metrics.csv", candidate_metrics, METRIC_FIELDNAMES)
    _write_csv(
        output_dir / "tuned_candidate_oof_predictions.csv",
        candidate_oof_rows,
        [
            "candidate_id",
            "candidate_display_name",
            "fold",
            "row_position",
            "row_index",
            "target",
            "score",
            "probability",
            "prediction",
            "score_kind",
        ],
    )
    _write_csv(output_dir / "ensemble_metrics.csv", ensemble_metrics, METRIC_FIELDNAMES)
    _write_csv(
        output_dir / "ensemble_oof_predictions.csv",
        ensemble_oof_rows,
        [
            "ensemble_id",
            "member_candidate_ids",
            "row_position",
            "row_index",
            "target",
            "probability",
            "prediction",
            "score_kind",
        ],
    )
    (output_dir / "final_procedure_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    selected_procedure = selection["selected_procedure"]
    (output_dir / "final_procedure_selection.md").write_text(
        "\n".join(
            [
                "# Fast Finalization Procedure Selection",
                "",
                f"Selected procedure: `{selected_procedure['procedure_id']}`",
                f"Procedure type: `{selected_procedure['procedure_type']}`",
                f"OOF average precision: `{float(selected_procedure['average_precision']):.12g}`",
                "",
                selection["selection_rule"],
                "",
                selection["warning"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "candidate_metrics": candidate_metrics,
        "ensemble_metrics": ensemble_metrics,
        "selection": selection,
    }


def print_dry_run(
    *,
    source_run_id: str,
    leading_candidates_file: Path,
    output_dir: Path,
    top_k: int,
) -> None:
    """Print the fast-finalization plan without creating artifacts or fitting models."""
    _assert_no_forbidden_loader_references()
    protocol = load_protocol()
    selected, source_metadata = load_leading_candidates(leading_candidates_file, top_k=top_k)
    print("Fast finalization dry run")
    print(f"Source run ID: {source_run_id}")
    print(f"Source evidence role: {source_metadata.get('source_evidence_role')}")
    print(f"Finalization evidence role: {protocol['evidence_role']}")
    print(f"Leading candidates file: {leading_candidates_file}")
    print(f"Output directory: {output_dir}")
    print(f"Selected candidate count: {len(selected)}")
    print("Selected candidates:")
    for candidate in selected:
        print(
            f"  {candidate['rank']:>2}. {candidate['candidate_id']} "
            f"({candidate['candidate_display_name']}) "
            f"profile={search_profile_for(candidate['candidate_id'])}"
        )
    print(
        "Tuning: "
        f"{protocol['tuning']['stage_a_trials']} trials, "
        f"{protocol['tuning']['inner_cv_folds']}-fold inner CV, "
        f"metric={protocol['tuning']['primary_metric']}"
    )
    print(
        "OOF evaluation: "
        f"{protocol['oof_evaluation']['outer_cv_folds']} folds x "
        f"{protocol['oof_evaluation']['outer_cv_repeats']} repeat"
    )
    print("Ensemble options:")
    for ensemble in protocol["ensembles"]:
        print(
            f"  {ensemble['ensemble_id']}: top {ensemble['member_count']}, "
            f"weighting={ensemble['weighting']}"
        )
    print(f"Selection rule: {protocol['selection_rule']['tie_break']}")
    print("Dry-run completed without creating artifacts or fitting models.")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    arguments = parse_arguments(argv)
    source_run_id = str(arguments.source_run_id)
    leading_candidates_file = (
        Path(arguments.leading_candidates_file)
        if arguments.leading_candidates_file is not None
        else default_leading_candidates_file(source_run_id)
    )
    output_dir = (
        Path(arguments.output_dir)
        if arguments.output_dir is not None
        else default_output_dir(source_run_id)
    )
    try:
        if arguments.dry_run:
            print_dry_run(
                source_run_id=source_run_id,
                leading_candidates_file=leading_candidates_file,
                output_dir=output_dir,
                top_k=int(arguments.top_k),
            )
            return
        if not arguments.confirm_fast_finalization:
            raise FastFinalizationError(
                "Non-dry-run fast finalization requires --confirm-fast-finalization."
            )
        result = run_finalization(
            source_run_id=source_run_id,
            leading_candidates_file=leading_candidates_file,
            output_dir=output_dir,
            top_k=int(arguments.top_k),
            random_state=int(arguments.random_state),
        )
    except FastFinalizationError as exc:
        raise SystemExit(str(exc)) from exc

    selected = result["selection"]["selected_procedure"]
    print(f"Fast finalization outputs written to: {output_dir}")
    print(
        f"Selected procedure: {selected['procedure_id']} "
        f"({selected['procedure_type']}), "
        f"OOF AP={float(selected['average_precision']):.12g}"
    )


if __name__ == "__main__":
    main()
