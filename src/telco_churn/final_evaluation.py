"""Readiness and one-time final evaluation helpers.

This module deliberately contains no project final-test data loader import. The
guarded CLI is responsible for calling that loader only after the one-time receipt
has been written.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from telco_churn.config import PROJECT_ROOT
from telco_churn.experiment_protocol import make_dataframe_fingerprint
from telco_churn.final_procedure import FrozenProbabilityVotingEnsemble


CONFIRMATION_PHRASE = "I_UNDERSTAND_THIS_CONSUMES_THE_FINAL_TEST_SET"
FINAL_EVALUATION_ROLE = "one_time_held_out_test_evaluation"
EXPECTED_REFIT_FILES = (
    "artifact_checksums.json",
    "feature_schema.json",
    "final_procedure_spec.json",
    "final_refit_manifest.json",
    "fitted_final_pipeline.joblib",
    "model_environment.json",
    "roundtrip_validation.json",
)
EVALUATION_FILENAMES = (
    "evaluation_receipt.json",
    "final_test_manifest.json",
    "final_test_metrics.json",
    "final_test_metrics.csv",
    "final_test_confusion_matrix.json",
    "final_test_confusion_matrix.csv",
    "final_test_predictions.csv",
    "final_test_bootstrap_confidence_intervals.json",
    "final_test_bootstrap_confidence_intervals.csv",
    "final_test_evaluation_report.md",
)


class FinalEvaluationError(ValueError):
    """Raised when final-evaluation guards fail."""


@dataclass(frozen=True)
class ReadinessResult:
    """Readiness audit details for the frozen final model."""

    ready: bool
    checks: tuple[dict[str, Any], ...]
    procedure_spec: dict[str, Any]
    refit_manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    roundtrip_validation: dict[str, Any]
    model_path: Path
    model_sha256: str
    spec_path: Path
    spec_sha256: str
    current_git_revision: str | None


def default_refit_dir(source_run_id: str) -> Path:
    """Return the default final-development refit artifact directory."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / "final_development_refit_v1"
    )


def default_procedure_spec_path(source_run_id: str) -> Path:
    """Return the default copied frozen procedure spec path."""
    return default_refit_dir(source_run_id) / "final_procedure_spec.json"


def default_evaluation_output_dir(source_run_id: str) -> Path:
    """Return the default one-time final-evaluation output directory."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_evaluation"
        / source_run_id
        / "held_out_test_v1"
    )


def sha256_file(path: Path) -> str:
    """Return SHA-256 for one file."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON next to the destination file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    if not path.exists():
        raise FinalEvaluationError(f"Required file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalEvaluationError(f"Expected JSON object at {path}.")
    return payload


def current_git_revision() -> str | None:
    """Return current Git revision when available."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def assert_output_dir_allowed(path: Path) -> None:
    """Refuse outputs under immutable comparison or selection artifact roots."""
    resolved = path.resolve()
    forbidden_roots = (
        (PROJECT_ROOT / "artifacts" / "final_comparison").resolve(),
        (PROJECT_ROOT / "artifacts" / "final_selection").resolve(),
    )
    for root in forbidden_roots:
        if resolved == root or root in resolved.parents:
            raise FinalEvaluationError(
                f"Final evaluation outputs cannot be written under {root}."
            )


def check_no_existing_receipt(source_run_id: str, output_dir: Path) -> None:
    """Refuse if any receipt already exists for this source run."""
    root = PROJECT_ROOT / "artifacts" / "final_evaluation" / source_run_id
    if output_dir.exists():
        receipt = output_dir / "evaluation_receipt.json"
        if receipt.exists():
            raise FinalEvaluationError(f"Evaluation receipt already exists: {receipt}")
    if root.exists():
        for receipt in root.rglob("evaluation_receipt.json"):
            raise FinalEvaluationError(
                f"An evaluation receipt already exists for {source_run_id}: {receipt}"
            )


def verify_refit_checksums(refit_dir: Path) -> dict[str, str]:
    """Verify all recorded refit artifact checksums."""
    checksum_path = refit_dir / "artifact_checksums.json"
    checksums = read_json(checksum_path)
    for filename, expected_sha256 in checksums.items():
        path = refit_dir / filename
        if not path.exists():
            raise FinalEvaluationError(f"Checksummed artifact is missing: {path}")
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise FinalEvaluationError(
                f"Checksum mismatch for {path}: expected {expected_sha256}, got {observed}."
            )
    return {str(key): str(value) for key, value in checksums.items()}


def _add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    """Append one readiness check row."""
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise FinalEvaluationError(detail)


def _validate_equal_thirds(weights: Sequence[Any]) -> tuple[float, ...]:
    """Validate exactly equal one-third weights."""
    numeric = tuple(float(weight) for weight in weights)
    if len(numeric) != 3:
        raise FinalEvaluationError("Final ensemble must contain exactly three weights.")
    expected = 1.0 / 3.0
    if any(weight != expected for weight in numeric):
        raise FinalEvaluationError("Final ensemble weights must be exactly one third each.")
    if not np.isclose(sum(numeric), 1.0, atol=1e-12):
        raise FinalEvaluationError("Final ensemble weights must sum to one.")
    return numeric


def run_readiness_audit(
    *,
    source_run_id: str,
    procedure_spec_path: Path | None = None,
    refit_dir: Path | None = None,
) -> ReadinessResult:
    """Run readiness checks without accessing final evaluation data."""
    refit_dir = refit_dir or default_refit_dir(source_run_id)
    procedure_spec_path = procedure_spec_path or default_procedure_spec_path(source_run_id)
    checks: list[dict[str, Any]] = []

    _add_check(checks, "refit_dir_exists", refit_dir.exists(), f"refit dir: {refit_dir}")
    for filename in EXPECTED_REFIT_FILES:
        path = refit_dir / filename
        _add_check(checks, f"exists_{filename}", path.exists(), f"required file: {path}")

    recorded_checksums = verify_refit_checksums(refit_dir)
    _add_check(checks, "refit_checksums_verified", True, "all recorded SHA-256 values match")

    spec = read_json(procedure_spec_path)
    refit_spec = read_json(refit_dir / "final_procedure_spec.json")
    _add_check(
        checks,
        "procedure_spec_matches_refit_copy",
        spec == refit_spec,
        "procedure spec matches refit copy",
    )
    manifest = read_json(refit_dir / "final_refit_manifest.json")
    feature_schema = read_json(refit_dir / "feature_schema.json")
    roundtrip = read_json(refit_dir / "roundtrip_validation.json")

    model_path = refit_dir / "fitted_final_pipeline.joblib"
    model_sha256 = sha256_file(model_path)
    expected_model_sha256 = recorded_checksums.get("fitted_final_pipeline.joblib")
    _add_check(
        checks,
        "fitted_model_checksum_verified",
        model_sha256 == expected_model_sha256,
        "fitted model checksum matches recorded value before loading",
    )
    model = joblib.load(model_path)
    _add_check(
        checks,
        "model_type_valid",
        isinstance(model, FrozenProbabilityVotingEnsemble),
        f"model type: {type(model).__name__}",
    )

    member_ids = tuple(spec.get("member_candidate_ids", ()))
    model_member_ids = tuple(getattr(model, "member_ids_", ()))
    _add_check(
        checks,
        "member_ids_match",
        member_ids == model_member_ids == tuple(manifest.get("member_candidate_ids", ())),
        f"members: {member_ids}",
    )
    weights = _validate_equal_thirds(spec.get("member_weights", ()))
    model_weights = tuple(getattr(model, "member_weights_", ()))
    _add_check(checks, "weights_match", weights == model_weights, f"weights: {weights}")
    threshold = float(spec.get("selected_decision_threshold"))
    _add_check(
        checks,
        "threshold_match",
        np.isclose(threshold, float(getattr(model, "decision_threshold_")), atol=0.0),
        f"threshold: {threshold}",
    )
    _add_check(
        checks,
        "calibration_status_valid",
        spec.get("calibration_method") == "none"
        and spec.get("calibration_status") == "deferred_fast_completion"
        and manifest.get("calibration_method") == "none"
        and manifest.get("calibration_status") == "deferred_fast_completion",
        "calibration method/status are frozen fast-completion values",
    )
    _add_check(
        checks,
        "feature_schema_valid",
        isinstance(feature_schema.get("feature_columns"), list)
        and bool(feature_schema.get("feature_columns"))
        and isinstance(feature_schema.get("feature_dtypes"), dict)
        and bool(feature_schema.get("target_column")),
        "feature schema has columns, dtypes, and target",
    )
    _add_check(
        checks,
        "development_rows_recorded",
        int(manifest.get("training_row_count", -1)) == 5634,
        "manifest records 5,634 development rows",
    )
    _add_check(
        checks,
        "refit_records_no_final_data_access",
        manifest.get("held_out_test_policy") == "not_loaded_or_referenced",
        "refit manifest records no final-evaluation access",
    )
    _add_check(
        checks,
        "roundtrip_successful",
        bool(roundtrip.get("probabilities_allclose"))
        and bool(roundtrip.get("predictions_equal"))
        and float(roundtrip.get("max_probability_abs_diff", 1.0)) == 0.0,
        "roundtrip validation reports matching probabilities and predictions",
    )
    _add_check(
        checks,
        "git_revisions_recorded",
        bool(manifest.get("source_git_commit")) and bool(current_git_revision()),
        "source and current Git revisions are available",
    )

    return ReadinessResult(
        ready=True,
        checks=tuple(checks),
        procedure_spec=spec,
        refit_manifest=manifest,
        feature_schema=feature_schema,
        roundtrip_validation=roundtrip,
        model_path=model_path,
        model_sha256=model_sha256,
        spec_path=procedure_spec_path,
        spec_sha256=sha256_file(procedure_spec_path),
        current_git_revision=current_git_revision(),
    )


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Return binary confusion matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Return a ratio or None when undefined."""
    if denominator == 0:
        return None
    return float(numerator / denominator)


def compute_final_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    predictions: Sequence[int],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Compute final fixed-threshold metrics without optimizing anything."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = np.asarray(predictions, dtype=int)
    if y.shape != p.shape or y.shape != pred.shape:
        raise FinalEvaluationError("Targets, probabilities, and predictions must align.")
    if y.ndim != 1 or y.size == 0:
        raise FinalEvaluationError("Final metric inputs must be non-empty vectors.")
    if not np.all(np.isfinite(p)):
        raise FinalEvaluationError("Final probabilities must be finite.")
    has_two_classes = len(np.unique(y)) == 2
    counts = confusion_counts(y, pred)
    tn = counts["true_negatives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    tp = counts["true_positives"]
    metrics: dict[str, float | None] = {
        "average_precision": float(average_precision_score(y, p)) if has_two_classes else None,
        "roc_auc": None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, np.clip(p, 0.0, 1.0))),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if has_two_classes else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "negative_predictive_value": safe_divide(tn, tn + fn),
        "positive_prediction_rate": float(np.mean(pred == 1)),
    }
    if has_two_classes:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    return metrics, counts


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    predictions: Sequence[int],
    *,
    n_replicates: int,
    random_state: int,
) -> dict[str, Any]:
    """Run paired row bootstrap confidence intervals for fixed predictions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = np.asarray(predictions, dtype=int)
    rng = np.random.default_rng(int(random_state))
    values: dict[str, list[float]] = {
        name: []
        for name in (
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "negative_predictive_value",
            "positive_prediction_rate",
        )
    }
    failed = 0
    for _replicate in range(int(n_replicates)):
        indices = rng.integers(0, y.shape[0], size=y.shape[0])
        try:
            metrics, _counts = compute_final_metrics(y[indices], p[indices], pred[indices])
        except Exception:
            failed += 1
            continue
        for name in values:
            value = metrics.get(name)
            if value is not None and np.isfinite(float(value)):
                values[name].append(float(value))

    intervals: dict[str, Any] = {}
    for name, metric_values in values.items():
        if metric_values:
            array = np.asarray(metric_values, dtype=float)
            intervals[name] = {
                "successful_replicates": int(array.size),
                "skipped_replicates": int(n_replicates - array.size),
                "failed_replicates": int(failed),
                "ci_lower": float(np.percentile(array, 2.5)),
                "ci_upper": float(np.percentile(array, 97.5)),
            }
        else:
            intervals[name] = {
                "successful_replicates": 0,
                "skipped_replicates": int(n_replicates),
                "failed_replicates": int(failed),
                "ci_lower": None,
                "ci_upper": None,
            }
    return {
        "method": "paired_row_nonparametric_percentile_bootstrap",
        "confidence_level": 0.95,
        "requested_replicates": int(n_replicates),
        "random_state": int(random_state),
        "metric_intervals": intervals,
    }


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries to CSV with a stable schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def update_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Update the durable receipt atomically."""
    atomic_write_json(path, payload)


def finalize_checksums(output_dir: Path) -> dict[str, str]:
    """Hash generated evaluation files except the checksum file itself."""
    checksums = {
        filename: sha256_file(output_dir / filename)
        for filename in EVALUATION_FILENAMES
        if (output_dir / filename).exists()
    }
    atomic_write_json(output_dir / "evaluation_artifact_checksums.json", checksums)
    return checksums


def execute_one_time_evaluation(
    *,
    source_run_id: str,
    readiness: ReadinessResult,
    output_dir: Path,
    final_data_loader: Callable[[], pd.DataFrame],
    bootstrap_replicates: int,
    random_state: int,
) -> dict[str, Any]:
    """Execute the one-time final evaluation after guards have passed."""
    assert_output_dir_allowed(output_dir)
    check_no_existing_receipt(source_run_id, output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    spec = readiness.procedure_spec
    receipt_path = output_dir / "evaluation_receipt.json"
    started_at = datetime.now(UTC).isoformat()
    receipt: dict[str, Any] = {
        "status": "started_before_test_load",
        "source_run_id": source_run_id,
        "procedure_id": spec["selected_procedure_id"],
        "fitted_model_sha256": readiness.model_sha256,
        "procedure_spec_sha256": readiness.spec_sha256,
        "frozen_threshold": spec["selected_decision_threshold"],
        "git_revision": current_git_revision(),
        "started_at_utc": started_at,
        "statement": "This run is intended to consume the final held-out test set.",
    }
    update_receipt(receipt_path, receipt)

    model = joblib.load(readiness.model_path)
    final_df = final_data_loader()
    feature_columns = list(readiness.feature_schema["feature_columns"])
    target_column = str(readiness.feature_schema["target_column"])
    missing = [column for column in feature_columns + [target_column] if column not in final_df.columns]
    if missing:
        raise FinalEvaluationError(f"Final evaluation data is missing columns: {missing}")
    X = final_df[feature_columns].copy()
    y = final_df[target_column].astype(int).to_numpy()
    fingerprint = make_dataframe_fingerprint(X, y)
    receipt.update(
        {
            "status": "test_loaded_and_consumed",
            "test_loaded_at_utc": datetime.now(UTC).isoformat(),
            "test_row_count": int(len(X)),
            "test_data_fingerprint": fingerprint,
        }
    )
    update_receipt(receipt_path, receipt)

    probabilities = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    predictions = np.asarray(model.predict(X), dtype=int)
    metrics, counts = compute_final_metrics(y, probabilities, predictions)
    bootstrap = bootstrap_confidence_intervals(
        y,
        probabilities,
        predictions,
        n_replicates=int(bootstrap_replicates),
        random_state=int(random_state),
    )

    target_distribution = {
        str(label): int(count)
        for label, count in pd.Series(y).value_counts().sort_index().items()
    }
    manifest = {
        "source_run_id": source_run_id,
        "evidence_role": FINAL_EVALUATION_ROLE,
        "procedure_id": spec["selected_procedure_id"],
        "procedure_type": spec["procedure_type"],
        "model_path": str(readiness.model_path),
        "model_sha256": readiness.model_sha256,
        "procedure_spec_path": str(readiness.spec_path),
        "procedure_spec_sha256": readiness.spec_sha256,
        "refit_dir": str(readiness.model_path.parent),
        "source_git_commit": readiness.refit_manifest.get("source_git_commit"),
        "evaluation_git_commit": current_git_revision(),
        "test_row_count": int(len(X)),
        "test_target_distribution": target_distribution,
        "test_data_fingerprint": fingerprint,
        "member_candidate_ids": spec["member_candidate_ids"],
        "member_weights": spec["member_weights"],
        "frozen_threshold": spec["selected_decision_threshold"],
        "threshold_origin": "development_data_oof_f1",
        "calibration_method": spec["calibration_method"],
        "calibration_status": spec["calibration_status"],
        "metric_definitions": {
            "primary_metric": "average_precision",
            "threshold_metrics": "computed using frozen development-selected threshold",
        },
        "bootstrap": {
            "method": bootstrap["method"],
            "random_state": int(random_state),
            "requested_replicates": int(bootstrap_replicates),
        },
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "no_post_selection_actions": (
            "No fitting, tuning, threshold selection, calibration fitting, reweighting, "
            "or model selection occurred during this final evaluation."
        ),
        "test_set_status": "consumed_by_final_evaluation",
        "warning": (
            "Further changes based on final evaluation results would invalidate the "
            "clean final-evaluation interpretation."
        ),
    }
    metric_rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    count_rows = [{"cell": key, "count": value} for key, value in counts.items()]
    prediction_rows = []
    for position, (index, target, probability, prediction) in enumerate(
        zip(X.index, y, probabilities, predictions)
    ):
        row = {
            "row_position": int(position),
            "dataframe_index": str(index),
            "target": int(target),
            "probability_class_1": float(probability),
            "frozen_threshold_prediction": int(prediction),
            "frozen_threshold": float(spec["selected_decision_threshold"]),
        }
        if "customerID" in final_df.columns:
            row["customerID"] = str(final_df.iloc[position]["customerID"])
        prediction_rows.append(row)
    interval_rows = [
        {"metric": metric, **details}
        for metric, details in bootstrap["metric_intervals"].items()
    ]

    atomic_write_json(output_dir / "final_test_manifest.json", manifest)
    atomic_write_json(output_dir / "final_test_metrics.json", metrics)
    write_csv_rows(output_dir / "final_test_metrics.csv", metric_rows, ["metric", "value"])
    atomic_write_json(output_dir / "final_test_confusion_matrix.json", counts)
    write_csv_rows(output_dir / "final_test_confusion_matrix.csv", count_rows, ["cell", "count"])
    write_csv_rows(
        output_dir / "final_test_predictions.csv",
        prediction_rows,
        [
            "row_position",
            "dataframe_index",
            "customerID",
            "target",
            "probability_class_1",
            "frozen_threshold_prediction",
            "frozen_threshold",
        ],
    )
    atomic_write_json(output_dir / "final_test_bootstrap_confidence_intervals.json", bootstrap)
    write_csv_rows(
        output_dir / "final_test_bootstrap_confidence_intervals.csv",
        interval_rows,
        [
            "metric",
            "successful_replicates",
            "skipped_replicates",
            "failed_replicates",
            "ci_lower",
            "ci_upper",
        ],
    )
    report = make_markdown_report(
        manifest=manifest,
        metrics=metrics,
        counts=counts,
        bootstrap=bootstrap,
    )
    (output_dir / "final_test_evaluation_report.md").write_text(report, encoding="utf-8")

    receipt.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "checksum_manifest": "evaluation_artifact_checksums.json",
        }
    )
    update_receipt(receipt_path, receipt)
    finalize_checksums(output_dir)
    return {"metrics": metrics, "confusion_matrix": counts, "manifest": manifest}


def make_markdown_report(
    *,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    counts: Mapping[str, int],
    bootstrap: Mapping[str, Any],
) -> str:
    """Create a concise final-evaluation Markdown report."""
    lines = [
        "# Final Held-Out Evaluation",
        "",
        "## Procedure",
        "",
        f"- Procedure: `{manifest['procedure_id']}`",
        f"- Type: `{manifest['procedure_type']}`",
        f"- Members: `{', '.join(manifest['member_candidate_ids'])}`",
        f"- Frozen threshold: `{manifest['frozen_threshold']}`",
        f"- Calibration: `{manifest['calibration_method']}` / `{manifest['calibration_status']}`",
        "",
        "## Test Set",
        "",
        f"- Rows: `{manifest['test_row_count']}`",
        f"- Class distribution: `{manifest['test_target_distribution']}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Confusion Matrix", ""])
    for key, value in counts.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## 95% Confidence Intervals", ""])
    for metric, details in bootstrap["metric_intervals"].items():
        lines.append(
            f"- {metric}: `[{details['ci_lower']}, {details['ci_upper']}]` "
            f"from `{details['successful_replicates']}` successful replicates"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The procedure was selected using fast development-data evidence.",
            "- The robust protocol-v2 benchmark was not run to completion.",
            "- No post-hoc test-driven adjustments are allowed.",
            "- Confidence intervals reflect finite held-out-sample uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)
