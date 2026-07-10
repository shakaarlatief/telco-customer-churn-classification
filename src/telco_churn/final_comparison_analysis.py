"""Read-only analysis helpers for persisted final-comparison runs.

The functions in this module inspect existing run artifacts only. They never resume
Optuna studies, fit models, mutate task registries, or write experiment artifacts. Any
CSV export is owned by the caller and is intentionally separate from the immutable run
directory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "final_comparison"
NON_SELECTION_ROLE_FRAGMENT = "non-selection"
METRIC_NAMES = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier_score",
    "log_loss",
)
TIMING_NAMES = ("inner_search", "outer_fit", "outer_prediction", "total_completed")
NON_COMPLETED_STATES = ("pending", "running", "failed", "interrupted")


@dataclass(frozen=True)
class CompletedTaskEntry:
    """One checksum-verified completed task result and its registry identity."""

    task_key: str
    candidate_id: str
    repeat_index: int
    fold_index: int
    split_hash: str
    result_path: Path
    result: Mapping[str, Any]


@dataclass(frozen=True)
class FinalComparisonRunAnalysis:
    """Read-only material loaded from one final-comparison run directory."""

    run_id: str
    run_directory: Path
    manifest: Mapping[str, Any]
    evidence_role: str
    completed_entries: tuple[CompletedTaskEntry, ...]
    task_state_counts: Mapping[str, int]
    candidate_task_state_counts: Mapping[str, Mapping[str, int]]
    integrity_problems: tuple[str, ...]


def open_read_only_sqlite(path: Path) -> sqlite3.Connection:
    """Open a SQLite database with an OS-enforced read-only URI."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read one JSON object without tolerating malformed or non-mapping payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_rows(run_directory: Path) -> list[dict[str, Any]]:
    registry_path = run_directory / "task_registry.sqlite"
    if not registry_path.exists():
        raise FileNotFoundError(f"Task registry does not exist: {registry_path}")

    with closing(open_read_only_sqlite(registry_path)) as connection:
        cursor = connection.execute(
            """
            SELECT task_key, candidate_id, repeat_index, fold_index, split_hash, status,
                   result_path, result_sha256
            FROM tasks
            ORDER BY candidate_id, repeat_index, fold_index, task_key
            """
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def derive_evidence_role(manifest: Mapping[str, Any]) -> str:
    """Classify the manifest's scientific role without using performance values."""
    protocol = manifest.get("protocol", {})
    metadata: Mapping[str, Any] = {}
    if isinstance(protocol, Mapping) and isinstance(protocol.get("metadata"), Mapping):
        metadata = protocol["metadata"]

    role_text = " ".join(
        str(value)
        for value in (
            metadata.get("workflow_role"),
            metadata.get("purpose"),
            metadata.get("candidate_set_role"),
            protocol.get("protocol_id") if isinstance(protocol, Mapping) else None,
        )
        if value is not None
    ).lower()

    explicit_role = metadata.get("workflow_role")
    if "admission" in role_text and "smoke" in role_text:
        return "non-selection implementation-admission evidence"
    if "search-budget" in role_text or "calibration" in role_text:
        return "non-selection search-budget calibration/runtime evidence"
    if explicit_role:
        return str(explicit_role)
    return "unknown or future final-comparison evidence role"


def _validate_result_identity(
    *,
    result: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    expected = {
        "candidate_id": str(row.get("candidate_id")),
        "outer_repeat_index": int(row.get("repeat_index")),
        "outer_fold_index": int(row.get("fold_index")),
        "split_hash": str(row.get("split_hash")),
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            problems.append(
                f"{row.get('task_key')}: result identity mismatch for {key}: "
                f"observed {result.get(key)!r}, expected {expected_value!r}."
            )
    return problems


def load_final_comparison_run(
    *,
    run_id: str,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> FinalComparisonRunAnalysis:
    """Load checksum-verified completed task artifacts from one run directory."""
    run_directory = Path(artifacts_root) / run_id
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")

    manifest = read_json_mapping(manifest_path)
    rows = _registry_rows(run_directory)
    state_counts: Counter[str] = Counter()
    candidate_state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    completed: list[CompletedTaskEntry] = []
    problems: list[str] = []

    for row in rows:
        task_key = str(row.get("task_key"))
        candidate_id = str(row.get("candidate_id"))
        status = str(row.get("status"))
        state_counts[status] += 1
        candidate_state_counts[candidate_id][status] += 1
        if status != "completed":
            continue

        result_path_value = row.get("result_path")
        recorded_sha = row.get("result_sha256")
        if not result_path_value or not recorded_sha:
            problems.append(f"{task_key}: completed registry row lacks result path or SHA-256.")
            continue

        result_path = run_directory / str(result_path_value)
        if not result_path.exists():
            problems.append(f"{task_key}: result artifact is missing: {result_path}.")
            continue
        observed_sha = _sha256_file(result_path)
        if observed_sha != str(recorded_sha):
            problems.append(f"{task_key}: result artifact SHA-256 mismatch.")
            continue

        try:
            wrapper = read_json_mapping(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"{task_key}: cannot read result artifact: {type(exc).__name__}: {exc}")
            continue
        result = wrapper.get("result")
        if not isinstance(result, Mapping):
            problems.append(f"{task_key}: result artifact has no mapping payload['result'].")
            continue
        identity_problems = _validate_result_identity(result=result, row=row)
        if identity_problems:
            problems.extend(identity_problems)
            continue

        completed.append(
            CompletedTaskEntry(
                task_key=task_key,
                candidate_id=candidate_id,
                repeat_index=int(row.get("repeat_index")),
                fold_index=int(row.get("fold_index")),
                split_hash=str(row.get("split_hash")),
                result_path=result_path,
                result=dict(result),
            )
        )

    return FinalComparisonRunAnalysis(
        run_id=run_id,
        run_directory=run_directory,
        manifest=manifest,
        evidence_role=derive_evidence_role(manifest),
        completed_entries=tuple(completed),
        task_state_counts=dict(sorted(state_counts.items())),
        candidate_task_state_counts={
            candidate: dict(sorted(counter.items()))
            for candidate, counter in sorted(candidate_state_counts.items())
        },
        integrity_problems=tuple(problems),
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return _finite_float(value) is not None


def distribution_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    """Return compact descriptive statistics for finite numeric values."""
    numbers = sorted(value for value in (_finite_float(item) for item in values) if value is not None)
    count = len(numbers)
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    median = statistics.median(numbers)
    lower = numbers[: count // 2]
    upper = numbers[(count + 1) // 2 :] if count % 2 else numbers[count // 2 :]
    q1 = statistics.median(lower) if lower else numbers[0]
    q3 = statistics.median(upper) if upper else numbers[-1]
    return {
        "count": count,
        "mean": statistics.mean(numbers),
        "std": statistics.stdev(numbers) if count > 1 else 0.0,
        "median": median,
        "iqr": q3 - q1,
        "min": numbers[0],
        "max": numbers[-1],
    }


def summarize_metrics(entries: Sequence[CompletedTaskEntry]) -> list[dict[str, Any]]:
    """Aggregate persisted outer-task metrics by candidate and metric name."""
    values: dict[tuple[str, str], list[Any]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for entry in entries:
        result = entry.result
        display_names[entry.candidate_id] = str(
            result.get("candidate_display_name") or entry.candidate_id
        )
        metrics = result.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in METRIC_NAMES:
            if metric_name in metrics and metrics[metric_name] is not None:
                values[(entry.candidate_id, metric_name)].append(metrics[metric_name])

    rows: list[dict[str, Any]] = []
    for (candidate_id, metric_name), metric_values in sorted(values.items()):
        row = {
            "candidate_id": candidate_id,
            "candidate_display_name": display_names.get(candidate_id, candidate_id),
            "metric": metric_name,
        }
        row.update(distribution_summary(metric_values))
        rows.append(row)
    return rows


def _candidate_non_completed_counts(
    analysis: FinalComparisonRunAnalysis,
    candidate_id: str,
) -> dict[str, int]:
    counts = analysis.candidate_task_state_counts.get(candidate_id, {})
    return {state: int(counts.get(state, 0)) for state in NON_COMPLETED_STATES}


def summarize_runtime(analysis: FinalComparisonRunAnalysis) -> list[dict[str, Any]]:
    """Aggregate persisted timing fields and append per-candidate task-state counts."""
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for entry in analysis.completed_entries:
        result = entry.result
        display_names[entry.candidate_id] = str(
            result.get("candidate_display_name") or entry.candidate_id
        )
        timing = result.get("timing_seconds", {})
        if not isinstance(timing, Mapping):
            continue
        component_values: dict[str, float] = {}
        for component in ("inner_search", "outer_fit", "outer_prediction"):
            numeric = _finite_float(timing.get(component))
            if numeric is not None:
                component_values[component] = numeric
                values[(entry.candidate_id, component)].append(numeric)
        if component_values:
            values[(entry.candidate_id, "total_completed")].append(
                sum(component_values.values())
            )

    rows: list[dict[str, Any]] = []
    for (candidate_id, timing_component), timing_values in sorted(values.items()):
        row = {
            "candidate_id": candidate_id,
            "candidate_display_name": display_names.get(candidate_id, candidate_id),
            "timing_component": timing_component,
            "completed_task_count": int(
                analysis.candidate_task_state_counts.get(candidate_id, {}).get("completed", 0)
            ),
        }
        row.update(_candidate_non_completed_counts(analysis, candidate_id))
        row.update(distribution_summary(timing_values))
        rows.append(row)
    for candidate_id in sorted(analysis.candidate_task_state_counts):
        row = {
            "candidate_id": candidate_id,
            "candidate_display_name": display_names.get(candidate_id, candidate_id),
            "timing_component": "task_state_counts",
            "completed_task_count": int(
                analysis.candidate_task_state_counts.get(candidate_id, {}).get("completed", 0)
            ),
        }
        row.update(_candidate_non_completed_counts(analysis, candidate_id))
        row.update(distribution_summary(()))
        rows.append(row)
    return rows


def summarize_warnings(entries: Sequence[CompletedTaskEntry]) -> list[dict[str, Any]]:
    """Summarize persisted outer-fit warning messages by candidate."""
    warning_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    display_names: dict[str, str] = {}
    completed_candidates: set[str] = set()
    for entry in entries:
        result = entry.result
        completed_candidates.add(entry.candidate_id)
        display_names[entry.candidate_id] = str(
            result.get("candidate_display_name") or entry.candidate_id
        )
        warnings = result.get("warnings", [])
        if not isinstance(warnings, Sequence) or isinstance(warnings, (str, bytes)):
            continue
        for message in warnings:
            warning_counts[(entry.candidate_id, str(message))].update([entry.task_key])

    rows: list[dict[str, Any]] = []
    for (candidate_id, message), task_counter in sorted(warning_counts.items()):
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_display_name": display_names.get(candidate_id, candidate_id),
                "warning_message": message,
                "task_count": len(task_counter),
                "occurrence_count": sum(task_counter.values()),
            }
        )
    candidates_with_warnings = {candidate_id for candidate_id, _message in warning_counts}
    for candidate_id in sorted(completed_candidates - candidates_with_warnings):
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_display_name": display_names.get(candidate_id, candidate_id),
                "warning_message": "",
                "task_count": 0,
                "occurrence_count": 0,
            }
        )
    return rows


def summarize_selected_parameters(entries: Sequence[CompletedTaskEntry]) -> list[dict[str, Any]]:
    """Summarize selected hyperparameters across completed outer tasks."""
    parameter_values: dict[tuple[str, str], list[Any]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for entry in entries:
        result = entry.result
        display_names[entry.candidate_id] = str(
            result.get("candidate_display_name") or entry.candidate_id
        )
        parameters = result.get("selected_parameters", {})
        if not isinstance(parameters, Mapping):
            continue
        for name, value in parameters.items():
            parameter_values[(entry.candidate_id, str(name))].append(value)

    rows: list[dict[str, Any]] = []
    for (candidate_id, parameter_name), values in sorted(parameter_values.items()):
        numeric_values = [_finite_float(value) for value in values]
        all_numeric = all(value is not None for value in numeric_values)
        if all_numeric and values:
            summary = distribution_summary(numeric_values)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_display_name": display_names.get(candidate_id, candidate_id),
                    "parameter": parameter_name,
                    "summary_type": "numeric",
                    "value": "",
                    "frequency": "",
                    **summary,
                }
            )
            continue

        frequencies = Counter(json.dumps(value, sort_keys=True, default=str) for value in values)
        for rendered_value, frequency in sorted(
            frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_display_name": display_names.get(candidate_id, candidate_id),
                    "parameter": parameter_name,
                    "summary_type": "categorical",
                    "value": json.loads(rendered_value),
                    "frequency": int(frequency),
                    "count": len(values),
                    "mean": "",
                    "std": "",
                    "median": "",
                    "iqr": "",
                    "min": "",
                    "max": "",
                }
            )
    return rows


def collect_oof_predictions(
    entries: Sequence[CompletedTaskEntry],
    *,
    evidence_role: str,
) -> list[dict[str, Any]]:
    """Collect completed outer-validation predictions without row aggregation."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        result = entry.result
        predictions = result.get("outer_validation_predictions", {})
        if not isinstance(predictions, Mapping):
            continue
        positions = list(predictions.get("training_row_positions", []))
        y_true = list(predictions.get("y_true", []))
        y_score = list(predictions.get("y_score", []))
        y_pred = list(predictions.get("y_pred", []))
        lengths = {len(positions), len(y_true), len(y_score), len(y_pred)}
        if len(lengths) != 1:
            continue
        for index, position in enumerate(positions):
            rows.append(
                {
                    "run_evidence_role": evidence_role,
                    "candidate_id": entry.candidate_id,
                    "candidate_display_name": result.get(
                        "candidate_display_name",
                        entry.candidate_id,
                    ),
                    "task_key": entry.task_key,
                    "outer_repeat_index": entry.repeat_index,
                    "outer_fold_index": entry.fold_index,
                    "split_hash": entry.split_hash,
                    "training_row_position": int(position),
                    "y_true": int(y_true[index]),
                    "y_score": float(y_score[index]),
                    "y_pred": int(y_pred[index]),
                    "score_kind": result.get("score_kind"),
                }
            )
    return rows


def task_state_summary_rows(analysis: FinalComparisonRunAnalysis) -> list[dict[str, Any]]:
    """Return long-form task-state counts by candidate, plus an all-candidate row."""
    rows: list[dict[str, Any]] = []
    for status, count in sorted(analysis.task_state_counts.items()):
        rows.append({"candidate_id": "__ALL__", "status": status, "count": int(count)})
    for candidate_id, counts in sorted(analysis.candidate_task_state_counts.items()):
        for status, count in sorted(counts.items()):
            rows.append({"candidate_id": candidate_id, "status": status, "count": int(count)})
    return rows


def is_non_selection_evidence(evidence_role: str) -> bool:
    """Return whether a run role should be reported as non-selection evidence."""
    return NON_SELECTION_ROLE_FRAGMENT in evidence_role.lower()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write mappings as a CSV file, creating a header even for empty rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_analysis_outputs(
    *,
    analysis: FinalComparisonRunAnalysis,
    output_dir: Path,
) -> None:
    """Write read-only derived summaries outside the immutable run artifact directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metric_summary.csv", summarize_metrics(analysis.completed_entries))
    write_csv(output_dir / "runtime_summary.csv", summarize_runtime(analysis))
    write_csv(output_dir / "warning_summary.csv", summarize_warnings(analysis.completed_entries))
    write_csv(
        output_dir / "selected_parameter_summary.csv",
        summarize_selected_parameters(analysis.completed_entries),
    )
    write_csv(
        output_dir / "oof_predictions.csv",
        collect_oof_predictions(
            analysis.completed_entries,
            evidence_role=analysis.evidence_role,
        ),
    )
    write_csv(output_dir / "task_state_summary.csv", task_state_summary_rows(analysis))
    (output_dir / "integrity_problems.txt").write_text(
        "\n".join(analysis.integrity_problems) + ("\n" if analysis.integrity_problems else ""),
        encoding="utf-8",
    )
