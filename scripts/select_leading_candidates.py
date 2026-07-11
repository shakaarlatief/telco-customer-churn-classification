"""Select leading candidates from read-only final-comparison summary CSVs.

This script consumes derived summary artifacts, not immutable run artifacts. It ranks
candidate families by a chosen development-data metric and writes a transparent leading
set for the next project-completion stage. The default source is the completed
``fast_completion_v1`` summary, whose evidence role is intentionally weaker than the
robust frozen protocol-v2 benchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "fast_completion_v1"
DEFAULT_METRIC = "average_precision"
DEFAULT_SOURCE_EVIDENCE_ROLE = "fast_completion_pipeline_evidence"
REQUIRED_COLUMNS = frozenset(
    {
        "candidate_id",
        "candidate_display_name",
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "max",
    }
)


class LeadingCandidateSelectionError(ValueError):
    """Raised when leading-candidate selection inputs are inconsistent."""


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file into string-keyed rows with its header."""
    if not path.exists():
        raise LeadingCandidateSelectionError(f"Metric summary CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise LeadingCandidateSelectionError(f"Metric summary CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise LeadingCandidateSelectionError(
            f"Metric summary CSV is missing required columns: {missing}."
        )
    return fieldnames, rows


def _float_value(row: Mapping[str, str], column: str) -> float:
    """Parse one required numeric value with a clear row-level error."""
    value = row.get(column, "")
    try:
        return float(value)
    except ValueError as exc:
        raise LeadingCandidateSelectionError(
            f"Column {column!r} for candidate {row.get('candidate_id')!r} "
            f"must be numeric, got {value!r}."
        ) from exc


def _int_value(row: Mapping[str, str], column: str) -> int:
    """Parse one required integer value with a clear row-level error."""
    value = row.get(column, "")
    try:
        return int(float(value))
    except ValueError as exc:
        raise LeadingCandidateSelectionError(
            f"Column {column!r} for candidate {row.get('candidate_id')!r} "
            f"must be integer-like, got {value!r}."
        ) from exc


def load_metric_ranking(metric_summary_path: Path, *, metric: str) -> list[dict[str, Any]]:
    """Load and rank candidate rows for one metric by mean descending."""
    _fieldnames, rows = _read_csv_rows(metric_summary_path)
    metric_rows = [row for row in rows if row.get("metric") == metric]
    if not metric_rows:
        raise LeadingCandidateSelectionError(
            f"Metric {metric!r} is absent from {metric_summary_path}."
        )

    candidate_ids = [row.get("candidate_id", "") for row in metric_rows]
    duplicates = sorted(
        candidate_id
        for candidate_id in set(candidate_ids)
        if candidate_ids.count(candidate_id) > 1
    )
    if duplicates:
        raise LeadingCandidateSelectionError(
            f"Metric {metric!r} has duplicate candidate rows: {duplicates}."
        )

    ranking: list[dict[str, Any]] = []
    for row in metric_rows:
        ranking.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_display_name": row["candidate_display_name"],
                "metric": metric,
                "count": _int_value(row, "count"),
                "mean": _float_value(row, "mean"),
                "std": _float_value(row, "std"),
                "min": _float_value(row, "min"),
                "max": _float_value(row, "max"),
            }
        )

    ranking.sort(
        key=lambda row: (
            -float(row["mean"]),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return ranking


def select_leading_candidates(
    *,
    metric_summary_path: Path,
    output_dir: Path,
    run_id: str,
    metric: str = DEFAULT_METRIC,
    top_k: int = 5,
    source_evidence_role: str = DEFAULT_SOURCE_EVIDENCE_ROLE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rank all candidates and write top-k leading-candidate outputs."""
    if top_k < 1:
        raise LeadingCandidateSelectionError("--top-k must be at least 1.")

    ranking = load_metric_ranking(metric_summary_path, metric=metric)
    if top_k > len(ranking):
        raise LeadingCandidateSelectionError(
            f"--top-k={top_k} exceeds available candidate rows ({len(ranking)})."
        )

    selected = [dict(row) for row in ranking[:top_k]]
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run_id": run_id,
        "source_metric_summary": str(metric_summary_path),
        "source_evidence_role": source_evidence_role,
        "metric": metric,
        "top_k": top_k,
        "selection_rule": (
            f"Filter metric_summary.csv to metric == {metric!r}; sort by mean "
            "descending with candidate_id as a deterministic tie-breaker; select "
            f"the first {top_k} candidates."
        ),
        "warning": (
            "This leading set is selected from fast-completion development-data "
            "evidence. It is not a robust frozen protocol-v2 benchmark result and "
            "is not a final held-out-test result."
        ),
        "held_out_test_policy": "not_loaded_or_referenced",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ranking_table.csv", ranking)
    write_csv(output_dir / "leading_candidates.csv", selected)
    (output_dir / "leading_candidates.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "selected_candidates": selected,
                "ranking_table": ranking,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected, ranking, metadata


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to CSV with stable leading-candidate columns."""
    fieldnames = [
        "rank",
        "candidate_id",
        "candidate_display_name",
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Select leading candidates from a read-only final-comparison metric summary."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--summary-dir",
        default=None,
        help=(
            "Directory containing metric_summary.csv. Defaults to "
            "artifacts/final_comparison_summaries/<run-id>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for leading-candidate outputs. Defaults to "
            "artifacts/final_selection/<run-id>."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run leading-candidate selection and print the selected table."""
    arguments = parse_arguments(argv)
    run_id = str(arguments.run_id)
    summary_dir = (
        Path(arguments.summary_dir)
        if arguments.summary_dir is not None
        else PROJECT_ROOT / "artifacts" / "final_comparison_summaries" / run_id
    )
    output_dir = (
        Path(arguments.output_dir)
        if arguments.output_dir is not None
        else PROJECT_ROOT / "artifacts" / "final_selection" / run_id
    )

    try:
        selected, _ranking, metadata = select_leading_candidates(
            metric_summary_path=summary_dir / "metric_summary.csv",
            output_dir=output_dir,
            run_id=run_id,
            metric=str(arguments.metric),
            top_k=int(arguments.top_k),
        )
    except LeadingCandidateSelectionError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Leading-candidate selection source: {metadata['source_metric_summary']}")
    print(f"Evidence role: {metadata['source_evidence_role']}")
    print(f"Metric: {metadata['metric']}")
    print(f"Top K: {metadata['top_k']}")
    print(f"Output directory: {output_dir}")
    print("Selected candidates:")
    for row in selected:
        print(
            f"  {row['rank']:>2}. {row['candidate_id']} "
            f"({row['candidate_display_name']}): "
            f"mean={row['mean']:.12g}, std={row['std']:.12g}, "
            f"min={row['min']:.12g}, max={row['max']:.12g}, count={row['count']}"
        )


if __name__ == "__main__":
    main()
