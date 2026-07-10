"""Summarize persisted final-comparison run artifacts without executing a workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.final_comparison_analysis import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    collect_oof_predictions,
    is_non_selection_evidence,
    load_final_comparison_run,
    summarize_metrics,
    summarize_runtime,
    summarize_selected_parameters,
    summarize_warnings,
    write_analysis_outputs,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read completed final-comparison task artifacts and print training-only "
            "summaries. This command never resumes a workflow or fits a model."
        )
    )
    parser.add_argument("--run-id", required=True, help="Run directory below artifacts root.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing final-comparison run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for derived CSV summaries. It must be outside the "
            "immutable run artifact directory."
        ),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve()


def _assert_output_outside_run(output_dir: Path, run_directory: Path) -> None:
    output = _resolve(output_dir)
    run = _resolve(run_directory)
    if output == run or run in output.parents:
        raise SystemExit(
            "--output-dir must not be inside the immutable run artifact directory."
        )


def _metric_preview(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    preview = [row for row in metric_rows if row.get("metric") == "average_precision"]
    return sorted(preview, key=lambda row: str(row.get("candidate_id")))[:12]


def main() -> None:
    args = parse_arguments()
    analysis = load_final_comparison_run(
        run_id=str(args.run_id),
        artifacts_root=Path(args.artifacts_root),
    )
    metric_rows = summarize_metrics(analysis.completed_entries)
    runtime_rows = summarize_runtime(analysis)
    warning_rows = summarize_warnings(analysis.completed_entries)
    parameter_rows = summarize_selected_parameters(analysis.completed_entries)
    oof_rows = collect_oof_predictions(
        analysis.completed_entries,
        evidence_role=analysis.evidence_role,
    )

    print(f"Run: {analysis.run_id}")
    print(f"Run directory: {analysis.run_directory}")
    print(f"Evidence role: {analysis.evidence_role}")
    protocol = analysis.manifest.get("protocol", {})
    if isinstance(protocol, dict):
        metadata = protocol.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("purpose"):
            print(f"Purpose: {metadata['purpose']}")
    if is_non_selection_evidence(analysis.evidence_role):
        print(
            "Warning: this run is explicitly non-selection evidence. Do not use these "
            "summaries to rank, select, or eliminate candidate procedures."
        )

    print("Task states:")
    for status, count in sorted(analysis.task_state_counts.items()):
        print(f"  {status}: {count}")
    print(f"Checksum/identity integrity problems: {len(analysis.integrity_problems)}")
    print(f"Completed task artifacts loaded: {len(analysis.completed_entries)}")
    print(f"Metric summary rows: {len(metric_rows)}")
    print(f"Runtime summary rows: {len(runtime_rows)}")
    print(f"Warning summary rows: {len(warning_rows)}")
    print(f"Selected-parameter summary rows: {len(parameter_rows)}")
    print(f"Development OOF prediction rows: {len(oof_rows)}")

    preview = _metric_preview(metric_rows)
    if preview:
        print("Average-precision summary preview by candidate:")
        for row in preview:
            print(
                "  {candidate}: n={count}, mean={mean:.4f}, median={median:.4f}".format(
                    candidate=row["candidate_id"],
                    count=int(row["count"]),
                    mean=float(row["mean"]),
                    median=float(row["median"]),
                )
            )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        _assert_output_outside_run(output_dir, analysis.run_directory)
        write_analysis_outputs(analysis=analysis, output_dir=output_dir)
        print(f"Wrote derived summary CSV files to: {output_dir}")


if __name__ == "__main__":
    main()
