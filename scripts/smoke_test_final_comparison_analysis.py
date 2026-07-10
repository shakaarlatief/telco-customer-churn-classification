"""Smoke-test read-only final-comparison artifact analysis.

The test inspects already-created local artifacts only. It does not create experiment
results, resume workflows, fit estimators, or import project data loaders.
"""

from __future__ import annotations

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
)


PREFERRED_RUN_IDS = (
    "search_budget_calibration_v1_warning_clean",
    "admission_smoke_c26_warning_clean_v2",
    "pilot_pruned_f2_v6_io_resilient",
)
REQUIRED_OOF_COLUMNS = {
    "run_evidence_role",
    "candidate_id",
    "task_key",
    "outer_repeat_index",
    "outer_fold_index",
    "split_hash",
    "training_row_position",
    "y_true",
    "y_score",
    "y_pred",
    "score_kind",
}


def _try_load_run(run_id: str):
    run_directory = DEFAULT_ARTIFACTS_ROOT / run_id
    if not run_directory.exists():
        return None
    try:
        return load_final_comparison_run(run_id=run_id, artifacts_root=DEFAULT_ARTIFACTS_ROOT)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _load_preferred_analysis():
    for run_id in PREFERRED_RUN_IDS:
        analysis = _try_load_run(run_id)
        if analysis is not None and analysis.completed_entries:
            return analysis

    if DEFAULT_ARTIFACTS_ROOT.exists():
        for run_directory in sorted(DEFAULT_ARTIFACTS_ROOT.iterdir()):
            if not run_directory.is_dir():
                continue
            analysis = _try_load_run(run_directory.name)
            if analysis is not None and analysis.completed_entries:
                return analysis
    return None


def _assert_no_data_loader_dependency() -> None:
    checked_paths = (
        PROJECT_ROOT / "src" / "telco_churn" / "final_comparison_analysis.py",
        PROJECT_ROOT / "scripts" / "summarize_final_comparison_run.py",
    )
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        forbidden_loader_name = "load" + "_test" + "_data"
        if forbidden_loader_name in text:
            raise AssertionError(f"Read-only analysis path references a forbidden loader: {path}")


def _assert_completed_summaries(analysis) -> None:
    metric_rows = summarize_metrics(analysis.completed_entries)
    runtime_rows = summarize_runtime(analysis)
    warning_rows = summarize_warnings(analysis.completed_entries)
    parameter_rows = summarize_selected_parameters(analysis.completed_entries)
    oof_rows = collect_oof_predictions(
        analysis.completed_entries,
        evidence_role=analysis.evidence_role,
    )

    if not metric_rows:
        raise AssertionError("Metric summary is empty despite completed artifacts.")
    if not any(row.get("metric") == "average_precision" for row in metric_rows):
        raise AssertionError("Metric summary does not include average precision.")
    if not runtime_rows:
        raise AssertionError("Runtime summary is empty despite completed artifacts.")
    if not warning_rows:
        raise AssertionError("Warning summary should include zero-warning rows if needed.")
    if not parameter_rows:
        raise AssertionError("Selected-parameter summary is empty despite completed artifacts.")
    if not oof_rows:
        raise AssertionError("OOF prediction export is empty despite completed artifacts.")

    missing_columns = REQUIRED_OOF_COLUMNS - set(oof_rows[0])
    if missing_columns:
        raise AssertionError(f"OOF prediction row is missing columns: {sorted(missing_columns)}")
    if len(oof_rows) != sum(
        int(entry.result.get("n_outer_validation", 0))
        for entry in analysis.completed_entries
    ):
        raise AssertionError("OOF prediction export row count does not match completed tasks.")


def _assert_non_selection_label(analysis) -> None:
    if analysis.run_id in {
        "search_budget_calibration_v1_warning_clean",
        "admission_smoke_c26_warning_clean_v2",
    } and not is_non_selection_evidence(analysis.evidence_role):
        raise AssertionError("Admission/calibration run was not labelled non-selection evidence.")


def _assert_partial_run_handling() -> None:
    analysis = _try_load_run("search_budget_calibration_v1_warning_clean")
    if analysis is None:
        return

    non_completed = sum(
        int(analysis.task_state_counts.get(status, 0))
        for status in ("pending", "running", "failed", "interrupted")
    )
    if non_completed <= 0:
        return

    completed_count = int(analysis.task_state_counts.get("completed", 0))
    if len(analysis.completed_entries) != completed_count:
        raise AssertionError("Partial-run loader included non-completed tasks as completed.")

    metric_candidates = {
        str(row["candidate_id"])
        for row in summarize_metrics(analysis.completed_entries)
    }
    for candidate_id, counts in analysis.candidate_task_state_counts.items():
        if int(counts.get("completed", 0)) == 0 and any(
            int(counts.get(status, 0)) > 0
            for status in ("pending", "running", "failed", "interrupted")
        ):
            if candidate_id in metric_candidates:
                raise AssertionError(
                    f"Candidate {candidate_id} has no completed tasks but appears in metrics."
                )


def main() -> None:
    _assert_no_data_loader_dependency()
    analysis = _load_preferred_analysis()
    if analysis is None:
        print(
            "SKIP: no existing local final-comparison run artifacts with completed tasks "
            "were found."
        )
        return

    if analysis.integrity_problems:
        raise AssertionError(
            "Expected preferred local run to have checksum-valid completed artifacts, "
            f"but found problems: {analysis.integrity_problems[:3]}"
        )

    _assert_completed_summaries(analysis)
    _assert_non_selection_label(analysis)
    _assert_partial_run_handling()
    print(
        "Final-comparison analysis smoke passed: "
        f"run={analysis.run_id}, completed={len(analysis.completed_entries)}, "
        f"role={analysis.evidence_role}"
    )


if __name__ == "__main__":
    main()
