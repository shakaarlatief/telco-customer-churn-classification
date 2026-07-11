"""Smoke test for read-only leading-candidate selection.

The test uses a temporary fake ``metric_summary.csv``. It does not inspect real run
artifacts, fit models, resume workflows, or touch immutable experiment outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from select_leading_candidates import (  # noqa: E402
    LeadingCandidateSelectionError,
    select_leading_candidates,
)


def write_fake_metric_summary(path: Path) -> None:
    """Write fake metric rows with intentionally unsorted candidate performance."""
    rows = [
        {
            "candidate_id": "C01_RIDGE_CLASSIFIER",
            "candidate_display_name": "Ridge classifier",
            "metric": "average_precision",
            "count": "2",
            "mean": "0.61",
            "std": "0.01",
            "median": "0.61",
            "iqr": "0.01",
            "min": "0.60",
            "max": "0.62",
        },
        {
            "candidate_id": "C19_CATBOOST",
            "candidate_display_name": "CatBoost",
            "metric": "average_precision",
            "count": "2",
            "mean": "0.72",
            "std": "0.02",
            "median": "0.72",
            "iqr": "0.02",
            "min": "0.70",
            "max": "0.74",
        },
        {
            "candidate_id": "C17_XGBOOST",
            "candidate_display_name": "XGBoost",
            "metric": "average_precision",
            "count": "2",
            "mean": "0.68",
            "std": "0.03",
            "median": "0.68",
            "iqr": "0.03",
            "min": "0.65",
            "max": "0.71",
        },
        {
            "candidate_id": "C02_LOGISTIC_REGRESSION",
            "candidate_display_name": "Regularized logistic regression",
            "metric": "roc_auc",
            "count": "2",
            "mean": "0.85",
            "std": "0.01",
            "median": "0.85",
            "iqr": "0.01",
            "min": "0.84",
            "max": "0.86",
        },
    ]
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one output CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_selection_outputs() -> None:
    """Verify rank order, top-k selection, and output file contents."""
    with tempfile.TemporaryDirectory(prefix="leading-selection-smoke-") as temporary:
        root = Path(temporary)
        summary_dir = root / "summaries"
        output_dir = root / "selection"
        summary_dir.mkdir()
        metric_summary_path = summary_dir / "metric_summary.csv"
        write_fake_metric_summary(metric_summary_path)

        selected, ranking, metadata = select_leading_candidates(
            metric_summary_path=metric_summary_path,
            output_dir=output_dir,
            run_id="fake_fast_run",
            metric="average_precision",
            top_k=2,
        )

        if [row["candidate_id"] for row in ranking] != [
            "C19_CATBOOST",
            "C17_XGBOOST",
            "C01_RIDGE_CLASSIFIER",
        ]:
            raise AssertionError("Ranking order should sort by mean average precision.")
        if [row["candidate_id"] for row in selected] != [
            "C19_CATBOOST",
            "C17_XGBOOST",
        ]:
            raise AssertionError("Top-k selected candidates are incorrect.")
        if metadata["source_evidence_role"] != "fast_completion_pipeline_evidence":
            raise AssertionError("Metadata must preserve fast-completion evidence role.")
        if metadata["held_out_test_policy"] != "not_loaded_or_referenced":
            raise AssertionError("Metadata must record no held-out test access.")

        for filename in (
            "leading_candidates.json",
            "leading_candidates.csv",
            "ranking_table.csv",
        ):
            if not (output_dir / filename).exists():
                raise AssertionError(f"Expected output file was not written: {filename}")

        json_payload = json.loads((output_dir / "leading_candidates.json").read_text(encoding="utf-8"))
        if json_payload["metadata"]["source_run_id"] != "fake_fast_run":
            raise AssertionError("JSON metadata must record source run ID.")
        if len(json_payload["selected_candidates"]) != 2:
            raise AssertionError("JSON output must contain two selected candidates.")

        csv_selected = read_csv_rows(output_dir / "leading_candidates.csv")
        if [row["candidate_id"] for row in csv_selected] != ["C19_CATBOOST", "C17_XGBOOST"]:
            raise AssertionError("Selected CSV output is incorrect.")
        csv_ranking = read_csv_rows(output_dir / "ranking_table.csv")
        if len(csv_ranking) != 3:
            raise AssertionError("Ranking CSV should include all metric candidate rows.")


def assert_error_paths() -> None:
    """Verify core validation failures raise clear selection errors."""
    with tempfile.TemporaryDirectory(prefix="leading-selection-errors-") as temporary:
        root = Path(temporary)
        metric_summary_path = root / "missing.csv"
        try:
            select_leading_candidates(
                metric_summary_path=metric_summary_path,
                output_dir=root / "out",
                run_id="fake",
            )
        except LeadingCandidateSelectionError as exc:
            if "missing" not in str(exc).lower():
                raise AssertionError(f"Unexpected missing-file error: {exc}") from exc
        else:
            raise AssertionError("Missing metric_summary.csv should fail.")

        existing = root / "metric_summary.csv"
        write_fake_metric_summary(existing)
        for kwargs, expected_text in (
            ({"top_k": 0}, "top-k"),
            ({"top_k": 99}, "exceeds"),
            ({"metric": "not_a_metric"}, "absent"),
        ):
            try:
                select_leading_candidates(
                    metric_summary_path=existing,
                    output_dir=root / "out",
                    run_id="fake",
                    **kwargs,
                )
            except LeadingCandidateSelectionError as exc:
                if expected_text not in str(exc):
                    raise AssertionError(f"Unexpected validation error: {exc}") from exc
            else:
                raise AssertionError(f"Expected validation failure for {kwargs!r}.")


def assert_no_loader_tokens() -> None:
    """Static guard that the selector does not import known held-out loading helpers."""
    source = (PROJECT_ROOT / "scripts" / "select_leading_candidates.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("load_" + "test_data", "split_" + "test")
    for token in forbidden:
        if token in source:
            raise AssertionError(f"Selector must not import or call {token}.")


def main() -> None:
    """Run the leading-candidate selection smoke test."""
    assert_selection_outputs()
    assert_error_paths()
    assert_no_loader_tokens()
    print("Leading-candidate selection smoke test passed.")


if __name__ == "__main__":
    main()
