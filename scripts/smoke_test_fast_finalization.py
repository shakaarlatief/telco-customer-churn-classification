"""Smoke tests for the fast-finalization scaffold.

The test uses temporary synthetic fixtures only. It does not inspect immutable run
artifacts, run experiments, fit real project models, or write under project artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from run_fast_finalization import (  # noqa: E402
    SOURCE_EVIDENCE_ROLE,
    build_ensemble_outputs,
    compute_metrics,
    load_leading_candidates,
    print_dry_run,
    select_final_procedure,
)


def write_fake_leading_file(path: Path) -> None:
    """Write a minimal leading-candidate JSON fixture."""
    payload = {
        "metadata": {
            "source_run_id": "fake_fast_completion",
            "source_evidence_role": SOURCE_EVIDENCE_ROLE,
            "held_out_test_policy": "not_loaded_or_referenced",
        },
        "selected_candidates": [
            {
                "rank": 1,
                "candidate_id": "C03_SPLINE_LOGISTIC_REGRESSION",
                "candidate_display_name": "Spline logistic regression",
                "mean": 0.67,
            },
            {
                "rank": 2,
                "candidate_id": "C20_EXPLAINABLE_BOOSTING_MACHINE",
                "candidate_display_name": "Explainable Boosting Machine",
                "mean": 0.66,
            },
            {
                "rank": 3,
                "candidate_id": "C18_LIGHTGBM",
                "candidate_display_name": "LightGBM",
                "mean": 0.65,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def assert_leading_parsing_and_dry_run() -> None:
    """Verify selected-candidate parsing and artifact-free dry-run."""
    with tempfile.TemporaryDirectory(prefix="fast-finalization-dry-run-") as temporary:
        root = Path(temporary)
        leading_path = root / "leading_candidates.json"
        output_dir = root / "outputs"
        write_fake_leading_file(leading_path)
        selected, metadata = load_leading_candidates(leading_path, top_k=2)
        if metadata["source_evidence_role"] != SOURCE_EVIDENCE_ROLE:
            raise AssertionError("Leading metadata must preserve source evidence role.")
        if [row["candidate_id"] for row in selected] != [
            "C03_SPLINE_LOGISTIC_REGRESSION",
            "C20_EXPLAINABLE_BOOSTING_MACHINE",
        ]:
            raise AssertionError("Leading parser returned the wrong top-k candidates.")

        print_dry_run(
            source_run_id="fake_fast_completion",
            leading_candidates_file=leading_path,
            output_dir=output_dir,
            top_k=2,
        )
        if output_dir.exists():
            raise AssertionError("Dry-run must not create the output directory.")


def assert_metric_and_selection_logic() -> None:
    """Verify AP ranking and the simplicity tie-break rule."""
    y_true = [0, 1, 0, 1, 1, 0]
    scores = [0.05, 0.8, 0.2, 0.65, 0.7, 0.1]
    metrics = compute_metrics(y_true, scores, score_kind="probability")
    if metrics["average_precision"] <= 0.99:
        raise AssertionError("Synthetic AP should be high for ordered probabilities.")
    if metrics["n_oof_rows"] != 6:
        raise AssertionError("Metric summary must preserve OOF row count.")

    individual = {
        "procedure_id": "C03_SPLINE_LOGISTIC_REGRESSION",
        "procedure_type": "individual",
        "candidate_id": "C03_SPLINE_LOGISTIC_REGRESSION",
        "candidate_display_name": "Spline logistic regression",
        "score_kind": "probability",
        "average_precision": 0.700,
    }
    ensemble = {
        "procedure_id": "top3_unweighted_soft_average",
        "procedure_type": "ensemble",
        "candidate_id": "",
        "candidate_display_name": "top3_unweighted_soft_average",
        "score_kind": "probability",
        "average_precision": 0.701,
    }
    selection = select_final_procedure([individual], [ensemble], tolerance=0.002)
    if selection["selected_procedure"]["procedure_type"] != "individual":
        raise AssertionError("Simplicity tie-break should prefer close individual model.")
    if not selection["tie_break_applied"]:
        raise AssertionError("Tie-break metadata should record that it was applied.")


def assert_ensemble_logic() -> None:
    """Verify probability averaging and ensemble output schemas."""
    candidate_metrics = [
        {
            "procedure_id": "C_A",
            "procedure_type": "individual",
            "candidate_id": "C_A",
            "candidate_display_name": "A",
            "score_kind": "probability",
            "average_precision": 0.80,
        },
        {
            "procedure_id": "C_B",
            "procedure_type": "individual",
            "candidate_id": "C_B",
            "candidate_display_name": "B",
            "score_kind": "probability",
            "average_precision": 0.70,
        },
        {
            "procedure_id": "C_C",
            "procedure_type": "individual",
            "candidate_id": "C_C",
            "candidate_display_name": "C",
            "score_kind": "probability",
            "average_precision": 0.60,
        },
    ]
    candidate_oof_rows = []
    targets = [0, 1, 0, 1]
    probabilities = {
        "C_A": [0.1, 0.9, 0.2, 0.8],
        "C_B": [0.2, 0.8, 0.3, 0.7],
        "C_C": [0.3, 0.7, 0.4, 0.6],
    }
    for candidate_id, values in probabilities.items():
        for position, probability in enumerate(values):
            candidate_oof_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_display_name": candidate_id,
                    "fold": 1,
                    "row_position": position,
                    "row_index": str(position),
                    "target": targets[position],
                    "score": probability,
                    "probability": probability,
                    "prediction": int(probability >= 0.5),
                    "score_kind": "probability",
                }
            )
    protocol = {
        "ensembles": [
            {
                "ensemble_id": "top3_unweighted_soft_average",
                "minimum_probability_candidates": 3,
                "member_count": 3,
                "weighting": "uniform",
            }
        ]
    }
    ensemble_metrics, ensemble_rows = build_ensemble_outputs(
        candidate_metrics=candidate_metrics,
        candidate_oof_rows=candidate_oof_rows,
        protocol=protocol,
    )
    if len(ensemble_metrics) != 1:
        raise AssertionError("Expected one top-3 ensemble metric row.")
    if len(ensemble_rows) != 4:
        raise AssertionError("Expected one ensemble OOF row per synthetic sample.")
    if abs(float(ensemble_rows[0]["probability"]) - 0.2) > 1e-12:
        raise AssertionError("Unweighted ensemble probability average is incorrect.")


def assert_no_loader_tokens() -> None:
    """Static guard that new scripts do not import known final-evaluation helpers."""
    for script_name in ("run_fast_finalization.py", "smoke_test_fast_finalization.py"):
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        for token in ("load_" + "test_data", "split_" + "test"):
            if token in source:
                raise AssertionError(f"{script_name} must not import or call {token}.")


def main() -> None:
    """Run the fast-finalization smoke test."""
    assert_leading_parsing_and_dry_run()
    assert_metric_and_selection_logic()
    assert_ensemble_logic()
    assert_no_loader_tokens()
    print("Fast-finalization smoke test passed.")


if __name__ == "__main__":
    main()
