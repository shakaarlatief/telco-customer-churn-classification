"""Synthetic smoke tests for the guarded final held-out evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluate_final_held_out_test import validate_confirmation_phrase  # noqa: E402
from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.experiment_protocol import make_dataframe_fingerprint  # noqa: E402
from telco_churn.final_evaluation import (  # noqa: E402
    CONFIRMATION_PHRASE,
    FinalEvaluationError,
    bootstrap_confidence_intervals,
    check_no_existing_receipt,
    compute_final_metrics,
    execute_one_time_evaluation,
    run_readiness_audit,
    sha256_file,
)
from telco_churn.final_procedure import FrozenProbabilityVotingEnsemble  # noqa: E402


class DummyProbabilityEstimator:
    """Small picklable estimator for synthetic final-evaluation tests."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray([0, 1], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        """Return deterministic probabilities for the requested row count."""
        n_rows = len(X)
        positive = self.probabilities[:n_rows]
        return np.column_stack([1.0 - positive, positive])


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def make_refit_fixture(root: Path, *, mismatch: bool = False, corrupt_checksum: bool = False) -> tuple[Path, Path]:
    """Create synthetic frozen/refit artifacts."""
    refit_dir = root / "final_development_refit_v1"
    refit_dir.mkdir(parents=True)
    member_ids = [
        "C03_SPLINE_LOGISTIC_REGRESSION",
        "C25_FT_TRANSFORMER",
        "C20_EXPLAINABLE_BOOSTING_MACHINE",
    ]
    if mismatch:
        spec_member_ids = list(reversed(member_ids))
    else:
        spec_member_ids = member_ids
    spec = {
        "selected_procedure_id": "top3_unweighted_soft_average",
        "procedure_type": "ensemble",
        "ensemble_aggregation": "arithmetic_mean_of_probabilities",
        "member_candidate_ids": spec_member_ids,
        "member_display_names": ["Spline logistic regression", "FT-Transformer", "EBM"],
        "member_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        "selected_decision_threshold": 0.4,
        "calibration_method": "none",
        "calibration_status": "deferred_fast_completion",
        "held_out_test_policy": "not_loaded_or_referenced",
    }
    model = FrozenProbabilityVotingEnsemble(
        member_ids=tuple(member_ids),
        member_display_names=("Spline logistic regression", "FT-Transformer", "EBM"),
        member_weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        estimators=(
            DummyProbabilityEstimator([0.1, 0.8, 0.2, 0.7]),
            DummyProbabilityEstimator([0.2, 0.7, 0.3, 0.8]),
            DummyProbabilityEstimator([0.3, 0.9, 0.4, 0.6]),
        ),
        decision_threshold=0.4,
    )
    X_dev = pd.DataFrame({"tenure": [1, 2, 3, 4], "MonthlyCharges": [10.0, 20.0, 30.0, 40.0]})
    y_dev = pd.Series([0, 1, 0, 1])
    manifest = {
        "training_row_count": 5634,
        "held_out_test_policy": "not_loaded_or_referenced",
        "source_git_commit": "synthetic",
        "member_candidate_ids": member_ids,
        "decision_threshold": 0.4,
        "calibration_method": "none",
        "calibration_status": "deferred_fast_completion",
        "development_data_fingerprint": make_dataframe_fingerprint(X_dev, y_dev),
    }
    feature_schema = {
        "feature_columns": ["tenure", "MonthlyCharges"],
        "feature_dtypes": {"tenure": "int64", "MonthlyCharges": "float64"},
        "target_column": "Churn_binary",
        "target_dtype": "int64",
    }
    roundtrip = {
        "probabilities_allclose": True,
        "predictions_equal": True,
        "max_probability_abs_diff": 0.0,
    }
    write_json(refit_dir / "final_procedure_spec.json", spec)
    write_json(refit_dir / "final_refit_manifest.json", manifest)
    write_json(refit_dir / "feature_schema.json", feature_schema)
    write_json(refit_dir / "roundtrip_validation.json", roundtrip)
    write_json(refit_dir / "model_environment.json", {"packages": {}})
    joblib.dump(model, refit_dir / "fitted_final_pipeline.joblib")
    checksums = {
        filename: sha256_file(refit_dir / filename)
        for filename in (
            "feature_schema.json",
            "final_procedure_spec.json",
            "final_refit_manifest.json",
            "fitted_final_pipeline.joblib",
            "model_environment.json",
            "roundtrip_validation.json",
        )
    }
    if corrupt_checksum:
        checksums["fitted_final_pipeline.joblib"] = "bad"
    write_json(refit_dir / "artifact_checksums.json", checksums)
    return refit_dir, refit_dir / "final_procedure_spec.json"


def synthetic_final_data() -> pd.DataFrame:
    """Return a tiny synthetic final-evaluation dataframe."""
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d"],
            "tenure": [1, 2, 3, 4],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
            "Churn_binary": [0, 1, 0, 1],
        }
    )


def assert_metric_and_bootstrap_logic() -> None:
    """Verify metrics, confusion counts, and reproducible bootstrap schema."""
    y = [0, 1, 0, 1]
    p = [0.2, 0.8, 0.3, 0.6]
    pred = [0, 1, 0, 1]
    metrics, counts = compute_final_metrics(y, p, pred)
    if counts != {"true_negatives": 2, "false_positives": 0, "false_negatives": 0, "true_positives": 2}:
        raise AssertionError("Confusion matrix counts are incorrect.")
    if metrics["specificity"] != 1.0 or metrics["negative_predictive_value"] != 1.0:
        raise AssertionError("Specificity or NPV calculation is incorrect.")
    first = bootstrap_confidence_intervals(y, p, pred, n_replicates=20, random_state=RANDOM_STATE)
    second = bootstrap_confidence_intervals(y, p, pred, n_replicates=20, random_state=RANDOM_STATE)
    if first != second:
        raise AssertionError("Bootstrap output should be reproducible for a fixed seed.")
    if "average_precision" not in first["metric_intervals"]:
        raise AssertionError("Bootstrap output is missing average precision.")


def assert_readiness_and_failure_modes() -> None:
    """Verify readiness, mismatch failure, checksum failure, and receipt guard."""
    with tempfile.TemporaryDirectory(prefix="final-eval-readiness-") as temporary:
        root = Path(temporary)
        refit_dir, spec_path = make_refit_fixture(root)
        readiness = run_readiness_audit(
            source_run_id="synthetic",
            procedure_spec_path=spec_path,
            refit_dir=refit_dir,
        )
        if not readiness.ready:
            raise AssertionError("Synthetic readiness fixture should pass.")

        bad_refit_dir, bad_spec_path = make_refit_fixture(root / "bad_mismatch", mismatch=True)
        try:
            run_readiness_audit(
                source_run_id="synthetic",
                procedure_spec_path=bad_spec_path,
                refit_dir=bad_refit_dir,
            )
        except FinalEvaluationError as exc:
            if "member" not in str(exc):
                raise AssertionError(f"Unexpected mismatch failure: {exc}") from exc
        else:
            raise AssertionError("Model/spec member mismatch should fail readiness.")

        checksum_refit_dir, checksum_spec_path = make_refit_fixture(
            root / "bad_checksum",
            corrupt_checksum=True,
        )
        try:
            run_readiness_audit(
                source_run_id="synthetic",
                procedure_spec_path=checksum_spec_path,
                refit_dir=checksum_refit_dir,
            )
        except FinalEvaluationError as exc:
            if "Checksum mismatch" not in str(exc):
                raise AssertionError(f"Unexpected checksum failure: {exc}") from exc
        else:
            raise AssertionError("Checksum mismatch should fail readiness.")

        receipt_dir = root / "existing_receipt"
        receipt_dir.mkdir()
        write_json(receipt_dir / "evaluation_receipt.json", {"status": "completed"})
        try:
            check_no_existing_receipt("synthetic", receipt_dir)
        except FinalEvaluationError:
            pass
        else:
            raise AssertionError("Existing receipt should block evaluation.")


def assert_confirmation_and_synthetic_execution() -> None:
    """Verify confirmation enforcement and synthetic one-time execution outputs."""
    try:
        validate_confirmation_phrase("wrong")
    except FinalEvaluationError:
        pass
    else:
        raise AssertionError("Wrong confirmation phrase should fail.")
    validate_confirmation_phrase(CONFIRMATION_PHRASE)

    with tempfile.TemporaryDirectory(prefix="final-eval-execution-") as temporary:
        root = Path(temporary)
        refit_dir, spec_path = make_refit_fixture(root)
        readiness = run_readiness_audit(
            source_run_id="synthetic",
            procedure_spec_path=spec_path,
            refit_dir=refit_dir,
        )
        output_dir = root / "final_evaluation" / "synthetic" / "held_out_test_v1"
        loader_called = {"value": False, "receipt_status_before_load": None}

        def loader() -> pd.DataFrame:
            loader_called["value"] = True
            receipt = json.loads((output_dir / "evaluation_receipt.json").read_text(encoding="utf-8"))
            loader_called["receipt_status_before_load"] = receipt["status"]
            return synthetic_final_data()

        result = execute_one_time_evaluation(
            source_run_id="synthetic",
            readiness=readiness,
            output_dir=output_dir,
            final_data_loader=loader,
            bootstrap_replicates=20,
            random_state=RANDOM_STATE,
        )
        if not loader_called["value"]:
            raise AssertionError("Synthetic final data loader should be called during execution.")
        if loader_called["receipt_status_before_load"] != "started_before_test_load":
            raise AssertionError("Receipt must be written before final data load.")
        receipt = json.loads((output_dir / "evaluation_receipt.json").read_text(encoding="utf-8"))
        if receipt["status"] != "completed":
            raise AssertionError("Receipt should finish in completed status.")
        for filename in (
            "final_test_manifest.json",
            "final_test_metrics.json",
            "final_test_metrics.csv",
            "final_test_confusion_matrix.json",
            "final_test_confusion_matrix.csv",
            "final_test_predictions.csv",
            "final_test_bootstrap_confidence_intervals.json",
            "final_test_bootstrap_confidence_intervals.csv",
            "final_test_evaluation_report.md",
            "evaluation_artifact_checksums.json",
        ):
            if not (output_dir / filename).exists():
                raise AssertionError(f"Expected synthetic output file missing: {filename}")
        if result["metrics"]["average_precision"] is None:
            raise AssertionError("Synthetic execution should compute primary metric.")


def assert_no_forbidden_imports() -> None:
    """Verify only the evaluator script imports the project final-data loader."""
    allowed = PROJECT_ROOT / "scripts" / "evaluate_final_held_out_test.py"
    for path in (
        PROJECT_ROOT / "src" / "telco_churn" / "final_evaluation.py",
        PROJECT_ROOT / "scripts" / "audit_final_test_readiness.py",
        PROJECT_ROOT / "scripts" / "smoke_test_final_held_out_evaluation.py",
    ):
        source = path.read_text(encoding="utf-8")
        for token in ("load_" + "test_data", "TEST" + "_DATA_PATH"):
            if token in source:
                raise AssertionError(f"{path.name} must not contain {token}.")
    evaluator_source = allowed.read_text(encoding="utf-8")
    if "load_" + "test_data" not in evaluator_source:
        raise AssertionError("Evaluator script should be the only new final loader import.")


def main() -> None:
    """Run synthetic final-evaluation smoke tests."""
    assert_metric_and_bootstrap_logic()
    assert_readiness_and_failure_modes()
    assert_confirmation_and_synthetic_execution()
    assert_no_forbidden_imports()
    print("Final held-out evaluation smoke test passed.")


if __name__ == "__main__":
    main()
