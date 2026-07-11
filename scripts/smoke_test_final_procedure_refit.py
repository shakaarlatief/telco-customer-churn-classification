"""Synthetic structural smoke tests for frozen final-procedure/refit tooling."""

from __future__ import annotations

import csv
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

from freeze_fast_final_procedure import (  # noqa: E402
    build_final_procedure_spec,
    default_output_file,
)
from refit_final_development_pipeline import default_output_dir  # noqa: E402
from telco_churn.final_procedure import FrozenProbabilityVotingEnsemble  # noqa: E402


class DummyProbabilityEstimator:
    """Small picklable binary probability estimator for serialization tests."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray([0, 1], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        """Return deterministic probabilities for the requested row count."""
        n_rows = len(X)
        positive = self.probabilities[:n_rows]
        return np.column_stack([1.0 - positive, positive])


def write_json(path: Path, payload: object) -> None:
    """Write JSON fixture."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write CSV fixture."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_fake_finalization_dir(root: Path) -> Path:
    """Create a minimal finalization fixture with real member configs."""
    finalization_dir = root / "fast_finalization_v1"
    finalization_dir.mkdir()
    selected = {
        "selected_procedure": {
            "procedure_id": "top3_unweighted_soft_average",
            "procedure_type": "ensemble",
            "score_kind": "probability",
            "best_f1_threshold": 0.4,
            "average_precision": 0.67,
        }
    }
    configs = [
        {
            "candidate_id": "C03_SPLINE_LOGISTIC_REGRESSION",
            "candidate_display_name": "Spline logistic regression",
            "search_profile": "full",
            "parameters": {
                "n_knots": 3,
                "degree": 2,
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "none",
                "max_iter": 8000,
                "feature_policy": "F0_RAW",
                "feature_selection_policy": "S0_NONE",
                "imbalance_policy": "I0_NONE",
            },
        },
        {
            "candidate_id": "C20_EXPLAINABLE_BOOSTING_MACHINE",
            "candidate_display_name": "Explainable Boosting Machine",
            "search_profile": "full",
            "parameters": {
                "interactions": 0,
                "outer_bags": 4,
                "learning_rate": 0.03,
                "max_rounds": 2000,
                "early_stopping_rounds": 100,
                "min_samples_leaf": 4,
                "max_leaves": 2,
                "feature_policy": "F0_RAW",
                "feature_selection_policy": "S0_NONE",
                "imbalance_policy": "I0_NONE",
            },
        },
        {
            "candidate_id": "C25_FT_TRANSFORMER",
            "candidate_display_name": "FT-Transformer",
            "search_profile": "full",
            "parameters": {
                "n_blocks": 1,
                "d_block": 96,
                "attention_n_heads": 8,
                "attention_dropout": 0.2,
                "ffn_d_hidden_multiplier": 1.0,
                "ffn_dropout": 0.05,
                "residual_dropout": 0.0,
                "learning_rate": 0.003,
                "weight_decay": 0.0001,
                "batch_size": 256,
                "max_epochs": 200,
                "patience": 20,
                "feature_policy": "F1_DOMAIN_ENRICHED",
                "feature_selection_policy": "S0_NONE",
                "imbalance_policy": "I0_NONE",
            },
        },
    ]
    member_ids = (
        "C03_SPLINE_LOGISTIC_REGRESSION|"
        "C20_EXPLAINABLE_BOOSTING_MACHINE|"
        "C25_FT_TRANSFORMER"
    )
    write_json(finalization_dir / "final_procedure_selection.json", selected)
    write_json(finalization_dir / "tuned_candidate_configs.json", configs)
    write_json(
        finalization_dir / "finalization_manifest.json",
        {
            "development_rows": 5634,
            "source_evidence_role": "fast_completion_pipeline_evidence",
            "evidence_role": "fast_finalization_pipeline_evidence",
        },
    )
    write_csv(
        finalization_dir / "ensemble_oof_predictions.csv",
        [
            {
                "ensemble_id": "top3_unweighted_soft_average",
                "member_candidate_ids": member_ids,
                "row_position": 0,
                "target": 0,
                "probability": 0.2,
            },
            {
                "ensemble_id": "top3_unweighted_soft_average",
                "member_candidate_ids": member_ids,
                "row_position": 1,
                "target": 1,
                "probability": 0.8,
            },
        ],
        ["ensemble_id", "member_candidate_ids", "row_position", "target", "probability"],
    )
    return finalization_dir


def assert_ensemble_prediction_and_roundtrip() -> None:
    """Verify averaging, thresholding, and joblib round trip."""
    ensemble = FrozenProbabilityVotingEnsemble(
        member_ids=("A", "B", "C"),
        member_display_names=("A", "B", "C"),
        member_weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        estimators=(
            DummyProbabilityEstimator([0.2, 0.6]),
            DummyProbabilityEstimator([0.4, 0.6]),
            DummyProbabilityEstimator([0.6, 0.6]),
        ),
        decision_threshold=0.5,
    )
    X = [0, 1]
    probabilities = ensemble.predict_proba(X)
    if not np.allclose(probabilities[:, 1], [0.4, 0.6], atol=1e-12):
        raise AssertionError("Soft-voting probabilities are incorrect.")
    if ensemble.predict(X).tolist() != [0, 1]:
        raise AssertionError("Thresholded predictions are incorrect.")
    with tempfile.TemporaryDirectory(prefix="final-procedure-roundtrip-") as temporary:
        path = Path(temporary) / "model.joblib"
        joblib.dump(ensemble, path)
        loaded = joblib.load(path)
        if not np.allclose(loaded.predict_proba(X), probabilities, atol=1e-12):
            raise AssertionError("Joblib round trip changed probabilities.")


def assert_spec_builder_and_dry_paths() -> None:
    """Verify spec derivation and default dry-run output paths."""
    with tempfile.TemporaryDirectory(prefix="final-procedure-spec-") as temporary:
        root = Path(temporary)
        finalization_dir = make_fake_finalization_dir(root)
        spec = build_final_procedure_spec(
            source_run_id="fake_fast_completion",
            finalization_dir=finalization_dir,
        )
        if spec["selected_procedure_id"] != "top3_unweighted_soft_average":
            raise AssertionError("Selected procedure ID was not preserved.")
        if len(spec["member_candidate_ids"]) != 3:
            raise AssertionError("Spec must contain exactly three members.")
        if spec["member_weights"] != [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]:
            raise AssertionError("Spec must use equal top-three weights.")
        if spec["selected_decision_threshold"] != 0.4:
            raise AssertionError("Spec must preserve selected OOF threshold.")
        c03 = next(
            member
            for member in spec["members"]
            if member["candidate_id"] == "C03_SPLINE_LOGISTIC_REGRESSION"
        )
        if "max_iter" not in c03["parameters"]:
            raise AssertionError("Full executable C03 config must preserve max_iter.")

    dry_spec = default_output_file("fast_completion_v1")
    dry_refit = default_output_dir("fast_completion_v1")
    if dry_spec.exists():
        raise AssertionError("Default spec output should not be created by smoke tests.")
    if dry_refit.exists():
        raise AssertionError("Default refit output should not be created by smoke tests.")


def assert_member_validation_rejects_inconsistent_members() -> None:
    """Verify inconsistent ensemble member rows are rejected."""
    with tempfile.TemporaryDirectory(prefix="final-procedure-bad-members-") as temporary:
        root = Path(temporary)
        finalization_dir = make_fake_finalization_dir(root)
        write_csv(
            finalization_dir / "ensemble_oof_predictions.csv",
            [
                {
                    "ensemble_id": "top3_unweighted_soft_average",
                    "member_candidate_ids": "A|B|C",
                    "row_position": 0,
                    "target": 0,
                    "probability": 0.2,
                },
                {
                    "ensemble_id": "top3_unweighted_soft_average",
                    "member_candidate_ids": "A|B|D",
                    "row_position": 1,
                    "target": 1,
                    "probability": 0.8,
                },
            ],
            ["ensemble_id", "member_candidate_ids", "row_position", "target", "probability"],
        )
        try:
            build_final_procedure_spec(
                source_run_id="fake_fast_completion",
                finalization_dir=finalization_dir,
            )
        except Exception as exc:
            if "inconsistent" not in str(exc):
                raise AssertionError(f"Unexpected inconsistent-member error: {exc}") from exc
        else:
            raise AssertionError("Inconsistent ensemble members should fail.")


def assert_no_loader_tokens() -> None:
    """Static guard that final-procedure scripts avoid known final-evaluation helpers."""
    for path in (
        PROJECT_ROOT / "scripts" / "freeze_fast_final_procedure.py",
        PROJECT_ROOT / "scripts" / "refit_final_development_pipeline.py",
        PROJECT_ROOT / "scripts" / "smoke_test_final_procedure_refit.py",
        PROJECT_ROOT / "src" / "telco_churn" / "final_procedure.py",
    ):
        source = path.read_text(encoding="utf-8")
        for token in ("load_" + "test_data", "split_" + "test", "TEST" + "_DATA_PATH"):
            if token in source:
                raise AssertionError(f"{path.name} must not import or call {token}.")


def main() -> None:
    """Run final-procedure structural smoke tests."""
    assert_ensemble_prediction_and_roundtrip()
    assert_spec_builder_and_dry_paths()
    assert_member_validation_rejects_inconsistent_members()
    assert_no_loader_tokens()
    print("Final-procedure refit smoke test passed.")


if __name__ == "__main__":
    main()
