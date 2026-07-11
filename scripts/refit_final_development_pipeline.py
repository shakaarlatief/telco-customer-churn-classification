"""Fit the frozen final procedure on all development rows."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import joblib  # noqa: E402
import numpy as np  # noqa: E402

from freeze_fast_final_procedure import (  # noqa: E402
    build_final_procedure_spec,
    default_finalization_dir,
)
from telco_churn.candidates import build_candidate_pipeline  # noqa: E402
from telco_churn.config import RANDOM_STATE, TARGET_COLUMN  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.experiment_protocol import (  # noqa: E402
    make_dataframe_fingerprint,
    make_environment_fingerprint,
)
from telco_churn.final_procedure import FrozenProbabilityVotingEnsemble  # noqa: E402


DEFAULT_SOURCE_RUN_ID = "fast_completion_v1"


class FinalDevelopmentRefitError(ValueError):
    """Raised before the full-development refit can proceed safely."""


def default_spec_path(source_run_id: str) -> Path:
    """Return the default frozen final-procedure spec path."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / "frozen_final_procedure_v1"
        / "final_procedure_spec.json"
    )


def default_output_dir(source_run_id: str) -> Path:
    """Return the default full-development refit output directory."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / "final_development_refit_v1"
    )


def _assert_no_forbidden_loader_references() -> None:
    """Static guard against known final-evaluation loaders in this script."""
    source = Path(__file__).read_text(encoding="utf-8")
    for token in ("load_" + "test_data", "split_" + "test", "TEST" + "_DATA_PATH"):
        if token in source:
            raise FinalDevelopmentRefitError(f"Forbidden final-evaluation reference found: {token}")


def _assert_output_dir_is_derived(path: Path) -> None:
    """Refuse writes under immutable final-comparison run artifacts."""
    resolved = path.resolve()
    immutable_root = (PROJECT_ROOT / "artifacts" / "final_comparison").resolve()
    if resolved == immutable_root or immutable_root in resolved.parents:
        raise FinalDevelopmentRefitError(
            "Final refit outputs must not be written under final-comparison artifacts."
        )
    if path.exists() and any(path.iterdir()):
        raise FinalDevelopmentRefitError(f"Output directory already exists and is not empty: {path}")


def _read_spec(path: Path) -> dict[str, Any]:
    """Read and validate the frozen final-procedure spec."""
    if not path.exists():
        raise FinalDevelopmentRefitError(f"Procedure spec is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("procedure_type") != "ensemble":
        raise FinalDevelopmentRefitError("Only frozen ensemble specs are supported.")
    if payload.get("ensemble_aggregation") != "arithmetic_mean_of_probabilities":
        raise FinalDevelopmentRefitError("Unsupported ensemble aggregation.")
    if payload.get("score_kind") != "probability":
        raise FinalDevelopmentRefitError("Final ensemble must use probability scores.")
    if payload.get("held_out_test_policy") != "not_loaded_or_referenced":
        raise FinalDevelopmentRefitError("Procedure spec must record no final-evaluation access.")
    members = payload.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise FinalDevelopmentRefitError("Procedure spec must contain exactly three members.")
    return payload


def _read_spec_for_dry_run(path: Path, *, source_run_id: str) -> dict[str, Any]:
    """Read a spec or derive it in memory for artifact-free dry-run inspection."""
    if path.exists():
        return _read_spec(path)
    return build_final_procedure_spec(
        source_run_id=source_run_id,
        finalization_dir=default_finalization_dir(source_run_id),
    )


def _git_revision() -> str | None:
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


def _sha256_file(path: Path) -> str:
    """Return SHA-256 for one artifact."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def fit_final_development_pipeline(*, procedure_spec: Path, output_dir: Path) -> dict[str, Any]:
    """Fit, serialize, and round-trip validate the frozen final ensemble."""
    _assert_no_forbidden_loader_references()
    _assert_output_dir_is_derived(output_dir)
    spec = _read_spec(procedure_spec)

    fit_started_at = datetime.now(UTC)
    train_df = load_train_data()
    X, y = split_features_target(train_df)
    fitted_estimators: list[Any] = []
    for member in spec["members"]:
        estimator = build_candidate_pipeline(
            str(member["candidate_id"]),
            dict(member["parameters"]),
            random_state=RANDOM_STATE,
        )
        fitted_estimators.append(estimator.fit(X, y))

    ensemble = FrozenProbabilityVotingEnsemble(
        member_ids=tuple(spec["member_candidate_ids"]),
        member_display_names=tuple(spec["member_display_names"]),
        member_weights=tuple(float(weight) for weight in spec["member_weights"]),
        estimators=tuple(fitted_estimators),
        decision_threshold=float(spec["selected_decision_threshold"]),
    )
    fit_finished_at = datetime.now(UTC)

    output_dir.mkdir(parents=True, exist_ok=False)
    spec_copy_path = output_dir / "final_procedure_spec.json"
    spec_copy_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    environment = make_environment_fingerprint(
        (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "joblib",
            "interpret-core",
            "torch",
            "rtdl_revisiting_models",
        )
    )
    feature_schema = {
        "feature_columns": list(X.columns),
        "feature_dtypes": {column: str(dtype) for column, dtype in X.dtypes.items()},
        "target_column": TARGET_COLUMN,
        "target_dtype": str(y.dtype),
    }
    data_fingerprint = make_dataframe_fingerprint(X, y)
    target_distribution = {
        str(label): int(count) for label, count in y.value_counts().sort_index().items()
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fit_started_at_utc": fit_started_at.isoformat(),
        "fit_finished_at_utc": fit_finished_at.isoformat(),
        "training_row_count": int(len(X)),
        "target_distribution": target_distribution,
        "development_data_fingerprint": data_fingerprint,
        "source_git_commit": _git_revision(),
        "environment_sha256": environment["sha256"],
        "member_candidate_ids": list(spec["member_candidate_ids"]),
        "member_configurations": spec["members"],
        "decision_threshold": float(spec["selected_decision_threshold"]),
        "calibration_method": spec["calibration_method"],
        "calibration_status": spec["calibration_status"],
        "held_out_test_policy": "not_loaded_or_referenced",
    }

    model_path = output_dir / "fitted_final_pipeline.joblib"
    joblib.dump(ensemble, model_path)
    sample = X.iloc[: min(25, len(X))]
    before_proba = ensemble.predict_proba(sample)
    before_pred = ensemble.predict(sample)
    loaded = joblib.load(model_path)
    after_proba = loaded.predict_proba(sample)
    after_pred = loaded.predict(sample)
    roundtrip = {
        "sample_row_count": int(len(sample)),
        "probabilities_allclose": bool(np.allclose(before_proba, after_proba, atol=1e-12, rtol=0.0)),
        "predictions_equal": bool(np.array_equal(before_pred, after_pred)),
        "max_probability_abs_diff": float(np.max(np.abs(before_proba - after_proba))),
    }
    if not roundtrip["probabilities_allclose"] or not roundtrip["predictions_equal"]:
        raise FinalDevelopmentRefitError("Serialized final pipeline failed round-trip validation.")

    _write_json(output_dir / "final_refit_manifest.json", manifest)
    _write_json(output_dir / "model_environment.json", environment)
    _write_json(output_dir / "feature_schema.json", feature_schema)
    _write_json(output_dir / "roundtrip_validation.json", roundtrip)
    artifact_names = [
        "final_procedure_spec.json",
        "final_refit_manifest.json",
        "fitted_final_pipeline.joblib",
        "model_environment.json",
        "feature_schema.json",
        "roundtrip_validation.json",
    ]
    checksums = {name: _sha256_file(output_dir / name) for name in artifact_names}
    _write_json(output_dir / "artifact_checksums.json", checksums)
    return {"manifest": manifest, "roundtrip": roundtrip, "output_dir": str(output_dir)}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI controls."""
    parser = argparse.ArgumentParser(
        description="Fit the frozen final procedure on all development rows."
    )
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--procedure-spec", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-final-development-refit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    arguments = parse_arguments(argv)
    source_run_id = str(arguments.source_run_id)
    procedure_spec = (
        Path(arguments.procedure_spec)
        if arguments.procedure_spec is not None
        else default_spec_path(source_run_id)
    )
    output_dir = (
        Path(arguments.output_dir)
        if arguments.output_dir is not None
        else default_output_dir(source_run_id)
    )
    try:
        _assert_no_forbidden_loader_references()
        spec = (
            _read_spec_for_dry_run(procedure_spec, source_run_id=source_run_id)
            if arguments.dry_run
            else _read_spec(procedure_spec)
        )
        _assert_output_dir_is_derived(output_dir)
        if arguments.dry_run:
            print("Final development refit dry run")
            print(f"Procedure spec: {procedure_spec}")
            print(f"Output directory: {output_dir}")
            print(f"Selected procedure: {spec['selected_procedure_id']}")
            print(f"Members: {', '.join(spec['member_candidate_ids'])}")
            print(f"Weights: {', '.join(str(weight) for weight in spec['member_weights'])}")
            print(f"Decision threshold: {spec['selected_decision_threshold']}")
            print("Dry-run completed without fitting models or writing artifacts.")
            return
        if not arguments.confirm_final_development_refit:
            raise FinalDevelopmentRefitError(
                "Full-development refit requires --confirm-final-development-refit."
            )
        result = fit_final_development_pipeline(
            procedure_spec=procedure_spec,
            output_dir=output_dir,
        )
    except FinalDevelopmentRefitError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Final development refit artifacts written to: {result['output_dir']}")
    print(f"Round-trip validation: {result['roundtrip']}")


if __name__ == "__main__":
    main()
