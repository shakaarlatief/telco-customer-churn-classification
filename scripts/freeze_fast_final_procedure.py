"""Create a frozen final-procedure spec from fast-finalization artifacts."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import build_candidate_pipeline  # noqa: E402
from telco_churn.final_procedure import validate_equal_weights  # noqa: E402


DEFAULT_SOURCE_RUN_ID = "fast_completion_v1"
FINALIZATION_SUBDIR = "fast_finalization_v1"
DEFAULT_SPEC_DIR = "frozen_final_procedure_v1"
SPEC_EVIDENCE_ROLE = "fast_final_procedure_specification"
FINALIZATION_EVIDENCE_ROLE = "fast_finalization_pipeline_evidence"
SOURCE_EVIDENCE_ROLE = "fast_completion_pipeline_evidence"


class FinalProcedureSpecError(ValueError):
    """Raised when the frozen final-procedure spec cannot be built safely."""


def default_finalization_dir(source_run_id: str) -> Path:
    """Return the default fast-finalization artifact directory."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / FINALIZATION_SUBDIR
    )


def default_output_file(source_run_id: str) -> Path:
    """Return the default derived final-procedure spec path."""
    return (
        PROJECT_ROOT
        / "artifacts"
        / "final_selection"
        / source_run_id
        / DEFAULT_SPEC_DIR
        / "final_procedure_spec.json"
    )


def _read_json(path: Path) -> Any:
    """Read JSON with a clear path-specific error."""
    if not path.exists():
        raise FinalProcedureSpecError(f"Required finalization artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows with a clear path-specific error."""
    if not path.exists():
        raise FinalProcedureSpecError(f"Required finalization artifact is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _assert_output_path_is_derived(path: Path) -> None:
    """Refuse writes under immutable final-comparison artifacts."""
    resolved = path.resolve()
    immutable_root = (PROJECT_ROOT / "artifacts" / "final_comparison").resolve()
    if resolved == immutable_root or immutable_root in resolved.parents:
        raise FinalProcedureSpecError(
            "Frozen final-procedure outputs must not be written under final-comparison artifacts."
        )


def _assert_no_forbidden_loader_references() -> None:
    """Static guard against known final-evaluation loaders in this script."""
    source = Path(__file__).read_text(encoding="utf-8")
    for token in ("load_" + "test_data", "split_" + "test", "TEST" + "_DATA_PATH"):
        if token in source:
            raise FinalProcedureSpecError(f"Forbidden final-evaluation reference found: {token}")


def _selected_procedure(selection_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return and validate the selected procedure payload."""
    selected = selection_payload.get("selected_procedure")
    if not isinstance(selected, Mapping):
        raise FinalProcedureSpecError("final_procedure_selection.json has no selected_procedure.")
    if selected.get("procedure_type") != "ensemble":
        raise FinalProcedureSpecError("Fast final procedure must currently be an ensemble.")
    if selected.get("score_kind") != "probability":
        raise FinalProcedureSpecError("Selected ensemble must expose probability scores.")
    threshold = float(selected.get("best_f1_threshold"))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise FinalProcedureSpecError("Selected OOF F1 threshold must be finite and in [0, 1].")
    return selected


def _derive_member_ids(ensemble_rows: Sequence[Mapping[str, str]], *, ensemble_id: str) -> tuple[str, ...]:
    """Read and validate the selected ensemble member list from OOF rows."""
    matching = [row for row in ensemble_rows if row.get("ensemble_id") == ensemble_id]
    if not matching:
        raise FinalProcedureSpecError(f"No ensemble OOF rows found for {ensemble_id!r}.")
    member_lists = {row.get("member_candidate_ids", "") for row in matching}
    if len(member_lists) != 1:
        raise FinalProcedureSpecError("Selected ensemble OOF rows contain inconsistent members.")
    member_ids = tuple(next(iter(member_lists)).split("|"))
    if len(member_ids) != 3 or any(not candidate_id for candidate_id in member_ids):
        raise FinalProcedureSpecError("Selected ensemble must contain exactly three members.")
    return member_ids


def _member_configs(
    tuned_configs: Sequence[Mapping[str, Any]],
    member_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return one complete executable configuration per member."""
    rows_by_candidate: dict[str, list[Mapping[str, Any]]] = {candidate_id: [] for candidate_id in member_ids}
    for row in tuned_configs:
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id in rows_by_candidate:
            rows_by_candidate[candidate_id].append(row)

    members: list[dict[str, Any]] = []
    for candidate_id in member_ids:
        rows = rows_by_candidate[candidate_id]
        if len(rows) != 1:
            raise FinalProcedureSpecError(
                f"Expected exactly one tuned config for {candidate_id}, got {len(rows)}."
            )
        row = rows[0]
        parameters = deepcopy(dict(row.get("parameters") or {}))
        for required in ("feature_policy", "feature_selection_policy", "imbalance_policy"):
            if required not in parameters:
                raise FinalProcedureSpecError(
                    f"{candidate_id} configuration is missing {required!r}."
                )
        build_candidate_pipeline(candidate_id, parameters, random_state=42)
        members.append(
            {
                "candidate_id": candidate_id,
                "display_name": str(row.get("candidate_display_name", candidate_id)),
                "search_profile": str(row.get("search_profile", "")),
                "parameters": parameters,
            }
        )
    return members


def build_final_procedure_spec(
    *,
    source_run_id: str,
    finalization_dir: Path,
) -> dict[str, Any]:
    """Build a frozen final-procedure specification from finalization outputs."""
    _assert_no_forbidden_loader_references()
    selection_payload = _read_json(finalization_dir / "final_procedure_selection.json")
    tuned_configs = _read_json(finalization_dir / "tuned_candidate_configs.json")
    manifest = _read_json(finalization_dir / "finalization_manifest.json")
    ensemble_rows = _read_csv(finalization_dir / "ensemble_oof_predictions.csv")

    selected = _selected_procedure(selection_payload)
    selected_id = str(selected["procedure_id"])
    member_ids = _derive_member_ids(ensemble_rows, ensemble_id=selected_id)
    members = _member_configs(tuned_configs, member_ids)
    weights = validate_equal_weights([1.0 / len(member_ids)] * len(member_ids), expected_count=3)
    threshold = float(selected["best_f1_threshold"])
    development_rows = int(manifest.get("development_rows"))
    if development_rows < 1:
        raise FinalProcedureSpecError("development row count must be positive.")

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "spec_id": "fast_final_procedure_v1",
        "evidence_role": SPEC_EVIDENCE_ROLE,
        "source_run_id": source_run_id,
        "finalization_id": FINALIZATION_SUBDIR,
        "finalization_dir": str(finalization_dir),
        "source_evidence_role": manifest.get("source_evidence_role", SOURCE_EVIDENCE_ROLE),
        "finalization_evidence_role": manifest.get("evidence_role", FINALIZATION_EVIDENCE_ROLE),
        "selected_procedure_id": selected_id,
        "procedure_type": str(selected["procedure_type"]),
        "ensemble_aggregation": "arithmetic_mean_of_probabilities",
        "member_candidate_ids": list(member_ids),
        "member_display_names": [member["display_name"] for member in members],
        "member_weights": list(weights),
        "members": members,
        "score_kind": "probability",
        "selected_decision_threshold": threshold,
        "threshold_objective": "OOF_F1",
        "selected_oof_average_precision": float(selected["average_precision"]),
        "calibration_method": "none",
        "calibration_status": "deferred_fast_completion",
        "development_row_count": development_rows,
        "held_out_test_policy": "not_loaded_or_referenced",
        "warning": (
            "This frozen final-procedure specification is based on fast development-data "
            "evidence rather than robust frozen protocol-v2 evidence."
        ),
    }


def write_spec(path: Path, spec: Mapping[str, Any]) -> None:
    """Write a frozen final-procedure spec without overwriting non-empty directories."""
    _assert_output_path_is_derived(path)
    parent = path.parent
    if parent.exists() and any(parent.iterdir()) and not path.exists():
        raise FinalProcedureSpecError(f"Output directory already exists and is not empty: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI controls."""
    parser = argparse.ArgumentParser(
        description="Create or inspect a frozen final-procedure spec from fast-finalization outputs."
    )
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--finalization-dir", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-freeze-final-procedure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    arguments = parse_arguments(argv)
    source_run_id = str(arguments.source_run_id)
    finalization_dir = (
        Path(arguments.finalization_dir)
        if arguments.finalization_dir is not None
        else default_finalization_dir(source_run_id)
    )
    output_file = (
        Path(arguments.output_file)
        if arguments.output_file is not None
        else default_output_file(source_run_id)
    )
    try:
        spec = build_final_procedure_spec(
            source_run_id=source_run_id,
            finalization_dir=finalization_dir,
        )
        if arguments.dry_run:
            print("Frozen final-procedure spec dry run")
            print(f"Source run ID: {spec['source_run_id']}")
            print(f"Finalization directory: {finalization_dir}")
            print(f"Output file: {output_file}")
            print(f"Selected procedure: {spec['selected_procedure_id']}")
            print(f"Procedure type: {spec['procedure_type']}")
            print(f"Members: {', '.join(spec['member_candidate_ids'])}")
            print(f"Weights: {', '.join(str(weight) for weight in spec['member_weights'])}")
            print(f"Selected OOF F1 threshold: {spec['selected_decision_threshold']}")
            print("Dry-run completed without writing the frozen spec.")
            return
        if not arguments.confirm_freeze_final_procedure:
            raise FinalProcedureSpecError(
                "Writing the frozen final-procedure spec requires --confirm-freeze-final-procedure."
            )
        write_spec(output_file, spec)
    except FinalProcedureSpecError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Frozen final-procedure spec written to: {output_file}")


if __name__ == "__main__":
    main()
