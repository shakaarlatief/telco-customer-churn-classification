"""Read-only readiness audit for the frozen final model.

This script never loads final evaluation data. It verifies frozen/refit artifacts and
the serialized model before the one-time evaluator is allowed to consume the final
held-out set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.final_evaluation import (  # noqa: E402
    FinalEvaluationError,
    default_procedure_spec_path,
    default_refit_dir,
    run_readiness_audit,
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse readiness audit controls."""
    parser = argparse.ArgumentParser(
        description="Audit frozen final-model readiness without loading final evaluation data."
    )
    parser.add_argument("--source-run-id", default="fast_completion_v1")
    parser.add_argument("--procedure-spec", default=None)
    parser.add_argument("--refit-dir", default=None)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run readiness audit and print check results."""
    args = parse_arguments(argv)
    source_run_id = str(args.source_run_id)
    procedure_spec = (
        Path(args.procedure_spec)
        if args.procedure_spec is not None
        else default_procedure_spec_path(source_run_id)
    )
    refit_dir = (
        Path(args.refit_dir) if args.refit_dir is not None else default_refit_dir(source_run_id)
    )
    try:
        readiness = run_readiness_audit(
            source_run_id=source_run_id,
            procedure_spec_path=procedure_spec,
            refit_dir=refit_dir,
        )
    except FinalEvaluationError as exc:
        print("Final test readiness: NOT READY")
        raise SystemExit(str(exc)) from exc

    print("Final test readiness: READY")
    print(f"Source run ID: {source_run_id}")
    print(f"Procedure ID: {readiness.procedure_spec['selected_procedure_id']}")
    print(f"Members: {', '.join(readiness.procedure_spec['member_candidate_ids'])}")
    print(f"Weights: {', '.join(str(weight) for weight in readiness.procedure_spec['member_weights'])}")
    print(f"Frozen threshold: {readiness.procedure_spec['selected_decision_threshold']}")
    print(f"Model path: {readiness.model_path}")
    print(f"Model SHA-256: {readiness.model_sha256}")
    print("Final evaluation data was not loaded, inspected, counted, or fingerprinted.")
    print("Readiness checks:")
    for check in readiness.checks:
        print(f"  PASS {check['name']}: {check['detail']}")

    if args.report_path is not None:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "ready": readiness.ready,
                    "checks": list(readiness.checks),
                    "procedure_id": readiness.procedure_spec["selected_procedure_id"],
                    "member_candidate_ids": readiness.procedure_spec["member_candidate_ids"],
                    "model_sha256": readiness.model_sha256,
                    "spec_sha256": readiness.spec_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Optional readiness report written to: {report_path}")


if __name__ == "__main__":
    main()
