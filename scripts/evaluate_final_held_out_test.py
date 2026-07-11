"""Guarded one-time final held-out evaluator.

Dry-run mode runs only readiness checks. Non-dry-run requires the exact confirmation
phrase and writes a durable receipt before loading the final evaluation data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_test_data  # noqa: E402
from telco_churn.final_evaluation import (  # noqa: E402
    CONFIRMATION_PHRASE,
    FinalEvaluationError,
    assert_output_dir_allowed,
    check_no_existing_receipt,
    default_evaluation_output_dir,
    default_procedure_spec_path,
    default_refit_dir,
    execute_one_time_evaluation,
    run_readiness_audit,
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluator controls."""
    parser = argparse.ArgumentParser(
        description="One-time held-out final evaluator for the frozen final ensemble."
    )
    parser.add_argument("--source-run-id", default="fast_completion_v1")
    parser.add_argument("--procedure-spec", default=None)
    parser.add_argument("--refit-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-held-out-test-evaluation", default=None)
    return parser.parse_args(argv)


def validate_confirmation_phrase(value: str | None) -> None:
    """Require the exact one-time final-evaluation confirmation phrase."""
    if value != CONFIRMATION_PHRASE:
        raise FinalEvaluationError(
            "Non-dry-run final evaluation requires "
            f"--confirm-held-out-test-evaluation {CONFIRMATION_PHRASE}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run dry-run or confirmed one-time final evaluation."""
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
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_evaluation_output_dir(source_run_id)
    )

    try:
        readiness = run_readiness_audit(
            source_run_id=source_run_id,
            procedure_spec_path=procedure_spec,
            refit_dir=refit_dir,
        )
        assert_output_dir_allowed(output_dir)
        if args.dry_run:
            print("Final held-out evaluation dry run")
            print(f"Source run ID: {source_run_id}")
            print(f"Procedure ID: {readiness.procedure_spec['selected_procedure_id']}")
            print(f"Members: {', '.join(readiness.procedure_spec['member_candidate_ids'])}")
            print(f"Weights: {', '.join(str(weight) for weight in readiness.procedure_spec['member_weights'])}")
            print(f"Frozen threshold: {readiness.procedure_spec['selected_decision_threshold']}")
            print(f"Model path: {readiness.model_path}")
            print(f"Requested bootstrap replicates: {int(args.bootstrap_replicates)}")
            print(f"Intended output directory: {output_dir}")
            print("Dry-run completed; final evaluation data was not loaded.")
            return

        validate_confirmation_phrase(args.confirm_held_out_test_evaluation)
        if int(args.bootstrap_replicates) < 1:
            raise FinalEvaluationError("--bootstrap-replicates must be positive.")
        check_no_existing_receipt(source_run_id, output_dir)
        if output_dir.exists():
            raise FinalEvaluationError(f"Output directory already exists: {output_dir}")
        result = execute_one_time_evaluation(
            source_run_id=source_run_id,
            readiness=readiness,
            output_dir=output_dir,
            final_data_loader=load_test_data,
            bootstrap_replicates=int(args.bootstrap_replicates),
            random_state=int(args.random_state),
        )
    except FinalEvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Final evaluation completed at: {output_dir}")
    print(f"Primary metric average_precision: {result['metrics']['average_precision']}")


if __name__ == "__main__":
    main()
