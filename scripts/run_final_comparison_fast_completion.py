"""Run or inspect the fast-completion final-comparison protocol.

This workflow is intentionally separate from the frozen protocol-v2 base comparison.
It includes C01-C26 but uses minimal development-only search and repeat settings so
the complete project pipeline can be finished quickly. Its evidence is fast-completion
pipeline evidence, not the robust protocol-v2 benchmark.
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


from telco_churn.protocol_v2_workflows import (  # noqa: E402
    PROJECT_ROOT as PACKAGE_PROJECT_ROOT,
    ProtocolV2WorkflowArguments,
    ProtocolV2WorkflowConfigurationError,
    load_protocol_v2_base_spec,
    run_declared_final_comparison_workflow,
)


FAST_PROTOCOL_PATH = (
    PACKAGE_PROJECT_ROOT / "protocols" / "final_comparison_fast_completion_v1.json"
)
DEFAULT_RUN_ID = "fast_completion_v1"
FORBIDDEN_RUN_IDS = frozenset(
    {
        "admission_smoke_c26_warning_clean_v2",
        "search_budget_calibration_v1_warning_clean",
        "protocol_v2_base_official_v1",
        "protocol_v2_base_dry_run_check",
    }
)


def parse_arguments(argv: Sequence[str] | None = None) -> tuple[ProtocolV2WorkflowArguments, bool]:
    """Parse fast-completion operational controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or run the fast-completion development-only final-comparison "
            "protocol. Dry-run mode creates no artifact directories and fits no models."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stop-after-completed", type=int, default=None)
    parser.add_argument(
        "--confirm-fast-completion-run",
        action="store_true",
        help="Required for non-dry-run fast-completion execution.",
    )
    namespace = parser.parse_args(argv)
    run_id = str(namespace.run_id)
    if run_id in FORBIDDEN_RUN_IDS:
        raise ProtocolV2WorkflowConfigurationError(
            f"Run ID {run_id!r} is reserved for another workflow and cannot be reused."
        )
    arguments = ProtocolV2WorkflowArguments(
        run_id=run_id,
        max_workers=int(namespace.max_workers),
        dry_run=bool(namespace.dry_run),
        resume=bool(namespace.resume),
        retry_failed=bool(namespace.retry_failed),
        stop_after_completed=(
            None
            if namespace.stop_after_completed is None
            else int(namespace.stop_after_completed)
        ),
        confirm_official_base_comparison=False,
    )
    if not arguments.run_id.strip():
        raise ProtocolV2WorkflowConfigurationError("--run-id must not be empty.")
    if arguments.max_workers < 1:
        raise ProtocolV2WorkflowConfigurationError("--max-workers must be at least one.")
    if arguments.retry_failed and not arguments.resume:
        raise ProtocolV2WorkflowConfigurationError("--retry-failed requires --resume.")
    if arguments.stop_after_completed is not None:
        if arguments.stop_after_completed < 1:
            raise ProtocolV2WorkflowConfigurationError(
                "--stop-after-completed must be positive."
            )
        if arguments.max_workers != 1:
            raise ProtocolV2WorkflowConfigurationError(
                "--stop-after-completed requires --max-workers 1."
            )
    return arguments, bool(namespace.confirm_fast_completion_run)


def main(argv: Sequence[str] | None = None) -> None:
    """Run or inspect the fast-completion workflow."""
    arguments, confirmed = parse_arguments(argv)
    spec = load_protocol_v2_base_spec(FAST_PROTOCOL_PATH)
    run_declared_final_comparison_workflow(
        spec=spec,
        arguments=arguments,
        confirmation_flag_name="--confirm-fast-completion-run",
        confirmation_granted=confirmed,
    )


if __name__ == "__main__":
    main()
