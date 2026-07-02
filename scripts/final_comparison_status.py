"""Read-only terminal status for a resumable final-comparison experiment.

The command never creates, resumes, modifies, or validates an experiment.  It reads the
run manifest, SQLite task registry, worker progress sidecars, and task-local Optuna
studies so a long-running comparison can be monitored from a second terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telco_churn.experiment_progress import collect_run_status, render_run_status  # noqa: E402


DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "final_comparison"


def parse_arguments() -> argparse.Namespace:
    """Parse read-only run-status options."""
    parser = argparse.ArgumentParser(
        description="Inspect a final-comparison run without changing any experiment state."
    )
    parser.add_argument("--run-id", required=True, help="Run-directory name to inspect.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing final-comparison run directories.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh the read-only terminal view until Ctrl+C is pressed.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=10.0,
        help="Refresh interval for --watch. Defaults to 10 seconds.",
    )
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Include completed task rows as well as active, failed, and pending work.",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Show failed and interrupted task rows only.",
    )
    return parser.parse_args()


def render_once(arguments: argparse.Namespace) -> None:
    """Collect and print one stable read-only status snapshot."""
    run_directory = Path(arguments.artifacts_root) / arguments.run_id
    snapshot = collect_run_status(run_directory)
    print(
        render_run_status(
            snapshot,
            include_completed=bool(arguments.show_completed),
            failures_only=bool(arguments.failed),
        ),
        flush=True,
    )


def main() -> None:
    """Run one status print or a terminal watch loop."""
    arguments = parse_arguments()
    if arguments.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive.")

    if not arguments.watch:
        render_once(arguments)
        return

    try:
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            render_once(arguments)
            print(
                f"\nRefreshing every {arguments.interval_seconds:g} seconds. "
                "Press Ctrl+C to stop watching without affecting the experiment.",
                flush=True,
            )
            time.sleep(arguments.interval_seconds)
    except KeyboardInterrupt:
        print("\nStatus watch stopped. The experiment was not modified.", flush=True)


if __name__ == "__main__":
    main()
