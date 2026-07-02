"""Read-only terminal dashboard for a resumable final-comparison experiment.

The command never creates, resumes, modifies, or validates an experiment. It reconstructs
the display from durable run artifacts: the task registry, progress sidecars, task-local
Optuna studies, and coordinator event log.
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

from telco_churn.experiment_progress import (  # noqa: E402
    collect_run_status,
    render_run_status,
    render_task_details,
)


DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "final_comparison"


def parse_arguments() -> argparse.Namespace:
    """Parse read-only status and dashboard options."""
    parser = argparse.ArgumentParser(
        description="Inspect a final-comparison run without changing experiment state."
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
        help="Redraw the dashboard until Ctrl+C is pressed.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Refresh interval for --watch. Defaults to two seconds.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Append watch snapshots instead of redrawing the terminal in place.",
    )
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Include completed task rows in the outstanding-work table.",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Show failed and interrupted task rows only.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show full active-task parameter telemetry in the dashboard.",
    )
    parser.add_argument(
        "--task-key",
        help="Show a full read-only detail view for one exact task key.",
    )
    return parser.parse_args()


def render_once(arguments: argparse.Namespace) -> str:
    """Collect and render one stable read-only snapshot."""
    run_directory = Path(arguments.artifacts_root) / arguments.run_id
    snapshot = collect_run_status(run_directory)
    if arguments.task_key:
        return render_task_details(snapshot, arguments.task_key)
    return render_run_status(
        snapshot,
        include_completed=bool(arguments.show_completed),
        failures_only=bool(arguments.failed),
        details=bool(arguments.details),
    )


def _clear_terminal() -> None:
    """Clear and home the screen without emitting repeated newline-separated snapshots."""
    print("\033[H\033[2J", end="", flush=True)


def main() -> None:
    """Run one status display or an in-place dashboard watch loop."""
    arguments = parse_arguments()
    if arguments.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive.")
    if arguments.task_key and arguments.watch:
        raise SystemExit("--task-key and --watch cannot be combined.")

    if not arguments.watch:
        print(render_once(arguments), flush=True)
        return

    interactive = sys.stdout.isatty() and not arguments.no_clear
    try:
        if interactive:
            print("\033[?25l", end="", flush=True)
        while True:
            if interactive:
                _clear_terminal()
            print(render_once(arguments), flush=True)
            print(
                (
                    f"\nRefreshing every {arguments.interval_seconds:g}s. "
                    "Ctrl+C here stops only this dashboard. "
                    "Ctrl+C once in the runner terminal requests a clean experiment pause."
                ),
                flush=True,
            )
            time.sleep(arguments.interval_seconds)
    except KeyboardInterrupt:
        if interactive:
            _clear_terminal()
        print("Status watch stopped. The experiment was not modified.", flush=True)
    finally:
        if interactive:
            print("\033[?25h", end="", flush=True)


if __name__ == "__main__":
    main()
