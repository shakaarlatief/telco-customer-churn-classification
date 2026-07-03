#!/usr/bin/env bash
set -euo pipefail

DEFAULT_RUN_ID="pilot_pruned_f2_v5_history"

MODE="dashboard"
RUN_ID="$DEFAULT_RUN_ID"
INTERVAL_SECONDS="10"
EVENT_LINES="15"
HISTORY_LINES="50"
TASK_KEY=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/monitor_final_comparison.sh [dashboard|details|events|history] [options]

Modes:
  dashboard            Standard live dashboard. This is the default.
  details              Dashboard plus recent completed configuration history for active tasks.
  events               Colored recent coordinator-event viewer.
  history              Durable configuration timing, score, and parameter history.

Options:
  --run-id ID          Run identifier. Defaults to pilot_pruned_f2_v5_history.
  --interval SECONDS   Refresh interval. Defaults to 10.
  --event-lines N      Number of recent events in events mode. Defaults to 15.
  --history-lines N    Number of recent configuration records in history mode. Defaults to 50.
  --task-key KEY       Filter history mode to one exact outer-task key and show full records.
  --help               Show this help text.

Examples:
  bash scripts/monitor_final_comparison.sh
  bash scripts/monitor_final_comparison.sh details
  bash scripts/monitor_final_comparison.sh events --interval 20
  bash scripts/monitor_final_comparison.sh history
  bash scripts/monitor_final_comparison.sh history --task-key c19_catboost__r00__f02
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    dashboard|details|events|history)
      MODE="$1"
      shift
      ;;
    --run-id)
      RUN_ID="${2:?Missing value after --run-id}"
      shift 2
      ;;
    --interval)
      INTERVAL_SECONDS="${2:?Missing value after --interval}"
      shift 2
      ;;
    --event-lines)
      EVENT_LINES="${2:?Missing value after --event-lines}"
      shift 2
      ;;
    --history-lines)
      HISTORY_LINES="${2:?Missing value after --history-lines}"
      shift 2
      ;;
    --task-key)
      TASK_KEY="${2:?Missing value after --task-key}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$TASK_KEY" && "$MODE" != "history" ]]; then
  echo "--task-key is available only in history mode." >&2
  exit 2
fi

REPOSITORY_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run this command from inside the Git repository." >&2
  exit 1
}

cd "$REPOSITORY_ROOT"

COMMAND=(
  python scripts/final_comparison_status.py
  --run-id "$RUN_ID"
  --watch
  --interval-seconds "$INTERVAL_SECONDS"
)

case "$MODE" in
  dashboard)
    ;;
  details)
    COMMAND+=(--details)
    ;;
  events)
    COMMAND+=(--events --event-lines "$EVENT_LINES")
    ;;
  history)
    COMMAND+=(--history --history-lines "$HISTORY_LINES")
    if [[ -n "$TASK_KEY" ]]; then
      COMMAND+=(--task-key "$TASK_KEY")
    fi
    ;;
esac

exec "${COMMAND[@]}"
