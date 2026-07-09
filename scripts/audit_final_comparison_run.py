"""Read-only operational and search-behaviour audit for a final-comparison run.

The command reconstructs a completed or partially completed run from its durable task
registry, result artifacts, Optuna SQLite studies, and coordinator logs. It never opens a
study through Optuna, resumes a task, writes a checkpoint, or mutates the source tree.

The audit is intentionally generic. It derives each task's expected Stage-A and Stage-B
budget from the immutable task payload rather than assuming the v6 pilot's twelve-trial,
top-three contract. It can therefore inspect the v6 pilot, the all-candidate admission
smoke, the search-budget calibration run, and future frozen protocols while preserving
the distinction between operational evidence and final model-selection evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable, Mapping


DEFAULT_RUN_ID = "pilot_pruned_f2_v6_io_resilient"
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "final_comparison"
CHECKPOINTS = (1, 3, 5, 8, 12, 24, 36)


def parse_arguments() -> argparse.Namespace:
    """Parse read-only artifact locations and report-detail options."""
    parser = argparse.ArgumentParser(
        description="Audit final-comparison artifacts without modifying any run state."
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "Run-directory name below --artifacts-root. Defaults to "
            f"{DEFAULT_RUN_ID}."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing final-comparison run directories.",
    )
    parser.add_argument(
        "--warning-limit",
        type=int,
        default=12,
        help="Maximum distinct warnings shown per source and candidate. Defaults to 12.",
    )
    parser.add_argument(
        "--include-stage-a-trajectories",
        action="store_true",
        help=(
            "Add aggregate cumulative-best Stage-A trajectories and Stage-B winner ranks. "
            "Useful for pre-master search-budget calibration runs."
        ),
    )
    return parser.parse_args()


def parse_timestamp(value: Any) -> datetime | None:
    """Parse one stored ISO-8601 timestamp into UTC without raising on missing data."""
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def format_duration(seconds: float | None) -> str:
    """Render a duration compactly without importing execution code."""
    if seconds is None:
        return "-"
    rounded = max(0, int(round(float(seconds))))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds_part:02d}s"
    if minutes:
        return f"{minutes}m {seconds_part:02d}s"
    return f"{seconds_part}s"


def compact_text(value: Any, limit: int = 180) -> str:
    """Normalize a visible message into one stable single-line terminal fragment."""
    rendered = " ".join(str(value).strip().split())
    return rendered if len(rendered) <= limit else rendered[: max(1, limit - 3)] + "..."


def concise_error_reason(value: Any, limit: int = 260) -> str:
    """Extract the most actionable terminal line from one persisted traceback or error."""
    lines = [compact_text(line, limit=limit) for line in str(value or "").splitlines() if line.strip()]
    if not lines:
        return "<no persisted error text>"
    error_lines = [
        line
        for line in lines
        if any(token in line.lower() for token in ("error", "exception", "traceback", "failed"))
    ]
    return error_lines[-1] if error_lines else lines[-1]


def canonical_json(value: Any) -> str:
    """Return deterministic JSON used only for read-only configuration comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def metric(value: Any) -> str:
    """Render a numeric metric or an explicit unavailable marker."""
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "-"


def format_mean_min_max(values: Iterable[float | None]) -> str:
    """Summarize a small runtime sample without implying a population estimate."""
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return "-"
    return (
        f"mean={format_duration(statistics.mean(usable))} | "
        f"min={format_duration(min(usable))} | max={format_duration(max(usable))}"
    )


def open_read_only_sqlite(path: Path) -> sqlite3.Connection:
    """Open SQLite with OS-enforced read-only mode so the audit cannot alter state."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON mapping or raise an actionable structural error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON artifact {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON artifact is not a mapping: {path}")
    return dict(payload)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read valid JSONL records defensively without modifying source artifacts."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    records.append(dict(payload))
    except OSError:
        return []
    return records


def normalize_optuna_state(value: Any) -> str:
    """Map common Optuna SQLite trial-state encodings into stable labels."""
    if isinstance(value, int):
        return {
            0: "running",
            1: "complete",
            2: "pruned",
            3: "fail",
            4: "waiting",
        }.get(int(value), str(value).lower())
    rendered = str(value).strip().lower()
    return rendered.split(".", 1)[1] if rendered.startswith("trialstate.") else rendered


def load_registry_rows(run_directory: Path) -> list[dict[str, Any]]:
    """Read all authoritative task rows in deterministic candidate/split order."""
    registry_path = run_directory / "task_registry.sqlite"
    if not registry_path.exists():
        raise FileNotFoundError(f"Task registry does not exist: {registry_path}")
    with closing(open_read_only_sqlite(registry_path)) as connection:
        cursor = connection.execute(
            """
            SELECT task_key, candidate_id, repeat_index, fold_index, split_hash, status, attempts,
                   started_at, heartbeat_at, completed_at, error_text, result_path,
                   result_sha256, payload_json
            FROM tasks
            ORDER BY candidate_id, repeat_index, fold_index, task_key
            """
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def task_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the registered nested-task payload, tolerating old malformed rows."""
    try:
        registered = json.loads(str(row.get("payload_json") or "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(registered, Mapping):
        return {}
    payload = registered.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def load_completed_results(
    run_directory: Path,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load checksum-verified result artifacts for every registry task marked completed."""
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for row in rows:
        task_key = str(row.get("task_key"))
        if str(row.get("status")) != "completed":
            problems.append(f"{task_key}: registry state is {row.get('status')!r}, not completed.")
            continue
        result_path = row.get("result_path")
        recorded_sha = row.get("result_sha256")
        if not result_path or not recorded_sha:
            problems.append(f"{task_key}: result path or checksum is missing.")
            continue
        path = run_directory / str(result_path)
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{task_key}: cannot read result artifact: {type(exc).__name__}: {exc}")
            continue
        if sha256(raw).hexdigest() != str(recorded_sha):
            problems.append(f"{task_key}: SHA-256 checksum mismatch.")
            continue
        if not isinstance(payload, Mapping) or not isinstance(payload.get("result"), Mapping):
            problems.append(f"{task_key}: result artifact has no mapping payload['result'].")
            continue
        result = dict(payload["result"])
        identity_fields = (
            ("candidate_id", str(row.get("candidate_id"))),
            ("outer_repeat_index", int(row.get("repeat_index"))),
            ("outer_fold_index", int(row.get("fold_index"))),
            ("split_hash", str(row.get("split_hash"))),
        )
        mismatches = [
            name
            for name, expected in identity_fields
            if result.get(name) != expected
        ]
        if mismatches:
            problems.append(
                f"{task_key}: result artifact identity differs from registry fields: "
                + ", ".join(mismatches)
                + "."
            )
            continue
        started = parse_timestamp(row.get("started_at"))
        completed = parse_timestamp(row.get("completed_at"))
        wall_seconds = (
            max(0.0, (completed - started).total_seconds())
            if started is not None and completed is not None
            else None
        )
        entries.append(
            {
                "registry": dict(row),
                "payload": dict(task_payload(row)),
                "result": result,
                "result_path": path,
                "wall_seconds": wall_seconds,
            }
        )
    return entries, problems


def selected_stage_b_record(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Locate the persisted Stage-B winner by exact selected-parameter identity."""
    inner = result.get("inner_search")
    selected_parameters = result.get("selected_parameters")
    if not isinstance(inner, Mapping) or not isinstance(selected_parameters, Mapping):
        return None
    records = inner.get("stage_b_records")
    if not isinstance(records, list):
        return None
    selected_key = canonical_json(dict(selected_parameters))
    for record in records:
        if isinstance(record, Mapping) and canonical_json(record.get("parameters", {})) == selected_key:
            return record
    return None


def recursive_warning_messages(value: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    """Collect warning-like string values from a nested JSON-compatible result object."""
    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if "warning" in key_text.lower():
                if isinstance(nested, str) and nested.strip():
                    found.append((child_prefix, compact_text(nested)))
                elif isinstance(nested, (list, tuple)):
                    found.extend(
                        (child_prefix, compact_text(item))
                        for item in nested
                        if isinstance(item, str) and item.strip()
                    )
            found.extend(recursive_warning_messages(nested, prefix=child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(recursive_warning_messages(nested, prefix=f"{prefix}[{index}]"))
    return found


def inspect_stage_a_study(study_path: Path) -> tuple[Counter[str], list[str], list[str]]:
    """Inspect trial states and warning attributes through read-only SQLite queries only."""
    states: Counter[str] = Counter()
    warnings: list[str] = []
    problems: list[str] = []
    if not study_path.exists():
        return states, warnings, [f"Missing Optuna study database: {study_path}"]
    try:
        with closing(open_read_only_sqlite(study_path)) as connection:
            table_names = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "trials" not in table_names:
                return states, warnings, [f"{study_path}: no Optuna trials table."]
            for (state,) in connection.execute("SELECT state FROM trials").fetchall():
                states[normalize_optuna_state(state)] += 1
            if "trial_user_attributes" not in table_names:
                return states, warnings, problems
            rows = connection.execute(
                """
                SELECT trials.number, trial_user_attributes.value_json
                FROM trial_user_attributes
                JOIN trials ON trial_user_attributes.trial_id = trials.trial_id
                WHERE trial_user_attributes.key = 'warning_messages'
                ORDER BY trials.number
                """
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        return states, warnings, [f"{study_path}: {type(exc).__name__}: {exc}"]
    for trial_number, raw_value in rows:
        try:
            parsed = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            warnings.extend(
                f"trial {trial_number}: {compact_text(message)}"
                for message in parsed
                if isinstance(message, str) and message.strip()
            )
    return states, warnings, problems


def print_heading(title: str) -> None:
    """Print a consistently separated terminal section heading."""
    print("\n" + title)
    print("-" * len(title))


def print_manifest_summary(manifest: Mapping[str, Any]) -> None:
    """Show immutable run purpose and protocol identity before inspecting outcomes."""
    protocol = manifest.get("protocol", {})
    protocol = protocol if isinstance(protocol, Mapping) else {}
    metadata = protocol.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    print("FINAL-COMPARISON READ-ONLY RUN AUDIT")
    print(f"Run: {manifest.get('run_id', '-')}")
    print(f"Protocol: {protocol.get('protocol_id', '-')} | version={protocol.get('version', '-')}")
    print(f"Purpose: {metadata.get('purpose', '-')}")
    print("Interpretation: audit output is operational and protocol evidence, not automatic model selection.")


def print_incomplete_task_audit(
    rows: Iterable[Mapping[str, Any]],
    run_directory: Path,
) -> None:
    """Show durable failure state and the actionable stored root cause for unfinished tasks."""
    incomplete = [row for row in rows if str(row.get("status")) != "completed"]
    if not incomplete:
        return
    print_heading("INCOMPLETE OR FAILED TASKS")
    for row in incomplete:
        task_key = str(row.get("task_key") or "-")
        state = str(row.get("status") or "-")
        attempts = row.get("attempts")
        log_path = run_directory / "logs" / "tasks" / f"{task_key}.log"
        print(f"{task_key}: state={state} | attempts={attempts}")
        print(f"  reason: {concise_error_reason(row.get('error_text'))}")
        print(f"  task log: {log_path}")
    print(
        "These task states prevent a completed-workflow conclusion. Resolve failed tasks "
        "before interpreting any partial score summary."
    )


def print_integrity_and_budget(entries: list[dict[str, Any]], integrity_problems: list[str]) -> None:
    """Verify each completed result against its own immutable task-level budget."""
    print_heading("INTEGRITY AND TASK-LEVEL BUDGET CHECK")
    budget_problems: list[str] = []
    for entry in entries:
        task_key = str(entry["registry"]["task_key"])
        payload = entry["payload"]
        result = entry["result"]
        inner = result.get("inner_search")
        if not isinstance(inner, Mapping):
            budget_problems.append(f"{task_key}: missing inner_search mapping.")
            continue
        expected_stage_a = payload.get("stage_a_n_trials")
        expected_stage_b = payload.get("confirmation_top_k")
        observed_stage_a = inner.get("stage_a_completed_trials")
        records = inner.get("stage_b_records")
        observed_stage_b = len(records) if isinstance(records, list) else "non-list"
        if expected_stage_a is None:
            budget_problems.append(f"{task_key}: registered payload has no stage_a_n_trials.")
        else:
            try:
                stage_a_matches = int(observed_stage_a) == int(expected_stage_a)
            except (TypeError, ValueError):
                stage_a_matches = False
            if not stage_a_matches:
                budget_problems.append(
                    f"{task_key}: Stage A completed {observed_stage_a!r}, expected {expected_stage_a!r}."
                )
        if expected_stage_b is None:
            budget_problems.append(f"{task_key}: registered payload has no confirmation_top_k.")
        else:
            try:
                stage_b_matches = observed_stage_b == int(expected_stage_b)
            except (TypeError, ValueError):
                stage_b_matches = False
            if not stage_b_matches:
                budget_problems.append(
                    f"{task_key}: Stage B record count is {observed_stage_b!r}, expected {expected_stage_b!r}."
                )
    if integrity_problems or budget_problems:
        print("Problems:")
        for problem in [*integrity_problems, *budget_problems]:
            print(f"  - {problem}")
    else:
        print(
            "Passed: every registry task is completed, checksum-verified, and reached its "
            "registered Stage-A and Stage-B budget."
        )


def print_runtime_audit(entries: list[dict[str, Any]]) -> None:
    """Print candidate-level wall-time and fitted-step runtime summaries."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["registry"]["candidate_id"])].append(entry)
    print_heading("RUNTIME BY CANDIDATE")
    print(
        "candidate                     tasks  task wall time                         "
        "inner search                           outer fit             outer prediction"
    )
    print("-" * 132)
    for candidate, group in sorted(grouped.items()):
        wall = [entry["wall_seconds"] for entry in group]
        inner: list[float] = []
        fit: list[float] = []
        prediction: list[float] = []
        for entry in group:
            timing = entry["result"].get("timing_seconds", {})
            if not isinstance(timing, Mapping):
                continue
            for key, target in (("inner_search", inner), ("outer_fit", fit), ("outer_prediction", prediction)):
                if isinstance(timing.get(key), (int, float)):
                    target.append(float(timing[key]))
        print(
            f"{candidate:29} {len(group):5d}  {format_mean_min_max(wall):38}  "
            f"{format_mean_min_max(inner):38}  {format_mean_min_max(fit):20}  "
            f"{format_mean_min_max(prediction):20}"
        )
    total_wall = sum(float(entry["wall_seconds"]) for entry in entries if entry["wall_seconds"] is not None)
    print("\nNote: task wall time includes loading, study setup, checkpointing, and coordinator overhead.")
    print(f"Sum of per-task wall times: {format_duration(total_wall)}")


def print_selection_audit(entries: list[dict[str, Any]]) -> None:
    """Print fold-level selected policy patterns and Stage-A/Stage-B selection traces."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["registry"]["candidate_id"])].append(entry)
    print_heading("SELECTED POLICY PATTERNS")
    for candidate, group in sorted(grouped.items()):
        print(candidate)
        for policy_key in ("feature_policy", "feature_selection_policy", "imbalance_policy"):
            counts: Counter[str] = Counter()
            for entry in group:
                parameters = entry["result"].get("selected_parameters", {})
                value = parameters.get(policy_key, "<not applicable>") if isinstance(parameters, Mapping) else "<missing>"
                counts[str(value)] += 1
            print("  " + policy_key + ": " + ", ".join(
                f"{value} ({count}/{len(group)})" for value, count in sorted(counts.items())
            ))
    print_heading("STAGE-A AND STAGE-B SELECTION TRACE")
    print(
        "task                                  Stage-A best  selected trial  selected Stage-A  "
        "selected Stage-B  all Stage-B AP values"
    )
    print("-" * 126)
    for entry in sorted(entries, key=lambda item: str(item["registry"]["task_key"])):
        result = entry["result"]
        inner = result.get("inner_search", {})
        if not isinstance(inner, Mapping):
            print(f"{entry['registry']['task_key']:37} <inner search unavailable>")
            continue
        selected = selected_stage_b_record(result)
        records = inner.get("stage_b_records", [])
        values = [float(record["stage_b_average_precision"]) for record in records if isinstance(record, Mapping) and isinstance(record.get("stage_b_average_precision"), (int, float))] if isinstance(records, list) else []
        print(
            f"{str(entry['registry']['task_key']):37} {metric(inner.get('stage_a_best_average_precision')):12}  "
            f"{str(selected.get('stage_a_trial_number')) if isinstance(selected, Mapping) else '-':14}  "
            f"{metric(selected.get('stage_a_average_precision') if isinstance(selected, Mapping) else None):16}  "
            f"{metric(selected.get('stage_b_average_precision') if isinstance(selected, Mapping) else None):16}  "
            f"{', '.join(f'{value:.4f}' for value in values) or '-'}"
        )


def print_warning_and_study_audit(entries: list[dict[str, Any]], warning_limit: int) -> None:
    """Print persistent Optuna states, captured warnings, and inspection problems."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["registry"]["candidate_id"])].append(entry)
    print_heading("OPTUNA TRIAL STATES AND WARNING AUDIT")
    all_states: Counter[str] = Counter()
    trial_warnings: dict[str, Counter[str]] = defaultdict(Counter)
    result_warnings: dict[str, Counter[str]] = defaultdict(Counter)
    issues: list[str] = []
    for candidate, group in sorted(grouped.items()):
        candidate_states: Counter[str] = Counter()
        for entry in group:
            inner = entry["result"].get("inner_search", {})
            study_value = inner.get("study_database_path") if isinstance(inner, Mapping) else None
            if study_value:
                states, warnings, problems = inspect_stage_a_study(Path(str(study_value)))
                candidate_states.update(states)
                trial_warnings[candidate].update(warnings)
                issues.extend(problems)
            result_warnings[candidate].update(message for _source, message in recursive_warning_messages(entry["result"]))
        all_states.update(candidate_states)
        print(candidate + ": " + (", ".join(f"{state}={count}" for state, count in sorted(candidate_states.items())) or "no readable study state"))
    print("\nAggregate Stage-A trial states: " + (", ".join(f"{state}={count}" for state, count in sorted(all_states.items())) or "none"))
    for title, warning_map in (("Stage-A trial warnings", trial_warnings), ("Persisted selected-configuration and outer-task warnings", result_warnings)):
        print("\n" + title + ":")
        found = False
        for candidate, counts in sorted(warning_map.items()):
            if not counts:
                continue
            found = True
            print(f"  {candidate}:")
            for message, count in counts.most_common(warning_limit):
                print(f"    [{count}x] {message}")
        if not found:
            print("  None recorded.")
    if issues:
        print("\nStudy-inspection issues:")
        for issue in issues:
            print(f"  - {issue}")


def stage_a_history_by_task(run_directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Load completed Stage-A history rows in Optuna-id order for optional trajectory audit."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_json_lines(run_directory / "logs" / "configuration_history.jsonl"):
        if record.get("event") != "stage_a_trial_terminal" or record.get("status") != "completed":
            continue
        identifier = str(record.get("identifier") or "")
        task_key = str(record.get("task_key") or "")
        if not task_key or "Optuna ID" not in identifier or record.get("average_precision") is None:
            continue
        try:
            trial_number = int(identifier.rsplit(" ", 1)[-1])
            score = float(record["average_precision"])
        except (TypeError, ValueError):
            continue
        grouped[task_key].append({
            "trial_number": trial_number,
            "average_precision": score,
            "duration_seconds": record.get("duration_seconds"),
        })
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["trial_number"]))
    return dict(grouped)


def rank_in_stage_a(rows: list[Mapping[str, Any]], selected_trial_number: int) -> int | None:
    """Return deterministic Stage-A rank for one Stage-B-selected trial."""
    for rank, row in enumerate(sorted(rows, key=lambda item: (-float(item["average_precision"]), int(item["trial_number"]))), start=1):
        if int(row["trial_number"]) == int(selected_trial_number):
            return rank
    return None


def print_stage_a_trajectories(entries: list[dict[str, Any]], run_directory: Path) -> None:
    """Print aggregate cumulative-best Stage-A trajectories and Stage-B winner ranks."""
    history = stage_a_history_by_task(run_directory)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["registry"]["candidate_id"])].append(entry)
    print_heading("OPTIONAL STAGE-A TRAJECTORIES AND STAGE-B WINNER RANKS")
    print(
        "candidate                     tasks  mean cumulative-best AP at checkpoints                 "
        "mean gain first->final  mean gain penultimate->final  Stage-B winner ranks"
    )
    print("-" * 165)
    for candidate, group in sorted(grouped.items()):
        best_at: dict[int, list[float]] = defaultdict(list)
        early_to_final: list[float] = []
        late_to_final: list[float] = []
        ranks: list[int] = []
        usable = 0
        for entry in group:
            rows = history.get(str(entry["registry"]["task_key"]), [])
            if not rows:
                continue
            usable += 1
            cumulative: list[float] = []
            running = float("-inf")
            for row in rows:
                running = max(running, float(row["average_precision"]))
                cumulative.append(running)
            for checkpoint in CHECKPOINTS:
                if len(cumulative) >= checkpoint:
                    best_at[checkpoint].append(cumulative[checkpoint - 1])
            early_to_final.append(cumulative[-1] - cumulative[0])
            if len(cumulative) >= 2:
                late_to_final.append(cumulative[-1] - cumulative[-2])
            selected = selected_stage_b_record(entry["result"])
            if isinstance(selected, Mapping) and selected.get("stage_a_trial_number") is not None:
                rank = rank_in_stage_a(rows, int(selected["stage_a_trial_number"]))
                if rank is not None:
                    ranks.append(rank)
        checkpoint_text = ", ".join(
            f"{checkpoint}:{statistics.mean(best_at[checkpoint]):.4f}"
            for checkpoint in CHECKPOINTS if best_at.get(checkpoint)
        ) or "-"
        early_text = f"{statistics.mean(early_to_final):.4f}" if early_to_final else "-"
        late_text = f"{statistics.mean(late_to_final):.4f}" if late_to_final else "-"
        ranks_text = ", ".join(str(rank) for rank in ranks) or "-"
        print(f"{candidate:29} {usable:5d}  {checkpoint_text:58}  {early_text:23}  {late_text:31}  {ranks_text}")
    print("\nTrajectory rows are diagnostic only. Sparse early evidence cannot prove an optimal master budget.")


def print_resilience_audit(run_directory: Path) -> None:
    """Inspect durable coordinator and task logs for persistent I/O or telemetry issues."""
    coordinator = read_json_lines(run_directory / "logs" / "coordinator_events.jsonl")
    task_records: list[dict[str, Any]] = []
    task_dir = run_directory / "logs" / "tasks"
    if task_dir.exists():
        for path in sorted(task_dir.glob("*.jsonl")):
            task_records.extend(read_json_lines(path))
    terms = ("permissionerror", "winerror", "access denied", "sharing violation", "lock violation", "telemetry degraded", "monitoring degraded", "retry")
    hits: list[str] = []
    for source, records in (("coordinator", coordinator), ("task", task_records)):
        for record in records:
            if any(term in canonical_json(record).lower() for term in terms):
                hits.append(
                    f"{source} | {record.get('occurred_at_local') or record.get('occurred_at_utc') or '-'} | "
                    f"{record.get('task_key') or 'run'} | {record.get('event') or 'event'} | "
                    f"{compact_text(record.get('message') or '-')}"
                )
    print_heading("FILESYSTEM-RESILIENCE AND TELEMETRY EVIDENCE")
    print(f"Coordinator events inspected: {len(coordinator)}")
    print(f"Task-local events inspected: {len(task_records)}")
    if hits:
        print("Persisted I/O or degraded-monitoring indicators:")
        for hit in hits[:30]:
            print("  " + hit)
        if len(hits) > 30:
            print(f"  ... {len(hits) - 30} additional matching records omitted")
    else:
        print("No persisted I/O-error, retry, or degraded-monitoring messages were found.")
    print(
        "Successful bounded replacement retries are intentionally not logged as events. "
        "Therefore an empty result proves no persistent I/O issue was recorded, not that no retry occurred."
    )


def main() -> None:
    """Render the requested read-only audit without changing any artifact or source file."""
    arguments = parse_arguments()
    if arguments.warning_limit < 1:
        raise SystemExit("--warning-limit must be at least one.")
    run_directory = Path(arguments.artifacts_root) / str(arguments.run_id)
    if not run_directory.exists():
        raise SystemExit(f"Run directory does not exist: {run_directory.resolve()}")
    manifest = read_json(run_directory / "run_manifest.json")
    print_manifest_summary(manifest)
    print(f"Directory: {run_directory.resolve()}")
    rows = load_registry_rows(run_directory)
    counts = Counter(str(row["status"]) for row in rows)
    print(f"Registry tasks: {len(rows)}")
    print("States: " + " | ".join(f"{state}={count}" for state, count in sorted(counts.items())))
    print_incomplete_task_audit(rows, run_directory)
    entries, integrity_problems = load_completed_results(run_directory, rows)
    print(f"Checksum-verified completed result artifacts: {len(entries)}")
    print_integrity_and_budget(entries, integrity_problems)
    if entries:
        print_runtime_audit(entries)
        print_selection_audit(entries)
        print_warning_and_study_audit(entries, arguments.warning_limit)
        if arguments.include_stage_a_trajectories:
            print_stage_a_trajectories(entries, run_directory)
    print_resilience_audit(run_directory)
    print_heading("AUDIT CONCLUSION")
    print(
        "This report describes durable execution, search behaviour, and stored diagnostics. "
        "Interpret all scores according to the run manifest's declared purpose before using "
        "them in any selection decision."
    )


if __name__ == "__main__":
    main()
