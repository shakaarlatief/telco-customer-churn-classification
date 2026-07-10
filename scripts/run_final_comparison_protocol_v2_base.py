"""Inspect the executable protocol-v2 base-comparison scaffold.

The checked-in protocol declaration is still a draft pending human review. Dry-run mode
is available for task-count and budget inspection. Non-dry-run execution is refused
until the protocol declaration is explicitly frozen and the user supplies the official
confirmation flag.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.protocol_v2_workflows import run_protocol_v2_base_workflow  # noqa: E402


def main() -> None:
    """Run the protocol-v2 scaffold command."""
    run_protocol_v2_base_workflow()


if __name__ == "__main__":
    main()
