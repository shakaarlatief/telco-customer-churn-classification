"""Run all implemented candidate procedures through bounded nested-CV admission checks.

This command is an implementation-admission workflow, not a final comparison. It uses
all development rows and the same persistent task worker used by the completed v6 pilot,
but deliberately keeps its evaluation budget small:

* all current implemented core candidate families (currently C01-C26);
* two outer folds and one repeat;
* three valid Stage-A configurations per outer task;
* the top two configurations confirmed with a separate three-fold Stage B;
* the bounded ``smoke`` search profile for expensive libraries.

A completed run establishes only that every currently registered candidate can traverse
its declared representation, feature-selection, imbalance, persistence, monitoring,
resume, and outer-evaluation route. It must not be used to rank models, eliminate
families, select a feature policy, or choose a final procedure. C27 TabPFN and C28
AutoGluon remain deferred. No candidate is master-admitted, protocol v2 is not frozen,
and the held-out test set is never loaded or referenced.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import INITIAL_CANDIDATE_REGISTRY  # noqa: E402
from telco_churn.pre_master_workflows import (  # noqa: E402
    PreMasterWorkflowSpec,
    run_pre_master_workflow,
)


ADMISSION_CANDIDATE_IDS: tuple[str, ...] = tuple(
    definition.candidate_id for definition in INITIAL_CANDIDATE_REGISTRY
)

WORKFLOW_SPEC = PreMasterWorkflowSpec(
    workflow_id="telco_all_candidate_admission_smoke_v1",
    protocol_id="telco_final_comparison_all_candidate_admission_smoke",
    protocol_version="v1",
    default_run_id="admission_smoke_all_candidates_v1",
    purpose=(
        "implementation-admission validation for every currently implemented candidate "
        "procedure in the C01-C26 registry universe; not a ranking, selection, "
        "or elimination run"
    ),
    candidate_set_role=(
        "all current implemented core candidate procedures discovered from the registry; "
        "C27_TABPFN and C28_AUTOGLUON remain deferred, and each implemented route must "
        "complete the real persistent nested-CV worker before it can be admitted to a "
        "later master run"
    ),
    candidate_ids=ADMISSION_CANDIDATE_IDS,
    outer_n_splits=2,
    outer_n_repeats=1,
    stage_a_n_splits=3,
    stage_b_n_splits=3,
    stage_a_n_trials=3,
    confirmation_top_k=2,
    search_profile="smoke",
    feature_policy_contract="F2 remains restricted to regularized linear procedures.",
    seed_namespace="all_candidate_admission_smoke_v1",
    default_max_workers=2,
)


def main() -> None:
    """Execute the immutable all-candidate implementation-admission workflow."""
    run_pre_master_workflow(WORKFLOW_SPEC)


if __name__ == "__main__":
    main()
