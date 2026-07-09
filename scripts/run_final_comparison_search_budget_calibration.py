"""Run a bounded search-budget calibration experiment for representative procedures.

This command is a pre-master calibration workflow, not a final comparison. It evaluates
ten representative candidate routes on all development data using two outer folds, one
repeat, a 36-trial Stage-A Optuna search, and five-fold confirmation of the top five
Stage-A configurations. This is not full-universe admission. The candidate set
deliberately spans cheap linear models, classical trees, bagging, histogram boosting,
native categorical boosting, kernel methods, and dense neural networks while leaving
C24 TabNet, C25 FT-Transformer, and C26 TabM to the bounded all-candidate admission
smoke and their dedicated implementation smokes.

The run is designed to quantify trajectory shape, late-search gains, Stage-B winner rank,
runtime, warnings, and failure modes before a master comparison protocol freezes its
candidate-specific search budgets. Its outer scores must not be used to select final
procedures, remove families, exclude advanced candidates, or inspect the held-out test set.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.candidates import (  # noqa: E402
    CANDIDATE_CATBOOST,
    CANDIDATE_DECISION_TREE,
    CANDIDATE_HIST_GRADIENT_BOOSTING,
    CANDIDATE_LIGHTGBM,
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_MLP,
    CANDIDATE_RANDOM_FOREST,
    CANDIDATE_RBF_SVM,
    CANDIDATE_RIDGE_CLASSIFIER,
    CANDIDATE_XGBOOST,
)
from telco_churn.pre_master_workflows import (  # noqa: E402
    PreMasterWorkflowSpec,
    run_pre_master_workflow,
)


CALIBRATION_CANDIDATE_IDS: tuple[str, ...] = (
    CANDIDATE_RIDGE_CLASSIFIER,
    CANDIDATE_LOGISTIC_REGRESSION,
    CANDIDATE_DECISION_TREE,
    CANDIDATE_RANDOM_FOREST,
    CANDIDATE_HIST_GRADIENT_BOOSTING,
    CANDIDATE_XGBOOST,
    CANDIDATE_LIGHTGBM,
    CANDIDATE_CATBOOST,
    CANDIDATE_RBF_SVM,
    CANDIDATE_MLP,
)

WORKFLOW_SPEC = PreMasterWorkflowSpec(
    workflow_id="telco_search_budget_calibration_v1",
    protocol_id="telco_final_comparison_search_budget_calibration",
    protocol_version="v1",
    default_run_id="search_budget_calibration_v1",
    purpose=(
        "search-budget and confirmation-policy calibration for representative candidate "
        "routes before a master protocol is frozen; not full-universe admission, "
        "candidate ranking, selection, or candidate-elimination evidence"
    ),
    candidate_set_role=(
        "representative computational and modelling routes: regularized linear, single "
        "tree, bagging, histogram boosting, external boosting, native categorical boosting, "
        "kernel margin, and dense neural network; C24-C26 advanced neural routes are "
        "intentionally excluded from this calibration subset for now"
    ),
    candidate_ids=CALIBRATION_CANDIDATE_IDS,
    outer_n_splits=2,
    outer_n_repeats=1,
    stage_a_n_splits=3,
    stage_b_n_splits=5,
    stage_a_n_trials=36,
    confirmation_top_k=5,
    search_profile="full",
    feature_policy_contract="F2 remains restricted to regularized linear procedures.",
    seed_namespace="search_budget_calibration_v1",
    default_max_workers=2,
)


def main() -> None:
    """Execute the immutable representative-route search-budget calibration workflow."""
    run_pre_master_workflow(WORKFLOW_SPEC)


if __name__ == "__main__":
    main()
