# Telco Customer Churn Classification: Chat Handoff Context 9

## Purpose

This is the current chat handoff for the Telco Customer Churn classification project. It supersedes `telco_churn_chat_handoff_context_8.md` for the current working state, while contexts 1 through 8 remain useful historical records.

This file is intentionally a handoff snapshot, not a replacement for the live coordination documents. For the current tactical source of truth, continue to use `docs/knowledge_notes/current_project_status_and_next_actions.md`.

## Repository and Working-Tree State

Repository:

```text
shakaarlatief/telco-customer-churn-classification
branch: main
```

Latest confirmed committed and pushed revision before this documentation handoff:

```text
6104dcf05495edbed76ea99318c5d9053f644929
Add guarded final held-out evaluation workflow
```

Git checks recorded while preparing this handoff:

```text
git status --short
 M docs/knowledge_notes/00_documentation_workflow.md
 M docs/knowledge_notes/01_model_inventory_and_roadmap.md
 M docs/knowledge_notes/02_candidate_status_register.md
 M docs/knowledge_notes/current_project_status_and_next_actions.md
?? artifacts/
```

```text
git log --oneline -8
6104dcf Add guarded final held-out evaluation workflow
5c59f84 Add frozen final procedure refit workflow
aa00323 Preserve complete tuned candidate parameters
573b3a6 Add fast finalization workflow
ab847cd Add leading candidate selection workflow
ba7c377 Add fast-completion comparison protocol
9b3a65a Freeze protocol v2 base comparison
05c6bdd Update status after protocol v2 scaffold
```

```text
git rev-parse HEAD
6104dcf05495edbed76ea99318c5d9053f644929
```

```text
git rev-parse origin/main
6104dcf05495edbed76ea99318c5d9053f644929
```

At handoff creation time, the four coordination documents had documentation-only local modifications:

```text
docs/knowledge_notes/00_documentation_workflow.md
docs/knowledge_notes/01_model_inventory_and_roadmap.md
docs/knowledge_notes/02_candidate_status_register.md
docs/knowledge_notes/current_project_status_and_next_actions.md
```

This context 9 file is newly created and uncommitted. The local `artifacts/` tree is untracked and must not be committed wholesale. Large run artifacts, out-of-fold predictions, row-level predictions, and fitted joblib files should stay local unless a deliberate publication policy is established.

The guarded final evaluator implementation is already committed and pushed at `6104dcf`. Do not assume this Git state remains current in a later chat without rerunning Git checks.

## Project Purpose and User Preferences

The project is a professional, portfolio-ready churn-classification project and a reusable reference project covering many classification methods, not only the final winner.

The user values:

```text
deep mathematical and methodological explanation
standalone report prose that does not refer to lectures, slides, or a course
LaTeX for the formal report
professional and detailed code and technical documentation
no emojis
no em dashes
explicit descriptions of every meaningful change
no silent modifications or deletion of useful content
```

Current working model:

```text
ChatGPT develops modelling/design specifications and precise Codex prompts.
Codex inspects and implements locally.
The user reviews diffs and validation.
The user controls staging, commits, and pushes.
```

Do not stage, commit, push, or directly modify repository state without explicit user instruction.

## Dataset and Test Boundary

Dataset state:

```text
Clean modelling dataset: 7043 rows
Development data:        5634 rows
Held-out test set:       1409 rows
Target:                  Churn_binary
```

The held-out test remains untouched. No final test metrics exist. The user explicitly deferred test evaluation, and test evaluation is not required before documentation and reporting work.

Do not run the final held-out evaluator unless the user explicitly requests it. Do not inspect or summarize test data. Future test results may not be used to alter model members, weights, threshold, calibration, preprocessing, features, or any other procedure component.

## Evidence Categories

Keep these evidence categories distinct:

```text
implementation-admission evidence
runtime evidence
fast-completion development evidence
robust protocol-v2 evidence
held-out test evidence
```

Implementation admission confirms runnable implementations. Runtime evidence concerns feasibility. Fast-completion evidence supports the completed development pipeline. Robust protocol-v2 evidence has not been completed. Held-out test evidence does not exist.

## Candidate and Comparison State

Candidate state:

```text
C01-C26 implemented
C27 TabPFN deferred
C28 AutoGluon deferred
C01-C26 warning-clean admission smoke completed
```

Fast-completion run:

```text
run_id:          fast_completion_v1
submitted:       52
completed:       52
failed:          0
interrupted:     0
paused:          0
data:            development data only
outer CV:        2 folds x 1 repeat
tuning trials:   2
inner CV:        2 folds
primary metric:  average precision
evidence role:   fast_completion_pipeline_evidence
```

The frozen robust protocol remains:

```text
protocols/final_comparison_protocol_v2_base.json
```

It remains frozen and untouched but was not completed. It is optional future work, not the immediate next step.

## Leading Candidates and Fast Finalization

Selected leading set:

1. `C03_SPLINE_LOGISTIC_REGRESSION`
2. `C20_EXPLAINABLE_BOOSTING_MACHINE`
3. `C25_FT_TRANSFORMER`
4. `C01_RIDGE_CLASSIFIER`
5. `C18_LIGHTGBM`

Finalization development OOF average precision:

```text
C03_SPLINE_LOGISTIC_REGRESSION:      approximately 0.666354114311
C25_FT_TRANSFORMER:                  approximately 0.665475534103
C20_EXPLAINABLE_BOOSTING_MACHINE:    approximately 0.664879135290
C01_RIDGE_CLASSIFIER:                approximately 0.649499243122
C18_LIGHTGBM:                        approximately 0.616658334430
```

Selected development procedure:

```text
procedure_id:    top3_unweighted_soft_average
procedure_type:  ensemble
development OOF average precision: approximately 0.669605382543
```

The ensemble exceeded the best individual by approximately `0.003251` average precision, more than the frozen `0.002` simplicity tolerance. Do not describe this as robust protocol-v2 evidence or final test performance.

## Frozen Final Procedure

Members in order:

1. `C03_SPLINE_LOGISTIC_REGRESSION`
2. `C25_FT_TRANSFORMER`
3. `C20_EXPLAINABLE_BOOSTING_MACHINE`

Weights:

```text
0.3333333333333333
0.3333333333333333
0.3333333333333333
```

Aggregation:

```text
arithmetic mean of positive-class probabilities
```

Frozen decision threshold:

```text
0.39106601395524887
```

Threshold origin:

```text
development-data OOF F1 maximization
```

Calibration:

```text
method: none
status: deferred_fast_completion
```

These elements are frozen and must not be changed based on future held-out test results.

## Final Development Refit and Serialization

The final procedure was fitted on all 5,634 development rows.

Serialized model:

```text
artifacts/final_selection/fast_completion_v1/final_development_refit_v1/fitted_final_pipeline.joblib
```

Model type:

```text
FrozenProbabilityVotingEnsemble
```

Validation state:

```text
serialization round trip passed
pre-save and post-load probabilities matched
pre-save and post-load predictions matched
independent manual loading passed
```

Standalone import requirement:

```text
PYTHONPATH=src ./.venv/Scripts/python.exe ...
```

Serialized object fields:

```text
member_ids
member_weights
decision_threshold
calibration_method
calibration_status
```

## Guarded Final Evaluator

Implemented files:

```text
src/telco_churn/final_evaluation.py
scripts/audit_final_test_readiness.py
scripts/evaluate_final_held_out_test.py
scripts/smoke_test_final_held_out_evaluation.py
```

Evaluator state:

```text
implementation committed and pushed
py_compile passed
smoke test passed
readiness audit reported READY
evaluator dry-run passed
real evaluator was not run
no final-evaluation output directory exists
no evaluation receipt exists
no test metrics exist
```

Exact confirmation phrase, recorded only as a safety reference:

```text
I_UNDERSTAND_THIS_CONSUMES_THE_FINAL_TEST_SET
```

Do not present running the evaluator as the next action.

## Important Resolved Implementation Issue

Optuna `best_trial.params` contains only `trial.suggest_*` values. Fixed executable parameters such as `max_iter` and `class_weight` were initially lost. The workflow was fixed to preserve the complete executable configuration for each trial, and reconstruction now uses the winning trial's complete configuration.

Regression coverage was added. Do not revert this behavior.

## Artifact Locations and Publication Policy

Artifact locations:

```text
fast comparison:
    artifacts/final_comparison/fast_completion_v1

leading selection:
    artifacts/final_selection/fast_completion_v1

fast finalization:
    artifacts/final_selection/fast_completion_v1/fast_finalization_v1

frozen procedure:
    artifacts/final_selection/fast_completion_v1/frozen_final_procedure_v1

development refit:
    artifacts/final_selection/fast_completion_v1/final_development_refit_v1

potential future test evaluation:
    artifacts/final_evaluation/fast_completion_v1/held_out_test_v1
```

The `artifacts/` tree remains local and untracked. Do not commit it wholesale. Large run artifacts, OOF predictions, row-level predictions, and fitted joblib files should remain local unless a deliberate publication policy is established.

Later, decide which compact tables, summaries, manifests, and report-ready outputs belong in Git.

## Documentation Structure

Stable workflow rules:

```text
docs/knowledge_notes/00_documentation_workflow.md
```

Strategic roadmap:

```text
docs/knowledge_notes/01_model_inventory_and_roadmap.md
```

Candidate register:

```text
docs/knowledge_notes/02_candidate_status_register.md
```

Live tactical status:

```text
docs/knowledge_notes/current_project_status_and_next_actions.md
```

Historical chat handoffs:

```text
docs/knowledge_notes/context_history/
```

Reusable theory:

```text
docs/knowledge_notes/methodology/
docs/knowledge_notes/models/
```

Report:

```text
reports/latex/
```

The current-status file remains the live tactical source of truth. Context 9 is a handoff snapshot, not a replacement for the coordination-document system.

## Current Priority and Next Actions

The modelling and full-development-refit pipeline is complete, but the complete portfolio project is not finished.

Current priority:

1. review existing report, knowledge notes, notebooks, tables, and figures;
2. organize the remaining documentation plan;
3. complete the LaTeX technical report;
4. add detailed mathematical and methodological explanations;
5. present development evidence clearly;
6. improve README and repository navigation;
7. document limitations of fast-completion evidence;
8. decide which compact artifacts should be version-controlled;
9. optionally return to robust protocol v2 later;
10. leave the held-out test untouched until explicitly requested.

## New-Chat Startup Checklist

1. Read context 9.
2. Read `current_project_status_and_next_actions.md`.
3. Read `02_candidate_status_register.md`.
4. Read `01_model_inventory_and_roadmap.md`.
5. Read `00_documentation_workflow.md`.
6. Run `git status --short`.
7. Run `git log --oneline -8`.
8. Confirm whether the documentation handoff changes were committed.
9. Confirm `artifacts/` remains untracked.
10. Do not run the held-out evaluator.
11. Continue with documentation/reporting unless the user changes priorities.
