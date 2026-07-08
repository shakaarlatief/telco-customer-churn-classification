# Telco Churn Chat Handoff Context 8

## Purpose

This handoff records the current working state after the C20 Explainable Boosting Machine implementation. It is intended to let a new chat continue the Telco Customer Churn final-comparison work without relying on the previous conversation.

## Repository and latest pushed state

Repository:

```text
shakaarlatief/telco-customer-churn-classification
branch: main
```

Latest pushed commit:

```text
d5d34cf Add explainable boosting machine candidate
```

Previous conventional expansion commit:

```text
98eeec006e69b6871f08874f398de75169ca81c0 Add conventional core candidate expansion
```

The C20 commit was pushed successfully to `origin/main`. After the push, the only local untracked files reported were the intentional pre-master workflow files listed below.

## Local files to preserve

The following local files are intentionally untracked and must not be touched, staged, or overwritten unless the user explicitly decides to continue the pre-master workflow implementation:

```text
scripts/audit_final_comparison_run.py
scripts/run_final_comparison_admission_smoke.py
scripts/run_final_comparison_search_budget_calibration.py
scripts/smoke_test_pre_master_workflows.py
src/telco_churn/pre_master_workflows.py
```

These files were reviewed previously as a local pre-master workflow baseline. They must be reconciled with the final candidate-admission scope before any real pre-master experiment is launched.

## Dataset and held-out test boundary

Dataset state:

```text
Clean modelling dataset: 7043 rows
Development training set: 5634 rows
Held-out test set:       1409 rows
Positive class:          Churn_binary = 1
```

The held-out test set remains untouched for model-family selection, preprocessing selection, feature-policy selection, hyperparameter search, calibration, threshold selection, stacking, and final candidate comparison. It is used exactly once only after one complete final procedure is frozen.

## Implemented candidate state

The implemented final-comparison core registry now contains 23 candidates, C01 through C23:

```text
C01  Ridge classifier
C02  Regularized logistic regression
C03  Spline logistic regression
C04  Shrinkage linear discriminant analysis
C05  Regularized quadratic discriminant analysis
C06  k-nearest neighbours
C07  Hybrid Gaussian-Bernoulli Naive Bayes
C08  Regularized decision tree
C09  Extra Trees
C10  Bagged decision trees
C11  Random forest
C12  Balanced random forest
C13  AdaBoost
C14  RUSBoost
C15  GradientBoostingClassifier
C16  HistGradientBoostingClassifier
C17  XGBoost
C18  LightGBM
C19  CatBoost
C20  Explainable Boosting Machine
C21  Linear SVM
C22  RBF-kernel SVM
C23  Dense multilayer perceptron
```

No candidate is master-admitted yet. Protocol v2 has not been frozen.

## C20 Explainable Boosting Machine design

C20 was added as an interpretable nonlinear additive comparator. Its design is:

```text
package:
    interpret-core==0.7.8

estimator:
    interpret.glassbox.ExplainableBoostingClassifier

representation:
    native categorical string columns

feature policies:
    F0_RAW
    F1_DOMAIN_ENRICHED

excluded feature policy:
    F2_LINEAR_EXPANDED

feature selection:
    S0_NONE only

imbalance policies:
    I0_NONE
    I1_CLASS_WEIGHT_BALANCED

excluded imbalance policies:
    I2_RANDOM_OVERSAMPLING
    I3_RANDOM_UNDERSAMPLING
    I4_SMOTENC

resource policy:
    n_jobs=1

initial deterministic speed policy:
    inner_bags=0
    validation_size=0.15
```

C20 uses a bounded smoke search space and a modest full search space. It is routed as weighted-only, consistent with tree and boosting-style procedures rather than logistic-style resampling.

## C20 validation already passed

The following validation passed after C20 implementation:

```text
compile check:
    passed

dedicated C20 EBM smoke:
    passed

complete core candidate registry smoke:
    passed, 23/23 candidates

feature-policy routing smoke:
    passed, 47/47 routes

feature-selection routing smoke:
    passed, 22/22 nontrivial selector routes

imbalance-routing smoke:
    passed

pip check:
    passed

git diff --check:
    passed
```

The C20 commit included:

```text
requirements.txt
scripts/smoke_test_final_comparison_explainable_boosting.py
scripts/smoke_test_final_comparison_feature_policy_routing.py
src/telco_churn/candidates.py
src/telco_churn/core_candidate_builders.py
src/telco_churn/imbalance_routing.py
```

## Advanced-candidate admission state

The remaining documented candidates are C24-C28. Current status:

```text
C24 TabNet:
    package/API smoke feasible;
    not implemented in the final-comparison registry yet

C25 FT-Transformer:
    use official rtdl_revisiting_models route;
    PyTorch Tabular route rejected because of dependency conflict;
    not implemented in the final-comparison registry yet

C26 TabM:
    package/API smoke feasible;
    not implemented in the final-comparison registry yet

C27 TabPFN:
    deferred because current CPU practicality and model-weight/licence constraints make it
    unsuitable for this project stage

C28 AutoGluon:
    deferred because the resolver would downgrade the numerical stack
```

Installed or verified package direction from the admission probes:

```text
interpret-core==0.7.8
torch==2.12.1
pytorch-tabnet==4.1.0
rtdl_revisiting_models==0.0.2
tabm==0.0.3
```

C24-C26 package/API smoke checks passed outside the final-comparison registry. They are not yet implemented as registry candidates.

## Workflow notes from the C20 implementation

The earlier generated patch-installer approach should not be reused for large multi-file source-code integration. It failed because exact text anchors in evolving source files no longer matched the local repository. The failures did not damage the repository, but they showed that brittle external patch installers are not the right workflow for large candidate integrations.

For C20, the safer workflow was:

1. Make the modelling and routing decisions explicitly.
2. Ask Codex to inspect the actual local repository and produce a plan.
3. Review the plan before implementation.
4. Let Codex implement only the approved source-code change.
5. Inspect the diff.
6. Run dedicated and shared training-only smokes.
7. Stage only the intended files.
8. Commit and push locally.

This worked well for C20 because Codex adapted the implementation to the real current repository structure while the modelling choices remained fixed.

For documentation updates, the preferred workflow is different:

1. Use exact local docs-only scripts or patches.
2. Avoid broad Codex rewrites.
3. Review the docs diff before staging.
4. Commit only documentation files.

The current coordination-docs update was produced as a docs-only local update after C20. It should be committed separately from the C20 source-code commit. The temporary helper script `update_coordination_docs_after_c20.py` must not be committed.

The user should continue to apply, validate, stage, commit, and push changes locally. Avoid direct assistant-side repository writes unless the user explicitly asks for them.

## Recommended next actions

The next safe sequence is:

```text
1. Commit and push this coordination-docs update as docs-only.

2. Preserve the local untracked pre-master workflow files.

3. Decide whether to implement C24-C26.

4. Keep C27 and C28 deferred unless their constraints materially change.

5. If C24-C26 are implemented, validate them through training-only smokes.

6. Reconcile the pre-master workflow files with the final admitted-candidate scope.

7. Run the all-admitted-candidate admission smoke.

8. Run representative search-budget calibration.

9. Freeze protocol v2.

10. Only after protocol v2 is frozen, launch the repeated nested-CV master comparison on development data.
```

Do not run the held-out test set. Do not treat pilot AP values as model-selection evidence.
