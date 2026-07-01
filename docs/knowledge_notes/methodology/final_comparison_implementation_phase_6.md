# Final Comparison Implementation Phase 6: Fold-Safe Feature-Selection Policies

## Purpose

Phase 5 made deterministic feature policy F0, F1, or F2 part of each candidate
procedure. Phase 6 adds the next fitted layer that can operate on a represented feature
matrix:

```text
raw rows
    -> feature policy
    -> fold-local representation preprocessing
    -> optional fold-local feature selection
    -> classifier
```

Feature selection is never fitted on all development rows before an inner or outer
validation split. The learned variance mask, mutual-information ranking, or L1 logistic
coefficient mask is estimated solely from the active training partition.

The held-out test set is not loaded or referenced.

## Implemented policies

```text
S0_NONE
    Retain every represented column.

S1_VARIANCE_MUTUAL_INFO
    Remove zero-variance represented columns, then select the strongest
    mutual-information features.

S2_L1_LOGISTIC_SELECT_FROM_MODEL
    Fit L1-regularized logistic regression and use SelectFromModel to retain
    coefficient-supported columns.
```

S1 receives the one-hot represented matrix after preprocessing. The selector records
which leading columns are continuous numeric features and which remaining columns are
one-hot indicators, so mutual information is evaluated with the correct continuous /
discrete declaration. A requested `k` is capped at the nonconstant width observed in a
training fold. This avoids failure when a rare category is absent in a small inner split.

S2 uses a sparse L1 logistic selector with a balanced training loss. The inner HPO search
tunes its L1 strength and standard SelectFromModel threshold. If a very strong penalty
would select no feature, a deterministic fallback retains the highest absolute
coefficient. This preserves a valid downstream pipeline while documenting the selected
feature count.

## Candidate compatibility matrix

Feature selection is not presumed beneficial for every model family.

```text
Ridge, logistic regression, linear SVM:
    S0, S1, S2

kNN, RBF SVM, dense MLP:
    S0, S1

Hybrid Naive Bayes, decision tree, Extra Trees, bagging, random forest,
AdaBoost, gradient boosting, histogram gradient boosting, XGBoost, LightGBM,
and CatBoost:
    S0 only
```

The second group can benefit from a compact ranking- or distance-oriented feature space,
but it has no equally strong rationale for a logistic-model selector. The final group
already contains internal split selection, nonlinear interaction construction, or a
native-category representation. Treating external one-hot feature selection as a default
improvement would duplicate their modelling role and needlessly expand the experiment
universe.

The F2 systematic linear expansion remains restricted to Ridge and logistic regression.
F2 therefore combines with S0, S1, or S2 only for those two families.

## Inner-HPO and resume contract

The candidate registry first samples a compatible feature policy, then samples a
compatible selection policy conditionally. Selection-specific hyperparameters are only
sampled on the corresponding branch:

```text
S0:
    no selector hyperparameters

S1:
    selected-feature upper bound k

S2:
    L1 logistic C and SelectFromModel threshold
```

The persistent-study fingerprint now includes a deterministic candidate-procedure
routing fingerprint. It binds both the feature-policy matrix and the feature-selection
matrix to study reuse. A study created under an earlier routing contract cannot silently
resume under a different selector universe.

## Smoke test

`smoke_test_final_comparison_feature_selection.py` fits every nontrivial declared S1/S2
route on a small stratified sample from `train.csv` only. It verifies:

```text
22 nontrivial compatible routes
correct pipeline order
correct selector class
nonempty fitted support mask
valid binary predictions and continuous scores
rejection of an undeclared Extra Trees plus S1 route
stable candidate-procedure contract fingerprints
```

The existing core-candidate and feature-policy-routing smoke tests continue to cover all
S0 routes, including native-categorical LightGBM and CatBoost paths.

## Non-actions

Phase 6 does not:

```text
run the master repeated nested-CV comparison
perform stability selection S3
add resampling or synthetic oversampling
calibrate scores or select thresholds
fit stacking or blending
load the held-out test set
```
