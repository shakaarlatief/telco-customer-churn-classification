# Final Comparison Implementation Phase 8B: Candidate-Specific Imbalance Routing

## Purpose

Phase 7 introduced and tested the imbalance-policy primitives. Phase 8A established the
fit-time pipeline topologies for balanced sample weights, random resampling, and raw-only
SMOTENC. This phase makes those verified mechanisms part of a complete candidate procedure:
the policy is selected inside inner HPO, recorded in every persisted trial, bound into the
persistent-study compatibility fingerprint, and re-created for the selected outer-fold fit.

The held-out test data is not loaded or referenced.

## One mutually exclusive treatment branch

Every candidate configuration now contains exactly one of the following policy identifiers:

```text
I0_NONE
I1_CLASS_WEIGHT_BALANCED
I2_RANDOM_OVERSAMPLING
I3_RANDOM_UNDERSAMPLING
I4_SMOTENC
```

I1 calculates balanced sample weights from the target vector in the active training
partition. I2 through I4 alter only fit-time training rows through an `imblearn` pipeline.
Validation and test rows never enter `fit_resample`.

The candidate builders now fix older estimator-specific `class_weight` and
`base_class_weight` controls to their neutral value. This prevents a procedure from
silently combining two imbalance corrections, such as class weighting plus random
over-sampling. The single selected policy remains visible in stored trial parameters.

## Candidate compatibility matrix

```text
Ridge and logistic regression:
    I0, I1, I2, I3, I4 when feature policy is F0
    I0, I1, I2, I3 when feature policy is F1 or F2

Linear SVM, RBF SVM, and MLP:
    I0, I1, I2, I3, I4 when feature policy is F0
    I0, I1, I2, I3 when feature policy is F1

kNN and hybrid Gaussian-Bernoulli Naive Bayes:
    I0, I2, I3, I4 when feature policy is F0
    I0, I2, I3 when feature policy is F1

Decision tree, Extra Trees, bagging, random forest, AdaBoost,
GradientBoostingClassifier, HistGradientBoostingClassifier,
XGBoost, LightGBM, and CatBoost:
    I0, I1
```

The matrix deliberately does not claim that any policy is superior. It merely prevents
incoherent or untested procedure combinations from entering the search.

## Why SMOTENC is raw-only

SMOTENC creates synthetic observations. F1 and F2 contain derived counts, products,
interactions, and nonlinear transforms. Independently synthesising those derived columns
would not ensure that they agree with a coherent raw customer profile. I4 therefore remains
available only with F0 until a later design can synthesize raw inputs first and recompute all
derived features deterministically afterward.

## Search-space controls

For I2 and I3, the inner search selects the post-resampling minority-to-majority ratio from
`0.50`, `0.75`, and `1.00`. I4 uses the same ratio and selects SMOTENC neighbour counts from
`3`, `5`, and `7`. The smoke profile fixes the ratio at `0.75` and the SMOTENC neighbour
count at `3` so that infrastructure validation remains deterministic and fast.

Optuna requires one fixed categorical distribution per parameter name within a persistent
study. Because F0 may select I4 whereas F1 and F2 cannot, the imbalance-policy draw uses a
policy-specific parameter name such as `imbalance_policy__f0_raw`,
`imbalance_policy__f1_domain_enriched`, or `imbalance_policy__f2_linear_expanded`. The
persisted candidate configuration continues to store the canonical `imbalance_policy` key;
the distinct Optuna parameter names only make the conditional search space valid and
resume-safe.

## Persistent-study safety

The candidate-procedure contract now records the complete nested map:

```text
candidate family
    -> feature policy
        -> feature-selection policy
            -> allowed imbalance policies
```

Because the HPO study fingerprint already includes the candidate-procedure contract,
resuming a study after this routing universe changes raises a compatibility error instead
of reusing trials created under a different procedure definition.

## Smoke-test coverage

The routing smoke test uses only a stratified development-data partition. It verifies:

```text
all declared I1 weighted routes fit and expose equal total class mass
representative I2 and I3 routes fit after their selected representation
representative I4 routes fit through the mixed-table SMOTENC topology
F1 plus I4 is rejected before estimator fitting
candidate procedure contracts include the imbalance routing matrix
valid continuous scores are available after every fitted route
```

This is a technical-contract test. Nested CV, not this smoke test, determines whether a
policy improves average precision for a candidate family.
