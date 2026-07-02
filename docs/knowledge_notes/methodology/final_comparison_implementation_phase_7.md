# Final Comparison Implementation Phase 7: Fold-Safe Imbalance-Treatment Primitives

## Purpose

The Telco development target has a minority churn class. Class imbalance is therefore
not handled as a post-processing convenience or by one global preprocessing operation.
It is part of the candidate-procedure definition and must be fitted only from the active
training partition of every inner and outer split.

Phase 7 establishes and independently validates the reusable imbalance-policy layer
before it is admitted to the complete candidate registry. This staged implementation is
intentional: ordinary random resampling is naturally placed after a fitted numerical
representation, whereas SMOTENC must operate earlier on a mixed numeric/categorical
DataFrame before one-hot encoding. The placement contracts need verification before
combining them with every model, feature policy, and feature selector.

The held-out test set is not loaded or referenced.

## Initial policy family

```text
I0_NONE
    Preserve the observed training-fold class distribution.

I1_CLASS_WEIGHT_BALANCED
    Compute balanced binary class weights from the active training target:
    weight_c = n / (2 n_c).

I2_RANDOM_OVERSAMPLING
    Duplicate minority-class training rows to a predeclared post-resampling ratio.

I3_RANDOM_UNDERSAMPLING
    Randomly discard majority-class training rows to a predeclared ratio.

I4_SMOTENC
    Synthetically oversample mixed numerical and nominal data before one-hot encoding.
```

The policies are alternatives, not operations to be stacked by default. A procedure
selects at most one imbalance policy inside the inner hyperparameter search. This keeps
interpretation clear and prevents a class-weighted, oversampled model from silently
receiving two unrelated corrections unless a later protocol explicitly declares that
combination as its own candidate.

## Why ordinary SMOTE is not used on one-hot Telco data

One-hot categorical indicators are not continuous measurements. Interpolating them with
ordinary SMOTE can generate fractional indicator values that do not correspond to a real
service, contract, or payment category. SMOTENC instead distinguishes numeric and
categorical columns, generates numeric coordinates from minority neighbours, and assigns
categorical coordinates by a category-aware rule.

The compatible data path is therefore:

```text
raw development training rows
    -> FeaturePolicyTransformer(F0 or F1)
    -> FeaturePolicySamplerImputer
    -> SMOTENC
    -> one-hot or native-categorical representation
    -> optional feature selector
    -> classifier
```

The sampler imputer preserves a pandas DataFrame and the fixed policy schema. It learns
only training-fold numerical medians and categorical modes. The fixed numeric-first,
categorical-second ordering provides the categorical boolean mask supplied to SMOTENC.

## Random resampling placement

Random over- and undersampling do not synthesize feature values. They can therefore work
after a compatible feature representation is fitted:

```text
raw development training rows
    -> FeaturePolicyTransformer
    -> representation preprocessor
    -> random over- or undersampler
    -> optional feature selector
    -> classifier
```

The later routing phase will use an `imblearn.pipeline.Pipeline` where a sampler is
executed during `fit` only. Validation rows never enter `fit_resample`; they flow through
ordinary prediction-time transformation only.

## Smoke-test coverage

The Phase 7 smoke test uses a stratified development-only training partition and validates
all policies on F0 and F1 feature-policy tables.

```text
I0:
    exact preservation of class counts

I1:
    equal total weighted mass for the two classes

I2:
    increased training-row count and requested minority-to-majority ratio

I3:
    reduced training-row count and requested ratio

I4:
    increased training-row count, requested ratio, preserved DataFrame schema,
    finite numeric values, complete data, and no unseen categorical levels
```

This is an implementation-contract test, not a performance comparison and not evidence
that a resampling method improves average precision. The later nested-CV routing phase
will determine whether a compatible procedure should select an imbalance policy.

## Dependency

Phase 7 adds `imbalanced-learn` to the locked environment. It supplies the cloneable
random samplers, SMOTENC, and the pipeline type required for correct fit-time-only
resampling in the subsequent routing phase.

## Non-actions

Phase 7 does not yet:

```text
wire imbalance policies into candidate-specific Optuna spaces
inject class-weight controls into estimator builders
place samplers into final candidate pipelines
add algorithm-specific balanced ensembles
run the master repeated nested-CV comparison
calibrate models, choose thresholds, stack models, or load the held-out test set
```

Those changes follow only after the mixed-data sampler contracts have passed on the
project's actual operating environment.
