# Final Comparison Implementation Phase 8A: Imbalance-Pipeline Adapter

## Purpose

Phase 7 established the sampler primitives independently. This follow-up phase adds the
reusable pipeline adapter that places one declared imbalance treatment in the fitted path
of an otherwise complete feature-policy pipeline. Candidate-specific Optuna routing is
intentionally deferred to the next phase, so the exact pipeline topology is validated
before the registry expands its search universe.

The held-out test set is not loaded or referenced.

## Pipeline topologies

```text
I0_NONE
    raw rows
        -> feature policy
        -> representation preprocessor
        -> feature selector
        -> classifier

I1_CLASS_WEIGHT_BALANCED
    raw rows
        -> feature policy
        -> representation preprocessor
        -> feature selector
        -> balanced-sample-weight classifier adapter

I2_RANDOM_OVERSAMPLING / I3_RANDOM_UNDERSAMPLING
    raw rows
        -> feature policy
        -> representation preprocessor
        -> fit-time-only random sampler
        -> feature selector
        -> classifier

I4_SMOTENC
    raw rows
        -> F0 feature policy
        -> mixed-table sampler imputer
        -> fit-time-only SMOTENC
        -> representation preprocessor
        -> feature selector
        -> classifier
```

The two random samplers operate after representation preprocessing because they only copy
or discard complete rows. SMOTENC is necessarily earlier: it must see categorical columns
as categorical variables rather than as one-hot indicators.

## Weighted estimator adapter

`BalancedSampleWeightClassifier` computes

```text
weight_c = n / (2 n_c)
```

from the active fitting target and passes the resulting per-row values to the wrapped
classifier through `sample_weight`. This makes I1 a fold-local pipeline operation rather
than a global class-weight calculation. It also allows model families with different
constructor conventions to use the same policy definition. A later registry phase will
remove overlapping model-specific class-weight hyperparameters so one candidate cannot
receive two separate imbalance corrections.

The CatBoost wrapper now accepts `sample_weight`, allowing the same adapter to be used
without bypassing its policy-dependent categorical-column contract.

## Why SMOTENC remains raw-only

F1 and F2 contain derived service counts, products, interactions, and nonlinear terms.
Synthesizing those coordinates independently would not guarantee that they agree with a
coherent synthetic raw customer row. For example, a synthetic charge-by-contract
interaction need not equal the product implied by separately synthesized charge and
contract fields. Therefore I4 is rejected for F1 and F2 until a future design can
synthesize raw inputs first and recompute all deterministic derived features afterward.

## Smoke-test coverage

The integration smoke test uses only a stratified partition of the development data. It
fits I0 through I4 through a logistic-regression route, checks the declared pipeline
steps, verifies balanced total class mass for I1, checks valid probability scores on a
validation partition, rejects F1 plus SMOTENC before fitting, and separately verifies
CatBoost compatibility with I1 sample weighting.

## Non-actions

This phase does not yet modify the candidate registry, Optuna search space, persistent
study contract, or master comparison runner. The following phase will route the validated
imbalance policies through candidate-compatible HPO branches and persist them as part of
each selected outer-fold procedure.
