# Final Comparison Implementation Phase 4: Deterministic Feature Policies

## Purpose

Phase 4 formalizes the feature-engineering representations that may later enter the
resumable nested-comparison system. A feature policy is part of a complete candidate
procedure. It must therefore be predeclared, reproducible, target-free, fitted only on
the relevant training partition, and stored as part of future task metadata.

This phase implements and smoke-tests the representation layer only. It does not yet
attach every policy to every candidate builder or begin a master model-comparison run.
That separation is deliberate: it validates the feature contract before the candidate
registry starts selecting among representations.

## Implemented policies

```text
F0_RAW
    The cleaned raw modelling columns.

F1_DOMAIN_ENRICHED
    Raw columns plus a compact domain-driven representation suitable for broad
    comparison across linear, tree, boosting, kernel, and neural candidates.

F2_LINEAR_EXPANDED
    Raw columns plus F1 structural features and a wider systematic expansion intended
    only for regularized linear procedures.
```

The policies are implemented by `FeaturePolicyTransformer`, a clone-safe
scikit-learn transformer. It returns a pandas DataFrame with a deterministic column
order. Its `fit` method estimates only training-partition medians and modes used to
construct derived values for incomplete future rows. It never uses the target.

## F1: compact domain-enriched representation

F1 contains three groups of additional variables.

### Service aggregates

```text
number of subscribed services
number of protection / support services
number of streaming services
```

The aggregate counts compress related binary service choices into lower-dimensional
summaries without discarding the original raw columns.

### Safe tenure and charge summaries

```text
tenure squared
log(1 + tenure)
historical total charges divided by max(tenure, 1)
zero-tenure indicator
```

The denominator convention makes the charge-to-tenure quantity defined for a new
customer with zero observed tenure. The zero-tenure indicator preserves the fact that
this convention was used rather than treating the resulting zero as ordinary history.

### Selected interactions

```text
MonthlyCharges × Month-to-month contract
MonthlyCharges × Two-year contract
MonthlyCharges × DSL service
MonthlyCharges × Fiber-optic service
MonthlyCharges × TechSupport = Yes
MonthlyCharges × OnlineSecurity = Yes
tenure × Month-to-month contract
Contract × PaymentMethod categorical cross
```

The chosen numerical interactions have a direct interpretation as deviations from a
reference slope. The omitted contract and internet-service categories act as reference
levels, avoiding an exact linear dependence in which all category-specific interactions
sum to the original numerical predictor.

## F2: systematic regularized-linear expansion

F2 is not an unrestricted all-pairs expansion. It is deliberately bounded:

```text
all original raw columns
F1 structural features, but not duplicate F1 interaction columns
all three numeric squared terms
all pairwise products of the three numeric features
numeric × nonreference-category products for every declared categorical feature
one Contract × PaymentMethod categorical cross
```

For a numeric predictor x and categorical variable G with levels g_0, ..., g_K, F2
creates x * 1(G = g_k) for k = 1, ..., K. The first declared level g_0 is the reference
and is omitted. This prevents the exact identity

```text
x = sum_k x * 1(G = g_k)
```

which would arise if interactions for all category levels were included alongside x.

F2 deliberately excludes blanket categorical-by-categorical interactions. Such terms
would produce a large number of sparse dummy cross-products, many without a strong
mechanistic interpretation. The selected Contract × PaymentMethod cross remains because
it is a compact, domain-plausible retention and billing interaction.

## Candidate compatibility to be applied next

```text
Ridge, logistic regression, and elastic-net logistic regression:
    F0, F1, and F2.
    F2 is controlled by regularization, with elastic net preferred when correlated
    groups of related main effects and interactions are present.

Linear SVM:
    F0 and F1 by default. F2 is admitted only after a dedicated compatibility smoke
    check because the broad expansion changes margin geometry substantially.

Trees, bagging, random forest, Extra Trees, boosting, LightGBM, CatBoost:
    F0 and F1. Do not apply F2 by default because these families learn split-based
    interactions internally.

RBF SVM, MLP, kNN, and Hybrid Naive Bayes:
    F0 and a narrow F1 assessment. F2 is not their default representation.
```

No candidate is allowed to consume a feature policy until the next integration phase
adds it explicitly to the corresponding pipeline builder and persistent task metadata.

## Smoke-test guarantees

`scripts/smoke_test_final_comparison_feature_policies.py` uses `train.csv` only and
verifies:

```text
stable F0/F1/F2 schemas
raw-column and row-index preservation
training-fold-only fit then validation transform
finite engineered numeric outputs
zero-tenure handling
service-count semantics
selected F1 interaction values
representative F2 quadratic and numeric-by-category interactions
```

The smoke test does not read the held-out test set, fit model candidates, tune
hyperparameters, select features, resample observations, or report predictive results.
