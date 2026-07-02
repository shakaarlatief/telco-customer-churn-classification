# Final Comparison Implementation Phase 5: Candidate-Specific Feature-Policy Routing

## Purpose

Phase 4 defined the deterministic, target-free feature policies used by the final
comparison. Phase 5 makes those policies part of complete, fold-safe candidate
procedures. This is not a notebook-side preprocessing switch. A policy is selected
only inside inner cross-validation and becomes a persisted hyperparameter of the
selected outer-fold procedure.

The implementation adds three layers:

```text
1. a policy-aware pipeline adapter;
2. an explicit candidate-to-policy compatibility matrix; and
3. persistent-study fingerprints that bind feature-policy routing and schema.
```

The held-out test set is not loaded or referenced.

## Policy matrix

```text
F0_RAW
    Every core candidate.

F1_DOMAIN_ENRICHED
    Every core candidate.

F2_LINEAR_EXPANDED
    Ridge classifier and regularized logistic regression only.
```

F2 is deliberately restricted. Its pruned basis retains the F1 structural terms,
MonthlyCharges squared, tenure × MonthlyCharges, and selected tenure- or
MonthlyCharges-by-category interactions. It retains raw TotalCharges as a main feature
but excludes every higher-order TotalCharges term because cumulative charges are almost
determined by tenure × MonthlyCharges in the development data. Ridge and logistic
regression require explicit basis terms to represent such effects. For logistic
regression, L1 and elastic-net regularization may remove or jointly shrink related
terms; Ridge retains all terms but can shrink correlated coefficients.

The other families are not denied nonlinear information. Trees, forests, boosting,
CatBoost, LightGBM, RBF SVMs, and MLPs have their own mechanisms for nonlinear or
interaction effects. Their default comparison receives raw and domain-enriched inputs,
not an indiscriminate high-dimensional cross-product matrix. kNN and hybrid Naive
Bayes are also restricted to F0/F1 because F2 would alter distance geometry or amplify
the conditional-independence burden without a sufficiently strong modelling rationale.

## Fold-safe pipeline order

Every routed pipeline now has the same high-level shape:

```text
raw development rows
    -> FeaturePolicyTransformer(policy_id)
    -> representation-specific preprocessing
    -> classifier
```

For one-hot representations, the second step is a policy-specific
`ColumnTransformer` that learns numerical imputers, scalers where needed, categorical
imputers, and category vocabularies from the fitted training partition. For native
categorical LightGBM and CatBoost routes, the second step preserves a pandas DataFrame
and applies fold-local median/mode imputation while retaining the full policy-dependent
categorical column list.

The CatBoost adapter is intentionally separate from the historical CatBoost wrapper.
F1 and F2 add the categorical `Contract × PaymentMethod` feature, so CatBoost must be
informed about the policy-specific categorical schema at fit time. The wrapper stores
that list as an immutable constructor parameter and supplies it during fitting, which
remains compatible with scikit-learn cloning.

## Inner-HPO selection

The candidate registry appends a categorical `feature_policy` hyperparameter to every
candidate's Optuna search space. Its candidate-compatible values are predeclared:

```text
Ridge / logistic regression:
    F0, F1, F2

all other implemented core candidates:
    F0, F1
```

Consequently, an outer-validation score evaluates a complete tuned procedure that may
select a different representation within each outer training partition. It does not
use outer validation or test information to choose a feature policy.

## Routing smoke test

The routing smoke test uses only a small stratified subset of `train.csv`. It fits and
scores every declared core candidate-policy route:

```text
17 core families
36 compatible feature-policy routes
```

For each route it checks:

```text
fresh pipeline construction and cloning
fold-local feature-policy placement
exact deterministic output schema
fit and binary prediction output
probability-versus-margin score declaration
single-thread model configuration where applicable
rejection of a deliberately incompatible F2 route for kNN
```

The persistent Optuna interruption-and-resume smoke test remains a separate test. It
continues to validate durable SQLite studies, atomic sidecars, retries, and controlled
outer-process parallelism.

## Non-actions

Phase 5 does not yet:

```text
run the master repeated nested-CV comparison
add feature-selection policy S1, S2, or S3
add resampling or synthetic oversampling
select a final model
calibrate scores or select thresholds
fit a stack or blend
load the held-out test set
```
