# Implemented Final-Comparison Methodology and Evidence Roles

## Purpose

This note reconciles the project’s earlier planning documents with the methodology that was actually implemented in code.

It distinguishes two different things that must not be conflated:

1. the intended robust final-comparison methodology;
2. the reduced-cost end-to-end verification profile that was executed to confirm that the complete pipeline works.

The reduced-cost profile was deliberately created so the full sequence could be exercised quickly, including candidate execution, artifact persistence, summarization, leading-set selection, finalization, refitting, serialization, and guarded final-evaluation preparation. It was not intended to replace the robust comparison protocol or to become the project’s definitive model-comparison evidence.

This note is therefore the current methodological reconciliation point for the final-comparison stage. Earlier notes remain useful for theory and design history, but references to planned future work should be interpreted through the implementation state described here.

## 1. Core methodological principle

The object compared in this project is a complete candidate procedure, not merely an estimator name.

A candidate procedure contains:

```text
feature-policy choice
preprocessing representation
feature-selection policy where compatible
imbalance-treatment policy
model family
hyperparameter search space
search algorithm and search budget
inner-validation design
random-state and resource policy
score-production rule
failure and warning policy
```

For a candidate indexed by j, the procedure can be represented conceptually as

\[
\mathcal{P}_j
=
\left(
F_j,
S_j,
I_j,
M_j,
\Lambda_j,
\mathcal{A}_j,
R_j
\right),
\]

where:

- \(F_j\) is the feature policy;
- \(S_j\) is the feature-selection policy;
- \(I_j\) is the imbalance policy;
- \(M_j\) is the model family;
- \(\Lambda_j\) is the hyperparameter domain;
- \(\mathcal{A}_j\) is the tuning and confirmation algorithm;
- \(R_j\) is the reproducibility and resource contract.

This framing matters because two procedures using the same estimator can still be materially different if they use different representations, imbalance treatments, feature selectors, calibration rules, or tuning policies.

## 2. Dataset boundary

The clean modelling dataset contains 7,043 observations.

```text
Development data: 5,634 rows
Held-out test data: 1,409 rows
Target: Churn_binary
Positive class: 1
```

All candidate comparison, tuning, threshold analysis, calibration decisions, complementarity analysis, ensemble design, and final refitting must use the development data only.

The held-out test set has not been evaluated. It remains reserved for one eventual evaluation of one frozen final procedure after the user explicitly decides that the project is ready to consume it.

## 3. Implemented candidate universe

The implemented final-comparison registry contains C01 through C26:

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
C21  Linear support vector machine
C22  RBF-kernel support vector machine
C23  Multilayer perceptron
C24  TabNet
C25  FT-Transformer
C26  TabM
```

The following candidates remain deferred:

```text
C27  TabPFN
C28  AutoGluon
```

TabPFN was deferred because of CPU practicality, model-weight, package, and project-practicality constraints. AutoGluon was deferred because its dependency resolution conflicted with the controlled numerical environment.

Deferral is a protocol decision, not a performance claim.

## 4. Intended robust comparison protocol

The intended robust methodology is encoded in:

```text
protocols/final_comparison_protocol_v2_base.json
```

Its declaration is frozen:

```text
protocol_id: telco_final_comparison_protocol_v2_base
protocol_version: v2-base-comparison-frozen
freeze_state: frozen
is_frozen: true
evidence_role: official_base_comparison_candidate_protocol
```

This is the methodology intended for the complete model-family comparison. It has not yet been run to completion.

### 4.1 Outer evaluation design

The robust protocol uses repeated stratified nested cross-validation on all 5,634 development rows:

```text
Outer folds: 5
Outer repeats: 3
Outer evaluation tasks per candidate: 15
Candidate count: 26
Total outer candidate tasks: 390
```

For each outer task:

1. one outer-validation fold is held aside;
2. Stage A and Stage B operate only on the outer-training partition;
3. the selected configuration is refitted on the complete outer-training partition;
4. the fitted procedure predicts the untouched outer-validation fold once.

The resulting outer scores estimate the performance of the full tuning-and-refit procedure, not the performance of one globally fixed hyperparameter vector.

### 4.2 Stage A exploration

Stage A performs persistent Optuna exploration using three-fold stratified inner cross-validation.

Its role is broad but bounded search over valid combinations of:

```text
feature policy
feature-selection policy
imbalance policy
model hyperparameters
```

The sampled parameter space is candidate-specific. Invalid combinations are prevented by the routing layer rather than discovered through failed full fits.

### 4.3 Stage B confirmation

Stage B re-evaluates the strongest Stage-A configurations using a separate five-fold stratified confirmation design inside the same outer-training partition.

Stage B is not another outer evaluation. It is an internal confirmation stage intended to reduce dependence on one exploratory inner split and one noisy Stage-A estimate.

Only after Stage B is complete is one configuration selected for the outer-training refit.

### 4.4 Candidate-specific search budgets

The robust protocol does not force every model family to receive the same number of Optuna trials. Equal trial counts would not imply equal or fair search effort because model families differ in dimensionality, runtime, and the number of consequential hyperparameters.

The frozen budget lanes are:

| Budget lane | Candidates | Stage-A trials | Stage-B top K | Search profile |
|---|---|---:|---:|---|
| Cheap | C01-C08 and C21 | 36 | 5 | full |
| Medium | C09-C18 except C19, plus C20, C22, C23 | 24 | 4 | full |
| Expensive | C24-C26 | 8 | 2 | full |
| Expensive runtime-limited | C19 CatBoost | 8 | 2 | catboost_v2 |

The budgets were frozen before robust comparison results were inspected. Changing them after seeing robust results would create a new protocol version.

### 4.5 CatBoost runtime policy

Runtime calibration identified C19 CatBoost as a practical bottleneck. That evidence was used only to design a bounded runtime policy. It was not used to rank or exclude CatBoost.

The frozen CatBoost profile is:

```text
iterations: 100 to 600, step 50
depth: 3 to 6
learning_rate: 0.003 to 0.2, logarithmic
l2_leaf_reg: 0.001 to 100, logarithmic
Stage-A trials: 8
Stage-B top K: 2
```

CatBoost remains part of the candidate universe. Its bounded profile is a resource-governance choice, not evidence that the family is weak.

## 5. Primary and secondary evidence

The robust protocol uses average precision as the primary ranking metric:

\[
\operatorname{AP}
=
\sum_n
\left(R_n-R_{n-1}\right)P_n,
\]

where \(P_n\) and \(R_n\) denote precision and recall at successive score thresholds.

Average precision is appropriate because churn is the minority class and the project is concerned with retrieving likely churners without allowing the majority class to dominate the evaluation.

Secondary evidence includes:

```text
ROC-AUC
log loss when probability scores are available
Brier score when probability scores are available
threshold-dependent metrics
runtime
warnings
failure or interruption state
selected-parameter stability
```

A candidate should not be selected solely because it has the largest displayed mean AP. The later interpretation should also consider uncertainty, practical equivalence, stability, reproducibility, operational complexity, probability quality when relevant, and whether an observed difference is large enough to matter.

## 6. Evidence-role taxonomy

The project now uses explicit evidence labels.

### 6.1 Implementation-admission evidence

Purpose:

```text
confirm that each candidate can be constructed, fitted, scored, persisted, and resumed
under bounded development-only conditions
```

It does not rank candidates.

Primary example:

```text
admission_smoke_c26_warning_clean_v2
```

### 6.2 Runtime evidence

Purpose:

```text
measure feasibility
identify bottlenecks
design bounded resource policies
```

It does not rank or eliminate candidates.

Primary example:

```text
search_budget_calibration_v1_warning_clean
```

### 6.3 Fast-completion development evidence

Purpose:

```text
exercise the complete end-to-end modelling pipeline with reduced computational settings
```

It can verify orchestration and produce provisional development outputs, but it is too small to support strong model-comparison claims.

Primary example:

```text
fast_completion_v1
```

### 6.4 Robust protocol-v2 evidence

Purpose:

```text
provide the intended repeated nested-CV comparison across the full C01-C26 registry
```

This evidence does not yet exist because the frozen robust protocol has not been completed.

### 6.5 Held-out test evidence

Purpose:

```text
estimate the final frozen procedure's performance on previously untouched observations
```

This evidence does not yet exist because the test set remains untouched.

## 7. Reduced-cost end-to-end verification profile

The executed fast-completion protocol is encoded in:

```text
protocols/final_comparison_fast_completion_v1.json
```

Its role is explicitly:

```text
fast_completion_pipeline_evidence
```

The reduced settings were:

| Component | Intended robust protocol | Fast-completion verification profile |
|---|---:|---:|
| Outer folds | 5 | 2 |
| Outer repeats | 3 | 1 |
| Stage-A inner folds | 3 | 2 |
| Stage-B confirmation folds | 5 | 2 |
| Stage-A trials, cheap lane | 36 | 2 |
| Stage-A trials, medium lane | 24 | 2 |
| Stage-A trials, expensive lane | 8 | 2 |
| Stage-B top K, cheap lane | 5 | 1 |
| Stage-B top K, medium lane | 4 | 1 |
| Stage-B top K, expensive lane | 2 | 1 |
| Total outer candidate tasks | 390 | 52 |

The reduced profile intentionally kept the same candidate universe and the same general runner architecture while shrinking the number of splits, repeats, trials, and confirmations.

Its purpose was to answer engineering questions such as:

```text
Can every candidate complete through the same runner?
Are artifacts written and checksum-verified correctly?
Can results be summarized automatically?
Can a leading set be derived programmatically?
Can candidate configurations be reconstructed completely?
Can the ensemble and threshold workflow run end to end?
Can the selected procedure be refitted, serialized, reloaded, and audited?
Can a guarded one-time test evaluator be prepared without touching the test set?
```

It was not designed to answer the substantive question:

```text
Which candidate procedure has the strongest and most stable generalization performance?
```

Therefore, the fast-completion point estimates, rankings, selected hyperparameters, and ensemble outcome must not be described as the definitive final comparison.

## 8. Statistical interpretation of the fast-completion run

The fast profile has only two outer folds and one repeat. This has several consequences:

1. fold-level variability is poorly estimated;
2. one particular split can influence the ranking strongly;
3. two Optuna trials provide only minimal coverage of most search spaces;
4. Stage-B top 1 does not provide a meaningful confirmation competition;
5. parameter-stability summaries are extremely limited;
6. corrected repeated-CV comparisons and strong paired uncertainty analysis are not supported;
7. the selected development score is vulnerable to selection optimism.

The run is still valuable because it validates the complete implementation. Its value is primarily procedural and operational rather than inferential.

## 9. Downstream methodology after the robust comparison

After the robust C01-C26 comparison is completed, the intended sequence is:

### 9.1 Summarize outer evidence

For every candidate, retain and summarize:

```text
outer-fold AP and secondary metrics
runtime and warnings
failure and interruption state
selected configuration per outer task
parameter-selection stability
OOF scores with stable row identifiers
```

### 9.2 Select a leading set

Select a defensible leading set using development-only evidence. The decision should consider:

```text
mean AP
paired outer differences
uncertainty
practical equivalence
stability
runtime
score semantics
implementation complexity
```

The leading set should not be chosen from one point estimate alone.

### 9.3 Obtain concrete final configurations

Nested CV evaluates tuning procedures. It does not directly produce one universal hyperparameter vector.

After family-level comparison, the leading procedures should be tuned on all development data using a frozen post-comparison tuning design. This stage produces concrete executable configurations for final selection and refitting.

### 9.4 Study calibration and threshold choice

Calibration and threshold selection are separate decisions.

Calibration should be evaluated only for probability-producing leading procedures and only with leakage-safe development predictions.

Threshold choice should use development-only OOF or cross-fitted scores. The selected threshold must be frozen before the held-out test is evaluated.

### 9.5 Study complementarity and ensembles

Soft voting, blending, or stacking may be considered only when leakage-safe OOF predictions exist for the constituent procedures.

Any ensemble is itself a candidate procedure. Its members, order, weights, meta-model if applicable, score semantics, and threshold rule must be frozen and evaluated using development-only evidence.

### 9.6 Choose and refit one final procedure

After all development decisions are complete:

1. choose one individual procedure or one justified ensemble;
2. freeze its complete configuration;
3. refit it on all 5,634 development rows;
4. serialize the fitted object and its feature schema;
5. verify checksums and prediction round trips;
6. evaluate it once on the held-out test only when the user explicitly authorizes that final step.

## 10. Current engineering verification state

The reduced-cost route completed successfully and exercised the complete pipeline.

It produced a fast-route frozen ensemble, refitted it on all development rows, serialized it, reloaded it, and prepared a guarded one-time test evaluator.

That state demonstrates that the pipeline is operational. It does not convert the reduced-cost comparison into robust scientific evidence.

Because the held-out test remains untouched, the project can still execute the robust protocol later and use that stronger development evidence to choose the procedure that will eventually receive the single held-out evaluation.

The phrase “frozen final procedure” in fast-route artifacts should therefore be read as:

```text
frozen within the completed fast-completion verification route
```

It should not be read as:

```text
irreversibly established as the final scientific winner before the robust comparison
```

## 11. Reporting rules

The report should use the following language consistently.

Appropriate:

```text
implemented candidate procedure
implementation-admission evidence
runtime evidence
reduced-cost end-to-end verification
fast-completion development evidence
intended robust protocol
frozen protocol-v2 design
provisional fast-route selection
held-out test not yet evaluated
```

Avoid:

```text
final winner
best model
statistically superior
unbiased final performance
robust benchmark result
```

unless the supporting evidence actually exists.

The fast-completion run may be reported as a successful end-to-end systems and methodology verification. Its numerical ranking should be labelled provisional and should not replace the future robust protocol-v2 analysis.

## 12. Future execution rule

When the project returns to the full comparison:

1. run the frozen robust protocol-v2 design unchanged;
2. do not use fast-completion rankings to expand or shrink search spaces selectively;
3. treat any protocol change as a new explicitly versioned protocol;
4. keep the held-out test untouched throughout robust comparison and final development selection;
5. only after one complete robust final procedure is frozen should the one-time held-out evaluator be considered.

The robust run is not required for updating the current methodology documentation. It is required before making strong comparative claims about the candidate procedures.
