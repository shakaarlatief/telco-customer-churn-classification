# Final Comparison Protocol v1: Comprehensive, Resumable, Training-Only Model Selection

## Status and purpose

This document defines the planned experiment system for selecting one final binary churn-classification pipeline from the Telco development data.

It is designed to satisfy two goals simultaneously:

1. produce a rigorous, leakage-safe, training-only basis for final model selection; and
2. create reusable infrastructure and documentation for future classification projects.

The held-out test set remains untouched until all selections are frozen. It must not be loaded by the experiment runner except in the separate final test-evaluation workflow.

This protocol replaces any informal interpretation of historical notebook scores as final evidence. Earlier notebook results remain valuable for model understanding and for defining broad, defensible search spaces. They do not determine the final winner.

---

## 1. Core principles

### 1.1 The object being evaluated

The unit of comparison is a complete candidate procedure:

```text
candidate procedure =
    input representation
    fold-internal preprocessing
    deterministic feature-engineering policy
    optional fold-internal feature-selection policy
    imbalance-treatment policy
    model family
    hyperparameter search method and search space
    random-seed policy
    score-production rule
```

Examples:

```text
"CatBoost" is not a complete candidate.

A complete CatBoost procedure is:
    raw Telco feature table
    -> fold-internal missing-value treatment and categorical handling
    -> a specified class-weight / resampling policy
    -> a predefined CatBoost Optuna search
    -> average precision as inner validation objective
    -> raw predicted probability as the ranking score
```

```text
"RBF SVM" is not automatically equivalent to "calibrated RBF SVM".

The uncalibrated procedure returns a signed decision score.
A calibrated procedure adds an additional fitted calibration model and is therefore
a different end-to-end procedure.
```

### 1.2 Strict separation of purposes

```text
Historical model-family notebooks:
    education, diagnostics, broad initial hyperparameter regions

Master final-comparison runner:
    training-only selection evidence across predeclared candidate procedures

Calibration and threshold workflow:
    converts ranking candidates into calibrated probability and hard-decision rules

Stacking and blending workflow:
    evaluates learned combinations of already competitive base procedures

Final test workflow:
    evaluates exactly one frozen pipeline once
```

### 1.3 No silent adaptation

The following are frozen before the comprehensive master run begins:

```text
candidate registry
candidate-specific representations
search spaces
search budgets
outer and inner split-generation rules
random seeds
imbalance options
feature-engineering options
feature-selection options
primary metric
practical-equivalence margin
statistical evidence hierarchy
checkpoint and resume contract
```

Follow-up experiments remain allowed, but they must be recorded as new protocol versions rather than silently replacing the original evidence.

---

## 2. Primary selection design

## 2.1 Primary estimand

The primary question is:

> When each candidate family receives the documented tuning procedure using only the available outer-training data, which complete tuned procedure provides the strongest and most stable training-only evidence for churn ranking?

The primary estimand is therefore the expected performance of a **tuned candidate procedure**, not the performance of one fixed hyperparameter vector found in an earlier notebook.

## 2.2 Repeated nested cross-validation

The main comparison will use repeated nested stratified cross-validation.

```text
Outer loop:
    RepeatedStratifiedKFold
    5 folds
    10 repeats
    50 outer evaluations per candidate family

Inner loop:
    StratifiedKFold
    5 folds
    fixed seed derived from:
        protocol version
        candidate identifier
        outer repeat
        outer fold

Inside each outer task:
    1. Receive the outer-training data only.
    2. Run the candidate's frozen hyperparameter / policy search using inner CV.
    3. Select one candidate configuration using inner average precision.
    4. Refit that selected configuration on all outer-training observations.
    5. Predict the untouched outer-validation observations.
    6. Persist metrics, predictions, selected parameters, timing, warnings,
       and provenance atomically.
```

This design deliberately evaluates the entire tune-and-fit procedure. The outer-validation observations have not influenced preprocessing fitting, resampling, feature selection, hyperparameter tuning, calibration fitting, or threshold selection.

## 2.3 Why repeated outer CV is retained

One five-fold nested CV pass gives only five outer scores. That is insufficient for a rich stability analysis in a project intended as a reusable reference.

Ten repeated outer partitions provide:

```text
50 paired outer evaluation units per candidate procedure
50 selected-hyperparameter records per candidate procedure
50 runtime records per candidate procedure
repeated outer out-of-fold score sets for pooled and aggregate diagnostics
a basis for corrected resampling and correlated Bayesian analyses
```

Repeated outer partitions are not treated as independent customer datasets. Their dependence is explicitly handled in the statistical analysis.

## 2.4 Inner hyperparameter optimization design

The inner search uses a two-stage strategy.

```text
Stage A: broad exploration
    candidate-specific Optuna search
    3-fold stratified CV
    many trials from the frozen search budget

Stage B: confirmation of the strongest configurations
    retain the top K Stage-A configurations
    evaluate them with 5-fold stratified CV
    select the exact configuration using the Stage-B mean average precision
```

This approach retains wide exploration while reducing the chance that one configuration wins only because of a particularly favourable small inner split.

For every candidate, the same outer task stores:

```text
all Optuna trial parameters and values
Stage-A rankings
Stage-B confirmation results
selected configuration
selection metric
all fold-level inner scores for the selected configuration
```

The value of `K` and the trial budgets are frozen in the candidate registry. A recommended initial value is:

```text
K = 20
```

## 2.5 Final fitting route after selection

After the master comparison, calibration and threshold studies, and any stacking study are complete:

```text
1. Choose one final family or one final stack using training-only evidence.
2. Rerun only that winning procedure's frozen search on all 5,634 development rows.
3. Select one exact final configuration.
4. Fit all learned preprocessing, feature policy, imbalance policy, model parameters,
   calibration method if selected, and threshold rule using development data only.
5. Save a complete final-pipeline manifest.
6. Evaluate the frozen pipeline once on the held-out test set.
```

---

## 3. Primary metric, secondary metrics, and practical equivalence

## 3.1 Primary metric

The primary ranking metric is:

```text
Average Precision (AP)
implemented by:
    sklearn.metrics.average_precision_score
```

The report can refer to “precision-recall ranking performance” in prose and show precision-recall curves. However, tables and final selection logic should use the exact name **average precision** for the implemented metric.

Average precision is preferable to ordinary accuracy for this problem because churn is the minority class and the project cares about prioritising actual churners above non-churners.

## 3.2 Secondary ranking and probability metrics

Every outer task records:

```text
Average precision
ROC-AUC
log loss
Brier score
balanced accuracy at the default threshold
precision, recall, specificity, F1, and predicted-positive rate at the default threshold
fit time
prediction time
peak transformed feature count where applicable
warning / failure indicator
```

Important interpretation rule:

```text
Average precision and ROC-AUC:
    evaluate ranking quality

log loss and Brier score:
    evaluate probabilistic quality

threshold metrics:
    evaluate one chosen operating point
```

A candidate is not selected solely because it has good default-threshold F1 if it has weaker ranking evidence or poor probability behaviour.

## 3.3 Practical-equivalence margin

The predeclared primary-metric practical-equivalence region is:

```text
Average precision difference within +/- 0.005
```

Interpretation:

```text
difference greater than 0.005:
    potentially meaningful primary-metric advantage

difference between -0.005 and +0.005:
    practical tie unless secondary evidence gives a compelling reason otherwise
```

The margin is intentionally defined before final comparison results. It prevents a 0.001 or 0.002 point-estimate difference from deciding a final model by itself.

## 3.4 Deterministic tie-breaking order

For candidates inside the AP practical-tie region, decide in this order:

```text
1. Higher probability quality:
       lower log loss and Brier score, if probabilities are operationally relevant.

2. More stable outer performance:
       lower dispersion of outer AP,
       fewer convergence failures,
       more stable selected hyperparameters.

3. Better threshold trade-off:
       stronger precision-recall / intervention-volume profile after cross-fitted
       threshold analysis.

4. Simpler and more reproducible procedure:
       fewer moving parts,
       lower implementation risk,
       fewer external dependencies.

5. Interpretability:
       cleaner global and local explanation when predictive evidence is otherwise tied.

6. Runtime:
       reported transparently but not used to discard a materially stronger model.
```

---

## 4. Candidate registry

The project will intentionally compare a broad library. Candidates are grouped by modelling principle rather than by package popularity.

## 4.1 Reference baselines

These appear in descriptive tables and sanity checks. They are not expected to be final deployment candidates.

```text
B01  most-frequent dummy classifier
B02  prior-probability dummy classifier
B03  stratified random dummy classifier
B04  uniform random dummy classifier
B05  EDA-inspired rule classifier
```

## 4.2 Core learned candidates

### Linear, generative, local, and additive procedures

```text
C01  Ridge classifier
C02  Logistic regression:
         L1, L2, and elastic-net regularization

C03  Spline logistic regression:
         numeric B-spline bases plus categorical indicators and regularized logistic output

C04  Shrinkage linear discriminant analysis

C05  Regularized quadratic discriminant analysis

C06  k-nearest neighbours

C07  Hybrid Gaussian-Bernoulli Naive Bayes
```

### Tree and bagging procedures

```text
C08  Regularized decision tree
C09  Extra Trees
C10  Bagged decision trees
C11  Random forest
C12  Balanced random forest
```

### Boosting, imbalance-aware ensemble, and interpretable nonlinear procedures

```text
C13  AdaBoost
C14  RUSBoost or EasyEnsemble:
         one imbalance-aware boosting / ensemble procedure, chosen after package smoke checks

C15  GradientBoostingClassifier
C16  HistGradientBoostingClassifier
C17  XGBoost
C18  LightGBM
C19  CatBoost
C20  Explainable Boosting Machine
```

### Margin-based procedures

```text
C21  Linear SVM
C22  RBF-kernel SVM
```

### Neural and tabular foundation procedures

```text
C23  Dense multilayer perceptron

C24  TabNet:
         attention-based neural tabular model with native categorical embeddings

C25  FT-Transformer:
         feature-token transformer for tabular data

C26  TabM:
         parameter-efficient tabular deep-learning ensemble

C27  TabPFN:
         tabular foundation model
```

### External AutoML challenge procedure

```text
C28  AutoML tabular ensemble:
         evaluated only as one fully nested end-to-end procedure,
         never as an unexamined shortcut or test-set optimiser
```

The AutoML procedure is intentionally separated from the core candidate families. It can be useful as a challenge benchmark, but its internal search, bagging, and stacking policies must be treated as part of the evaluated procedure.

## 4.3 Advanced-candidate admission rule

The core procedures C01 through C23 are ordinary local-library procedures and will be implemented directly in the project.

C24 through C28 require a package, GPU, or licence smoke check. They remain part of the documented candidate universe, but a procedure may only enter the master comparison after it passes:

```text
installation check
licence / model-weight check
reproducible fit-and-predict smoke test
fold-safe preprocessing smoke test
checkpoint and resume smoke test
CPU or GPU resource scheduling smoke test
```

A failure is not silently ignored. It is persisted as an excluded candidate with a technical reason.

## 4.4 Why the library is broad but not arbitrary

The project will not include every model ever proposed. A new family is added when it is either:

```text
a materially different modelling principle
a modern and credible tabular-learning approach
a practically relevant imbalance-aware method
a highly interpretable nonlinear alternative
a useful combination / AutoML benchmark
```

The project will not add models that are clearly dominated in conceptual role, unsupported on the operating environment, or unsuitable for this sample size merely to make the list longer.

Examples that are not primary candidates:

```text
unregularized decision tree:
    retained as an overfitting diagnostic only

polynomial and sigmoid SVM:
    retained as documented screened kernels unless a smoke study shows a compelling reason

Gaussian process classifier:
    not a primary candidate because exact training scales poorly for this sample size

generic deep architectures with no tabular-specific rationale:
    not added merely because they are neural networks
```

---

## 5. Preprocessing and representation policy

## 5.1 Raw feature table

The raw modelling table consists of:

```text
numeric:
    tenure
    MonthlyCharges
    TotalCharges

binary and nominal categorical features:
    the documented Telco service, contract, and payment variables
```

Customer identifier columns remain excluded.

## 5.2 Representation families

```text
Scaled one-hot:
    ridge, logistic regression, LDA/QDA where appropriate, kNN, SVM

Dense scaled one-hot:
    MLP and dense scikit-learn procedures

Dense unscaled one-hot:
    AdaBoost, GradientBoostingClassifier, HistGradientBoostingClassifier,
    XGBoost where one-hot representation is selected

Unscaled sparse / one-hot:
    decision tree, Extra Trees, bagging, random forest

Native categorical DataFrame:
    LightGBM, CatBoost, Explainable Boosting Machine where compatible

Raw category-aware neural representation:
    TabNet, FT-Transformer, TabM, and TabPFN where their package contracts support it
```

All preprocessing remains inside the fitted pipeline or inner task. No imputation, scaling, encoding, feature selection, resampling, or learned target transformation may be fitted once on all development rows before a validation split.

## 5.3 Deterministic feature-engineering variants

The master registry contains three deterministic feature policies:

```text
F0: raw cleaned feature set

F1: domain-enriched feature set

F2: systematic regularized-linear expansion
```

F1 contains only predeclared transformations that are available at prediction time and do not use the target. It includes service aggregates, a safe charge-to-tenure summary with zero-tenure handling, selected nonlinear tenure summaries, selected charge-by-contract and charge-by-service interactions, and a `Contract × PaymentMethod` categorical interaction.

F2 is not a universal feature expansion. It is available only to Ridge classifier and regularized logistic regression. It contains the F1 structural features, numeric squares, pairwise numeric products, and numeric-by-nonreference-category interactions. It intentionally excludes a blanket categorical-by-categorical expansion and does not duplicate F1's curated numeric interactions. The purpose is to offer regularized linear procedures a controlled basis for nonlinear and interaction effects that tree, boosting, kernel, and neural procedures can learn through their own model structure.

The exact output columns, categorical reference-level convention, and zero-tenure rule are declared in the reusable feature-policy contract. A policy is selected within the relevant inner CV search only from candidate-compatible predeclared options:

```text
Ridge classifier and logistic regression:
    F0, F1, F2

all other implemented core candidates:
    F0, F1
```

Neither F1 nor F2 is a free-form manual feature-creation playground. The exact output columns must be frozen before running the master comparison.

## 5.4 Feature-selection policies

Feature selection is tested only where it has a coherent role.

```text
S0: no selection

S1: variance filtering followed by mutual-information SelectKBest

S2: L1-logistic SelectFromModel

S3: stability-selection study:
        separate training-only interpretability / reduction analysis,
        not automatically part of every model family
```

For tree ensembles, boosted trees, CatBoost, LightGBM, TabPFN, and similar representation-learning procedures, feature selection is not assumed beneficial. Their no-selection version remains the primary candidate. Selection variants may be assessed as ablations rather than assumed to be universal improvements.

## 5.5 Spline logistic regression

Spline logistic regression is included because it gives a useful middle ground:

```text
linear logistic regression:
    interpretable but only linear in numeric predictors

tree ensembles:
    flexible but less directly interpretable

spline logistic regression:
    smooth nonlinear numeric effects plus interpretable categorical log-odds effects
```

Numeric spline knot locations are estimated inside the relevant training partition. The number of knots, knot placement strategy, and logistic regularization strength are tuned only inside inner CV.

---

## 6. Imbalance-treatment policy

## 6.1 General rule

Class imbalance treatment is part of a candidate procedure. It is not an afterthought applied after cross-validation.

Every fitted imbalance method must be applied only to the training portion of the relevant split.

## 6.2 Initial policy options

The master candidate search may choose from the following compatible policies:

```text
I0: no explicit imbalance treatment

I1: class weighting:
        class_weight = None or "balanced"
        or an explicitly tuned positive-class weight where the estimator supports it

I2: random oversampling inside the training pipeline

I3: random undersampling inside the training pipeline

I4: SMOTENC for mixed numerical and categorical raw data

I5: algorithm-specific balanced ensembles:
        BalancedRandomForest, EasyEnsemble, or RUSBoost
```

## 6.3 Important SMOTE rule

Ordinary SMOTE must not be applied after one-hot encoding nominal categorical features. Interpolating one-hot indicator columns can create invalid fractional category values.

For mixed Telco data, synthetic oversampling must use a mixed-feature method such as SMOTENC, applied before the final one-hot representation or within a purpose-built category-aware pipeline.

## 6.4 Candidate compatibility

Not every imbalance policy is compatible with every model:

```text
tree and boosting models:
    class-weight or native scale-positive-weight controls where available

kNN and some neural procedures:
    resampling or explicit weighted loss where supported

native categorical models:
    class weights first; SMOTENC only if the pipeline safely preserves their required input contract

balanced random forest / EasyEnsemble / RUSBoost:
    imbalance handling is intrinsic to the estimator
```

The candidate registry must record incompatible combinations explicitly rather than treating an unavailable policy as a failed trial.

---

## 7. Hyperparameter optimization policy

## 7.1 Search methods

```text
Finite, small, categorical search spaces:
    exhaustive grid search or Optuna GridSampler

Mixed or conditional spaces:
    Optuna multivariate TPE

Iterative boosting and neural procedures:
    Optuna TPE plus appropriate pruning callbacks where valid

Optional secondary HPO sensitivity analysis:
    CMA-ES or Gaussian-process sampler for continuous low-dimensional spaces
```

The main HPO method remains Optuna because it supports persistent studies, resumption, trial-level metadata, conditional search spaces, pruning, and programmatic inspection.

## 7.2 Search budgets

Budgets are trial-count based, not wall-clock based.

Initial planned budgets:

```text
Tier A: 250 trials per outer task
    ridge
    logistic regression
    spline logistic regression
    LDA/QDA
    kNN
    hybrid Naive Bayes
    decision tree

Tier B: 400 trials per outer task
    Extra Trees
    bagging
    random forest
    balanced random forest
    AdaBoost
    EasyEnsemble / RUSBoost
    GradientBoostingClassifier
    HistGradientBoostingClassifier
    Explainable Boosting Machine
    linear SVM
    RBF SVM

Tier C: 600 trials per outer task
    XGBoost
    LightGBM
    CatBoost
    MLP
    TabNet
    FT-Transformer
    TabM

Tier D: model-specific evaluation plan
    TabPFN and AutoML procedures
```

These budgets may be adjusted only after a documented pilot establishes that a particular search domain has substantially more or fewer effective degrees of freedom than expected. Any change creates a new protocol revision.

## 7.3 Search-space principles

Every search space must:

```text
cover low, moderate, and high-complexity regimes
use logarithmic distributions for regularization and learning-rate parameters
encode conditional parameters explicitly
include class weighting / imbalance options where compatible
include representation and feature-policy choices only when coherent
set hard validity constraints before a trial begins
record package version and estimator configuration
```

## 7.4 Convergence and failed-trial policy

```text
warnings:
    captured and persisted with trial metadata

numerical convergence warning:
    does not automatically invalidate a trial;
    the candidate's policy states whether it is tolerated, penalised, or retried

invalid parameter combination:
    marked PRUNED or FAIL with a structured reason

unexpected exception:
    captured with traceback;
    task continues according to retry policy

repeated failure of one family:
    reported as a family-level technical result, not silently dropped
```

---

## 8. Statistical evidence hierarchy

The project intentionally uses multiple complementary analyses. No single test is treated as a universal answer.

## 8.1 Descriptive evidence for all candidates

For every candidate:

```text
mean, median, standard deviation, IQR, and quantiles of outer AP
mean, median, standard deviation, IQR, and quantiles of outer ROC-AUC
outer score distributions and paired difference plots
hyperparameter-selection frequencies
runtime and failure summaries
cross-fitted pooled prediction diagnostics, clearly labelled
```

## 8.2 Primary frequentist sensitivity analysis: corrected repeated-CV t-test

Yes, the corrected repeated-CV test associated with Nadeau and Bengio is included.

For paired score differences \(d_1,\ldots,d_{rk}\) from \(r\) repeats and \(k\) folds:

\[
\bar d = \frac{1}{rk}\sum_{i=1}^{rk}d_i,
\]

with sample variance \(s_d^2\), the corrected standard-error form is:

\[
\sqrt{
\left(
\frac{1}{rk}
+
\frac{n_{\mathrm{test}}}{n_{\mathrm{train}}}
\right)s_d^2
}.
\]

The resulting statistic is:

\[
t_{\mathrm{corr}}
=
\frac{\bar d}{
\sqrt{
\left(
\frac{1}{rk}
+
\frac{n_{\mathrm{test}}}{n_{\mathrm{train}}}
\right)s_d^2
}
}.
\]

For a five-fold split, the train-test ratio term is approximately:

\[
\frac{n_{\mathrm{test}}}{n_{\mathrm{train}}}
\approx
\frac{1}{4}.
\]

This correction reflects that repeated-CV scores are dependent because training sets overlap. It is a useful frequentist sensitivity analysis, but not an oracle. The result is reported together with effect sizes, uncertainty intervals, and practical equivalence.

## 8.3 Focused 5x2 CV tests

For a small number of predeclared leading candidate pairs, run:

```text
Dietterich-style 5x2 CV t-test
Alpaydin 5x2 CV F-test
```

These are focused robustness checks for pairwise comparisons. They should not be inflated into a full pairwise matrix across every candidate in the library.

## 8.4 Bayesian correlated comparison with ROPE

For leading candidate pairs, use a Bayesian correlated comparison based on repeated-CV differences and the AP ROPE:

```text
left region:
    candidate A is practically worse than B

rope:
    AP difference within +/- 0.005

right region:
    candidate A is practically better than B
```

Report posterior probabilities for the three regions rather than only a binary “significant / not significant” statement.

## 8.5 Prediction-level paired bootstrap

Use prediction-level paired bootstrap analyses for:

```text
average precision difference
ROC-AUC difference
Brier-score difference
log-loss difference
threshold-metric difference at a frozen training-only threshold
```

Repeated-CV predictions must not be concatenated and treated as independent customer rows. Use one of:

```text
one outer-CV pass:
    one OOF prediction per customer

repeated outer CV:
    aggregate repeated OOF scores per customer before row bootstrap

resampling-level approach:
    use fold/repeat differences with corrected or correlated methods
```

## 8.6 Metric-specific analyses

```text
Average precision:
    paired bootstrap difference intervals

ROC-AUC:
    paired bootstrap and DeLong-style comparison

Hard predictions at a fixed threshold:
    McNemar test on paired validation predictions

Calibration:
    Brier score, log loss, calibration slope/intercept,
    reliability diagrams with bootstrap bands

Signal:
    permutation tests as a supplementary check that the observed validation score
    exceeds a no-association reference
```

## 8.7 Multiple comparisons

The full candidate library receives descriptive comparison, not an uncontrolled all-pairs p-value matrix.

Formal comparisons are restricted to:

```text
predeclared reference contrasts:
    each leading candidate versus regularized logistic regression

predeclared leading-candidate contrasts:
    the top candidates after the primary outer-score and ROPE rule

post-hoc exploratory contrasts:
    explicitly labelled exploratory
```

Frequentist p-values in a family of planned comparisons receive Holm correction. False-discovery-rate adjustment may be shown as a secondary exploratory view, but not used as the primary final-decision rule.

## 8.8 Tests explicitly not misused

```text
Friedman / Nemenyi:
    designed for comparison across multiple independent datasets;
    not used as formal inference across dependent folds from one dataset

ordinary paired t-test across CV folds:
    not used as primary inference because folds are dependent

Wilcoxon signed-rank across dependent repeated-CV folds:
    not used as a formal substitute for a correlated resampling test

McNemar or DeLong on the final test set:
    never used to choose among candidate models
```

The report will document these exclusions, since understanding why a test is inappropriate is part of the reusable reference value.

---

## 9. Calibration and threshold-selection stage

## 9.1 Ranking first, operating policy second

Family selection begins with raw ranking scores and AP as primary metric. Calibration and threshold optimisation occur after a leading set of ranking procedures has been identified.

This avoids accidentally allowing an arbitrary threshold choice to determine the best ranking model.

## 9.2 Calibration candidates

For the leading set of base procedures, evaluate:

```text
P0: no calibration
P1: sigmoid / Platt-style calibration
P2: isotonic calibration
```

Calibration must be cross-fitted:

```text
base model:
    fit on a training partition

calibrator:
    fit only on held-out calibration predictions

evaluation:
    use predictions from a model-calibrator sequence that did not fit on that
    evaluation observation
```

Evaluate:

```text
Brier score
log loss
calibration intercept
calibration slope
reliability diagrams
bootstrap uncertainty bands
```

## 9.3 Threshold selection

Without a defensible real retention-cost model, select and report several training-only policy operating points:

```text
T1: threshold maximising F1
T2: threshold maximising balanced accuracy
T3: threshold achieving predeclared minimum recall
T4: threshold achieving predeclared minimum precision
T5: threshold producing a predeclared intervention volume
```

The eventual final deployment threshold is chosen before the test set is touched. If a plausible cost model is later defined, expected utility becomes the preferred threshold-selection criterion.

---

## 10. Stacking, blending, and model combinations

Stacking is deferred until base-model results exist.

The project will first inspect:

```text
pairwise score correlation
paired outer-fold AP differences
complementarity of false positives and false negatives
calibration differences
whether candidates in the AP tie region make distinct mistakes
```

Only then build a planned ensemble stage.

## 10.1 Candidate combinations

```text
simple unweighted soft voting
validation-weighted soft voting
rank averaging
logistic-regression stacking
regularized linear stacking
possibly a shallow nonlinear combiner if justified
```

## 10.2 Leakage-safe stacking rule

The meta-model must train only on out-of-fold base-model predictions.

```text
incorrect:
    fit base models on all data
    use their in-sample predictions as meta-features
    fit stacker on those same predictions

correct:
    create cross-fitted base-model predictions
    fit the stacker on those out-of-fold predictions
    evaluate the completed stacking procedure in an outer loop
```

A stack is itself a candidate procedure. It must be evaluated in the same nested framework before it can compete for final selection.

---

## 11. Resumable experiment architecture

## 11.1 Execution principle

The master comparison is run by a command-line script, not inside a notebook.

Notebooks inspect completed artifacts and generate interpretation. They do not own a multi-process, resumable, checkpointed HPO run.

This reduces Windows/Jupyter worker-shutdown problems and makes the run restartable after:

```text
intentional stop
laptop sleep or shutdown
power loss
Python crash
package crash
one failed model family
interrupted Optuna study
```

## 11.2 Run directory

Each immutable protocol run receives a run identifier:

```text
artifacts/final_comparison/<run_id>/
    protocol.yaml
    protocol_hash.txt
    data_fingerprint.json
    environment_fingerprint.json
    git_revision.txt
    split_definitions/
    task_registry.sqlite
    task_manifests/
    optuna_studies/
    checkpoints/
    results/
    predictions/
    selected_parameters/
    logs/
    status.json
```

## 11.3 Task identity

One atomic master task is:

```text
candidate procedure
x outer repeat
x outer fold
```

The task key contains:

```text
protocol hash
candidate identifier
outer repeat
outer fold
split-index hash
```

A task is complete only after all required outputs are atomically persisted:

```text
outer metrics
selected configuration
prediction rows
warnings and errors
timing metadata
task manifest
artifact hashes
```

## 11.4 Resume contract

The resume command verifies:

```text
same protocol hash
same development-data fingerprint
same target definition
same split definitions
same package / environment policy
same candidate registry
same code revision policy
```

A mismatch blocks unsafe resume by default. A separate explicit `--fork-run` command creates a new run identifier rather than mixing incomparable results.

## 11.5 Optuna persistence

Each outer task receives its own persistent Optuna study database:

```text
optuna_studies/
    C17_xgboost/
        repeat_00_fold_00.sqlite
        repeat_00_fold_01.sqlite
        ...
```

This design avoids many unrelated workers writing simultaneously to the same SQLite study. A stopped task resumes its own incomplete study. A completed trial remains recorded; only missing trials are run after resume.

Sampler and pruner state are checkpointed alongside each study where required for exact reproducibility.

## 11.6 Stale task recovery

Each task writes a heartbeat with:

```text
process identifier
hostname
start timestamp
latest heartbeat
current stage
current trial number
```

On restart:

```text
completed:
    skip after integrity check

running with recent heartbeat:
    preserve unless user explicitly takes over

running with stale heartbeat:
    mark interrupted and resume safely

failed:
    retain error record;
    retry only under explicit retry policy
```

## 11.7 Graceful stopping

```text
Ctrl+C once:
    stop scheduling new outer tasks;
    request a clean stop after the current atomic unit where possible;
    write paused state.

Ctrl+C twice:
    terminate immediately;
    unfinished task resumes later from persistent trial state.
```

---

## 12. Parallel execution policy

## 12.1 One active parallelism layer

The runner uses one deliberate parallelism layer:

```text
top-level parallelism:
    independent outer tasks run in separate processes

inner Optuna trials:
    sequential within each outer-task worker

model-internal threads:
    limited to one thread inside an outer-task worker
```

This avoids nested worker multiplication such as:

```text
many outer processes
x many Optuna threads
x many estimator threads
x BLAS/OpenMP threads
```

## 12.2 Worker configuration

```text
CPU worker count:
    detected physical cores minus one, with a configurable cap

inside every worker:
    OMP_NUM_THREADS = 1
    MKL_NUM_THREADS = 1
    OPENBLAS_NUM_THREADS = 1
    BLIS_NUM_THREADS = 1
    threadpoolctl limit = 1

estimator-specific:
    RandomForest / ExtraTrees / Bagging: n_jobs = 1
    XGBoost: n_jobs = 1
    LightGBM: n_jobs = 1
    CatBoost: thread_count = 1
    EBM: n_jobs = 1
```

If the runner is configured with only one outer worker, model-internal parallelism may be enabled through a separate, explicit profile. The system never enables both layers at full capacity by accident.

## 12.3 GPU scheduling

GPU-capable procedures are scheduled through a separate resource pool:

```text
GPU task capacity:
    one active heavy training task per GPU by default

CPU worker pool:
    continues non-GPU candidate tasks independently

GPU task metadata:
    device identifier
    CUDA / driver information
    memory policy
```

This prevents several CatBoost, TabNet, TabM, FT-Transformer, or TabPFN tasks from competing for the same GPU memory.

## 12.4 Dynamic monitoring

The coordinator owns terminal output. Workers send structured events to the coordinator rather than printing directly.

Live terminal view includes:

```text
run identifier
overall tasks completed / total
candidate currently running
outer repeat and fold
inner HPO stage
trial count and best inner AP
elapsed time
task failure count
estimated remaining work
CPU / GPU worker occupancy
```

Additional commands:

```text
python scripts/final_comparison_status.py --run-id <run_id>
python scripts/final_comparison_status.py --run-id <run_id> --watch
python scripts/final_comparison_status.py --run-id <run_id> --failed
```

---

## 13. Windows and Jupyter reliability rules

The previous notebook warnings are treated as an infrastructure concern even though the completed results were usable.

The final comparison avoids this failure mode through:

```text
main-process runner:
    no full multi-process HPO inside a notebook cell

Windows entry point:
    if __name__ == "__main__" guard for all process-launching code

picklable task functions:
    top-level functions and serialisable task specifications only

one controlled process pool:
    no accidental nested joblib pool

no n_jobs = -1 inside outer worker processes

explicit cleanup:
    executor shutdown
    Optuna storage close where applicable
    temporary-file cleanup
    status flush in finally blocks

worker logs:
    separate file per task
```

The smoke test intentionally exercises an interrupted and resumed multi-process run on the actual operating system.

---

## 14. Source-code and script architecture

Reusable modules:

```text
src/telco_churn/
    candidates.py
        candidate registry, compatibility matrix, estimator factories

    feature_engineering.py
        deterministic Telco feature policies

    feature_selection.py
        fold-safe selectors and selection reporting

    imbalance.py
        fold-safe resampling / weighting pipeline builders

    hpo.py
        Optuna objectives, samplers, pruners, trial persistence

    experiment_protocol.py
        validated protocol dataclasses and hash generation

    experiment_splits.py
        deterministic repeated nested split definitions

    experiment_tasks.py
        serialisable task definitions and task execution

    experiment_store.py
        atomic artifacts, manifests, resume checks, provenance

    experiment_runner.py
        scheduler, worker management, pause / resume logic

    experiment_metrics.py
        metric computation, score extraction, calibration summaries

    experiment_statistics.py
        corrected CV tests, Bayesian ROPE comparisons, bootstrap,
        DeLong, McNemar, multiplicity handling

    experiment_progress.py
        Rich progress and structured status output
```

Scripts:

```text
scripts/smoke_test_final_comparison.py
scripts/run_final_comparison.py
scripts/final_comparison_status.py
scripts/export_final_comparison_tables.py
scripts/run_calibration_and_threshold_selection.py
scripts/run_stacking_comparison.py
scripts/run_final_test_evaluation.py
```

Notebook roles:

```text
13_final_candidate_comparison.py
14_calibration_and_threshold_selection.py
15_stacking_and_model_combination.py
16_final_test_evaluation.py
17_interpretability_and_ablation.py
```

Notebook numbering may be updated to match the existing repository convention.

---

## 15. Smoke-test contract

The smoke test uses a tiny fixed candidate subset and small trial budgets. It is not a performance benchmark.

It verifies:

```text
protocol validates
candidate registry validates
split definitions are reproducible
outer and inner partitions remain disjoint
preprocessing is fit only on the correct training partition
resampling happens only during fit on a training partition
feature engineering is fold-safe
one Optuna study is persisted
intentional interruption leaves an incomplete but valid state
resume skips completed tasks and completes unfinished work
task output hashes validate
protocol mismatch blocks unsafe resume
worker failure is captured without corrupting completed results
parallel worker settings avoid nested oversubscription
progress monitoring and status commands work
all required result tables and prediction files are created
```

The smoke test must intentionally:

```text
1. begin a small run;
2. stop after a preselected task;
3. restart with --resume;
4. verify that completed tasks were not recomputed;
5. verify that final artifacts are identical to an uninterrupted smoke run.
```

---

## 16. Planned implementation sequence

```text
Step 1:
    Add this protocol and revise the candidate audit with the expanded registry.

Step 2:
    Implement generic experiment protocol, artifact, task, and resume infrastructure.

Step 3:
    Implement a small candidate registry:
        logistic regression
        Extra Trees
        CatBoost
        linear SVM
        MLP

Step 4:
    Add and run the interruption / resume smoke test.

Step 5:
    Expand to the complete core library.

Step 6:
    Implement feature-engineering, feature-selection, and imbalance branches.

Step 7:
    Run the full repeated nested master comparison.

Step 8:
    Produce comparison tables, uncertainty analyses, and candidate interpretation.

Step 9:
    Run calibration and threshold-selection workflows for the leading set.

Step 10:
    Test stacking / blending only if base-model complementarity supports it.

Step 11:
    Freeze one final procedure, tune it on all development data, and save it.

Step 12:
    Evaluate that one frozen pipeline once on the held-out test set.

Step 13:
    Complete post-selection interpretation, ablation, and final report integration.
```

---

## 17. Research and implementation notes

### Corrected repeated-CV testing

The corrected resampling family of tests is included as a frequentist sensitivity analysis because repeated-CV score differences are dependent. It does not replace effect sizes, bootstrap intervals, or practical equivalence.

### Nested CV

Nested cross-validation is the primary model-family comparison design because it estimates the performance of the tuning procedure while avoiding the direct optimism caused by using the same data to tune and evaluate a searched configuration.

### Extra Trees

Extra Trees is included as a distinct randomized-tree ensemble. It uses randomized tree construction and averaging to improve predictive accuracy and control overfitting.

### Explainable Boosting Machine

EBM is included because it provides a modern, nonlinear, interaction-capable generalized additive model that can remain globally interpretable.

### TabM and TabPFN

TabM and TabPFN are included as advanced tabular deep-learning and tabular-foundation candidates. They remain conditional on reliable package, licence, hardware, and reproducibility smoke checks. TabPFN requires additional attention because modern releases use externally distributed model weights and GPU execution is strongly recommended for data larger than small CPU-scale problems.

### Parallelism

Parallelism is applied at the outer-task level, with inner model threads limited. This is both faster and more stable than allowing every library layer to launch all CPU workers simultaneously.
