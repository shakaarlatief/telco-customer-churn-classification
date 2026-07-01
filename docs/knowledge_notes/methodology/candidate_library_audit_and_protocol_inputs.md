# Candidate Library Audit and Final-Comparison Protocol Inputs

## Purpose

This document is the bridge between the completed model-family workflows and the later comprehensive candidate-comparison notebook. It is **not yet the frozen experimental protocol** and it does not contain final performance claims. Its purpose is to make the candidate universe explicit before any new final-comparison result is examined.

The project deliberately studies a broad classification library. Therefore, the final comparison should not be reconstructed informally from notebook memories or from one selected row per workflow. It should begin with a written inventory of:

- which fitted models are only baselines or educational diagnostics;
- which model families are eligible for the full training-only comparison;
- which settings belong inside a family's tuning search rather than becoming separate candidate models;
- which preprocessing, imbalance, score-output, calibration, and threshold choices are part of each procedure; and
- which unresolved choices must be frozen before the comparison is executed.

This note should be read together with:

```text
methodology/final_model_selection_designs_and_candidate_comparison.md
methodology/cross_validation_and_model_selection.md
methodology/statistical_uncertainty_and_tests.md
methodology/final_model_comparison_plan.md
```

---

## 1. The object that will be compared

The relevant unit is neither a single trained estimator nor an individual row in an old grid-search table. It is a **candidate procedure**.

```text
candidate procedure =
    data representation and preprocessing
    feature-engineering and feature-selection policy
    imbalance-treatment policy
    model family
    hyperparameter search space and search method
    random-seed and compute-budget policy
    score-production rule
    optional calibration policy
    optional threshold-selection policy
```

For example, the candidate is not simply “XGBoost.” A complete XGBoost procedure might mean:

```text
raw Telco features
-> fold-internal imputation and encoding
-> XGBClassifier
-> a predefined hyperparameter search
-> PR-oriented inner validation scoring
-> raw model score for ranking evaluation
```

Similarly, a linear SVM that exposes signed margins is not automatically the same procedure as a calibrated linear SVM that produces estimated probabilities. Calibration changes the pipeline and must be documented as such.

---

## 2. Four candidate roles

The project will report a wide model comparison, but not every earlier model should be treated as equally eligible for deployment. The following roles preserve both completeness and methodological coherence.

### 2.1 Reference baselines

Reference baselines are included in descriptive tables to show what learned models add. They are not normally expected to win final selection.

```text
majority / prior dummy classifier
random dummy classifiers
EDA-inspired rule classifier
```

A baseline would only become a final candidate under an exceptional and substantive reason, such as a strict interpretability constraint or a finding that complex models add no meaningful predictive value.

### 2.2 Full-selection candidates

These are defensible learned procedures that should enter the comprehensive training-only comparison with a predefined search policy.

```text
regularized linear classification
regularized logistic regression
k-nearest neighbours
hybrid Gaussian-Bernoulli Naive Bayes
single decision tree
bagged trees
random forest
AdaBoost
GradientBoostingClassifier
HistGradientBoostingClassifier
XGBoost
LightGBM
CatBoost
linear support vector machine
RBF-kernel support vector machine
multilayer perceptron
```

The final selection process may decide that several are practically tied or materially weaker. Inclusion means they receive an honest, documented chance under the frozen comparison protocol. It does not imply that every model must receive identical search dimensionality or that every model will be statistically tested against every other model.

### 2.3 Within-family search configurations

These are alternative settings **inside** one candidate procedure. They are not separate final candidate models unless the change creates a substantively different pipeline or operational objective.

Examples:

```text
logistic regression:
    penalty, C, class_weight

kNN:
    number of neighbours, distance metric, weighting rule

random forest:
    number of trees, depth, minimum leaf size, max_features

XGBoost:
    learning rate, tree complexity, regularization, subsampling,
    column sampling, number of estimators
```

A family is allowed to select one of these settings using its defined inner tuning process.

### 2.4 Diagnostic or conditionally eligible variants

Some variants are useful for learning, ablation, or implementation comparison but should not automatically become separate full-selection candidates.

Examples include:

```text
unrestricted decision tree:
    diagnostic for overfitting, not a likely final procedure

polynomial or sigmoid SVM kernels:
    screened alternatives; carry forward only if a defined protocol includes them

Gaussian-only or Bernoulli-only Naive Bayes on an incoherent representation:
    descriptive comparisons, whereas the hybrid likelihood is the main coherent candidate

post-hoc calibrated variants:
    separate candidates only if calibrated probabilities are central to the final use case

SMOTE or other resampling variants:
    include only after a leakage-safe resampling workflow and search policy are defined
```

---

## 3. Candidate inventory for the comprehensive comparison

The table below is the initial inventory. The “role” column states the intended comparison role, not an inferred performance ranking.

| ID | Procedure family | Role | Primary score output | Representation / preprocessing | Main tuning dimensions to freeze later |
|---|---|---|---|---|---|
| B01 | Most-frequent / prior dummy | Reference baseline | Constant class probability or label | No learned feature transformation | Dummy strategy only |
| B02 | Stratified / uniform dummy | Reference baseline | Randomized labels or scores | No learned feature transformation | Strategy, fixed seed |
| B03 | EDA-inspired rule classifier | Reference baseline | Rule-derived score | Raw interpretable feature conditions | Risk threshold only |
| C01 | Ridge classifier | Full-selection candidate | Signed decision score | Scaled one-hot representation | alpha, class weighting |
| C02 | Logistic regression | Full-selection candidate | Probability | Scaled one-hot representation | penalty, C, class weighting, possibly elastic-net mix |
| C03 | k-nearest neighbours | Full-selection candidate | Neighbour class proportion | Scaled one-hot representation | k, distance, weighting |
| C04 | Hybrid Gaussian-Bernoulli Naive Bayes | Full-selection candidate | Probability | Numeric Gaussian block plus one-hot Bernoulli block | alpha, variance smoothing |
| C05 | Regularized decision tree | Full-selection candidate | Leaf churn proportion | Unscaled one-hot representation | criterion, depth, leaf constraints, pruning |
| C06 | Bagged trees | Full-selection candidate | Mean tree probability | Unscaled one-hot representation | number of estimators, sample fraction, base-tree complexity |
| C07 | Random forest | Full-selection candidate | Mean tree probability | Unscaled one-hot representation | number of estimators, depth, leaf size, max_features |
| C08 | AdaBoost | Full-selection candidate | Ensemble score / probability | Dense one-hot representation | estimator complexity, learning rate, number of estimators |
| C09 | GradientBoostingClassifier | Full-selection candidate | Probability | Dense one-hot representation | tree depth, learning rate, estimators, subsampling |
| C10 | HistGradientBoostingClassifier | Full-selection candidate | Probability | Dense one-hot representation | leaf constraints, learning rate, iterations, L2 regularization |
| C11 | XGBoost | Full-selection candidate | Probability | One-hot representation | learning rate, estimators, depth, child-weight / regularization, sampling |
| C12 | LightGBM | Full-selection candidate | Probability | Native categorical representation | leaves, learning rate, estimators, regularization, sampling |
| C13 | CatBoost | Full-selection candidate | Probability | Native categorical representation | depth, learning rate, iterations, L2 regularization, bagging / randomization controls |
| C14 | Linear SVM | Full-selection candidate | Signed margin | Scaled one-hot representation | C, loss, class weighting |
| C15 | RBF SVM | Full-selection candidate | Signed margin | Scaled one-hot representation | C, gamma, class weighting |
| C16 | Multilayer perceptron | Full-selection candidate | Probability | Dense scaled one-hot representation | architecture, activation, alpha, learning rate, batch size, early-stopping settings |

The table is intentionally broad. It preserves the project’s goal of comparing every meaningful learned family while still separating true candidate procedures from intentionally weak dummy baselines and from individual hyperparameter rows.

---

## 4. Important procedure-specific notes

### 4.1 Score semantics and calibration

The candidate library does not have identical score semantics.

```text
Probability-producing models:
    logistic regression, kNN, hybrid Naive Bayes, trees, ensembles, boosting,
    and MLP

Margin-producing models:
    ridge classifier, LinearSVC, and SVC with probability=False
```

Ranking metrics can use either probabilities or continuous margin scores. However, a score should only be interpreted as a churn probability after appropriate calibration assessment. This matters most for:

```text
cost-based threshold selection
risk communication
expected-value calculations
comparing operational action thresholds across model families
```

The final protocol must decide whether family selection is based on uncalibrated ranking scores first, followed by calibration only for the selected final family, or whether calibrated pipelines are compared directly. The first route is usually simpler and avoids multiplying the candidate library too early. The second route is appropriate only if probability quality is central to the primary decision problem.

### 4.2 Class weights and resampling

Class weighting and resampling are not merely implementation details. They can change the fitted ranking function, score distribution, calibration, and threshold behaviour.

For the first comprehensive library comparison, use only imbalance treatments that already have a clear, leakage-safe implementation. A binary class-weight choice can normally be placed inside the relevant family’s hyperparameter space:

```text
class_weight in {None, "balanced"}
```

Resampling methods such as random undersampling, random oversampling, or SMOTE should not be inserted ad hoc into a final comparison. They require a dedicated workflow in which resampling occurs only inside each training partition. Once that workflow exists, each resampling policy should be recorded as a distinct candidate procedure or a clearly specified preprocessing hyperparameter.

### 4.3 Feature selection and engineered features

Feature engineering and feature selection must be treated consistently across families.

A reasonable first final-comparison design is:

```text
primary comparison:
    use one common, documented feature representation for every compatible family

family-specific representation:
    use native categorical processing only where it is a defining and deliberately
    documented feature of the library, such as LightGBM and CatBoost
```

A later ablation can compare:

```text
common full feature set
versus
one or more fold-internal feature-selection procedures
```

Feature selection cannot be fitted once on the whole training set before cross-validation. It must be part of the training pipeline inside each relevant split.

### 4.4 MLP internal early stopping

The MLP procedure has an additional internal validation split when `early_stopping=True`. That split is part of the estimator’s optimization policy, not the outer comparison metric.

The frozen protocol must record:

```text
early_stopping enabled or disabled
validation_fraction
n_iter_no_change
max_iter
random seed policy
```

Outer validation remains the source of comparison evidence. The internal validation score used by scikit-learn for stopping does not replace outer PR-oriented evaluation.

### 4.5 Native categorical boosting

LightGBM and CatBoost may use native categorical preprocessing while several other methods use one-hot encoded features. This is legitimate because representation is part of the modelling procedure.

The report should therefore say that the comparison is between complete procedures, not between different algorithms forced into an artificially identical representation. The preprocessing recipe must remain fold-internal and fully documented.

---

## 5. Search-space governance

The objective is not to make every candidate have the same number of combinations. It is to make the comparison defensible.

### 5.1 Required rule

Each full-selection candidate must have:

```text
one written search space
one written search method
one documented compute budget
one seed policy
one failure / convergence policy
one primary selection metric
```

The search space should be designed using model knowledge and the learning-stage notebooks, then frozen before final comparison results are inspected.

### 5.2 Comparable effort does not mean identical grids

Different model families have different numbers of consequential hyperparameters.

```text
logistic regression:
    a compact search may be sufficient

RBF SVM:
    C and gamma require a two-dimensional, usually logarithmic search

boosted trees:
    multiple coupled complexity, learning-rate, and regularization controls matter

MLP:
    architecture, regularization, and optimization controls interact
```

Fairness therefore requires transparent, reasonable effort rather than literal equality in the number of evaluated combinations. A final protocol can use search-budget tiers, for example:

```text
compact search tier:
    simple linear, kNN, Naive Bayes, and single-tree families

medium search tier:
    bagging, random forest, AdaBoost, and standard gradient boosting

high-complexity search tier:
    modern boosting libraries, RBF SVM, and MLP
```

The exact trial counts or grid sizes remain to be determined only after a computational pilot. They should not be increased selectively after seeing that a favourite procedure is almost winning.

---

## 6. Outputs that the eventual comparison workflow must retain

A final comparison workflow should generate more than one “best score” table. It should save enough evidence to support later interpretation and statistical analysis.

For every candidate procedure and relevant fold or repeat, retain:

```text
candidate identifier
outer / repeated-CV split identifier
training and validation sample counts
selected hyperparameters
fit time and prediction time
convergence warnings or failed fits
fold-level primary and secondary metrics
out-of-fold score vector with observation identifier
score type: probability or margin
random seed
```

At candidate level, retain:

```text
mean and dispersion of fold metrics
pooled OOF diagnostics, clearly labelled as pooled
hyperparameter-selection frequency or stability summary
runtime summary
calibration summary when applicable
threshold trade-off summaries
```

These records permit careful later analysis without rerunning a large comparison merely to answer a bookkeeping question.

---

## 7. Decision points that must be resolved before code

The documentation now contains the theoretical alternatives. The following are the actual unresolved project choices.

### 7.1 Primary comparison design

Choose exactly one primary selection route:

```text
Route A: flat repeated CV
    tune all procedures with repeated CV;
    select the best family and exact configuration directly.

Route B: per-family nested CV
    compare tuned family procedures using outer folds;
    select the family;
    tune that winning family once on all development data to select exact final settings.
```

Any additional design, such as bias-corrected flat CV or repeated nested CV, should be labelled as supplementary research or sensitivity analysis rather than quietly replacing the primary route.

### 7.2 Primary metric terminology and implementation

The project needs one explicit statement of whether the primary metric is:

```text
average precision as implemented by sklearn.metrics.average_precision_score
or
a numerical area under an interpolated precision-recall curve
```

These quantities are related but not identical in all implementations. The final protocol, notebooks, tables, and report should use one consistent label.

### 7.3 Practical-tie rule

Before observing final comparison results, define:

```text
a practical-equivalence margin for the primary metric
and
a deterministic tie-breaking order
```

A defensible tie-break order may prioritize:

```text
1. primary ranking metric outside the practical-tie region;
2. calibration or threshold behaviour if operational probabilities matter;
3. stability across splits and seeds;
4. simpler procedure;
5. lower runtime and implementation complexity;
6. interpretability.
```

The exact practical margin should be motivated by what difference would affect realistic retention targeting, not selected because it changes the preferred winner.

### 7.4 Calibration and threshold policy

Choose whether:

```text
family comparison:
    uses ranking scores only, with calibration and threshold choice after a family wins

or
candidate procedure:
    includes calibration and a threshold-selection rule inside each validation loop
```

The first is appropriate when the primary goal is discrimination/ranking. The second is needed when expected utility at a fixed operational policy is the primary criterion.

### 7.5 Statistical comparison scope

The final plan should predefine a limited evidence hierarchy:

```text
all candidates:
    descriptive tables, stability summaries, and graphical comparison

leading candidates:
    focused paired comparisons, practical-equivalence analysis, and suitable
    uncertainty summaries

selected final model:
    one test-set evaluation with bootstrap confidence intervals
```

Do not plan exhaustive pairwise p-value testing for every pair in a broad library.

---

## 8. Recommended immediate next action

The immediate next action is a **candidate-registry review and protocol decision session**, not final-comparison code.

The session should do the following in order:

```text
1. Confirm the candidate inventory and classify each entry as reference-only,
   full-selection, conditional, or diagnostic.

2. Decide whether the first comprehensive comparison uses flat repeated CV or
   per-family nested CV as its primary route.

3. Decide whether the primary ranking metric will be reported precisely as
   average precision or as another PR-curve summary.

4. Decide the treatment of class weighting, resampling, feature selection,
   calibration, and thresholds.

5. Specify a practical-tie rule and a limited statistical evidence hierarchy.

6. Inspect current reusable source modules, scripts, and the model notebooks to
   translate the registry into exact estimator factories and search spaces.

7. Freeze the written protocol before running a new full-comparison experiment.
```

Only after these decisions are documented should the project add reusable evaluation code, a smoke test, and a comprehensive candidate-comparison notebook.

---

## 9. Explicit non-actions at this stage

Do not yet:

```text
run the held-out test set
select a final model from old section-level point estimates
retune only the current apparent winner
add resampling or feature selection without a fold-safe pipeline
expand one procedure's search only because its score looks promising
turn close prior development scores into final superiority claims
```

The purpose of this audit is to ensure that the eventual comparison answers a clean question: which documented and fairly tuned candidate procedure should be selected using training-only evidence?
