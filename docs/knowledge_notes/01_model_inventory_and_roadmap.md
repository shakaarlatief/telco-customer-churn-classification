# Telco Customer Churn Classification: Model Inventory and Roadmap

## Purpose

This document is the strategic modelling roadmap for the Telco Customer Churn classification project.

The project has two linked goals:

```text
1. Build a professional, portfolio-ready churn-classification project.
2. Preserve reusable knowledge of classification modelling, evaluation, and implementation.
```

The project deliberately studies more than one strong model. A model family remains valuable even when it is not the strongest observed candidate, because it teaches a distinct modelling principle, preprocessing requirement, loss function, geometry, or error tradeoff.

This roadmap is strategic rather than tactical. For immediate work, use:

```text
docs/knowledge_notes/current_project_status_and_next_actions.md
```

For a new-chat continuation snapshot, use the newest file in:

```text
docs/knowledge_notes/context_history/
```

## Dataset and evaluation boundary

```text
Clean modelling dataset: 7043 observations
Training set:            5634 observations
Held-out test set:       1409 observations
Positive class:          Churn_binary = 1
Training churn rate:     approximately 26.54%
```

The test set remains untouched until one final end-to-end modelling procedure is frozen.

All current model-family results are training-only, development-stage estimates. They support understanding, tuning, and candidate shortlisting. They do not establish final test performance or prove that small metric differences represent genuine population-level superiority.

## Completed project foundations

The following stages are complete:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_statistical_evaluation_methodology
04_preprocessing_and_simple_baselines
```

These stages establish:

```text
raw-schema inspection and deterministic TotalCharges correction
clean train-test split discipline
training-set-only exploratory analysis
positive-class definition
positive-first confusion-matrix convention
cross-validation and pooled out-of-fold diagnostics
threshold and calibration concepts
baseline classifiers
the strict single-final-test-model policy
```

## Completed model-family inventory

### 05. Linear classification and logistic regression

Core ideas:

```text
linear scores and decision boundaries
least-squares classification
logistic probabilities and log odds
binary cross-entropy
L1 and L2 regularization
class weighting
coefficient interpretation
probability thresholds and calibration
```

Project role:

```text
first learned-model family
strong, interpretable benchmark
reference point for whether nonlinear models add meaningful value
```

Observed development-stage position:

```text
L1 and L2 logistic regression are essentially tied.
The representative L2 logistic model has PR-AUC about 0.658 and ROC-AUC about 0.846.
```

### 06. k-nearest neighbours

Core ideas:

```text
local non-parametric prediction
Minkowski, Manhattan, and Euclidean distance
feature scaling
the bias-variance role of k
uniform versus distance weighting
probability estimates from neighbour proportions
```

Project role:

```text
contrast global linear scores with local similarity-based prediction
```

Observed development-stage position:

```text
The selected kNN model uses k = 101, uniform weighting, and Manhattan distance.
Its PR-AUC is about 0.628 and its ROC-AUC is about 0.836.
```

### 07. Naive Bayes

Core ideas:

```text
Bayes classifier and Bayes risk
generative modelling
class priors and class-conditional likelihoods
conditional independence
Gaussian, Bernoulli, and hybrid likelihood blocks
additive smoothing
```

Project role:

```text
introduce a probabilistic generative perspective for mixed tabular data
```

Observed development-stage position:

```text
The selected HybridGaussianBernoulliNB has PR-AUC about 0.615 and ROC-AUC about 0.822.
The hybrid likelihood is more coherent than treating one-hot categorical indicators as Gaussian.
```

### 08. Decision trees

Core ideas:

```text
recursive partitioning
Gini impurity and entropy
greedy split selection
leaf churn proportions
tree probabilities and tied rankings
pre-pruning
cost-complexity pruning
feature importance
```

Project role:

```text
first nonlinear rule-based model
foundation for all later tree ensembles
```

Observed development-stage position:

```text
The selected pre-pruned tree has PR-AUC about 0.628 and ROC-AUC about 0.824.
The unrestricted tree overfits strongly, which demonstrates why tree regularization is necessary.
```

### 09. Bagging and random forests

Core ideas:

```text
bootstrap aggregation
variance reduction through averaging
tree correlation
feature subsampling
out-of-bag observations
ensemble probability scores
feature-importance diagnostics
```

Project role:

```text
show how an ensemble can improve the instability of a single tree
```

Observed development-stage position:

```text
Selected bagged trees:
    pooled OOF PR-AUC about 0.662
    pooled OOF ROC-AUC about 0.846

Selected random forest:
    pooled OOF PR-AUC about 0.660
    pooled OOF ROC-AUC about 0.847
```

The relevant conclusion is that both ensembles materially improve on the single tree. Their close differences do not establish that one is meaningfully superior.

### 10. Boosting

Core ideas:

```text
sequential additive modelling
AdaBoost reweighting
gradient boosting and pseudo-residuals
learning rate, depth, and number of estimators
histogram split construction
first-order and second-order boosting intuition
XGBoost, LightGBM, and CatBoost design differences
native categorical handling
```

Project role:

```text
evaluate the strongest observed tree-based model group
```

Observed development-stage position:

```text
CatBoost, GradientBoostingClassifier, and XGBoost form a very close leading group.
The representative XGBoost pooled OOF diagnostic has PR-AUC about 0.670 and ROC-AUC about 0.850.
```

The top boosting point estimates are too close to support a conclusion that one boosting library is definitively best.

### 11. Support vector machines

Core ideas:

```text
linear score functions
maximum-margin separation
support vectors
hard and soft margins
hinge and squared-hinge loss
C as inverse margin-violation penalty strength
class weighting
kernel trick
polynomial and RBF kernels
gamma and nonlinear complexity
decision-function scores versus calibrated probabilities
```

Project role:

```text
introduce margin-based linear classification and nonlinear kernel classification
```

Observed development-stage position:

```text
Selected LinearSVC:
    squared hinge
    C = 0.1
    balanced class weights
    mean fold PR-AUC about 0.6594
    mean fold ROC-AUC about 0.8453

Selected RBF SVC:
    C = 10
    gamma = 0.001
    balanced class weights
    mean fold PR-AUC about 0.6595
    mean fold ROC-AUC about 0.8424
```

The selected linear and RBF candidates are effectively tied within the tried grid. The RBF point estimate is only about 0.0001 higher in mean fold PR-AUC and does not justify a nonlinear-advantage claim. The linear SVM is retained as the representative SVM diagnostic model because it is faster, interpretable, and stronger on the pooled OOF diagnostic.

## Remaining model-family roadmap

### 12. Multilayer perceptrons and feed-forward neural networks

This is the immediate next model-family section.

Core topics:

```text
perceptrons and affine transformations
hidden layers and nonlinear activation functions
feed-forward computation
sigmoid output for binary classification
binary cross-entropy
backpropagation
gradient descent, mini-batches, and adaptive optimization
learning rate, batch size, epochs, and early stopping
weight decay, dropout, and capacity control
scaling and one-hot encoded tabular inputs
validation behaviour, calibration, and thresholds
```

The MLP section should compare shallow and moderately deep tabular neural-network candidates while maintaining training-only preprocessing and cross-validation discipline. The purpose is not to assume that neural networks will beat boosted trees. It is to understand whether learned nonlinear representations add useful predictive signal for this relatively small mixed tabular dataset.

## Cross-cutting modelling topics before final evaluation

The project has already introduced several cross-cutting issues, including preprocessing, imbalance, thresholds, and calibration. Before final test evaluation, the remaining serious candidates may require targeted, training-only work on the following topics where justified:

```text
feature engineering and transformations
feature selection or ablation
class weighting versus resampling
probability calibration
threshold policy
business-value or cost-sensitive evaluation
repeated cross-validation
nested cross-validation
paired comparisons and uncertainty analysis
```

These should not be applied indiscriminately to every model family. They should be introduced when they answer a specific modelling question and must remain inside the training-only development process.

## Final training-only selection roadmap

After the remaining model-family work, the project should transition from educational family sections to a dedicated finalist-selection stage.

The intended sequence is:

```text
1. Define a limited set of serious finalists.
2. Freeze comparable candidate procedures, including preprocessing and hyperparameters.
3. Compare candidates with training-only evidence.
4. Use stronger stability or uncertainty methods when they add decision value.
5. Decide whether calibration is needed for the intended decision use.
6. Define and freeze a threshold policy from retention economics, contact capacity,
   and false-positive versus false-negative costs.
7. Select one final end-to-end pipeline.
8. Fit the frozen pipeline on the complete training set.
9. Evaluate once on the untouched test set.
10. Report final metrics and uncertainty intervals where feasible.
```

## Model-selection language

Use:

```text
selected within the tried development grid
representative strong candidate
development-stage cross-validated estimate
small differences should be interpreted cautiously
final test evaluation is deferred
```

Avoid:

```text
definitively best
uniquely optimal
proven superior
final performance
```

## Deliberately deferred work

The following are not current modelling-section tasks:

```text
using the test set to compare candidates
using the test set to choose a threshold
using the test set to decide whether calibration is useful
claiming a final production model before the finalist-selection stage
treating close cross-validation point estimates as proof of superiority
```

## Documentation convention

The strategic inventory belongs in this roadmap. Immediate work belongs in the live status file. New-chat context belongs in the newest handoff file. Stable documentation architecture belongs in `00_documentation_workflow.md`.
