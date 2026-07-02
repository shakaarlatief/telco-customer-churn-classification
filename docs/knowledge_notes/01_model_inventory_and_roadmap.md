# Telco Customer Churn Classification: Model Inventory and Roadmap

## Purpose

This document is the strategic modelling roadmap for the Telco Customer Churn classification project.

The project has two linked goals:

```text
1. Build a professional, portfolio-ready churn-classification project.
2. Preserve reusable knowledge of classification modelling, evaluation, implementation,
   and statistically responsible final selection.
```

The project deliberately studies more than one strong model. A model family remains valuable even when it is not the strongest observed candidate, because it teaches a distinct modelling principle, preprocessing requirement, loss function, geometry, or error trade-off.

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
Development training set: 5634 observations
Held-out test set:       1409 observations
Positive class:          Churn_binary = 1
Development churn rate:  approximately 26.54%
```

The held-out test set remains untouched until one final end-to-end procedure is frozen. All current model-family and candidate-comparison evidence is development-stage evidence. It supports understanding, tuning, shortlisting, and procedure design. It does not establish final test performance or prove that small metric differences represent genuine population-level superiority.

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

### Workflow-number convention

The labels below refer to repository workflow and notebook identifiers, not LaTeX report section numbers.

```text
Workflow 11:
    support-vector-machine workflow identifier 11

Workflow 12:
    multilayer-perceptron workflow identifier 12

LaTeX report:
    these workflows appear later because the report contains data-audit, EDA, and
    methodology sections before the model-family sequence.
```

This distinction prevents the documentation roadmap from implying that workflow numbers and report section numbers must match.

### Workflow 05: Linear classification and logistic regression

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
The representative L2 logistic model has pooled OOF average precision about 0.658
and pooled OOF ROC-AUC about 0.846.
```

### Workflow 06: k-nearest neighbours

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
Its pooled OOF average precision is about 0.628 and its pooled OOF ROC-AUC is about 0.836.
```

### Workflow 07: Naive Bayes

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
The selected HybridGaussianBernoulliNB has pooled OOF average precision about 0.615
and pooled OOF ROC-AUC about 0.822.
The hybrid likelihood is more coherent than treating one-hot categorical indicators as Gaussian.
```

### Workflow 08: Decision trees

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
foundation for later tree ensembles
```

Observed development-stage position:

```text
The selected pre-pruned tree has pooled OOF average precision about 0.628
and pooled OOF ROC-AUC about 0.824.
The unrestricted tree overfits strongly, which demonstrates why tree regularization is necessary.
```

### Workflow 09: Bagging and random forests

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
    pooled OOF average precision about 0.662
    pooled OOF ROC-AUC about 0.846

Selected random forest:
    pooled OOF average precision about 0.660
    pooled OOF ROC-AUC about 0.847
```

Both ensembles materially improve on the single tree. Their close difference does not establish that one is meaningfully superior.

### Workflow 10: Boosting

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
The representative XGBoost pooled OOF diagnostic has average precision about 0.670
and ROC-AUC about 0.850.
```

The top boosting point estimates are too close to support a conclusion that one boosting library is definitively best.

### Workflow 11: Support vector machines

Core ideas:

```text
linear score functions
maximum-margin separation
support vectors
hard and soft margins
hinge and squared-hinge loss
C as the penalty weight on margin violations and inverse regularization strength
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
    mean-fold average precision about 0.6594
    mean-fold ROC-AUC about 0.8453

Selected RBF SVC:
    C = 10
    gamma = 0.001
    balanced class weights
    mean-fold average precision about 0.6595
    mean-fold ROC-AUC about 0.8424
```

The selected linear and RBF candidates are effectively tied within the tried grid. The RBF point estimate is only about 0.0001 higher in mean-fold average precision and does not justify a nonlinear-advantage claim. The linear SVM remains the representative SVM diagnostic because it is faster, interpretable, and stronger on pooled out-of-fold evidence.

### Workflow 12: Multilayer perceptrons and feed-forward neural networks

Core ideas:

```text
perceptrons and affine transformations
hidden layers and nonlinear activation functions
feed-forward computation
sigmoid output for binary classification
binary cross-entropy
backpropagation
gradient descent, mini-batches, and adaptive optimization
learning rate, batch size, epochs, regularization, and early stopping
scaling and one-hot encoded tabular inputs
validation behaviour, calibration, and threshold behaviour
```

Project role:

```text
test whether a feed-forward neural network can extract useful nonlinear evidence from
scaled one-hot Telco data without assuming that it should outperform tree boosting
```

Observed development-stage position:

```text
The representative MLP has pooled OOF average precision about 0.654.
```

The MLP is a legitimate finalist family but does not establish a material advantage over the leading boosted, bagged, regularized-linear, or SVM procedures from historical workflow evidence alone.

## Core candidate library and final-comparison infrastructure

The individual educational workflows now transition into a systematic final-selection stage. The implemented core registry contains 17 candidate families from the documented C01-C23 universe:

```text
C01  Ridge classifier
C02  Regularized logistic regression
C06  k-nearest neighbours
C07  Hybrid Gaussian-Bernoulli Naive Bayes
C08  Decision tree
C09  Extra Trees
C10  Bagging
C11  Random forest
C13  AdaBoost
C15  GradientBoostingClassifier
C16  HistGradientBoostingClassifier
C17  XGBoost
C18  LightGBM
C19  CatBoost
C21  Linear SVM
C22  RBF SVM
C23  Dense multilayer perceptron
```

The registry compares complete procedures, not bare estimator names. A procedure includes its feature policy, preprocessing representation, feature-selection policy where compatible, imbalance treatment, model hyperparameters, and random-state contract.

The reusable infrastructure now provides:

```text
deterministic repeated outer splits
persistent Optuna studies and trial continuation
Stage-A exploration and Stage-B confirmation
SQLite task coordination and atomic result artifacts
resume-safety fingerprints for protocol, data, environment, and candidate procedure
single-threaded native estimators inside process-level outer parallelism
training-only smoke tests for every major reusable component
```

## Cross-cutting modelling policies before final evaluation

### Feature policies

```text
F0_RAW:
    raw cleaned predictor table

F1_DOMAIN_ENRICHED:
    target-free service aggregates, tenure summaries, selected interactions, and one
    categorical contract-by-payment interaction

F2_LINEAR_EXPANDED:
    controlled nonlinear and interaction basis available only to ridge and logistic
    regression procedures
```

F2 requires a final protocol review before the master run. The final feature contract must not duplicate semantic constructs and must justify any interaction involving `TotalCharges`, which is a cumulative quantity related to tenure and monthly charges.

### Feature selection

```text
S0_NONE:
    no selection

S1_VARIANCE_MUTUAL_INFO:
    variance filtering plus mutual-information SelectKBest

S2_L1_LOGISTIC_SELECT_FROM_MODEL:
    embedded L1-logistic feature selection
```

Feature selection remains limited to candidate families for which it has a coherent modelling role. It is not presumed beneficial for trees, boosted trees, or native-categorical procedures.

### Imbalance treatment

```text
I0_NONE:
    observed training-fold class distribution

I1_CLASS_WEIGHT_BALANCED:
    fold-local balanced sample weighting

I2_RANDOM_OVERSAMPLING:
    fit-time-only random oversampling after representation preprocessing

I3_RANDOM_UNDERSAMPLING:
    fit-time-only random undersampling after representation preprocessing

I4_SMOTENC:
    raw-only mixed-data synthetic oversampling before one-hot encoding
```

The policies are mutually exclusive. The registry records compatibility explicitly, and all sampling occurs only during fitting inside the relevant inner or outer training partition.

## Final training-only selection roadmap

The intended sequence is:

```text
1. Resolve the final F2 feature-policy review and freeze all candidate contracts.
2. Run a realistic persistent-run pilot and inspect artifacts, runtimes, resume behavior,
   warnings, and failures.
3. Freeze the master comparison revision.
4. Run the 5 outer-fold x 10-repeat nested-CV comparison using average precision as the
   primary ranking metric.
5. Analyze outer-fold stability, paired differences, practical equivalence, runtime, and
   selected-configuration stability.
6. Define a defensible finalist set using training-only evidence.
7. Compare calibration only when probability outputs are operationally relevant.
8. Perform cross-fitted threshold, capacity, cost, and intervention-volume analysis.
9. Consider stacking only after constituent procedures and out-of-fold evidence are frozen.
10. Select one final procedure or justified stack.
11. Rerun the frozen search on all 5,634 development rows and fit the complete pipeline.
12. Evaluate once on the untouched test set and report final metrics with uncertainty where
    feasible.
```

The master comparison should use:

```text
5 outer folds x 10 repeats
Stage A: 3-fold persistent Optuna exploration
Stage B: 5-fold confirmation of the strongest Stage-A configurations
primary metric: average precision
```

## Advanced-candidate admission rule

The documented candidate universe also contains advanced procedures such as balanced ensembles, explainable boosting, TabNet, FT-Transformer, TabM, TabPFN, and an AutoML benchmark. They are not admitted automatically merely to enlarge the model list.

A later advanced candidate may enter only after it passes:

```text
installation and licensing check
reproducible fit-and-predict smoke test
fold-safe preprocessing smoke test
checkpoint and resume smoke test
CPU or GPU resource scheduling smoke test
```

A failure must be recorded as a technical exclusion rather than silently ignored.

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

The following remain prohibited or deferred until the correct stage:

```text
using the test set to compare candidates
using the test set to choose a threshold
using the test set to decide whether calibration is useful
claiming a final production model before the finalist-selection stage
using close historical cross-validation point estimates as proof of superiority
adding advanced model families without the documented admission checks
```

## Documentation convention

The strategic inventory belongs in this roadmap. Immediate work belongs in the live status file. New-chat context belongs in the newest handoff file. Stable documentation architecture belongs in `00_documentation_workflow.md`.
