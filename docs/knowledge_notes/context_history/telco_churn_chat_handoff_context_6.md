# Telco Customer Churn Classification: Chat Handoff Context 6

## Purpose

Use this file when continuing the Telco Customer Churn classification project in a new chat.

This handoff supersedes earlier handoff files for the current working state. Earlier handoffs remain useful for detailed history of the raw audit, splitting, EDA, individual model workflows, and earlier documentation decisions.

## Project identity

Project:

```text
Telco Customer Churn binary classification
```

Repository:

```text
shakaarlatief/telco-customer-churn-classification
```

Goal:

```text
Build a professional, portfolio-ready churn-classification project and use it as a
reusable technical reference for classification modelling, evaluation, implementation,
and reporting.
```

The project deliberately studies many relevant model families. The aim is not merely to find a high score. It is to create a rigorous reusable reference for preprocessing, feature engineering, evaluation, model comparison, imbalance handling, calibration, thresholds, and final test discipline.

## User preferences and working rules

The user prefers:

```text
professional, portfolio-ready work
deep technical explanations with mathematics when useful
standalone report prose that does not read like course notes
long report sections when detail is useful
deep reusable knowledge notes
notebooks that are educational but not giant textbooks
LaTeX for the formal report
no emojis in technical or professional responses
no em dashes
explicit description of meaningful changes
no silent deletion or shortening of good content
```

When changing code or documents:

```text
inspect the current file state first
preserve good existing content
state every meaningful change
validate through appropriate checks
do not use Codex for foundational project content
```

Codex may be considered only for narrow cleanup, formatting, or repetitive edits after the design and content are already established. Direct authored work is preferred for core methodology, model logic, and report content.

### Repository-write and artifact-handoff rule

The default workflow remains local-first and user-controlled:

```text
1. Prepare a patch or downloadable artifact in chat.
2. State every meaningful change and intended destination.
3. User reviews, applies locally, runs checks, and inspects the diff.
4. User stages, commits, and pushes by default.
```

Do not directly write to GitHub, modify repository files through a connected tool, stage changes, create commits, or push commits unless the user explicitly authorizes that specific action. Authorization for one action does not create continuing permission for later writes.

### Reusable source and smoke-test rule

Before a substantial model-family notebook or final-selection component is prepared, inspect:

```text
src/telco_churn/
scripts/
the closest previous workflow
```

Reusable preprocessing, model factories, evaluation functions, visualization utilities, and final-comparison coordination belong in `src/telco_churn/`. One-off experimental settings and result-specific notebook narrative can remain in the notebook.

Every new or materially changed reusable component requires a small, training-only smoke test in `scripts/`. The smoke test must exercise the same shared implementation used by the full workflow and should pass before a long notebook or master-comparison run is started.

## Dataset state

```text
Clean modelling dataset: 7043 observations
Development training set: 5634 observations
Held-out test set:       1409 observations
Positive class:          Churn_binary = 1
Development churn rate:  approximately 26.54%
```

Feature groups:

```text
Numeric:
    tenure
    MonthlyCharges
    TotalCharges

Categorical:
    SeniorCitizen
    gender
    Partner
    Dependents
    PhoneService
    PaperlessBilling
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaymentMethod
```

`customerID` is excluded from modelling as a unique identifier.

## Strict evaluation policy

The held-out test set has not been used for model selection, threshold selection, calibration selection, feature-policy selection, or model-family comparison.

All current model tables and figures are development-stage estimates based on training-only cross-validation. Use:

```text
selected within the tried development grid
representative strong candidate
development-stage cross-validated estimate
small differences should be interpreted cautiously
final test evaluation is deferred
```

Do not claim:

```text
definitively best
uniquely optimal
proven superior
final performance
```

Final test policy:

```text
1. Complete final procedure selection on development data only.
2. Freeze one complete end-to-end pipeline or justified stack.
3. Rerun its frozen search on the 5,634 development rows.
4. Fit all learned steps using development data only.
5. Evaluate once on the untouched 1,409-row test set.
6. Report uncertainty for that one final evaluation where feasible.
```

The final frozen procedure includes:

```text
candidate family or stack
feature policy
feature-selection policy
imbalance policy
model hyperparameters
calibration choice, when selected
threshold or operating-point policy
```

## Completed model-family workflows

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_statistical_evaluation_methodology
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
06_k_nearest_neighbours
07_naive_bayes
08_decision_trees
09_bagging_and_random_forests
10_boosting
11_support_vector_machines
12_multilayer_perceptrons_and_neural_networks
```

The LaTeX report contains the completed MLP section as Section 13 and has been compiled locally at 126 pages.

## Key development-stage results

These values orient the next final-selection stage. They are not a final ranking.

### Regularized logistic regression

```text
Representative L2 logistic regression:
    C = 1
    pooled OOF ROC-AUC about 0.846
    pooled OOF average precision about 0.658
```

L1 and L2 are effectively tied in the historical grid. L2 remains a strong, interpretable reference procedure.

### k-nearest neighbours

```text
Selected kNN:
    k = 101
    uniform weights
    Manhattan distance
    pooled OOF ROC-AUC about 0.836
    pooled OOF average precision about 0.628
```

Large, smooth neighbourhoods are preferable to very small local neighbourhoods for this one-hot representation.

### Hybrid Gaussian-Bernoulli Naive Bayes

```text
Selected model:
    HybridGaussianBernoulliNB
    alpha = 1
    pooled OOF ROC-AUC about 0.822
    pooled OOF average precision about 0.615
    recall about 0.809
```

The hybrid likelihood is more coherent than treating the one-hot categorical block as Gaussian, but conditional independence remains a material limitation.

### Decision tree

```text
Selected pre-pruned tree:
    gini
    max_depth = 6
    min_samples_split = 25
    min_samples_leaf = 10
    ccp_alpha = 0.0
    pooled OOF ROC-AUC about 0.824
    pooled OOF average precision about 0.628
```

The unrestricted tree overfits strongly.

### Bagging and random forest

```text
Selected bagged trees:
    pooled OOF average precision about 0.662
    pooled OOF ROC-AUC about 0.846

Selected random forest:
    pooled OOF average precision about 0.660
    pooled OOF ROC-AUC about 0.847
```

Both ensembles materially improve on the single tree. Their difference is too small to overclaim.

### Boosting

```text
Strong fixed-grid candidates:
    CatBoost mean CV average precision about 0.673
    GradientBoostingClassifier mean CV average precision about 0.672
    XGBoost mean CV average precision about 0.672

Representative pooled-OOF diagnostic:
    XGBoost average precision about 0.670
    XGBoost ROC-AUC about 0.850
```

Boosted trees form the strongest observed group so far, but the close values do not establish a definitive ordering.

### Support vector machines

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

The RBF point estimate exceeds the linear one by about 0.0001 in mean-fold average precision. This is negligible relative to fold-to-fold variation. Linear SVM remains the representative SVM diagnostic because it is faster, directly interpretable, and stronger on pooled OOF evidence.

### Multilayer perceptron

```text
Representative MLP:
    pooled OOF average precision about 0.654
```

The MLP belongs in the broad finalist library but does not establish a material advantage over the leading boosted, bagged, regularized-linear, or SVM procedures from the historical workflow evidence alone.

## Final-comparison design

The project now has reusable implementation through Phase 8B for a final training-only comparison of complete procedures.

### Candidate registry

The implemented core candidate registry contains 17 families:

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
C23  Multilayer perceptron
```

Every procedure remains an unfitted, fold-safe pipeline. Native parallel estimators are configured with one worker or one thread because the outer runner controls process-level parallelism.

### Evaluation and HPO protocol

The predeclared target architecture, pending the final F2 audit, is:

```text
5 outer folds x 10 repeats

Stage A:
    candidate-specific persistent Optuna exploration
    3-fold stratified inner CV

Stage B:
    5-fold confirmation of top Stage-A configurations

Primary metric:
    average precision
```

The outer task stores:

```text
selected configuration
inner trial history
Stage-A and Stage-B results
outer-fold scores and predictions
runtime information
warnings and failures
procedure metadata and fingerprints
```

Persistent-study and task-resume safety is implemented through protocol, data, environment, and candidate-procedure fingerprints. A changed routing contract must create a new compatible run rather than silently reuse old trials.

### Feature policies

```text
F0_RAW:
    raw cleaned predictors

F1_DOMAIN_ENRICHED:
    target-free service aggregates, tenure summaries, selected interactions, and one
    categorical contract-by-payment interaction

F2_LINEAR_EXPANDED:
    controlled nonlinear and interaction basis, available only to ridge and logistic
    regression
```

Important pending review: F2 needs a protocol-freeze audit before the master run. Review potential duplicate tenure-squared constructs and the rationale for `TotalCharges` interactions, because `TotalCharges` is cumulative and strongly related to tenure and monthly charges. Any approved change must update the feature-policy contract, smoke test, methodology note, candidate fingerprint, and related documentation before the master comparison begins.

### Feature selection

```text
S0_NONE:
    no feature selection

S1_VARIANCE_MUTUAL_INFO:
    variance filtering followed by mutual-information SelectKBest

S2_L1_LOGISTIC_SELECT_FROM_MODEL:
    embedded L1-logistic selection
```

Selection is restricted to families for which it is coherent. Tree and native categorical boosting procedures retain S0 only.

### Imbalance treatment

```text
I0_NONE:
    no explicit adjustment

I1_CLASS_WEIGHT_BALANCED:
    fold-local balanced sample weights from the active fitting target

I2_RANDOM_OVERSAMPLING:
    fit-time-only random oversampling after representation preprocessing

I3_RANDOM_UNDERSAMPLING:
    fit-time-only random undersampling after representation preprocessing

I4_SMOTENC:
    raw-only mixed-data synthetic oversampling before one-hot encoding
```

The policies are mutually exclusive. F1 and F2 are not compatible with I4 because synthetic derived features could be internally inconsistent with their raw inputs.

The Phase 8B compatibility matrix is:

```text
Ridge and logistic regression:
    I0, I1, I2, I3, I4 with F0
    I0, I1, I2, I3 with F1 or F2

Linear SVM, RBF SVM, and MLP:
    I0, I1, I2, I3, I4 with F0
    I0, I1, I2, I3 with F1

kNN and hybrid Gaussian-Bernoulli Naive Bayes:
    I0, I2, I3, I4 with F0
    I0, I2, I3 with F1

Decision tree, Extra Trees, bagging, random forest, AdaBoost,
GradientBoostingClassifier, HistGradientBoostingClassifier,
XGBoost, LightGBM, and CatBoost:
    I0, I1
```

Optuna uses feature-policy-specific imbalance parameter names, such as `imbalance_policy__f0_raw`, so each persistent study retains a fixed categorical distribution. Saved candidate configurations retain the canonical `imbalance_policy` field.

## Final-comparison implementation state

Completed phases:

```text
Phase 1:
    protocol, deterministic repeated splits, SQLite coordination, atomic artifacts,
    fingerprint and resume foundations

Phase 2:
    persistent two-stage Optuna HPO with Windows-safe study-resource cleanup and actual
    serial/interrupted/resumed/two-worker smoke coverage

Phase 3:
    complete 17-family core candidate registry and single-threaded builder contract

Phase 4:
    deterministic F0/F1/F2 feature-policy transformer layer

Phase 5:
    candidate-specific feature-policy routing inside HPO and pipelines

Phase 6:
    S0/S1/S2 feature-selection routing inside HPO and pipelines

Phase 7:
    I0-I4 fold-safe imbalance primitives and standalone sampler smoke coverage

Phase 8A:
    fit-time imbalance pipeline adapter, including balanced sample weights, random
    sampling, raw-only SMOTENC, and CatBoost sample-weight compatibility

Phase 8B:
    candidate-specific imbalance routing, conditional Optuna parameters, static
    categorical-distribution safety, and procedure-contract persistence
```

Smoke tests passed after Phase 8B for:

```text
persistent nested HPO and interruption/resume
the complete core candidate registry
F0/F1/F2 feature policies
candidate feature-policy routing
candidate feature-selection routing
imbalance policy primitives
imbalance pipeline topology
candidate-specific imbalance routing and static Optuna distributions
```

No master repeated nested-CV run has been performed yet.

## Report and documentation state

The LaTeX report compiles locally through TinyTeX and is currently 126 pages. The MLP section is Section 13.

Known deferred presentation cleanup:

```text
long-title collision in the table of contents
label-width cleanup for Figures 58 and 59
```

Documentation roles:

```text
00_documentation_workflow.md:
    stable rules and documentation architecture

01_model_inventory_and_roadmap.md:
    strategic model-family and final-selection roadmap

current_project_status_and_next_actions.md:
    live tactical state and immediate actions

context_history/telco_churn_chat_handoff_context_6.md:
    current standalone continuation snapshot
```

## Immediate next actions

1. Resolve and freeze the F2 feature-policy design review before any master repeated nested-CV result is used for selection.

2. Confirm candidate inclusion, HPO budgets, compatibility maps, artifact locations, and runtime assumptions. Run a realistic persistent-run pilot that verifies stored predictions, selected configurations, resumability, and diagnostics.

3. Freeze the full comparison protocol revision and run the 5 x 10 repeated nested-CV candidate comparison on development data only.

4. Analyze average precision, secondary metrics, practical equivalence, paired uncertainty, selected-hyperparameter stability, runtime, warnings, and failure rates. Define a defensible finalist set.

5. Build calibration and threshold-selection comparisons only after candidate ranking is understood. Calibration and threshold decisions must be based on training-only cross-fitted evidence, not the test set.

6. Consider stacking only after component procedures and their out-of-fold evidence are frozen.

7. Choose one final procedure or justified stack, refit on all development rows, write a complete manifest, and evaluate once on the held-out test set.
