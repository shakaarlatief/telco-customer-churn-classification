# Telco Customer Churn Classification: Chat Handoff Context 5

## Purpose

Use this file when continuing the Telco Customer Churn classification project in a new chat.

This handoff supersedes earlier handoff files for current project state. Earlier handoffs remain useful for detailed background on the raw audit, splitting, EDA, methodology design, and the earlier model sections.

## Project identity

Project:

```text
Telco Customer Churn binary classification
```

Repository:

```text
https://github.com/shakaarlatief/telco-customer-churn-classification
```

Goal:

```text
Build a professional, portfolio-ready classification project and use it as a reusable
technical reference for classification modelling, evaluation, implementation, and reporting.
```

The project deliberately studies many relevant model families rather than only selecting the current highest-scoring one.

## User preferences and working rules

The user prefers:

```text
professional, portfolio-ready work
deep technical explanations with mathematics when useful
standalone report prose that does not read like course notes
long report sections when the detail is useful
deep reusable knowledge notes
notebooks that are educational but not giant textbooks
LaTeX for the formal report
no emojis in technical or professional responses
no em dashes
explicit description of changes made
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

Codex may only be considered for narrow cleanup, formatting, or repetitive edits after the design and content are already established. In normal project work, direct authored changes are preferred.


### Repository-write and artifact-handoff rule

The assistant must work through downloadable artifacts by default:

```text
1. Prepare the requested file, replacement, or patch in the chat.
2. State every meaningful change and its intended repository location.
3. The user reviews the artifact, places it locally, runs checks, and controls Git.
4. The user stages, commits, and pushes by default.
```

The assistant must not directly write to GitHub, modify repository files through a connected tool, stage changes, create commits, or push commits unless the user explicitly authorizes that particular action.

The assistant may suggest a direct write and ask for a green light. Approval must be specific. It does not create ongoing permission for later writes.

### Reusable source and smoke-test rule

Before a substantial model-family notebook is prepared, inspect:

```text
src/telco_churn/
scripts/
the closest previous model-family notebook
```

Reusable preprocessing, model factories, evaluation functions, and plotting utilities belong in `src/telco_churn/`. The notebook should use those shared factories instead of maintaining a duplicate local implementation. One-off experimental settings and section-specific interpretation can remain local to the notebook.

Any new or materially changed reusable workflow component requires a corresponding small training-only smoke test in `scripts/`. The smoke test must exercise the same shared factory or utility used by the full notebook and should pass before the full notebook is executed.

## Latest confirmed repository checkpoint

The latest confirmed completed modelling commit before this handoff is:

```text
f784e781d0f674fc4b2265ea53ad601302536c6a
Add support vector machine section
```

This commit is pushed to GitHub and contains the full SVM workflow. It also removes the accidentally staged LaTeX SyncTeX busy lock file and adds this ignore rule:

```text
*.synctex(busy)
```

When continuing in a later chat, check GitHub for commits after this checkpoint. The coordination-document update containing this handoff may itself be newer.

## Dataset state

```text
Clean modelling dataset: 7043 observations
Training set:            5634 observations
Held-out test set:       1409 observations
Positive class:          Churn_binary = 1
Training churn rate:     approximately 26.54%
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

The held-out test set has not been used for model selection, threshold selection, calibration selection, or model-family comparison.

All current result tables and figures are development-stage estimates based on training-only cross-validation.

Use this wording:

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
1. Complete remaining model-family and finalist work on the training data only.
2. Freeze one full end-to-end pipeline.
3. Fit that frozen pipeline on the full training set.
4. Evaluate it once on the untouched test set.
5. Report uncertainty for the one final test evaluation where feasible.
```

## Completed project stages

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
```

## Key model results

### Logistic regression

```text
Representative L2 logistic regression:
    C = 1
    pooled OOF ROC-AUC about 0.846
    pooled OOF PR-AUC about 0.658
```

L1 and L2 are effectively tied. Class weighting materially increases recall but also increases false positives.

### k-nearest neighbours

```text
Selected kNN:
    k = 101
    uniform weights
    Manhattan distance
    pooled OOF ROC-AUC about 0.836
    pooled OOF PR-AUC about 0.628
```

The stable lesson is that smooth large-neighbourhood models work better than small local neighbourhoods for the one-hot tabular representation.

### Naive Bayes

```text
Selected model:
    HybridGaussianBernoulliNB
    alpha = 1
    pooled OOF ROC-AUC about 0.822
    pooled OOF PR-AUC about 0.615
    recall about 0.809
```

The hybrid implementation is in `src/telco_churn/models.py`.

### Decision tree

```text
Selected pre-pruned tree:
    gini
    max_depth = 6
    min_samples_split = 25
    min_samples_leaf = 10
    ccp_alpha = 0.0
    pooled OOF ROC-AUC about 0.824
    pooled OOF PR-AUC about 0.628
```

The unrestricted tree overfits strongly.

### Bagging and random forests

```text
Selected bagged trees:
    pooled OOF PR-AUC about 0.662
    pooled OOF ROC-AUC about 0.846

Selected random forest:
    pooled OOF PR-AUC about 0.660
    pooled OOF ROC-AUC about 0.847
```

The ensemble improvement over the single tree is material. The bagging versus random-forest difference is too small to overclaim.

### Boosting

```text
Strong fixed-grid candidates:
    CatBoost mean CV PR-AUC about 0.673
    GradientBoostingClassifier mean CV PR-AUC about 0.672
    XGBoost mean CV PR-AUC about 0.672

Representative pooled-OOF diagnostic:
    XGBoost PR-AUC about 0.670
    XGBoost ROC-AUC about 0.850
```

Boosted trees form the strongest observed group so far. CatBoost, GradientBoostingClassifier, and XGBoost are too close to claim a meaningful ordering.

### Support vector machines

The SVM knowledge note, notebook, report section, artifacts, factories, and smoke test are complete.

Selected linear SVM:

```text
LinearSVC:
    loss = squared_hinge
    C = 0.1
    class_weight = balanced
    mean fold PR-AUC about 0.6594
    mean fold ROC-AUC about 0.8453
    mean balanced accuracy about 0.7648
    mean F1 about 0.6267
```

Selected RBF SVM:

```text
SVC:
    kernel = rbf
    C = 10
    gamma = 0.001
    class_weight = balanced
    mean fold PR-AUC about 0.6595
    mean fold ROC-AUC about 0.8424
    mean balanced accuracy about 0.7464
    mean F1 about 0.6003
```

Interpretation:

```text
The RBF point estimate exceeds the linear point estimate by only about 0.0001 in
mean fold PR-AUC. This is negligible relative to fold-to-fold variation.

The nonlinear RBF kernel does not show a material advantage in the tried grid.

The linear SVM is the representative SVM diagnostic model because it is essentially tied
on the selection metric, faster, directly interpretable, and stronger on pooled OOF PR-AUC.
```

Class weighting changes the natural operating point:

```text
Selected LinearSVC, score threshold 0:
    precision about 0.513
    recall about 0.807
    specificity about 0.723
    predicted positive rate about 0.418
```

These are raw SVM decision scores, not probabilities. The threshold curve is diagnostic only.

## SVM implementation details

Reusable SVM factories are in:

```text
src/telco_churn/models.py
```

Relevant functions:

```text
make_linear_svc_classifier
make_linear_svc_pipeline
make_kernel_svc_classifier
make_kernel_svc_pipeline
make_rbf_svc_pipeline
```

The threshold plotting helper in:

```text
src/telco_churn/visualization.py
```

was generalized to accept arbitrary decision-score thresholds and an optional reference threshold.

The SVM smoke test is:

```text
scripts/smoke_test_svm_workflow.py
```

It passed after the shared factories and plot helper were added.

## Report and local build state

The LaTeX report is compiled locally and now contains the SVM section as Section 12. The compiled report is 113 pages.

TinyTeX is the chosen local LaTeX distribution. `tocloft` was installed after the compiler reported that the package was missing. The report compiles successfully after that installation.

LaTeX auxiliary files are ignored. The repository also ignores:

```text
*.synctex(busy)
```

The accidental `main.synctex(busy)` lock file was removed from the SVM commit before push.

## Documentation roles

Use these names in conversation:

```text
live project coordination documents:
    current_project_status_and_next_actions.md
    01_model_inventory_and_roadmap.md
    newest context_history handoff file
```

Their roles:

```text
current_project_status_and_next_actions.md:
    tactical state and immediate actions

01_model_inventory_and_roadmap.md:
    strategic model-family inventory and longer-term plan

context_history/telco_churn_chat_handoff_context_<number>.md:
    standalone state for starting a new chat
```

`00_documentation_workflow.md` defines stable documentation roles and workflow rules. It should not contain an exhaustively current file inventory. It is updated only when the documentation architecture or workflow changes.

`current_notebook_documentation_audit.md` is an audit snapshot, not a live task list.

## Immediate next step

Start the neural-network model-family section:

```text
12_multilayer_perceptrons_and_neural_networks
```

Recommended sequence:

```text
1. Create the MLP knowledge note first.
2. Cover architecture, activations, binary cross-entropy, backpropagation,
   optimisation, regularisation, early stopping, scaling, and validation.
3. Design the training-only notebook.
4. Run it locally and return the executed notebook, tables, figures, and logs.
5. Update interpretation from observed outputs only.
6. Write the LaTeX report section.
7. Compile and inspect the report.
8. Update the live coordination documents and create a new handoff if needed.
9. Commit only after review.
```

After remaining model-family work, build a dedicated finalist-selection stage using training-only evidence. Only after calibration and threshold policy are frozen should the test set be evaluated once.
