# Current Project Status and Next Actions

## Purpose

This file is the live tactical status tracker for the Telco Customer Churn classification project.

Use it to answer:

```text
Where is the project now?
What completed work is available in the repository?
What is the immediate next modelling step?
What must happen before the held-out test set is used?
```

It is intentionally shorter and more operational than the model knowledge notes and the LaTeX report.

## Latest confirmed modelling checkpoint

The latest confirmed modelling checkpoint before this coordination update is:

```text
f784e781d0f674fc4b2265ea53ad601302536c6a
Add support vector machine section
```

That commit contains the complete Support Vector Machine workflow and also corrects the LaTeX temporary-file policy by ignoring `*.synctex(busy)`.

The coordination-document commit that follows this file may be newer than the checkpoint above. The purpose of recording the SVM commit here is to identify the most recent completed modelling milestone, not to create a self-referential latest-commit field that becomes stale immediately after every documentation update.

## Current project state

Completed and committed workflow stages:

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

The held-out test set remains untouched.

```text
Clean modelling dataset: 7043 rows
Training set:            5634 rows
Held-out test set:       1409 rows
Positive class:          Churn_binary = 1
Training churn rate:     approximately 26.54%
```

The compiled report now includes the completed SVM section and is 113 pages long.

## Governing evaluation policy

All current model-family results are development-stage estimates.

```text
Allowed before final test evaluation:
    training-set cross-validation;
    training-only preprocessing;
    fixed-grid model-family experiments;
    candidate comparison;
    threshold diagnostics;
    calibration diagnostics;
    training-only uncertainty analysis.

Not allowed before final model freeze:
    selecting models with the held-out test set;
    selecting thresholds with the held-out test set;
    selecting calibration methods with the held-out test set;
    repeatedly checking test metrics while development continues.
```

Preferred language:

```text
selected within the tried development grid
representative strong candidate
development-stage cross-validated estimate
small differences should be interpreted cautiously
final test evaluation is deferred
```

The final test set is used exactly once after the final model family, preprocessing, hyperparameters, calibration policy, and decision threshold have all been frozen.

## Completed model-family checkpoints

### Section 05: linear classification and logistic regression

Status:

```text
complete and committed
```

Representative result:

```text
Selected L2 logistic regression:
    C = 1
    pooled OOF ROC-AUC about 0.846
    pooled OOF PR-AUC about 0.658
```

Main lesson:

```text
L1 and L2 logistic regression are effectively tied in the development grid.
L2 logistic regression remains a stable, interpretable linear benchmark.
Class weighting shifts the default operating point toward higher recall and more false positives.
```

### Section 06: k-nearest neighbours

Status:

```text
complete and committed
```

Representative result:

```text
Selected kNN:
    n_neighbors = 101
    uniform weights
    Manhattan distance
    pooled OOF ROC-AUC about 0.836
    pooled OOF PR-AUC about 0.628
```

Main lesson:

```text
Very small neighbourhoods are too noisy for this representation.
A smoother, large-neighbourhood model performs better within the tried grid.
kNN remains below the strongest linear and ensemble candidates by ranking metrics.
```

### Section 07: Naive Bayes

Status:

```text
complete and committed
```

Important reusable addition:

```text
HybridGaussianBernoulliNB in src/telco_churn/models.py
```

Representative result:

```text
Selected hybrid Gaussian-Bernoulli Naive Bayes:
    alpha = 1
    pooled OOF ROC-AUC about 0.822
    pooled OOF PR-AUC about 0.615
    recall about 0.809
```

Main lesson:

```text
The hybrid likelihood is more appropriate than treating one-hot categorical indicators
as Gaussian variables. Conditional independence remains a material limitation.
```

### Section 08: decision trees

Status:

```text
complete and committed
```

Representative result:

```text
Selected pre-pruned decision tree:
    criterion = gini
    max_depth = 6
    min_samples_split = 25
    min_samples_leaf = 10
    ccp_alpha = 0.0
    pooled OOF ROC-AUC about 0.824
    pooled OOF PR-AUC about 0.628
```

Main lesson:

```text
The unrestricted tree overfits strongly.
Regularization is essential.
A single tree is useful for interpretable nonlinear rules, but does not match the
stronger linear and ensemble candidates.
```

### Section 09: bagging and random forests

Status:

```text
complete and committed
```

Representative results:

```text
Selected bagged trees:
    grid mean PR-AUC about 0.668
    pooled OOF PR-AUC about 0.662
    pooled OOF ROC-AUC about 0.846

Selected random forest:
    grid mean PR-AUC about 0.664
    pooled OOF PR-AUC about 0.660
    pooled OOF ROC-AUC about 0.847
```

Main lesson:

```text
Averaging trees improves materially on the single decision tree.
Bagging and random forests are close enough that their small observed differences
do not establish a meaningful ordering.
```

### Section 10: boosting

Status:

```text
complete and committed
```

Representative results:

```text
Strongest fixed-grid family point estimates:
    CatBoost mean CV PR-AUC about 0.673
    GradientBoostingClassifier mean CV PR-AUC about 0.672
    XGBoost mean CV PR-AUC about 0.672

Representative pooled-OOF diagnostic model:
    XGBoost PR-AUC about 0.670
    XGBoost ROC-AUC about 0.850
```

Main lesson:

```text
Boosted-tree methods form the strongest observed group so far.
CatBoost, GradientBoostingClassifier, and XGBoost are too close to support a
claim that one implementation is definitively superior.
```

### Section 11: support vector machines

Status:

```text
complete and committed
```

The SVM section includes:

```text
linear maximum-margin classification
soft margins and hinge loss
class weighting
linear, polynomial, and RBF kernel screening
linear and RBF fixed grids
margin-score threshold diagnostics
linear coefficient directions
RBF support-vector diagnostics
a dedicated smoke test
report integration
```

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
The mean fold PR-AUC difference between the selected linear and RBF candidates is
approximately 0.0001, far below observed fold-to-fold variation.

The RBF model therefore does not show a material nonlinear advantage in the tried grid.

The linear SVM is retained as the representative SVM diagnostic model because it is
essentially tied on the selection metric, faster, directly interpretable, and stronger
on the pooled OOF diagnostic.
```

Important operating-point result:

```text
The balanced LinearSVC uses a natural score threshold of 0.

At that boundary:
    precision about 0.513
    recall about 0.807
    specificity about 0.723
    predicted positive rate about 0.418

This is a high-recall operating point, not a final decision rule.
The raw SVM decision scores are uncalibrated margins rather than probabilities.
```

Important reusable additions:

```text
src/telco_churn/models.py
    make_linear_svc_classifier
    make_linear_svc_pipeline
    make_kernel_svc_classifier
    make_kernel_svc_pipeline
    make_rbf_svc_pipeline

src/telco_churn/visualization.py
    threshold plot helper generalized for arbitrary score thresholds
    optional reference threshold and reference label

scripts/smoke_test_svm_workflow.py
    smoke test for reusable SVM factories, decision scores, and plots
```

## Documentation and build state

The current documentation roles are:

```text
00_documentation_workflow.md:
    stable rules, directory conventions, and model-section workflow

01_model_inventory_and_roadmap.md:
    strategic model-family inventory and longer-term roadmap

current_project_status_and_next_actions.md:
    live tactical state and immediate actions

context_history/telco_churn_chat_handoff_context_<number>.md:
    standalone continuation snapshot for a new chat
```

The LaTeX report compiles locally through TinyTeX.

```text
TinyTeX package added:
    tocloft

Generated LaTeX temporary files:
    auxiliary files are ignored
    *.synctex(busy) is explicitly ignored
```

## Immediate next actions

### Next model-family stage

The next planned model-family section is:

```text
12_multilayer_perceptrons_and_neural_networks
```

The section should begin with a knowledge note covering:

```text
feed-forward neural-network architecture
input layer, hidden layers, and output layer
weights, biases, affine transformations, and activation functions
binary cross-entropy
backpropagation at a conceptual and mathematical level
stochastic gradient descent and adaptive optimizers
batch size, epochs, learning rate, regularization, and early stopping
scaling and one-hot encoded tabular inputs
overfitting and validation discipline
probability outputs, calibration, and threshold behaviour
```

Then follow the established model-section process:

```text
1. Create the MLP knowledge note.
2. Design the training-only notebook workflow.
3. Keep preprocessing inside pipelines or fold-safe model workflows.
4. Run the notebook locally and collect artifacts.
5. Interpret observed results only after execution.
6. Write the report section from observed results.
7. Compile and inspect the report.
8. Update the live coordination documents and create a new handoff when needed.
9. Commit only after the section is checked.
```

### After remaining model-family work

After the MLP section and any justified cross-cutting candidate refinements, the project should perform a dedicated final training-only selection stage:

```text
1. Define a serious finalist set.
2. Compare the frozen candidate procedures fairly using training-only evidence.
3. Consider repeated CV, nested CV, paired bootstrap differences, or related
   uncertainty tools where justified.
4. Decide whether probability calibration is needed.
5. Define a threshold policy from retention value, contact capacity, and error costs.
6. Freeze one final end-to-end pipeline.
7. Fit on the complete training set.
8. Evaluate once on the untouched held-out test set.
9. Report final metrics with uncertainty intervals where feasible.
```

## Files not to use as live task lists

Do not use the following files as the source of immediate project status:

```text
00_documentation_workflow.md
current_notebook_documentation_audit.md
model knowledge notes
methodology knowledge notes
LaTeX report sections
```

Use this status file for tactical decisions and the newest handoff file when beginning a new chat.
