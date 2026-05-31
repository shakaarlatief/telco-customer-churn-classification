# Current Project Status and Next Actions

## Purpose

This file is the tactical status file for the Telco Customer Churn classification project.

Use it to answer:

```text
Where are we now?
What has been committed?
What was prepared but may still need to be added?
What should happen before continuing modelling?
What is the immediate next step?
```

This file is intentionally shorter and more operational than the knowledge notes and the report.

## Latest known GitHub state at time of this update

Latest confirmed GitHub commit at the time this file was last updated:

```text
d91fa9bf086bdfba58e4118d371c882e71452b05
Update project status after decision trees
```

Important note:

```text
The previous status-update commit was pushed after the decision-tree modelling commit, but the replacement files used there still contained stale checkpoint text in some places. This version corrects those live status and roadmap references.
```

The actual section 08 modelling commit is:

```text
08fb64873d4c8a929cfde529638d2e1ed49fcd5d
Add decision tree modelling section
```

That commit added the decision-tree knowledge note, notebook source, executed notebook, generated tables and figures, report section, and compiled report update.

At the current confirmed state, the project includes completed and committed sections through:

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
```

The next modelling step is:

```text
09_bagging_and_random_forests
```

## Important methodological decision that still governs the project

Section-level cross-validation results are development-stage estimates.

They are useful for:

```text
model learning
tuning
candidate comparison
understanding model-family behaviour
```

They are not:

```text
final performance claims
proof that a small metric difference is a population-level superiority result
proof that one hyperparameter value is uniquely optimal
```

Preferred language:

```text
selected within the development grid
strongest configuration in this tried grid
development-stage cross-validated estimate
representative candidate
small differences should be interpreted cautiously
final test evaluation is deferred
```

Avoid:

```text
this model is definitively best
this hyperparameter is uniquely optimal
this small metric difference proves superiority
final performance is ...
```

The held-out test set remains unused and should be evaluated only once after one final model, threshold, preprocessing strategy, calibration decision, and any final comparison decisions are fixed.

## Current status of model sections

### Section 04: preprocessing, evaluation, and simple baselines

Status:

```text
complete and committed
```

Key role:

```text
establishes binary classification metrics, positive-first confusion matrix,
preprocessing infrastructure, simple baselines, class imbalance logic,
thresholds, and calibration introduction
```

Main lesson:

```text
accuracy alone is misleading under churn imbalance.
The majority and prior baselines have high ordinary accuracy but zero recall.
The EDA-inspired rule has very high recall but many false positives.
```

### Section 05: linear classification and logistic regression

Status:

```text
complete and committed
```

Main development results:

```text
Logistic L1 C=1:
    PR-AUC about 0.659
    ROC-AUC about 0.846

Logistic L2 C=1:
    PR-AUC about 0.658
    ROC-AUC about 0.846

Class-weighted L2:
    recall about 0.801
    more false positives
```

Interpretation:

```text
L1 and L2 are effectively tied in development-stage ranking metrics.
L2 C=1 is retained as a stable, interpretable linear benchmark.
The exact C value should not be overinterpreted because several values are close.
```

### Section 06: k-nearest neighbours

Status:

```text
complete and committed
```

Main development result:

```text
Selected kNN:
    k = 101
    uniform weights
    Manhattan distance
    ROC-AUC about 0.836
    PR-AUC about 0.628
```

Interpretation:

```text
The strong lesson is that larger, smoother neighbourhoods work better than very small k.
The exact value k=101 is a grid-selected representative, not a proven unique optimum.
kNN improves strongly over default k=5 but remains below logistic regression in ranking metrics.
```

### Section 07: Naive Bayes

Status:

```text
complete and committed
```

Important source-code addition:

```text
Added custom HybridGaussianBernoulliNB in src/telco_churn/models.py
```

Main development result:

```text
Selected Naive Bayes:
    Hybrid Gaussian-BernoulliNB alpha=1
    ROC-AUC about 0.822
    PR-AUC about 0.615
    recall about 0.809
```

Interpretation:

```text
The hybrid model is theoretically cleaner for mixed numeric and one-hot categorical features.
It improves modestly over full transformed GaussianNB.
The result is a practical and theoretical Naive Bayes selection, not a final statistical superiority claim.
```

### Section 08: decision trees

Status:

```text
complete and committed
```

Actual section commit:

```text
08fb64873d4c8a929cfde529638d2e1ed49fcd5d
Add decision tree modelling section
```

Files added or updated in the section:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
reports/latex/main.tex
reports/latex/main.pdf
reports/figures/decision_tree_*.png
reports/tables/decision_tree_*.csv
```

Main concepts covered:

```text
recursive partitioning
internal split nodes and terminal leaves
Gini impurity
entropy and information gain
leaf churn proportions
ranking with tree probabilities
stepwise tied rankings
pre-pruning
cost-complexity pruning
validation discipline for pruning and hyperparameter selection
feature importance and tree-structure interpretation
```

Main development result:

```text
Selected decision tree:
    Gini criterion
    max_depth = 6
    min_samples_split = 25
    min_samples_leaf = 10
    ccp_alpha = 0.0

Cross-validated metrics:
    accuracy about 0.789
    balanced accuracy about 0.701
    precision about 0.624
    recall about 0.514
    F1 about 0.564
    ROC-AUC about 0.824
    PR-AUC about 0.628
```

Decision-tree comparison:

```text
Selected pre-pruned tree:
    PR-AUC about 0.628
    ROC-AUC about 0.824

Best cost-complexity-pruned tree:
    PR-AUC about 0.615
    ROC-AUC about 0.822

Decision stump:
    PR-AUC about 0.413
    ROC-AUC about 0.726

Default unrestricted tree:
    PR-AUC about 0.371
    ROC-AUC about 0.648
```

Interpretation:

```text
The default unrestricted tree overfits strongly.
Regularization is essential for useful single-tree performance.
The selected pre-pruned tree is a useful transparent nonlinear model, but it does not overtake logistic regression.
Its PR-AUC is close to selected kNN and stronger than selected Naive Bayes by PR-AUC.
Single trees are useful mainly as the foundation for bagging, random forests, and boosting.
```

## Current source-module status

`src/telco_churn/models.py` contains reusable model utilities through Naive Bayes, including:

```text
EDAInspiredRuleClassifier
DummyClassifier factories
RidgeClassifier factory
LogisticRegression factories
HybridGaussianBernoulliNB
make_hybrid_gaussian_bernoulli_nb_classifier
make_classifier_pipeline
```

The decision-tree section kept section-specific tree factories and plotting helpers inside the notebook source rather than moving them to `src/`, because they are currently notebook-specific workflow code rather than stable shared package utilities.

## Immediate next actions before continuing modelling

1. Replace the stale status/roadmap/handoff files with the corrected versions generated after the section 08 commit.
2. Commit the corrected status files.
3. Start section 09: bagging and random forests.
4. Follow the collaborative model-section workflow in `docs/knowledge_notes/00_documentation_workflow.md`.
5. Create the bagging/random-forest knowledge note first.
6. Build notebook 09 using training-set cross-validation only.
7. User runs it locally and sends executed outputs.
8. Update notebook interpretation from observed results.
9. Write the report section after results are known.
10. Compile and check the report.

Suggested corrective commit message for these status files:

```text
Correct project status after decision trees
```

## Next modelling stage

Next section:

```text
09_bagging_and_random_forests
```

Expected workflow:

```text
1. Create or update the bagging and random forest knowledge note.
2. Explain bootstrap aggregation, variance reduction, tree decorrelation, feature subsampling, and out-of-bag intuition.
3. Build notebook 09 using training-set CV only.
4. Evaluate bagged trees and random forests.
5. Compare against the selected single decision tree and earlier model families.
6. Save tables and figures.
7. Write report section after actual outputs are known.
```

Important evaluation language for section 09:

```text
Use ordinary stratified CV for section-level development.
Do not claim final superiority.
Treat ensemble hyperparameters as selected within a development grid.
Reserve final model-family comparison for later.
```

## Files that should not carry live next-step information

Do not use these as the main live task list:

```text
00_documentation_workflow.md
01_model_inventory_and_roadmap.md
current_notebook_documentation_audit.md
model knowledge notes
methodology knowledge notes
report sections
```

Use this file and the newest context handoff instead.

## Strict final test-set policy clarification

The final test set should be used for exactly one frozen final model.

Do not use the test set to compare multiple candidate models, additional candidate models, alternative thresholds, alternative calibration methods, or alternative preprocessing decisions. All model-family comparison, repeated CV, nested CV, statistical tests, paired bootstrap differences, McNemar-style comparisons, DeLong-style comparisons, threshold selection, calibration selection, and ablation decisions should happen before final test evaluation using training-only validation evidence.

After one final model is selected and frozen, evaluate that model once on the untouched test set. Bootstrap confidence intervals may be reported for that single final model's test metrics.
