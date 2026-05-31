# Current Project Status and Next Actions

## Purpose

This file is the tactical status file for the Telco Customer Churn classification project.

Use it to answer:

```text
Where are we now?
What has been prepared or completed?
What should happen before continuing modelling?
What is the immediate next step?
```

This file is intentionally shorter and more operational than the knowledge notes and the report.

## Latest project state at time of this update

Section 08, decision trees, has been completed in the collaborative workflow.

The following files were prepared for section 08:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
```

The report was compiled locally after adding:

```latex
\newpage
\input{sections/08_decision_trees}
```

inside `reports/latex/main.tex` after the Naive Bayes section.

The compiled report includes the decision-tree section as report Section 9, with subsections from recursive partitioning through the decision-tree summary.

If this file is read in a later chat, first check GitHub for newer commits and confirm whether the section 08 files were committed and pushed. This file records the state after the local section 08 workflow was completed in the chat, not a guaranteed remote commit hash.

Completed model/report stages now include:

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

## Important methodological policy still active

Section-level cross-validation results are development-stage estimates.

They are useful for:

```text
model learning
tuning inside a model family
candidate comparison
understanding bias-variance behaviour
selecting representative configurations within tried grids
```

They are not:

```text
final test-set performance claims
proof that a small metric difference is population-level superiority
proof that a hyperparameter value is uniquely optimal
```

Preferred language remains:

```text
selected within the development grid
strongest configuration in this tried grid
development-stage cross-validated estimate
representative candidate
small differences should be interpreted cautiously
final test evaluation is deferred
```

The held-out test set remains unused and must remain unused until one final model, threshold rule, calibration decision, and preprocessing strategy are frozen.

## Current status of model sections

### Section 04: preprocessing, evaluation, and simple baselines

Status:

```text
complete
```

Key role:

```text
establishes binary classification metrics, positive-first confusion matrix,
preprocessing infrastructure, simple baselines, class imbalance logic,
thresholds, and calibration introduction
```

Main lesson:

```text
accuracy alone is misleading under churn imbalance
majority/prior baselines have high accuracy but zero recall
EDA rule has very high recall but many false positives
```

### Section 05: linear classification and logistic regression

Status:

```text
complete
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
complete
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
complete
```

Important source-code addition:

```text
HybridGaussianBernoulliNB in src/telco_churn/models.py
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
Logistic regression remains the strongest model family so far by development-stage PR-AUC and ROC-AUC.
```

### Section 08: decision trees

Status:

```text
complete locally in the section 08 workflow
commit/push status should be checked in GitHub when this file is read later
```

Files prepared:

```text
docs/knowledge_notes/models/08_decision_trees.md
notebooks/08_decision_trees.py
notebooks/08_decision_trees.ipynb
reports/latex/sections/08_decision_trees.tex
reports/latex/main.tex updated to include sections/08_decision_trees
```

Main concepts covered:

```text
recursive partitioning
internal split rules and terminal leaves
leaf churn proportions
Gini impurity
entropy and information gain
impurity reduction
decision stumps
pre-pruning
cost-complexity pruning
ranking with leaf probabilities
validation discipline for pruning and hyperparameter tuning
```

Main development result:

```text
Selected decision tree:
    variant = pre-pruned grid
    criterion = gini
    max_depth = 6
    min_samples_split = 25
    min_samples_leaf = 10
    ccp_alpha = 0

    Accuracy about 0.789
    Balanced accuracy about 0.701
    Precision about 0.624
    Recall about 0.514
    Specificity about 0.888
    F1 about 0.564
    ROC-AUC about 0.824
    PR-AUC about 0.628
```

Comparison results:

```text
Selected pre-pruned tree:
    ROC-AUC about 0.824
    PR-AUC about 0.628

Best cost-complexity-pruned tree:
    ROC-AUC about 0.822
    PR-AUC about 0.615

Decision stump:
    ROC-AUC about 0.726
    PR-AUC about 0.413
    default hard prediction detects no churners

Default unrestricted tree:
    ROC-AUC about 0.648
    PR-AUC about 0.371
```

Interpretation:

```text
The unrestricted default tree overfits badly.
Regularization is essential for single decision trees.
Pre-pruning with moderate depth and leaf-size constraints gives the strongest single-tree result in this grid.
Cost-complexity pruning substantially improves over the unrestricted tree but does not beat the best pre-pruned tree here.
Single trees learn meaningful nonlinear churn structure but remain below logistic regression in ranking metrics.
The result motivates bagging and random forests as the next stage, because ensembles can reduce the variance and instability of single trees.
```

Important decision-tree validation note:

```text
Cost-complexity pruning is treated as a tree hyperparameter and is tuned by training-set cross-validation in this section.
This is acceptable for development-stage model selection.
If a separate validation set were reserved for higher-level model-family comparison, that same validation set should not also be used to select pruning strength.
A stricter evaluation of the full tune-and-select procedure would require nested validation.
```

## Current source-module status

`src/telco_churn/models.py` contains reusable model classes and factories from earlier sections, including:

```text
EDAInspiredRuleClassifier
DummyClassifier factories
RidgeClassifier factory
LogisticRegression factories
HybridGaussianBernoulliNB
make_hybrid_gaussian_bernoulli_nb_classifier
make_classifier_pipeline
```

For section 08, decision-tree helper functions were kept notebook-local because they were section-specific workflow helpers rather than reusable custom estimators. This matches the documentation workflow: reusable stable components go in `src/`, while section-specific plotting and grid helpers can remain in the notebook source.

## Immediate next actions before continuing modelling

1. Confirm the section 08 files are in the correct paths locally.
2. Confirm the report compiles after including `sections/08_decision_trees`.
3. Commit and push the completed decision-tree section if not already pushed.
4. After the commit, optionally update this file with the exact Git commit hash.
5. Start section 09: bagging and random forests.

Suggested commit message for section 08:

```text
Add decision tree modelling section
```

## Next modelling stage

Next section:

```text
09_bagging_and_random_forests
```

Expected workflow:

```text
1. Create or update the bagging/random-forest knowledge note.
2. Use the trees and methodology slides before writing the section.
3. Explain bootstrap aggregation, variance reduction, tree instability, feature subsampling, random forests, and out-of-bag intuition.
4. Build notebook 09 using training-set cross-validation only.
5. Compare bagged trees and random forests against the selected single decision tree.
6. Save model-comparison tables, threshold diagnostics, ROC/PR curves, feature importance diagnostics, and any useful forest-size or max-feature plots.
7. User runs the notebook locally and returns executed outputs.
8. Update notebook interpretation from actual results.
9. Write the LaTeX report section.
10. Compile/check before committing.
```

Expected evaluation language for section 09:

```text
Use ordinary stratified CV for development.
Treat forest hyperparameters as selected within a development grid.
Compare against the single-tree section as development evidence.
Do not claim final test performance.
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
