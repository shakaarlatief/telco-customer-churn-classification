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
4d6a61991110dde65b8720197d3c18a2be982f3b
Enforce single final test model evaluation policy
```

This commit includes the documentation cleanup, the statistical evaluation methodology knowledge notes, the report methodology rewrite, and the strict single-final-test-model policy.

When starting from this file in a later chat, first check GitHub for newer commits. Treat this commit as the latest confirmed checkpoint only as of the time this status file was written.

At this commit, the project includes completed and committed sections through:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_statistical_evaluation_methodology
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
06_k_nearest_neighbours
07_naive_bayes
```

The four methodology knowledge notes are committed:

```text
docs/knowledge_notes/methodology/evaluation_foundations.md
docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
docs/knowledge_notes/methodology/final_model_comparison_plan.md
```

There are no known prepared methodology/report cleanup files left to add from the previous chat. The next modelling step is section 08: decision trees.

## Important methodological decision made before continuing

Before continuing to decision trees, the project paused to improve statistical evaluation methodology.

The central decision:

```text
Section-level cross-validation results are development-stage estimates.
They are useful for model learning, tuning, and candidate comparison.
They are not final performance claims.
```

This affects how previous and future model sections should be written.

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

### Section 05: linear classification and logistic regression

Status:

```text
complete and committed at latest confirmed GitHub state
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
complete and committed at latest confirmed GitHub state
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
complete and committed at latest confirmed GitHub state
```

Important change already committed:

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
Logistic regression remains the strongest model family so far by development-stage PR-AUC and ROC-AUC.
```

## Current source-module status

`src/telco_churn/models.py` contains:

```text
EDAInspiredRuleClassifier
DummyClassifier factories
RidgeClassifier factory
LogisticRegression factories
HybridGaussianBernoulliNB
make_hybrid_gaussian_bernoulli_nb_classifier
make_classifier_pipeline
```

Reason for adding hybrid NB to `src/`:

```text
It is a reusable custom estimator, not just notebook glue.
It combines Gaussian numeric likelihoods with Bernoulli one-hot categorical likelihoods.
```

Notebook-local helper functions can remain notebook-local when they are section-specific and not reusable.

## Immediate next actions before continuing modelling

1. Start section 08: decision trees.
2. Create or update the decision-tree knowledge note.
3. Build notebook 08 using training-set cross-validation only.
4. Evaluate a stump, default tree, tuned tree, and possibly cost-complexity-pruned tree.
5. Save tables and figures.
6. Write the report section with development-stage wording.
7. Compile the report.
8. Commit after the section is checked.

Suggested future commit message after section 08 is complete:

```text
Add decision tree modelling section
```

## Next modelling stage

Next section:

```text
08_decision_trees
```

Expected workflow:

```text
1. Create/update decision tree knowledge note.
2. Explain recursive partitioning, impurity, entropy, Gini, overfitting, max depth, min samples leaf, pruning/cost-complexity.
3. Build notebook 08 using training-set CV only.
4. Evaluate simple tree, stump, tuned tree, and possibly pruned tree.
5. Save tables/figures.
6. Write report section.
```

Important evaluation language for section 08:

```text
Use ordinary stratified CV for development.
Do not claim final superiority.
Treat depth/pruning choices as selected within a development grid.
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
