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

## Latest known GitHub state

Latest confirmed GitHub commit:

```text
b85658e4bcfddfe7f2255f9f4f42324209a09227
Revise naive Bayes section with hybrid model
```

At that commit, the project includes completed sections through:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
06_k_nearest_neighbours
07_naive_bayes
```

The latest committed repository does not yet necessarily include the documentation cleanup and report methodology rewrite prepared after this commit. Check local files before committing.

## Current uncommitted/prepared changes from the current chat

The current chat prepared several files that should be added or used to replace existing files.

### New methodology knowledge notes

Place these in:

```text
docs/knowledge_notes/methodology/
```

Files prepared:

```text
evaluation_foundations.md
cross_validation_and_model_selection.md
statistical_uncertainty_and_tests.md
final_model_comparison_plan.md
```

Purpose:

```text
- explain true metric versus sample metric
- explain sampling uncertainty
- explain train/validation/test discipline
- explain leakage
- explain CV, repeated CV, nested CV
- explain hyperparameter tuning and selection optimism
- explain statistical uncertainty and tests
- define the final model comparison plan
```

### Report methodology rewrite

Files prepared:

```text
reports/latex/main.tex
reports/latex/sections/04_statistical_evaluation_methodology.tex
reports/latex/sections/05_linear_classification_and_logistic_regression.tex
reports/latex/sections/06_k_nearest_neighbours.tex
reports/latex/sections/07_naive_bayes.tex
```

Purpose:

```text
- add a new statistical evaluation methodology section before the model sections
- update the abstract to state that section-level CV results are development-stage estimates
- revise sections 05, 06, and 07 so close differences and tuned model choices are interpreted cautiously
- clarify that final test evaluation is deferred
- clarify that threshold curves are diagnostic, not final threshold choices
```

### Documentation cleanup prepared now

This cleanup adds or replaces:

```text
docs/knowledge_notes/00_documentation_workflow.md
docs/knowledge_notes/01_model_inventory_and_roadmap.md
docs/knowledge_notes/current_project_status_and_next_actions.md
docs/knowledge_notes/current_notebook_documentation_audit.md
docs/knowledge_notes/context_history/telco_churn_chat_handoff_context_3.md
```

Purpose:

```text
- remove stale "next task" content from the wrong files
- make file roles explicit
- create a single tactical next-actions file
- create a standalone handoff for a new chat
```

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
complete and committed at latest GitHub state
report rewrite prepared afterward
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
complete and committed at latest GitHub state
report rewrite prepared afterward
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
complete and committed at latest GitHub state
report rewrite prepared afterward
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

1. Add the four methodology knowledge notes to `docs/knowledge_notes/methodology/`.
2. Add the report methodology rewrite files.
3. Add the documentation cleanup files.
4. Compile `reports/latex/main.tex`.
5. Inspect the compiled PDF.
6. Fix any formatting, table, or wording issues.
7. Commit the cleanup and methodology rewrite.
8. Start section 08: decision trees.

Suggested commit message after checks:

```text
Add statistical evaluation methodology and documentation handoff
```

## Next modelling stage after cleanup

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
