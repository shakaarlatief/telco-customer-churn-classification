# Current Notebook and Documentation Audit

## Purpose

This file records the current audit of notebook and documentation roles. It is a review snapshot, not the main live task list.

For immediate next actions, use:

```text
docs/knowledge_notes/current_project_status_and_next_actions.md
```

For new-chat continuation, use the newest file in:

```text
docs/knowledge_notes/context_history/
```

## Audit decision

The project structure is good, but the documentation roles needed cleanup.

The main issue found was that several files had started to mix stable workflow rules, strategic roadmap content, historical audit notes, and immediate next actions. This made it harder to know which file should be updated when the project moves forward.

The corrected organization is:

```text
00_documentation_workflow.md:
    stable documentation roles and workflow rules

01_model_inventory_and_roadmap.md:
    strategic modelling roadmap

current_project_status_and_next_actions.md:
    immediate operational status and next actions

current_notebook_documentation_audit.md:
    audit snapshot

context_history/:
    standalone handoff files for continuing in new chats
```

## Notebook role review

### 01_raw_data_audit

Status:

```text
keep
```

Reason:

```text
- narrow purpose
- raw schema and data-quality inspection only
- no target-based EDA before splitting
- appropriate for professional workflow
```

### 02_cleaning_and_splitting

Status:

```text
keep
```

Reason:

```text
- focused on deterministic cleaning and train/test split
- separates modelling table construction from later EDA and modelling
- defines feature groups without premature model-specific preprocessing
```

### 03_training_set_eda

Status:

```text
keep
```

Reason:

```text
- EDA notebooks can be longer because they generate many tables and figures
- uses training set only
- interpretation sections are useful
- report-ready figure style is centralized and reusable
```

### 04_preprocessing_and_simple_baselines

Status:

```text
keep
```

Reason:

```text
- first evaluation notebook
- establishes preprocessing, CV, metrics, confusion matrix, threshold logic, calibration introduction, and baselines
- future notebooks should not repeat all metric theory in full
```

### 05_linear_classification_and_logistic_regression

Status:

```text
keep
```

Reason:

```text
- first learned-model notebook
- contains useful mathematical and practical explanation
- produces report tables/figures
- coefficient interpretation is important for a reusable classification reference project
```

### 06_knn

Status:

```text
keep
```

Reason:

```text
- introduces non-parametric local learning
- grid-search behaviour is useful for learning bias-variance effects
- threshold and curve outputs match the established evaluation workflow
```

### 07_naive_bayes

Status:

```text
keep
```

Reason:

```text
- introduces Bayes classifier logic, Bayes rule, generative classification, conditional independence, smoothing, and mixed-feature likelihoods
- includes the custom hybrid Gaussian-Bernoulli Naive Bayes estimator
- results now include the theoretically cleaner hybrid model
```

## Notebook length policy

Notebooks may contain explanation and interpretation, but they should not become giant textbooks.

Use this split:

```text
knowledge notes:
    deepest reusable theory and mathematics

notebooks:
    executable workflow, concise explanation, output generation, result interpretation

LaTeX report:
    polished standalone explanation, selected math, tables, figures, and interpretation
```

A notebook section can refer conceptually to the methodology or model knowledge notes rather than repeating every derivation.

## Documentation role review

### Good structure

The corrected documentation structure is:

```text
docs/knowledge_notes/
    00_documentation_workflow.md
    01_model_inventory_and_roadmap.md
    current_project_status_and_next_actions.md
    current_notebook_documentation_audit.md

    context_history/
        telco_churn_chat_handoff_context_2.md
        telco_churn_chat_handoff_context_3.md

    methodology/
        evaluation_foundations.md
        cross_validation_and_model_selection.md
        statistical_uncertainty_and_tests.md
        final_model_comparison_plan.md
        hyperparameter_tuning.md

    models/
        05_linear_classification_and_logistic_regression.md
        06_knn.md
        07_naive_bayes.md
```

### Files that were stale

Before cleanup:

```text
00_documentation_workflow.md:
    contained stale near-term tasks from before section 05

01_model_inventory_and_roadmap.md:
    said completed stages only through section 04

current_notebook_documentation_audit.md:
    contained an outdated recommended documentation structure and "going forward" steps

telco_churn_chat_handoff_context_2.md:
    still described the project as being around section 06 in some later parts
```

The cleanup should make these files consistent with the current state.

## Report role review

The report can be long. The important requirement is that it remains polished, structured, and professionally written.

The report should now include:

```text
- data audit
- cleaning and splitting
- EDA
- statistical evaluation methodology
- preprocessing and simple baselines
- linear classification and logistic regression
- kNN
- Naive Bayes
```

The statistical evaluation methodology section is important because it explains how to interpret later model comparisons.

## Current methodology audit

The project now explicitly distinguishes:

```text
true population metric:
    unobserved model quality under p(x, y)

sample metric:
    finite-sample estimate

cross-validation score:
    development-stage training-set estimate

selected CV score:
    estimate after model/hyperparameter selection, possibly mildly optimistic

test-set score:
    final estimate after all choices are fixed
```

This is a major improvement and should guide all future model sections.

## Future notebook guidance

For each future model section:

```text
1. Write or update a model knowledge note.
2. Build the notebook using training-set CV only.
3. Keep preprocessing inside the pipeline.
4. Save tables and figures.
5. Interpret close hyperparameter differences cautiously.
6. Write the report section.
7. Update current_project_status_and_next_actions.md and the newest handoff if needed.
```

## Future report wording guidance

Use:

```text
development-stage cross-validated estimate
selected within the tried grid
representative strong candidate
small differences should be interpreted cautiously
final test evaluation is deferred
```

Avoid:

```text
definitively best
uniquely optimal
proves superiority
final performance
```

## Audit conclusion

The project can continue after the documentation cleanup and report methodology rewrite are added, compiled, checked, and committed.

The next modelling section should be decision trees.
