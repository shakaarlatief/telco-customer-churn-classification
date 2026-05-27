# Current Notebook and Documentation Audit

## Decision

The current project structure is good enough to continue after a small cleanup.

The main adjustment is to separate reusable theory from executable notebooks:

```text
knowledge_notes = deep reusable explanations
notebooks = executable workflow with concise but sufficient explanation
LaTeX report = polished public explanation with results
```

## Notebook review

### 01_raw_data_audit

Status: keep.

Reason:

- The notebook has a narrow purpose.
- It avoids target-based EDA before splitting.
- It explains why raw auditing is allowed before splitting.
- It is professional and not unnecessarily broad.

### 02_cleaning_and_splitting

Status: keep.

Reason:

- The notebook is focused on creating the modelling table and train/test split.
- It clearly separates deterministic dataset construction from later EDA and modelling.
- It defines semantic feature groups without performing model-specific preprocessing too early.

### 03_training_set_eda

Status: keep.

Reason:

- EDA notebooks are naturally longer because they create many tables and figures.
- The notebook correctly uses only the training set.
- The interpretation sections are useful and should remain.
- The report-ready figure style is centralized and should be reused later.

### 04_preprocessing_and_simple_baselines

Status: keep with small wording cleanup.

Reason:

- This is the first evaluation notebook, so more explanation is justified.
- It establishes cross-validation, confusion-matrix logic, metrics, thresholds, calibration, and baseline interpretation.
- Future notebooks should not repeat all this theory in full. They should refer back to section 04 and the evaluation knowledge note.

Small cleanup:

- Replace "lecture order" wording with "positive-first order".
- Use the new evaluation knowledge note as the reusable reference.

## Documentation review

Current recommended structure:

```text
docs/knowledge_notes/
  00_documentation_workflow.md
  01_model_inventory_and_roadmap.md
  methodology/
    evaluation_metrics.md
    hyperparameter_tuning.md
  models/
    05_linear_classification_and_logistic_regression.md
```

## Going forward

For each new topic:

```text
1. Write or update the relevant knowledge note.
2. Implement the notebook.
3. Run and inspect outputs.
4. Interpret results.
5. Write the LaTeX report section.
6. Update knowledge notes only if the project introduced reusable new concepts.
```

Future notebooks should be professional and self-contained, but not overloaded with full theory. Deep theory belongs in knowledge notes and the report.
