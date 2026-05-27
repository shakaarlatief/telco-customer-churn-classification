# Documentation Workflow

## Purpose

This document defines how the project documentation should be organized from this point onward.

The project has two goals:

1. build a professional portfolio-ready churn-classification project;
2. preserve and deepen machine-learning knowledge by explaining the theory, mathematics, implementation choices, and interpretation behind each modelling step.

To keep the project organized, the repository uses several document types. Each type has a different purpose.

## 1. Roadmap and inventory documents

Roadmap files describe the full modelling plan.

They should contain:

```text
- model families to cover
- intended modelling order
- which models are suitable for this dataset
- which models are saved for future projects
- high-level preprocessing expectations
- how each future section fits into the project
```

They should not contain:

```text
- full mathematical derivations
- result interpretation
- long implementation details
```

Current roadmap file:

```text
docs/knowledge_notes/01_model_inventory_and_roadmap.md
```

## 2. Knowledge notes

Knowledge notes are deep technical references written before implementation.

They should contain:

```text
- standalone theory
- mathematical definitions
- intuition
- assumptions and limitations
- preprocessing requirements
- evaluation implications
- implementation plan
- what should later appear in the report
```

They should not contain:

```text
- messy running notes
- direct lecture/source-file references
- raw result dumps
- TODOs that belong in code
```

Model knowledge notes belong in:

```text
docs/knowledge_notes/models/
```

Methodology knowledge notes belong in:

```text
docs/knowledge_notes/methodology/
```

## 3. Notebooks

Notebooks are the executable workflow.

They should contain:

```text
- enough theory to understand the workflow
- code
- checks and outputs
- tables and figures
- interpretation of observed results
- saved artifacts
```

They should not contain every detail from the knowledge notes. The notebook should be educational, but still executable and readable.

Notebook files belong in:

```text
notebooks/
```

## 4. Source code modules

Source code modules contain reusable project functions.

They should contain:

```text
- reusable data loading helpers
- preprocessing factories
- model factories
- evaluation utilities
- plotting utilities
- feature interpretation utilities
```

They should not contain project narrative or report-style explanation. Technical docstrings are useful, but source modules should remain reusable and professional.

Source code belongs in:

```text
src/telco_churn/
```

## 5. LaTeX report

The LaTeX report is the polished portfolio artifact.

It should contain:

```text
- standalone explanations
- important mathematics
- modelling decisions
- result tables and figures
- interpretation
- limitations
- implications for later stages
```

It should not contain:

```text
- lecture references
- messy planning notes
- raw notebook dumps
- implementation TODOs
```

Report files belong in:

```text
reports/latex/
```

## 6. Standard workflow for each major topic

For each major modelling or methodology topic, use the following workflow:

```text
1. Create or update a knowledge note.
2. Implement the workflow in a notebook/script.
3. Run the notebook and inspect outputs.
4. Interpret results carefully.
5. Write the polished LaTeX report section.
6. Optionally update the knowledge note with short project lessons.
```

The knowledge note comes before coding because it forces the modelling decision to be understood before implementation.

The report comes after coding because it should include the actual results.

## 7. Current documentation roles

Current files:

```text
docs/knowledge_notes/01_model_inventory_and_roadmap.md
```

Role:

```text
Project-wide modelling roadmap and inventory.
```

```text
docs/knowledge_notes/methodology/hyperparameter_tuning.md
```

Role:

```text
Reusable methodology note for tuning, validation discipline, search strategies, threshold tuning, and project tuning policy.
```

```text
docs/knowledge_notes/models/05_linear_classification_and_logistic_regression.md
```

Role:

```text
Deep model note for section 05, including linear classifiers, least-squares classification, logistic regression, log loss, regularization, preprocessing, evaluation, coefficients, and threshold behaviour.
```

## 8. Near-term documentation tasks

Before implementing section 05, the project should have:

```text
docs/knowledge_notes/00_documentation_workflow.md
docs/knowledge_notes/01_model_inventory_and_roadmap.md
docs/knowledge_notes/methodology/hyperparameter_tuning.md
docs/knowledge_notes/models/05_linear_classification_and_logistic_regression.md
```

Later, create:

```text
docs/knowledge_notes/methodology/evaluation_metrics.md
docs/knowledge_notes/methodology/class_imbalance_and_resampling.md
docs/knowledge_notes/methodology/model_comparison_uncertainty.md
```

The evaluation metrics note can be created retroactively from section 04 because that section already contains the core evaluation theory.
