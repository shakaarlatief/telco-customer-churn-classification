# Documentation Workflow

## Purpose

This document defines the roles of the project documentation files. It should stay relatively stable. It should not be used as a running task list.

The Telco Customer Churn project has two goals:

1. build a professional, portfolio-ready churn-classification project;
2. preserve and deepen machine-learning knowledge through standalone mathematical explanations, careful implementation, and statistically responsible interpretation.

The repository therefore separates reusable theory, executable workflows, polished reporting, and current project status.

## Core rule

Each file type should have one clear job.

```text
stable workflow rules:
    00_documentation_workflow.md

strategic modelling roadmap:
    01_model_inventory_and_roadmap.md

current tactical status and next actions:
    current_project_status_and_next_actions.md

historical notebook/documentation audit:
    current_notebook_documentation_audit.md

chat handoff memory:
    context_history/

deep reusable theory:
    methodology/ and models/ knowledge notes

executable workflow:
    notebooks/

reusable implementation:
    src/telco_churn/

polished portfolio report:
    reports/latex/
```

When a file starts mixing these roles, move the content to the appropriate place instead of letting all files become informal task lists.

## 1. Roadmap and inventory documents

Roadmap files describe the strategic modelling plan.

Current roadmap file:

```text
docs/knowledge_notes/01_model_inventory_and_roadmap.md
```

It should contain:

```text
- project goal and modelling philosophy
- completed modelling stages
- planned model-family sequence
- which model families are relevant for this dataset
- which methods are deferred to later projects
- high-level methodology milestones
- high-level future sections
```

It may contain strategic next stages, but it should not become a day-to-day task list.

It should not contain:

```text
- long mathematical derivations
- raw result dumps
- copy-paste notebook outputs
- temporary instructions for the next chat
- local compile troubleshooting
```

## 2. Current project status and next actions

The current status file is the only normal place for tactical next steps.

Current status file:

```text
docs/knowledge_notes/current_project_status_and_next_actions.md
```

It should contain:

```text
- latest known committed state
- uncommitted or prepared changes
- immediate next actions
- short commit checklist
- what to do before continuing modelling
- short risk/cleanup notes
```

It should be updated whenever a new chat handoff is created or when the immediate next action changes.

It should not contain:

```text
- full theory
- complete model notes
- long report prose
- full result tables unless needed for a checklist
```

## 3. Historical notebook and documentation audits

Audit files record a snapshot of a review.

Current audit file:

```text
docs/knowledge_notes/current_notebook_documentation_audit.md
```

Despite the word "current" in its name, this file should be treated as an audit snapshot. It should not be the main live task list.

It should contain:

```text
- review of notebook/report/documentation quality at the time of the audit
- what was judged acceptable
- what cleanup was recommended
- whether notebooks are too long, too short, or appropriately balanced
```

It should not contain the only source of truth for next steps. Put those in `current_project_status_and_next_actions.md`.

## 4. Chat handoff context files

Chat handoff files preserve conversation state when a chat becomes too long or slow.

Location:

```text
docs/knowledge_notes/context_history/
```

Recommended naming:

```text
telco_churn_chat_handoff_context_1.md
telco_churn_chat_handoff_context_2.md
telco_churn_chat_handoff_context_3.md
```

Each handoff should be standalone enough that a new chat can continue without reading every previous handoff. Newer handoffs may summarize and supersede earlier ones, but older handoffs remain useful for detailed background.

A handoff should contain:

```text
- project identity
- user preferences
- current repository state
- completed sections
- important decisions
- key numerical results
- documentation structure
- pending local/uncommitted changes
- exact next steps
```

A handoff should not contain every minor conversation turn. It should preserve decisions and project state.

## 5. Knowledge notes

Knowledge notes are deep reusable technical references. They should be understandable outside the immediate chat.

Locations:

```text
docs/knowledge_notes/methodology/
docs/knowledge_notes/models/
```

Methodology notes should cover reusable ideas such as evaluation, cross-validation, hyperparameter tuning, uncertainty, resampling, calibration, and feature selection.

Model notes should cover specific model families such as logistic regression, kNN, Naive Bayes, decision trees, ensembles, SVMs, and MLPs.

Knowledge notes should contain:

```text
- standalone theory
- mathematical definitions
- intuition
- assumptions and limitations
- preprocessing implications
- evaluation implications
- implementation plan
- report plan
```

They should not contain:

```text
- messy running notes
- raw result dumps
- local compile instructions
- chat handoff details
- day-to-day task lists
```

A knowledge note may include a short "project placement" or "implementation plan" section. That is not the same as a live task list.

## 6. Notebooks

Notebook source files and rendered notebooks are the executable workflow.

Location:

```text
notebooks/
```

The `.py` files are the source workflow files. The `.ipynb` files are kept for readability, execution, saved outputs, and sharing results.

Notebooks should contain:

```text
- concise but sufficient explanation
- code
- checks and outputs
- saved tables and figures
- interpretation of observed results
- links back conceptually to knowledge notes and report sections
```

They should not contain every detail from the knowledge notes. The notebook should be educational, but still executable and readable.

## 7. Source code modules

Source modules contain reusable project functions.

Location:

```text
src/telco_churn/
```

They should contain:

```text
- reusable data loading helpers
- preprocessing factories
- model factories
- evaluation utilities
- plotting utilities
- feature interpretation utilities
```

They should not contain report narrative. Technical docstrings are useful, but source modules should remain professional and reusable.

Model-specific helper functions can stay in a notebook when they are one-off, experimental, or section-specific. Move them into `src/` when they become reusable, stable, or part of the project-wide toolkit.

## 8. LaTeX report

The LaTeX report is the polished portfolio artifact.

Location:

```text
reports/latex/
```

It should contain:

```text
- standalone explanations
- important mathematics
- modelling decisions
- result tables and figures
- interpretation
- limitations
- statistical caution around estimates and model comparisons
- implications for later stages
```

It should not contain:

```text
- direct lecture/source-file references in the main prose
- messy planning notes
- raw notebook dumps
- implementation TODOs
```

The report can be long. Length is acceptable when the explanation is useful, mathematically clear, and professionally written.

## 9. Standard workflow for each major topic

For each major modelling or methodology topic:

```text
1. Create or update the relevant knowledge note.
2. Implement the workflow in a notebook/script.
3. Run the notebook and inspect outputs.
4. Interpret results carefully.
5. Save tables and figures.
6. Write or revise the polished LaTeX report section.
7. Update the current status file and handoff context if the chat is becoming long.
8. Commit after the section is checked.
```

The knowledge note usually comes before coding because it forces the modelling decision to be understood before implementation.

The report usually comes after coding because it should include actual results.

## 10. Collaborative model-section workflow

For model-family sections, the preferred workflow is more specific than the general topic workflow above. It should be followed unless there is a clear reason to deviate.

```text
1. Knowledge note first:
       Create or update the relevant `.md` knowledge note in
       `docs/knowledge_notes/models/` or `docs/knowledge_notes/methodology/`.

       This note should preserve reusable theory, assumptions, mathematics,
       preprocessing implications, evaluation implications, and an implementation plan.

2. Assistant prepares executable source:
       Create or update the notebook source `.py` file.

       The `.py` file is the primary editable notebook source. It should contain
       professional code, concise markdown-style explanations, saved tables/figures,
       and enough interpretation to make the executed notebook understandable.

3. User runs the notebook locally:
       The user executes the `.py` workflow locally and generates the rendered
       `.ipynb`, tables, figures, and other saved artifacts.

       This is important because the assistant may not have access to the local
       environment, data paths, package versions, or generated outputs.

4. User sends the executed outputs back:
       The user sends the executed `.ipynb` and relevant generated files, such as
       CSV tables, PNG figures, logs, or screenshots.

       These outputs are treated as the observed results. The assistant should not
       invent results that were not produced.

5. Assistant updates interpretation:
       After seeing the actual results, update the `.py` source and, when useful,
       the rendered `.ipynb` interpretation.

       This step should add result-specific interpretation, fix unclear wording,
       adjust plots/tables if needed, and make sure the notebook remains professional.

6. Assistant writes or revises the LaTeX report:
       Only after the executed results are known, write or update the polished
       LaTeX report section.

       The report should include selected mathematics, modelling decisions,
       result tables/figures, interpretation, limitations, and statistically careful
       language around development-stage estimates.

7. User compiles/checks the report:
       The user compiles the LaTeX report locally and sends the PDF, screenshots,
       or errors if review is needed.

8. Fix and commit:
       Fix formatting, wording, code, or interpretation issues.
       Commit only after the section is checked.
```

This workflow separates theory, execution, observed results, and polished reporting. The assistant should not write final report claims before seeing the actual executed results. If a file may have changed locally and the assistant does not have the current version, ask the user to upload it or provide an exact copy-paste replacement rather than overwriting unknown work.


## 11. Current documentation map

The expected documentation map is:

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

Additional methodology notes can be added later for:

```text
class_imbalance_and_resampling.md
feature_selection.md
calibration.md
final_test_evaluation_template.md
```

## 12. Where to put "what should we do next?"

Use this rule:

```text
Strategic future modelling sequence:
    01_model_inventory_and_roadmap.md

Immediate next actions:
    current_project_status_and_next_actions.md

New chat continuation memory:
    context_history/telco_churn_chat_handoff_context_X.md

Deep theory to preserve:
    methodology/ or models/ knowledge notes

Polished public explanation:
    reports/latex/

Executable next code:
    notebooks/
```

This prevents the project from becoming messy as the chat and documentation grow.
