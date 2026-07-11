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

candidate-universe and admission status:
    02_candidate_status_register.md

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

## Collaboration and repository-write boundary

The default collaboration model is deliberately local-first and user-controlled.

```text
Assistant default:
    prepare new or revised files as downloadable artifacts in the chat;
    explain every meaningful change and the intended destination;
    do not directly modify the repository or Git hosting service.

User default:
    review the delivered artifacts;
    place or replace files in the local repository;
    run local checks and workflows;
    inspect `git diff`;
    stage, commit, and push changes.
```

The assistant must not create, update, delete, stage, commit, or push repository files by default. This includes direct writes through connected Git-hosting tools.

The assistant may recommend a direct repository write when it would be useful, but it must first ask for explicit approval. The user may give a clear green light for a specific action, such as:

```text
Yes, update this one file on GitHub.
Yes, commit the prepared changes.
Yes, push this commit to main.
```

Approval for one write action does not imply standing permission for later repository writes. If the requested write scope is ambiguous, the assistant must ask before acting.

Remote Git state and the user's local working tree must be treated as potentially different. Before proposing a replacement for a locally changed file, inspect the available current version or ask the user to provide it.

In shared Codex workspace sessions, the user may explicitly ask the assistant to edit repository files directly. In that mode, direct edits are allowed only within the approved scope, and the assistant must still avoid staging, committing, pushing, deleting unrelated work, or modifying artifacts unless the user explicitly requests those actions. Always inspect `git status --short` before and after substantial repository edits, and preserve untracked local artifact trees unless a deliberate artifact-publication policy says otherwise.

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

## 1.1 Candidate universe and admission register

Candidate inclusion is a governance decision that should not be inferred from an incomplete
implementation registry or from pilot metric values. The candidate-status register records:

```text
- the documented candidate universe;
- which candidates are implemented and runnable;
- which candidates remain pending conventional implementation;
- which advanced candidates require package, licence, hardware, reproducibility,
  preprocessing, and resume admission checks;
- which candidates are formally admitted to a frozen master protocol;
- technical exclusions and their recorded reasons.
```

Current register:

```text
docs/knowledge_notes/02_candidate_status_register.md
```

The register is not a day-to-day task list, a score table, or a final-ranking document. Use
the current status file for immediate actions and the final-comparison protocol version for
the frozen master design.

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


### Reusable implementation and smoke-test requirements

Before designing a substantial new model-family notebook, first inspect the existing reusable modules and workflow tests:

```text
src/telco_churn/
scripts/
previous notebook source files
```

Use the following boundary:

```text
Reusable preprocessing, estimator construction, evaluation, plotting, and interpretation helpers:
    implement or extend in src/telco_churn/

One-off experiment-specific grid definitions, result narrative, and model-family interpretation:
    keep in the notebook source
```

Do not duplicate a stable project-wide factory inside a notebook merely because a new workflow needs it. If a needed helper already exists, reuse it. If a helper is expected to be reused by smoke tests, later notebooks, or final model selection, create or extend it in `src/` first.

Every new or materially changed reusable implementation must have a corresponding smoke test in `scripts/`, normally named:

```text
scripts/smoke_test_<workflow_name>.py
```

The smoke test should be small, fast, training-only, and deterministic where possible. It should validate the same shared factories and utilities used by the full notebook. Depending on the workflow, it should check the following where relevant:

```text
- import and construction of reusable factories;
- preprocessing output shape and expected dense/sparse representation;
- a small stratified training-only split or cross-validation run;
- prediction or score shape, finite values, and class alignment;
- out-of-fold helper paths and threshold/calibration primitives when used;
- generated plot or table paths when a reusable plotting helper changed.
```

Run the smoke test successfully after the shared source changes and before executing the full notebook. A smoke test is not a substitute for the full workflow, but it prevents avoidable long-run failures and verifies that the notebook uses the intended reusable implementation.

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


### Mandatory implementation gate before full notebook execution

For a new model-family workflow that adds or changes reusable preprocessing, estimator factories, evaluation helpers, visualization helpers, or scripts:

```text
1. Inspect the relevant current src/ modules, scripts/, and comparable earlier notebooks.
2. Decide explicitly which logic is reusable and therefore belongs in src/.
3. Implement or revise the shared source functions first.
4. Create or revise the matching smoke test in scripts/.
5. The user runs the smoke test locally and reviews its output.
6. Only then run the complete notebook workflow.
```

If a notebook deliberately contains one-off helper logic instead of a src/ factory, the notebook should explain why the helper is local and why it is not intended for reuse.

## 11. Stable documentation directory conventions

This section records stable directory conventions rather than a literal inventory of every
currently existing knowledge note, handoff, or generated model document. A literal inventory
would become stale whenever a model family or handoff file is added.

The stable documentation layout is:

```text
docs/knowledge_notes/
    00_documentation_workflow.md
    01_model_inventory_and_roadmap.md
    02_candidate_status_register.md
    current_project_status_and_next_actions.md
    current_notebook_documentation_audit.md

    context_history/
        telco_churn_chat_handoff_context_<number>.md

    methodology/
        <methodology_topic>.md

    models/
        <section_number>_<model_family>.md
        figures/
```

Use the live coordination documents for current inventory and progress:

```text
completed and remaining model-family inventory:
    01_model_inventory_and_roadmap.md

candidate implementation, admission, exclusion, and master-freeze state:
    02_candidate_status_register.md

immediate current work:
    current_project_status_and_next_actions.md

new-chat continuation state:
    newest file in context_history/
```

## 12. Where to put "what should we do next?"

Use this rule:

```text
Strategic future modelling sequence:
    01_model_inventory_and_roadmap.md

Candidate implementation, admission, technical exclusion, and master-freeze state:
    02_candidate_status_register.md

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
