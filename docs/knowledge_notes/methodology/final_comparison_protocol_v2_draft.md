# Final Comparison Protocol V2 Draft: Official Base-Model Comparison

## Status

This document is a protocol-v2 draft. It is not the frozen master protocol, does not
master-admit any candidate, does not select a model, and does not authorize held-out
test evaluation.

Current phase:

```text
project phase:
    pre-master protocol-v2 freeze preparation

implemented and admission-smoke-passed candidates:
    C01 through C26

deferred candidates:
    C27_TABPFN
    C28_AUTOGLUON

master-admitted candidates:
    none

protocol v2:
    not frozen

held-out test set:
    untouched
```

The warning-clean admission smoke established implementation readiness only. It is not
model-selection evidence. The paused representative search-budget calibration run is
runtime evidence only. It is not candidate-ranking, candidate-selection, or
candidate-elimination evidence.

## Official Base-Comparison Draft

The official base comparison should evaluate complete candidate procedures on
development data only:

```text
data scope:
    all 5,634 development rows only

candidate universe:
    C01 through C26

deferred from this base comparison:
    C27_TABPFN
    C28_AUTOGLUON

primary ranking metric:
    average_precision
```

C27 TabPFN remains deferred because CPU practicality, model-weight/licence terms, and
reproducible resource scheduling have not been resolved for this project stage. C28
AutoGluon remains deferred because its resolver would downgrade the numerical stack,
and any future AutoML comparison must be a separate bounded end-to-end candidate
procedure.

The base comparison should persist and later summarize:

```text
primary metric:
    average_precision

secondary ranking metrics:
    ROC-AUC

probability metrics where valid:
    Brier score
    log loss

default-threshold diagnostics:
    balanced accuracy
    precision
    recall
    F1

operational evidence:
    runtime
    warnings
    failures and interruptions
    selected-parameter stability
    Stage-A and Stage-B search behavior
```

The read-only final-comparison analysis scaffold added in commit
`441f331 Add read-only final-comparison analysis scaffold` should be used after runs to
summarize completed task artifacts. It does not select winners and must not reinterpret
admission smoke or paused calibration as model-selection evidence.

## Base-Comparison Design

The practical protocol-v2 draft should use repeated nested cross-validation on the
development set:

```text
outer evaluation:
    5 stratified folds x 3 repeats

inner search:
    two-stage HPO retained

Stage A:
    3-fold persistent Optuna exploration

Stage B:
    independent confirmation of the top Stage-A configurations
```

The earlier protocol-v1 design of 5 folds x 10 repeats remains a useful reference, but
it is too expensive for the current C01-C26 universe unless a later explicit resource
decision justifies it. The 5 x 3 design is the current practical draft because it keeps
paired repeated outer evidence while avoiding an unnecessarily long first official base
comparison.

Protocol v2 should not use one universal full search budget. Search budgets should be
candidate-specific and frozen before any official base-comparison results are
inspected.

Draft budget lanes:

```text
cheap lane:
    linear, simple, and fast classical procedures

medium lane:
    tree ensembles, most boosting libraries, RBF SVM, and MLP where runtime is moderate

expensive lane:
    CatBoost and advanced neural tabular candidates
```

The exact lane assignments, Stage-A trial counts, Stage-B top-K values, and any
candidate-specific caps must be recorded before official base-comparison execution.
Changing budgets after seeing official base-comparison results would create a new
protocol version.

## CatBoost Runtime Policy

C19 CatBoost remains implemented and admission-smoke-passed. It must not be silently
removed.

The paused representative search-budget calibration run
`search_budget_calibration_v1_warning_clean` provides runtime evidence:

```text
observed runtime issue:
    C19_CATBOOST Stage-A trials took multiple minutes each

pause state:
    clean pause during C19_CATBOOST

interpretation:
    runtime evidence only
    not model-selection evidence
    not candidate-elimination evidence
```

Protocol v2 should treat C19 as a runtime-limited candidate. A practical draft treatment
is:

```text
C19 CatBoost lane:
    expensive

Stage A:
    smaller candidate-specific trial count than cheap and medium lanes

Stage B:
    smaller top-K confirmation depth

search profile:
    capped CatBoost iterations and depth

resource policy:
    single native worker per outer task
```

This preserves CatBoost as a serious base candidate while making the official
comparison executable. If CatBoost later exceeds a predeclared runtime limit under the
frozen protocol, that outcome should be recorded as a runtime-limited result rather than
converted into silent exclusion.

## Downstream Selection Route

After the frozen base comparison completes, the project should proceed in stages:

```text
1. Summarize base-candidate performance, runtime, warnings, failures, and selected
   hyperparameter stability using read-only completed artifacts.

2. Define a leading set using training-only evidence, practical-equivalence reasoning,
   uncertainty, stability, complexity, runtime, and warning behavior.

3. Run statistical uncertainty and paired-comparison summaries for the leading group.

4. Run calibration and threshold studies only for leading ranking candidates.

5. Inspect whether leading candidates make complementary errors.

6. Evaluate stacking, blending, or soft voting only if justified by leakage-safe
   out-of-fold predictions.

7. Treat any stack, blend, or vote as its own candidate procedure.

8. Choose one final family or one final stack using training-only evidence.

9. Rerun only the winning procedure's frozen search on all 5,634 development rows.

10. Fit the complete frozen final pipeline on development data.

11. Evaluate that one frozen pipeline once on the held-out test set.
```

Calibration, threshold selection, and stacking are model-selection decisions. They must
remain training-only until one complete final procedure is frozen.

## Freeze Checklist

Before protocol v2 becomes frozen, record:

```text
candidate registry:
    C01-C26 included
    C27-C28 deferred with rationale

feature policies:
    F0/F1/F2 compatibility and any F2 final decision

feature selection:
    S0/S1/S2 compatibility

imbalance policies:
    I0-I4 compatibility

outer split design:
    fold count, repeat count, seed policy

inner search:
    Stage-A folds and candidate-specific trial budgets
    Stage-B folds and candidate-specific top-K

runtime policy:
    candidate lanes
    CatBoost runtime-limited treatment
    native-thread limits

analysis policy:
    read-only completed-artifact summaries
    no use of admission smoke or paused calibration for model selection

downstream policy:
    leading-set criteria
    uncertainty, calibration, threshold, and stacking route
    one final held-out test evaluation only after final procedure freeze
```
