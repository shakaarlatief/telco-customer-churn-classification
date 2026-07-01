# Final model selection and single-test-set evaluation plan for the Telco churn project

This note defines the project-specific evaluation plan after the individual model-family sections are complete.

The previous methodology notes explain the underlying theory:

```text
evaluation_foundations.md
cross_validation_and_model_selection.md
statistical_uncertainty_and_tests.md
final_model_selection_designs_and_candidate_comparison.md
```

This note answers a practical question:

> After we have explained, implemented, tuned, and interpreted all relevant model families, how should the project compare candidates, choose one final model, and evaluate it without overstating the evidence?

The central rule is:

> The held-out test set is used for exactly one frozen final model. It is not used to compare candidate models, additional candidate models, hyperparameter settings, thresholds, calibration choices, feature sets, or modelling strategies.

All candidate comparison, statistical testing, repeated cross-validation, nested cross-validation, paired bootstrap comparison, threshold selection, calibration selection, and ablation decisions should happen before the final test set is touched.

---

## 1. Current stage of the project

The project has completed the broad model-family learning stage and is now entering the
methodology-design stage for final candidate comparison. The completed sections are not a
final ranking. Each section established a technically meaningful implementation, explored
a development-stage search region, and produced evidence that will inform a later
training-only comparison.

Completed modelling coverage includes:

```text
baseline classifiers
regularized logistic regression
k-nearest neighbours
Naive Bayes variants
decision trees
bagging and random forests
boosting methods, including gradient boosting, XGBoost, and CatBoost
linear and RBF-kernel support vector machines
multilayer perceptrons
```

The next stage is deliberately different from an ordinary model-family notebook. It will
not introduce another unrelated learner. It will define a reproducible candidate library,
a fair comparison protocol, a method for selecting one final model, and a strict route to
a single final test-set evaluation.

The individual model sections remain important because they provide:

```text
mathematical understanding of each family
development-stage evidence about useful search regions
diagnostics for underfitting, overfitting, calibration, and threshold behaviour
initial evidence about stability, complexity, and interpretability
```

They do not by themselves establish a final winner. The later comparison stage must make
the family-selection and final-hyperparameter-selection process explicit.

## 2. What section-level CV results mean

The section-level CV tables should be interpreted as:

```text
development-stage cross-validated estimates
```

They are useful for:

```text
understanding each model family
choosing reasonable hyperparameter regions
identifying obvious underfitting or overfitting
studying precision-recall tradeoffs
deciding which models are promising enough for later comparison
```

They should not be interpreted as:

```text
final test performance
proof that one close model is truly superior
proof that one exact hyperparameter is uniquely optimal
```

This distinction matters because each section often compares several variants. The highest observed cross-validated score may contain selection optimism.

---

## 3. Recommended wording for model sections

When a model configuration wins within a section, the report should use language like:

> Within the development-stage cross-validation grid, this configuration obtains the highest mean PR-AUC.

or:

> This result suggests that this region of the hyperparameter space is preferable within the tried grid.

Avoid language like:

> This is the best model.

or:

> This hyperparameter is optimal.

or:

> This model is statistically superior.

Better language:

```text
selected for this section
representative model
strongest configuration within the development grid
promising candidate
development-stage evidence
```

This keeps the interpretation honest while still allowing clear project progress.

---

## 4. What should happen after all model-family sections

After all relevant model families have been studied, the project should add a dedicated
stage:

```text
Candidate comparison, uncertainty, and final selection
```

This stage should compare a broad, predefined candidate library using training-only
evidence. It should not silently reduce the evidence to only the strongest score from
each earlier notebook, because those results arose from different search scopes, random
seeds, fold structures, and modelling purposes.

The purpose is to answer:

```text
Which complete candidate procedures are included in the final comparison?
Which procedures have the strongest and most stable training-only evidence?
Are apparent performance gaps practically meaningful?
Which family should be selected?
How should exact final hyperparameters be selected?
Which threshold should be selected for the final decision rule?
Should probability calibration be used?
How uncertain are the candidate comparisons?
```

This stage must not use the held-out test set. It is where broad learning-stage evidence
is turned into one pre-specified, reproducible final-selection protocol.

## 5. Candidate library and candidate-procedure registry

The project is not restricted to a small finalist shortlist. Its purpose is to compare
the full set of relevant classification families in a disciplined way. The final
comparison should therefore begin with a broad candidate library that includes the
completed baseline, linear, probabilistic, tree, ensemble, kernel, and neural-network
approaches.

A library-level inventory is:

```text
regularized logistic regression
k-nearest neighbours
Naive Bayes
single decision tree
bagged trees
random forest
gradient boosting
XGBoost
CatBoost
linear SVM
RBF SVM
multilayer perceptron
```

The exact registry may contain multiple conceptually meaningful procedures within a
family, for example class-weighted versus unweighted estimation, or a calibrated versus
uncalibrated probability workflow. Such variants should be included only when they
represent a defensible modelling choice rather than cosmetic duplication.

A candidate is not one already fitted model, and it is not every individual grid point.
The correct object is a **candidate procedure**:

```text
candidate procedure =
    preprocessing recipe
    feature-engineering and feature-selection policy
    imbalance-treatment policy
    model family
    hyperparameter search space and search method
    validation design inside tuning
    seed and compute-budget policy
    primary scoring metric
    prediction-score rule
```

For threshold-dependent or probability-dependent deployment decisions, the registry must
also state whether calibration and threshold selection are outside the family comparison
or are part of the candidate procedure. That choice changes what the comparison
evaluates.

Before any new comparison result is examined, create a candidate-procedure registry with
one row per procedure. It should document:

```text
candidate identifier
model family and implementation
preprocessing and feature policy
resampling or class-weighting policy
search strategy and full search space
search budget, early-stopping rule, and failure handling
random-seed policy
primary and secondary evaluation metrics
calibration and threshold-treatment policy
reason for inclusion
```

This registry turns the comparison into a reproducible experiment. It prevents
unrecorded adaptive changes, such as expanding only the currently favoured model's search
space after observing intermediate results.

## 6. Comparison and selection criteria

The full candidate library should be evaluated using a hierarchy of criteria, rather
than selecting a final model solely because it has the largest displayed point estimate.

Important criteria include:

```text
PR-AUC / average precision:
    positive-class retrieval under class imbalance

ROC-AUC:
    overall ranking ability

score stability:
    sensitivity to validation partitioning, random seeds, and tuning variation

calibration:
    reliability of predicted probabilities, if probabilities are interpreted as risks

threshold behaviour:
    precision, recall, specificity, F1, balanced accuracy, and intervention volume
    at relevant operating points

predicted positive rate:
    operational size of the retention-intervention group

complexity and runtime:
    fitting cost, prediction cost, implementation burden, and reproducibility

interpretability:
    usefulness for understanding churn drivers and communicating the decision rule
```

PR-AUC can remain the primary development ranking metric because churn is the minority
positive class and positive-class retrieval matters. However, before the final comparison
is run, the project must verify the precise implementation and terminology used for this
quantity. In particular, if scikit-learn's `average_precision_score` is used, the report
should call it average precision or explain the relationship to the broader PR-curve
area terminology precisely.

A small observed primary-metric advantage does not automatically determine the final
choice. The final protocol should predefine how practical ties are handled. For example,
if two procedures are practically indistinguishable within a justified margin, the
decision rule may prefer the simpler, more stable, better-calibrated, or easier-to-explain
procedure.

The practical-equivalence margin must be justified before final comparison results are
used for selection. It must not be chosen after seeing which value favours a preferred
model.

## 7. Flat repeated cross-validation across the full candidate library

Flat repeated cross-validation is one valid route for choosing the final model. Under
this design, every candidate procedure is tuned and compared using the same repeated
cross-validation design:

```text
For each candidate procedure:
    run the predefined hyperparameter search using repeated CV;
    select the configuration with the best repeated-CV primary metric;
    store fold-level metrics, out-of-fold predictions, selected configuration,
    secondary metrics, and stability summaries.

Across candidate procedures:
    compare the selected repeated-CV results;
    apply the predefined practical-tie and secondary-criteria rule;
    choose one final family and one final configuration.
```

This route directly produces both the selected family and the selected hyperparameter
configuration. The winning configuration is then fitted once on all development data.

Its advantages are efficiency, directness, and a transparent final training route. Its
limitation is selection optimism: the displayed winning score has benefited from
searching over configurations and from selecting the highest-looking family among many
estimated scores. Repetition reduces sensitivity to one fold partition, but it does not
eliminate that selection effect.

The complete candidate library can be compared this way. The critical requirements are
that all candidate procedures, search spaces, search budgets, seeds, and decision rules
are frozen before the new results are examined.

## 8. Per-family nested cross-validation across the full candidate library

Per-family nested CV is the main alternative when the project prioritizes a cleaner
comparison of tuned model-family procedures.

For each outer fold and each candidate procedure:

```text
1. Hold out one outer-validation partition.
2. Tune that procedure using only the outer-training data.
3. Fit the selected configuration on all outer-training data.
4. Evaluate it once on the outer-validation data.
```

After all outer folds, the outer-fold evidence compares the tuned procedures under the
same held-out outer observations.

This answers:

> If each family is allowed to tune itself fairly using only its available training data,
> which tuned family procedure generalizes better within the development data?

Nested CV therefore supports a more protected **family-selection** decision. It does not
directly provide one final deployable hyperparameter configuration, because different
outer folds can select different configurations. After choosing the winning family, rerun
that family's frozen tuning procedure on all development data, select one final
configuration, and fit it on all development data.

A nested design can also evaluate a complete automated rule that tunes every family and
selects an inner-CV winner separately in each outer fold. That estimates the performance
of the automated selection policy itself. It does not directly yield the final tuned
XGBoost, random forest, or other deployed model. It is a different estimand and is not
the core route for this project's final family and hyperparameter choice.

## 9. Choosing and freezing the final comparison design

Flat repeated CV and per-family nested CV are both defensible. The project must choose
one primary design before comprehensive candidate-comparison results are viewed.

Flat repeated CV is especially reasonable when:

```text
the project prioritizes a direct joint choice of family and exact configuration
the full candidate library makes nested computation prohibitively expensive
the protocol includes strong stability summaries and practical-tie safeguards
the final report clearly acknowledges selection optimism in the winner's CV score
```

Per-family nested CV is especially reasonable when:

```text
the project prioritizes a cleaner comparison of tuned model-family procedures
many families have flexible searches and close training-only scores
the extra computation is manageable
the project can include a separate all-development-data tuning run for the winning family
```

Repeated nested CV can offer more stable procedure-level evidence but may be expensive
for a broad library containing ensembles, kernel methods, and neural networks. Other
documented alternatives, including a fixed internal holdout, Monte Carlo
cross-validation, bias-corrected flat CV, and bootstrap-based correction methods, are
useful reference designs. They should not be added merely because they are more
complicated.

Before the final comparison is executed, freeze:

```text
candidate-procedure registry
primary metric and secondary metrics
validation design and fold or seed-generation rules
search spaces and budgets
early-stopping and convergence rules
handling of failed fits
random-seed policy
practical-equivalence margin and tie-breaking rule
calibration and threshold-treatment policy
planned statistical comparison summaries
```

After results are observed, changes may be made only as clearly labelled follow-up
experiments. They must not silently replace the original planned comparison.

## 10. Statistical comparison and uncertainty before final selection

Statistical analysis should support model selection, not create an illusion that a large
number of p-values can mechanically identify a single true winner.

The comparison stage should preserve paired structure. Candidates should be evaluated on
the same validation observations whenever possible, and the workflow should retain:

```text
fold-level metrics for every candidate
out-of-fold scores with observation and repeat identifiers
selected hyperparameters for every tuning run
fit failures, runtime, and convergence information
```

The following distinction is essential:

```text
outer-fold or repeated-CV metric differences:
    describe performance variation across resampling partitions

paired bootstrap on fixed validation or OOF prediction rows:
    describes conditional uncertainty in the metric difference for those predictions

a full resampling-and-retuning experiment:
    additionally reflects tuning and refitting variability, but is much more expensive
```

Therefore, a paired bootstrap on repeated-CV OOF predictions is valuable supplementary
evidence, but it should not be described as a complete sampling distribution of the full
tune-and-select pipeline.

Naive ordinary t-tests over repeated-CV fold scores should not be treated as valid
default inference. Fold scores are dependent because validation observations and training
sets are reused. Suitable analyses may include:

```text
descriptive paired metric-difference distributions
corrected repeated-CV tests as a frequentist sensitivity analysis
5x2 CV tests for focused two-model comparisons
Bayesian correlated comparisons with a practical-equivalence region
paired bootstrap intervals for PR-AUC / average-precision differences
Holm adjustment if a family of frequentist pairwise tests is reported
```

For a broad candidate library, formal pairwise testing should be selective and
pre-specified. The project should not calculate every possible pairwise test simply
because it can. A sensible structure is:

```text
all candidates:
    descriptive comparison tables, uncertainty summaries, rank/stability summaries

leading candidates after the predefined rule:
    focused paired comparisons and practical-equivalence analysis

final chosen candidate:
    no test-set comparison against alternatives
```

McNemar's test and DeLong-style ROC-AUC comparisons may be used only as training-only
supplementary analyses on fixed paired predictions. They must never be used to compare
candidate models on the final test set. PR-AUC / average-precision comparison should rely
on approaches applicable to that metric, such as paired bootstrap differences.

The final decision should combine primary-metric evidence, practical equivalence,
calibration and threshold behaviour where relevant, stability, complexity, runtime, and
interpretability. A non-significant p-value does not prove equality. A statistically
detectable difference does not prove operational importance.

## 11. Fair tuning effort in the final comparison

The final comparison must avoid giving one model family substantially more opportunity to
win merely because it receives more informal tuning attention.

For example, this would be unfair:

```text
logistic regression:
    only default C = 1

boosting:
    500 adaptive trials across many parameters
```

A fairer strategy is:

```text
define reasonable search spaces for every candidate procedure
use the same primary tuning metric
use the same outer validation partitions across procedures
use comparable search budgets where feasible
record all search spaces, budgets, seeds, and early-stopping rules
avoid manually expanding only the currently favoured model after results are viewed
```

Search effort does not have to be literally identical. Different families have different
numbers of consequential hyperparameters and different fitting costs. Fairness means
that the unequal effort is principled, pre-specified, and documented, rather than driven
by a desire to make a preferred family win.

A search strategy is not an evaluation design. Grid search, random search, successive
halving, and Bayesian optimization may all be used inside flat repeated CV or inside the
inner loop of nested CV. The final protocol should state both layers separately.

## 12. Protocol-design checklist before the final comparison

The final comparison should not be coded immediately after this note is updated. First,
the project should make a written protocol decision based on the central methodological
reference and the candidate registry.

The protocol must state:

```text
1. Complete candidate library and exact candidate-procedure definitions.
2. Primary metric, secondary metrics, and the precise PR-AUC / average-precision
   implementation used.
3. Whether the primary route is flat repeated CV or per-family nested CV.
4. The number of folds, repeats if applicable, seed policy, and stratification policy.
5. The preprocessing, feature-engineering, feature-selection, and imbalance-treatment
   policy inside each validation split.
6. Search spaces, search methods, compute budgets, early stopping, and failure handling.
7. The practical-equivalence margin and the rule for resolving ties.
8. The statistical summaries and focused comparison methods.
9. Whether calibration and threshold selection are outside the family comparison or part
   of each candidate procedure.
10. The exact final full-development-data fitting route.
11. The final test-set report and bootstrap confidence-interval plan.
```

Only after this protocol is frozen should the project run the comprehensive final
candidate-comparison workflow. The notebook and reusable source code should implement
the frozen protocol rather than inventing modelling choices while results are being
viewed.

## 13. Metric hierarchy for final model selection

The project should define a metric hierarchy before final selection.

```text
Primary development metric:
    PR-AUC / average precision, with the exact implementation verified

Secondary ranking metric:
    ROC-AUC

Threshold-dependent metrics:
    recall, precision, specificity, F1, and balanced accuracy

Operational metric:
    predicted positive rate or intervention capacity

Business metric if costs are defined:
    expected cost or expected utility

Probability metrics if probabilities are used:
    Brier score, calibration curve, calibration slope, and calibration intercept
```

PR-AUC can remain the primary development ranking metric because churn is the minority
positive class and positive-class retrieval matters. However, before the final comparison
is run, the project must verify the precise implementation and terminology used for this
quantity. In particular, if scikit-learn's `average_precision_score` is used, the report
should call it average precision or explain the relationship to broader PR-curve area
terminology precisely.

PR-AUC alone is not enough. A model with high ranking performance can still have an
undesirable operating threshold, poor calibration, unstable behaviour, or excessive
complexity. The final selection rule should therefore combine the primary metric with
practical equivalence, stability, threshold behaviour, calibration where relevant,
runtime, complexity, and interpretability.

## 14. Threshold-selection plan

Threshold selection should happen after candidate model comparison, not separately in every early model section.

The model-family sections can show threshold curves, but those are diagnostic.

A final threshold should be selected using training data only, possibly through validation, cross-validation, or a pre-specified cost rule.

Possible threshold-selection rules:

```text
maximize F1
maximize balanced accuracy
choose threshold for target recall
choose threshold for target precision
choose threshold for target predicted positive rate
minimize expected cost
maximize expected utility
```

For churn, a cost-sensitive rule may be most meaningful if costs can be defined.

Example cost structure:

```text
false negative:
    missed churner, possible lost customer value

false positive:
    unnecessary retention action or discount
```

If business costs are unknown, the report can present several training-only operating points rather than pretending one threshold is objectively optimal.

The final threshold must be frozen before final test evaluation.

---

## 15. Calibration plan

Calibration should be considered if the final model's probabilities are interpreted as risks or used in cost calculations.

Questions:

```text
Do predicted probabilities correspond to observed churn frequencies?
Does the model systematically overestimate or underestimate churn risk?
Does calibration improve Brier score or calibration curves?
Does calibration affect threshold decisions?
```

Potential calibration methods:

```text
sigmoid calibration / Platt scaling
isotonic regression
temperature scaling for neural networks
```

Calibration must be fitted without test leakage.

Possible workflow:

```text
training folds:
    fit base model

validation/calibration folds:
    fit calibrator

test set:
    evaluate the already-frozen calibrated final model once
```

For this project, calibration can be a later dedicated section after major model families are compared.

---

## 16. Final model fitting

The final-fitting route depends on the chosen comparison design, but both routes preserve
the same test-set discipline.

### 16.1 If the project uses flat repeated CV

```text
1. Tune every candidate procedure with the frozen repeated-CV design.
2. Select the final family and exact configuration using the frozen decision rule.
3. Freeze preprocessing, feature policy, imbalance treatment, calibration policy, and
   threshold rule.
4. Fit that complete pipeline once on all development data.
5. Evaluate it once on the untouched test set.
```

No additional tuning run is required solely because the selected configuration is later
fitted on all development observations. The repeated-CV tuning already used the complete
development dataset across its resampling partitions.

### 16.2 If the project uses per-family nested CV

```text
1. Compare tuned candidate procedures using the frozen outer-fold evidence.
2. Select the final family using the frozen decision rule.
3. Rerun only the winning family's frozen tuning procedure on all development data.
4. Select one exact final configuration from that all-development-data tuning run.
5. Freeze preprocessing, feature policy, imbalance treatment, calibration policy, and
   threshold rule.
6. Fit that complete pipeline once on all development data.
7. Evaluate it once on the untouched test set.
```

The later tuning run is necessary because nested CV evaluates a family-level tuning
procedure. Its outer folds can select different hyperparameters and do not themselves
define one final deployable configuration.

### 16.3 Definition of the frozen final pipeline

The final pipeline must include every step that learns from data:

```text
imputation
scaling
encoding
feature engineering
feature selection
resampling if used during training
model fitting
calibration if used
threshold rule
```

Everything that learns from data must be fitted only on development data before the test
set is evaluated. If a practical tie is resolved in favour of a simpler or more
interpretable candidate, that decision must occur before the test set is touched.

## 17. Final test-set evaluation

The final test set should answer one question:

> How well does the single frozen final modelling procedure perform on unseen data?

The final test evaluation should report metrics for one model only:

```text
confusion matrix
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
predicted positive rate
possibly expected cost
possibly calibration metrics
bootstrap confidence intervals for the single final model
```

The report should explicitly say:

```text
The test set was not used for model-family selection, hyperparameter tuning,
threshold selection, calibration fitting, candidate comparison, or ablation decisions.
```

If this is true, the test estimate is the clean final performance evidence for the selected model.

---

## 18. Final uncertainty reporting

Final test metrics should be reported with uncertainty where possible.

Recommended for the single final model:

```text
bootstrap confidence intervals for final metrics
```

For the single final model:

```text
metric point estimate
95 percent bootstrap confidence interval
```

Metrics suitable for bootstrap CIs:

```text
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
expected cost
Brier score
```

Candidate model-difference methods such as paired bootstrap, McNemar-style tests, or DeLong-style tests should be used before final selection using training-only validation evidence. They should not be applied on the final test set to compare multiple candidate models.

---

## 19. No test-set model comparison

The final test set should not be used to compare multiple candidate models.

Do not do:

```text
fit model A, model B, model C
evaluate all on the test set
choose the best test-set result
```

Also avoid:

```text
evaluate final model and additional candidate models on the test set
use comparisons on the test set to argue which model should have been selected
change the final model after seeing test results
```

All of that turns the test set into another validation set.

If the project wants paired model comparisons, they should be performed before final test evaluation using training-only validation evidence.

---

## 20. Ablation study plan

An ablation study measures the contribution of model components by removing or changing them.

Possible ablations for this project:

```text
with versus without engineered features
with versus without feature selection
with versus without class weighting
with versus without resampling
with versus without calibration
with versus without threshold tuning
full feature set versus reduced feature set
complex model versus interpretable baseline
```

Ablation studies should be used after a strong final candidate exists, but before final test evaluation if they influence model selection.

The purpose is not only performance ranking. It is understanding:

```text
Which modelling decisions mattered?
Which components improved performance?
Which components added complexity without clear benefit?
```

For example:

```text
If SMOTE barely improves PR-AUC but substantially complicates the pipeline,
the project may decide not to use it.

If class weighting improves recall but damages precision too much,
the tradeoff should be reported.

If calibration improves probability reliability but not ranking,
the report should separate calibration from discrimination.
```

If ablation results are used to decide the final pipeline, they must use training/validation data only.

---

## 21. Feature-importance and explanation plan

After final model selection, model explanation should be handled carefully.

Possible tools:

```text
logistic regression coefficients
decision-tree rules
random-forest impurity importance
permutation importance
partial dependence plots
SHAP values, if appropriate
```

Important caution:

```text
feature importance is model-specific
correlated features can share or distort importance
importance does not imply causality
permutation importance should be computed on training-validation data before final selection
```

If feature-importance results are used to decide whether to change the model, they must be computed before final test evaluation. After final model selection, feature explanations can be reported as interpretation of the selected model, but they should not cause model changes after the test result is known.

For the final report, interpretability should be linked to model type. A logistic regression model may provide cleaner coefficient interpretation. A tree ensemble may provide stronger predictive performance but more complex explanations.

---

## 22. Strict test-set use policy

The project should follow a strict test-set policy.

Allowed before final evaluation:

```text
define the test split
store the test file
verify file existence and schema if necessary
```

Not allowed before final evaluation:

```text
checking test metrics to choose models
checking test metrics to compare candidate models
checking test metrics to choose hyperparameters
checking test metrics to choose threshold
checking test metrics to decide whether to add features
checking test metrics to choose calibration
checking test metrics to decide whether to accept or reject ablation choices
repeatedly rerunning final evaluation and adapting the model
```

If test-set results cause model changes, the test set has become validation data. The final performance estimate is then compromised.

---

## 23. How to revise earlier report sections

Earlier model sections should use development-stage language consistently. They may identify
a representative strong configuration or a useful hyperparameter region, but they should
not claim final population superiority.

For every completed family, use wording such as:

```text
Within the development-stage cross-validation grid, this configuration achieved the
strongest observed primary metric.

The result identifies a promising representative candidate or a useful hyperparameter
region. It does not prove that the configuration is uniquely optimal or that the model
family is finally superior to all other families.
```

This applies to logistic regression, kNN, Naive Bayes, decision trees, bagging and random
forests, boosting, SVMs, and MLPs. Differences among neighbouring configurations should
be interpreted in light of evaluation uncertainty, hyperparameter-selection noise,
random-training variation where applicable, and the later all-model comparison stage.

The later candidate-comparison chapter should make the final selection claim. The final
test chapter should report performance only for the already frozen selected pipeline.

## 24. Suggested structure for the later report

A later report structure could be:

```text
1. Introduction
2. Data and problem framing
3. Methodology: splitting, metrics, evaluation discipline, and selection protocol
4. Exploratory data analysis
5. Preprocessing and feature engineering
6. Baseline classifiers
7. Model-family sections
       logistic regression
       kNN
       Naive Bayes
       decision trees
       ensembles
       SVM
       MLP
8. Candidate comparison, uncertainty, and final selection
       candidate-procedure registry
       flat repeated CV or per-family nested CV
       stability and uncertainty summaries
       focused practical-equivalence comparisons
       calibration and threshold selection
       ablations if used
       frozen final pipeline definition
9. Final test-set evaluation
       one final model only
       final metrics
       bootstrap confidence intervals
       final confusion matrix
       practical interpretation
10. Conclusion
```

The current report already has several of these components. The main future addition is a
dedicated candidate-comparison and final-selection chapter before the final test section.

## 25. Suggested implementation additions later

Before implementation begins, inspect the existing reusable modules, scripts, and
comparable notebooks. Reusable final-selection logic should live in `src/telco_churn/`,
with a matching training-only smoke test in `scripts/`.

Likely reusable additions include:

```text
evaluation.py or equivalent:
    candidate-procedure registry validation
    fold-level metric storage
    repeated-CV and nested-CV orchestration helpers
    pooled OOF versus fold-mean metric summaries
    paired bootstrap metric differences
    corrected-comparison and practical-equivalence summaries
    threshold-selection utilities
    calibration evaluation utilities
    final-test bootstrap confidence intervals

models.py or equivalent:
    candidate factories with fully documented search spaces
    reproducible seed handling
    final frozen-pipeline factory
```

Likely executable workflows include:

```text
candidate-comparison notebook:
    runs the frozen all-model protocol and saves comparison artifacts

threshold-and-calibration notebook:
    uses training-only data to select the operating rule, if required

final-test notebook:
    evaluates exactly one frozen final model and produces final report artifacts
```

No reusable code should be written until the protocol design in Section 12 is frozen.

## 26. Proposed later notebooks

The precise numbering should follow the existing notebook sequence. The conceptual stages
are more important than the numbers:

```text
candidate_comparison_and_selection
calibration_and_threshold_selection
final_test_evaluation
ablation_and_interpretability
```

Candidate comparison, threshold/calibration selection, and final test evaluation should
remain separate from the model-family learning notebooks. The final test evaluation
workflow must evaluate exactly one frozen final model.

## 27. Decision rule for choosing the final comparison design

Before implementing the final comparison stage, decide whether the primary selection route
will be flat repeated CV or per-family nested CV. This is a methodology decision, not a
result-driven decision to be made after seeing which method favours a particular model.

Flat repeated CV is especially reasonable when:

```text
the project prioritizes a direct joint choice of family and exact configuration
the full candidate library makes nested computation prohibitively expensive
the protocol includes strong stability summaries and practical-tie safeguards
the final report clearly acknowledges selection optimism in the winner's CV score
```

Per-family nested CV is especially reasonable when:

```text
the project prioritizes a cleaner comparison of tuned model-family procedures
many families have flexible searches and close training-only scores
the extra computation is manageable
the final project can include a separate all-development-data tuning run for the winning family
```

Repeated nested CV, bias-corrected flat CV, or a holdout-reference analysis may be
included as supplementary evidence only when their additional cost and interpretation are
justified in the frozen protocol.

Whichever primary route is chosen, the test set remains untouched until one final pipeline
has been defined.

## 28. Final selection checklist

Before touching the test set, confirm:

```text
candidate-procedure registry frozen
primary comparison design frozen
primary and secondary metrics frozen
all candidate comparisons completed using training data only
practical-equivalence and tie-breaking rule applied
one final model family selected
final hyperparameters fixed
preprocessing and feature set fixed
resampling or class-weighting policy fixed
calibration decision fixed
threshold or threshold-selection rule fixed
final test metrics and bootstrap confidence-interval procedure fixed
```

Only after this checklist is complete should the test set be evaluated.

## 29. Final test evaluation checklist

When evaluating the test set, compute for the single final model:

```text
final confusion matrix
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
predicted positive rate
bootstrap CIs for final metrics
calibration metrics if probabilities matter
business-cost metric if costs are defined
```

Then interpret:

```text
What tradeoff does the selected threshold create?
How many customers would be flagged?
How uncertain are the metrics?
Is the performance practically useful?
Are probabilities reliable enough to interpret as risks?
```

Do not compare against other test-set models.

---

## 30. Final conclusion style

The final report conclusion should avoid saying only:

```text
Model X performed best.
```

A better conclusion style:

```text
Model X was selected before final test evaluation because it provided the
strongest training-only evidence under the project selection criteria. After
the model, hyperparameters, preprocessing, calibration decision, and threshold
were frozen, the model was fitted on the full training set and evaluated once
on the untouched test set. The final test-set estimates are ..., with bootstrap
confidence intervals indicating finite-test-sample uncertainty.
```

Avoid:

```text
After comparing several models on the test set, model X was best.
```

This conclusion style is more professional because it connects:

```text
selection criterion
threshold tradeoff
single-model test performance
uncertainty
practical meaning
model complexity
```

---

## 31. Summary

The project should proceed in layers:

```text
Layer 1:
    completed model-family learning sections using ordinary stratified CV

Layer 2:
    protocol design for a broad candidate-procedure library, including fair search
    budgets, validation design, uncertainty summaries, and practical-tie rules

Layer 3:
    comprehensive training-only candidate comparison using the frozen primary route

Layer 4:
    final family, hyperparameter, calibration, and threshold selection using only
    development data

Layer 5:
    one untouched-test evaluation of exactly one frozen final pipeline, with bootstrap
    uncertainty reporting for its test metrics

Layer 6:
    ablation and interpretability analysis, using training-only evidence whenever the
    analysis could affect the selected pipeline
```

This structure preserves the educational purpose of the project while making the final
selection process, final test evaluation, and uncertainty claims statistically careful
and reproducible.
