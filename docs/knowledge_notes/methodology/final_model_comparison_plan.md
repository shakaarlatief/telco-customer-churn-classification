# Final model comparison plan for the Telco churn project

This note defines the project-specific evaluation plan after the individual model-family sections are complete.

The previous methodology notes explain the theory:

```text
evaluation_foundations.md
cross_validation_and_model_selection.md
statistical_uncertainty_and_tests.md
```

This note answers a practical question:

> After we have explained, implemented, tuned, and interpreted all model families, how should the project compare them and choose a final model without overstating the evidence?

The project should not treat every section-level cross-validation table as a final ranking. The section-level tables are development-stage evidence. The final comparison should be a separate stage with stricter language, more careful uncertainty analysis, and a clean final test-set evaluation.

---

## 1. Current stage of the project

The project is currently in the model-family learning stage.

Each model-family section has three purposes:

```text
1. explain the model mathematically;
2. implement the model on the Telco churn data;
3. study development-stage behaviour using training-set cross-validation.
```

Examples already completed:

```text
simple baseline classifiers
linear classification and logistic regression
k-nearest neighbours
Naive Bayes
```

Planned later model families include:

```text
decision trees
bagging
random forests
boosting
support vector machines
multilayer perceptrons
possibly calibration and resampling variants
```

In these sections, ordinary stratified cross-validation is appropriate because the goal is not final statistical proof. The goal is to learn, compare broad behaviour, and build candidate models.

---

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

After all relevant model families have been studied, the project should add a dedicated stage:

```text
Model comparison, uncertainty, and final selection
```

This stage should not introduce a new model family. It should compare the strongest candidate models from the earlier sections.

The purpose is to answer:

```text
Which candidate models remain competitive under a more rigorous comparison?
Which model should be selected as the final model?
Which threshold should be selected for the final decision rule?
How uncertain are the final claims?
```

---

## 5. Candidate shortlist construction

The first step is to construct a candidate shortlist.

A candidate is not every hyperparameter setting ever tried. A candidate should be a reasonable representative of a model family or modelling strategy.

Possible shortlist examples:

```text
best regularized logistic regression candidate
best kNN candidate
best Naive Bayes candidate
best shallow decision tree candidate
best pruned decision tree candidate
best random forest candidate
best gradient boosting candidate
best SVM candidate
best MLP candidate
```

The shortlist should include:

```text
strong performers
interpretable baselines
models with different error profiles
models with different complexity levels
```

The shortlist should not include dozens of nearly identical hyperparameter settings unless the goal is explicitly to study hyperparameter stability.

---

## 6. Candidate selection criteria

The candidate shortlist should not be based only on one metric.

Important criteria:

```text
PR-AUC:
    positive-class retrieval under class imbalance

ROC-AUC:
    overall ranking ability

recall:
    ability to detect churners

precision:
    fraction of flagged customers who actually churn

specificity:
    ability to avoid unnecessary false alarms

F1:
    fixed-threshold balance between precision and recall

balanced accuracy:
    average of recall and specificity

predicted positive rate:
    operational size of the intervention group

calibration:
    reliability of probabilities, if probabilities are used as risks

simplicity:
    ease of explanation and robustness

interpretability:
    usefulness for understanding churn drivers

computational cost:
    fitting and prediction time

stability:
    sensitivity to fold splits, random seeds, and hyperparameters
```

PR-AUC can remain the primary development ranking metric, but final selection should consider tradeoffs.

---

## 7. Repeated CV for stable tuning of serious candidates

Before the final model is chosen, repeated cross-validation can be used for serious candidate models.

Example:

```text
RepeatedStratifiedKFold:
    n_splits = 5
    n_repeats = 5 or 10
```

Repeated CV is useful because it reduces dependence on one particular fold split.

For each candidate family, repeated CV can help answer:

```text
Is the model's performance stable across splits?
Does the selected hyperparameter region remain similar?
Are small metric differences consistent or fragile?
```

Repeated CV is especially useful for:

```text
models with close performance
models with random training procedures
models with flexible hyperparameters
models being considered for final selection
```

Repeated CV should not necessarily be used for every early educational experiment because it increases computation and report complexity.

---

## 8. Nested CV for comparing tuned procedures

Nested cross-validation can be used to compare tuned model-family procedures.

The evaluated object is not a single fixed fitted model. It is a procedure:

```text
Given training data:
    tune hyperparameters using inner CV;
    fit the selected model on the available training data;
    produce predictions.
```

For each outer fold:

```text
1. Hold out one outer validation fold.
2. Tune each model family using only the outer-training data.
3. Fit each selected model on the full outer-training data.
4. Evaluate each selected model on the outer-validation fold.
```

This gives outer-fold scores for each tuned family.

Nested CV can answer:

> If each model family is allowed to tune itself fairly using only training data, which model-family procedure generalizes best?

Nested CV is most useful when the project wants a more statistically careful model-family comparison before touching the test set.

---

## 9. Whether nested CV should be used in this project

Nested CV is useful but computationally heavier and more complex to explain.

A balanced plan is:

```text
Use ordinary CV in individual model-family sections.
Use repeated CV for stable final tuning of serious candidates.
Use nested CV if the top model families are close or if final model-family comparison needs stronger support.
Always keep the untouched test set for final evaluation.
```

Nested CV should be considered especially if:

```text
several tuned families have very similar PR-AUC or ROC-AUC
the final selected model is much more complex than the runner-up
the report wants to claim one tuned family is clearly preferable
the tuning search spaces are large and selection optimism may matter
```

Nested CV may be skipped if:

```text
one model is clearly superior across many metrics
the computational cost is too high
the final comparison will rely mainly on held-out test uncertainty
the project scope would become too large
```

If skipped, the report should explicitly say that the final test set provides the primary final performance estimate, while the development-stage CV rankings are used for selection.

---

## 10. Fair tuning effort in the final comparison

The final comparison must avoid giving one model family much more tuning attention than others.

For example, this would be unfair:

```text
logistic regression:
    default C only

boosting:
    500 Optuna trials
```

A fairer strategy:

```text
define reasonable search spaces for each serious candidate
use comparable search budgets where feasible
use the same primary tuning metric
use the same CV splitter or repeated-CV design
document search spaces and search budgets
```

Search effort does not need to be identical in a literal sense. Some models have more sensitive hyperparameters than others. But the comparison should be transparent.

---

## 11. Suggested final candidate search strategy

A practical final search strategy could be:

```text
1. Use previous sections to identify reasonable hyperparameter ranges.
2. For each serious candidate family, define a compact but fair search space.
3. Use repeated stratified CV for tuning.
4. Use PR-AUC as the primary tuning metric.
5. Store secondary metrics for interpretation.
6. Inspect whether selected configurations are stable.
7. Shortlist the top few models for final threshold and calibration analysis.
```

If using Optuna or Bayesian optimization:

```text
use a fixed trial budget per serious model family
fix random seeds
record the search space
record the selected configuration
do not keep expanding the search only for the favourite model
```

---

## 12. Metric hierarchy for final model selection

The project should define a metric hierarchy before final selection.

A possible hierarchy:

```text
Primary development metric:
    PR-AUC

Secondary ranking metric:
    ROC-AUC

Threshold-dependent metrics:
    recall, precision, specificity, F1, balanced accuracy

Operational metric:
    predicted positive rate

Business metric if costs are defined:
    expected cost or expected utility

Probability metric if probabilities are used:
    Brier score, calibration curve, calibration slope/intercept
```

Why PR-AUC as primary?

```text
churn is the positive minority class
positive-class retrieval matters
ROC-AUC can look strong even when precision is weak under imbalance
```

But PR-AUC alone is not enough. A model with high PR-AUC may still have an undesirable operating threshold, poor calibration, or excessive complexity.

---

## 13. Threshold-selection plan

Threshold selection should happen after candidate model comparison, not separately in every early model section.

The model-family sections can show threshold curves, but those are diagnostic.

A final threshold should be selected using training data only, possibly through validation or cross-validation.

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

If business costs are unknown, the report can present several operating points rather than pretending one threshold is objectively optimal.

---

## 14. Calibration plan

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
    evaluate calibrated final model once
```

For this project, calibration can be a later dedicated section after major model families are compared.

---

## 15. Final model fitting

After final model family, hyperparameters, threshold, and optional calibration are chosen using training data only:

```text
1. Fit the final pipeline on the full training set.
2. Apply the fitted pipeline to the untouched test set.
3. Compute final metrics once.
```

The final pipeline should include all preprocessing steps:

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

Everything that learns from data must be fitted only on the training data.

---

## 16. Final test-set evaluation

The test set should answer:

> How well does the frozen final modelling procedure perform on unseen data?

The final test evaluation should report:

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
```

The report should explicitly say:

```text
The test set was not used for model-family selection, hyperparameter tuning,
threshold selection, or calibration fitting.
```

If this is true, the test estimate is the cleanest final performance evidence.

---

## 17. Final uncertainty reporting

Final test metrics should be reported with uncertainty where possible.

Recommended:

```text
bootstrap confidence intervals for final metrics
paired bootstrap intervals for differences against runner-up models
```

For one final model:

```text
metric point estimate
95 percent bootstrap confidence interval
```

For model differences:

```text
metric_A - metric_B
95 percent paired bootstrap confidence interval
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

Special tests can be mentioned or optionally included:

```text
McNemar's test:
    paired hard-classification error comparison

DeLong test:
    paired ROC-AUC comparison

permutation test:
    predictive signal compared with label-randomized null
```

But paired bootstrap is the most flexible main method.

---

## 18. Final paired comparison against runner-up models

The final selected model should be compared with at least one strong runner-up.

Example:

```text
Final model:
    gradient boosting

Runner-up:
    regularized logistic regression
```

Why compare with logistic regression?

```text
it is strong
it is interpretable
it is simpler
it is a natural baseline for tabular binary classification
```

A complex final model should justify its complexity. If the complex model improves PR-AUC only slightly and the paired interval includes zero, the report should not overstate the improvement.

Possible language:

> The final model has the highest observed PR-AUC on the test set, but the paired bootstrap interval for the difference relative to logistic regression includes zero. Therefore, the evidence for a meaningful ranking-performance improvement is limited. The simpler logistic model remains a competitive alternative.

or:

> The final model improves PR-AUC and recall relative to logistic regression, and the paired bootstrap interval for PR-AUC difference is mostly positive. This supports the final model choice, although the practical value of the improvement should still be considered relative to model complexity.

---

## 19. Ablation study plan

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

Ablation studies should be used after a strong final candidate exists.

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

---

## 20. Feature-importance and explanation plan

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
permutation importance should be computed on validation or test data, not training data only
```

For the final report, interpretability should be linked to model type.

A logistic regression model may provide cleaner coefficient interpretation. A tree ensemble may provide stronger predictive performance but more complex explanations.

---

## 21. Test-set use policy

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
checking test metrics to choose hyperparameters
checking test metrics to choose threshold
checking test metrics to decide whether to add features
checking test metrics to choose calibration
repeatedly rerunning final evaluation and adapting the model
```

If test-set results cause model changes, the test set has become validation data. The final performance estimate is then compromised.

---

## 22. How to revise earlier report sections

Earlier model sections should be revised with careful wording.

### 22.1 Logistic regression

Add language such as:

```text
The regularization comparison is a development-stage tuning analysis.
Differences between neighbouring C values should be interpreted as evidence
about a useful regularization region, not as proof of a uniquely optimal C.
```

### 22.2 kNN

Add language such as:

```text
The kNN grid is used to understand the effect of neighbourhood size and
distance weighting. Several neighbouring settings perform similarly, so the
selected configuration should be interpreted as a representative strong kNN
candidate rather than a statistically final optimum.
```

### 22.3 Naive Bayes

Add language such as:

```text
The hybrid Naive Bayes model is selected within the development comparison
because it has the strongest PR-AUC and the most coherent mixed-feature
likelihood specification. The observed gap relative to other Naive Bayes
variants is useful development evidence, but final model-family claims are
deferred to the later comparison stage.
```

### 22.4 Future tree and ensemble sections

Use the same pattern:

```text
select representative models within each section
describe close hyperparameter results cautiously
avoid final superiority claims
reserve final claims for the final comparison and test evaluation
```

---

## 23. Suggested structure for the later report

A later report structure could be:

```text
1. Introduction
2. Data and problem framing
3. Methodology: splitting, metrics, and evaluation discipline
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
8. Model comparison, uncertainty, and final selection
       candidate shortlist
       repeated CV / nested CV if used
       candidate comparison
       threshold selection
       calibration if used
       ablations if used
9. Final test-set evaluation
       final metrics
       confidence intervals
       paired comparisons
       final confusion matrix
       practical interpretation
10. Conclusion
```

The current report already has some of these sections. The main future addition is a dedicated section for model comparison and uncertainty.

---

## 24. Suggested implementation additions later

Later code additions may include:

```text
evaluation.py:
    fold-level metric storage
    repeated-CV evaluation utilities
    pooled OOF versus fold-mean metric summaries
    bootstrap confidence intervals
    paired bootstrap metric differences
    threshold-selection utilities
    calibration evaluation utilities

models.py:
    final model factories for selected candidate families

notebooks:
    model-comparison notebook
    final-test-evaluation notebook
    calibration notebook, if needed
    ablation notebook, if needed
```

The project does not need these utilities immediately before decision trees, but the evaluation plan should guide future development.

---

## 25. Proposed final comparison notebooks

Possible later notebooks:

```text
08_decision_trees.py
09_ensemble_methods.py
10_support_vector_machines.py
11_multilayer_perceptron.py
12_model_comparison_and_selection.py
13_calibration_and_threshold_selection.py
14_final_test_evaluation.py
15_ablation_and_interpretability.py
```

Exact numbering can change depending on the final project structure.

The important point is that final comparison and final test evaluation should be separate from model-family learning notebooks.

---

## 26. Decision rule for whether to use nested CV

Before implementing the final comparison stage, decide whether nested CV is worth the cost.

Use nested CV if:

```text
top model families are close
the final report wants stronger evidence about tuned-family comparison
compute cost is manageable
the tuning procedures can be clearly defined
```

Skip nested CV and rely on repeated CV plus final test bootstrap if:

```text
the top model is clearly stronger across metrics
the final comparison would become too computationally heavy
the project narrative benefits from simplicity
the final test set is sufficiently large and untouched
```

If nested CV is skipped, say so explicitly and explain the alternative:

```text
The project uses training-set cross-validation for model development and
a final held-out test set with bootstrap uncertainty for final performance.
Nested CV is discussed as a stricter procedure-level evaluation method but
is not used because the project already preserves an untouched final test set
and prioritizes a readable model-family learning workflow.
```

---

## 27. Final selection checklist

Before touching the test set, confirm:

```text
model family selected
hyperparameters selected
preprocessing fixed
feature set fixed
resampling or class weighting fixed
calibration decision fixed
threshold fixed or threshold-selection rule fixed
primary and secondary metrics fixed
runner-up comparison models fixed
bootstrap procedure fixed
```

Only after this checklist is complete should the test set be evaluated.

---

## 28. Final test evaluation checklist

When evaluating the test set, compute:

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
bootstrap CIs
paired bootstrap differences versus runner-up
calibration metrics if probabilities matter
business-cost metric if costs are defined
```

Then interpret:

```text
Does the final model improve enough over simpler baselines?
What tradeoff does the selected threshold create?
How many customers would be flagged?
How uncertain are the metrics?
Is the improvement practically meaningful?
Are probabilities reliable enough to interpret as risks?
```

---

## 29. Final conclusion style

The final report conclusion should avoid saying only:

```text
Model X performed best.
```

A better conclusion style:

```text
Model X was selected because it provided the strongest development-stage
positive-class ranking performance while maintaining an acceptable recall,
precision, and operational alert rate. On the untouched test set, it achieved
PR-AUC ..., recall ..., precision ..., and F1 ..., with bootstrap confidence
intervals indicating the remaining test-sample uncertainty. The paired
comparison against the strongest simpler baseline suggests that ...
```

This conclusion style is more professional because it connects:

```text
selection criterion
threshold tradeoff
test performance
uncertainty
practical meaning
model complexity
```

---

## 30. Summary

The project should proceed in layers:

```text
Layer 1:
    individual model-family learning sections using ordinary stratified CV

Layer 2:
    later rigorous candidate comparison using repeated CV and possibly nested CV

Layer 3:
    final model, threshold, and calibration selection using training data only

Layer 4:
    final untouched test-set evaluation with bootstrap uncertainty and paired comparisons

Layer 5:
    ablation and interpretability analysis for the selected model
```

This structure preserves the educational purpose of the project while also making the final evaluation statistically careful and professionally credible.
