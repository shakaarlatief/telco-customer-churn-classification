# Statistical uncertainty, confidence intervals, and model comparison tests

This note explains statistical uncertainty in model evaluation and introduces practical methods for confidence intervals and statistical comparisons.

The central idea is:

> Model metrics are estimates computed from finite samples. A serious model comparison should ask not only which point estimate is higher, but also how uncertain the estimate is and whether the observed difference is large enough to be meaningful.

This note focuses on uncertainty and tests. It builds on the previous notes:

```text
evaluation_foundations.md
cross_validation_and_model_selection.md
```

---

## 1. Why uncertainty matters

Suppose two models are evaluated on the same test set:

```text
Model A PR-AUC = 0.657
Model B PR-AUC = 0.660
```

Model B has a higher point estimate. But the difference is only \(0.003\). Without uncertainty analysis, we do not know whether this difference is meaningful.

The observed difference may reflect:

```text
true model superiority
finite test-sample noise
threshold choice
hyperparameter-selection noise
random training variation
random split variation
```

The correct interpretation is not automatically:

```text
Model B is better.
```

A better interpretation is:

```text
Model B has the higher observed estimate on this evaluation sample.
We need uncertainty analysis to judge whether the difference is practically
or statistically meaningful.
```

This is especially important in this project because many models can have similar ROC-AUC, PR-AUC, or \(F_1\) values.

---

## 2. Different sources of uncertainty

Model evaluation uncertainty has several layers.

### 2.1 Test-sample uncertainty

The test set is a finite sample from the population. If a different test sample were drawn, the metric would change.

This affects all metrics:

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
calibration error
```

### 2.2 Cross-validation split uncertainty

A cross-validation estimate depends on how the folds are constructed. A different random fold split may produce a slightly different mean score.

Repeated cross-validation helps estimate and reduce this source of instability.

### 2.3 Model-training randomness

Some algorithms are random even with the same training data.

Examples:

```text
random forests
gradient boosting with subsampling
neural networks
stochastic optimization
randomized feature selection
randomized hyperparameter search
```

The same model configuration can produce different fitted models under different random seeds.

### 2.4 Hyperparameter-selection uncertainty

When a tuning procedure selects hyperparameters, the selected configuration may change if the data split changes.

For example, kNN might choose:

```text
outer fold 1: k = 51
outer fold 2: k = 101
outer fold 3: k = 31
```

This does not necessarily mean the method is unstable in a bad way. It may mean several neighbouring values perform similarly.

### 2.5 Model-family selection uncertainty

When many model families are compared, the selected family may depend on validation noise.

This is why small differences between tuned families should be interpreted cautiously and why nested CV or final paired test-set comparisons can be useful.

---

## 3. Point estimates, standard errors, and confidence intervals

A point estimate is one number computed from a sample:

\[
\widehat{M}.
\]

The unknown target is the true population metric:

\[
M.
\]

A standard error estimates the typical sampling variability of \(\widehat{M}\):

\[
\operatorname{SE}(\widehat{M}).
\]

A confidence interval gives a range of plausible values for the true metric. A rough normal-approximation interval has the form:

\[
\widehat{M}
\pm
z_{1-\alpha/2}
\operatorname{SE}(\widehat{M}).
\]

For a 95 percent interval:

\[
z_{0.975}
\approx
1.96.
\]

Interpretation:

> If the same evaluation process were repeated many times and a confidence interval were computed each time, approximately 95 percent of those intervals would contain the true metric, under the assumptions of the method.

A confidence interval is not a guarantee that the true value lies inside a particular computed interval. It is a long-run procedure statement.

---

## 4. Accuracy confidence intervals

Accuracy is relatively simple because each test observation is either correct or incorrect.

For a fixed trained classifier \(h\), define:

\[
Z_i
=
\mathbf{1}\{h(x_i)=y_i\}.
\]

Then:

\[
\widehat{A}
=
\frac{1}{n}
\sum_{i=1}^{n}
Z_i.
\]

If test observations are independent and identically distributed and the model is fixed, \(Z_i\) can be treated as Bernoulli with success probability \(A\), the true accuracy.

A simple approximate standard error is:

\[
\widehat{\operatorname{SE}}(\widehat{A})
=
\sqrt{
\frac{\widehat{A}(1-\widehat{A})}{n}
}.
\]

A rough 95 percent interval is:

\[
\widehat{A}
\pm
1.96
\sqrt{
\frac{\widehat{A}(1-\widehat{A})}{n}
}.
\]

This illustrates why larger test sets give narrower confidence intervals.

### 4.1 Example

If:

```text
accuracy = 0.80
test set size = 100
```

then:

\[
\widehat{\operatorname{SE}}
=
\sqrt{\frac{0.8(0.2)}{100}}
=
0.04.
\]

A rough 95 percent interval is:

\[
0.80 \pm 1.96(0.04)
=
0.80 \pm 0.0784.
\]

So the interval is roughly:

\[
[0.722,\;0.878].
\]

That is wide. A model with observed accuracy \(0.80\) on only 100 cases is not estimated very precisely.

If the test set size were 10000 instead:

\[
\widehat{\operatorname{SE}}
=
\sqrt{\frac{0.8(0.2)}{10000}}
=
0.004.
\]

The interval becomes approximately:

\[
0.80 \pm 0.00784.
\]

The same point estimate is much more precise with a larger evaluation set.

---

## 5. Confidence intervals for recall and specificity

Recall is accuracy restricted to the positive class:

\[
\mathrm{Recall}
=
\frac{TP}{TP+FN}.
\]

If the set of actual positives is treated as the relevant evaluation sample, then recall can be viewed as a binomial proportion over positive cases:

\[
\widehat{\mathrm{Recall}}
=
\frac{TP}{n_+},
\]

where:

\[
n_+ = TP+FN.
\]

A rough standard error is:

\[
\sqrt{
\frac{
\widehat{\mathrm{Recall}}(1-\widehat{\mathrm{Recall}})
}{
n_+
}
}.
\]

Specificity is analogous over actual negatives:

\[
\mathrm{Specificity}
=
\frac{TN}{TN+FP}.
\]

A rough standard error is:

\[
\sqrt{
\frac{
\widehat{\mathrm{Specificity}}(1-\widehat{\mathrm{Specificity}})
}{
n_-
}
},
\]

where:

\[
n_- = TN+FP.
\]

These intervals are useful because recall and specificity are conditional accuracies on class-specific subsets.

In imbalanced classification, the positive class can be small. Therefore, recall may have a much wider interval than ordinary accuracy.

---

## 6. Why precision and F1 are harder

Precision is:

\[
\mathrm{Precision}
=
\frac{TP}{TP+FP}.
\]

It is conditional on predicted positives, not actual positives. The denominator \(TP+FP\) is random because it depends on the classifier's predictions.

F1 is:

\[
F_1
=
\frac{
2 \cdot \mathrm{Precision}\cdot \mathrm{Recall}
}{
\mathrm{Precision}+\mathrm{Recall}
}.
\]

It is a nonlinear function of two random ratios.

Because of this, simple binomial formulas are less straightforward. Bootstrap confidence intervals are often more practical for precision and \(F_1\).

---

## 7. Bootstrap: general idea

Bootstrap is a resampling method for approximating the sampling distribution of a statistic.

Given an evaluation dataset of size \(n\), a bootstrap sample is created by sampling \(n\) rows **with replacement** from the original evaluation dataset.

The procedure is:

```text
1. Start with evaluation rows and model predictions.
2. Resample rows with replacement.
3. Compute the metric on the resampled rows.
4. Repeat many times.
5. Use the distribution of bootstrap metrics to estimate uncertainty.
```

If \(B\) bootstrap samples are drawn, we obtain:

\[
\widehat{M}^{*(1)},
\widehat{M}^{*(2)},
\ldots,
\widehat{M}^{*(B)}.
\]

This empirical distribution approximates the sampling distribution of the metric estimate.

Bootstrap is attractive because it can be used for many metrics, including metrics whose analytic standard errors are difficult.

---

## 8. Bootstrap confidence interval for one model

For one final model evaluated on a test set, bootstrap confidence intervals can be computed for:

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
calibration error
```

A simple percentile bootstrap interval is:

```text
lower bound:
    alpha / 2 quantile of bootstrap metrics

upper bound:
    1 - alpha / 2 quantile of bootstrap metrics
```

For a 95 percent interval:

\[
[
q_{0.025},
q_{0.975}
].
\]

where \(q_{0.025}\) and \(q_{0.975}\) are bootstrap quantiles.

### 8.1 Practical algorithm

```text
Input:
    y_test
    y_score or y_pred
    metric function
    number of bootstrap samples B

For b = 1,...,B:
    sample n indices with replacement from {1,...,n}
    compute metric on sampled y and predictions
    store metric

Return:
    observed metric on original test set
    2.5th and 97.5th percentiles of bootstrap metrics
```

Important practical detail:

For metrics such as ROC-AUC and PR-AUC, a bootstrap sample must contain at least one positive and one negative observation. If a resample contains only one class, the metric is undefined and that bootstrap draw should be skipped or redrawn.

---

## 9. Bootstrap for paired model differences

Often the most important question is not:

```text
What is the uncertainty around model A?
```

but:

```text
Is model A meaningfully better than model B?
```

Because both models are evaluated on the same observations, the comparison is paired.

The paired bootstrap preserves this pairing.

For each bootstrap resample of test rows:

```text
1. sample row indices with replacement;
2. compute metric for model A on those rows;
3. compute metric for model B on those same rows;
4. compute the difference:
       metric_A - metric_B
```

This produces a bootstrap distribution of differences:

\[
\Delta^{*(b)}
=
\widehat{M}^{*(b)}_A
-
\widehat{M}^{*(b)}_B.
\]

A confidence interval for \(\Delta\) is obtained from the bootstrap quantiles.

Interpretation:

```text
If the interval is mostly above 0:
    evidence that model A improves the metric.

If the interval includes 0:
    the observed difference may not be clearly distinguishable from sampling noise.

If the interval is tiny but positive:
    statistically positive, but maybe not practically important.

If the interval is wide:
    evaluation sample may be too small or metric too variable.
```

This is one of the most useful final-comparison tools for this project.

---

## 10. Bootstrap for threshold uncertainty

Threshold-dependent metrics depend on the chosen threshold.

For a fixed threshold \(\tau\), bootstrap can estimate uncertainty around:

```text
precision at threshold tau
recall at threshold tau
specificity at threshold tau
F1 at threshold tau
expected cost at threshold tau
predicted positive rate at threshold tau
```

However, there is a subtle distinction.

### 10.1 Fixed-threshold bootstrap

If the threshold has already been selected using training/validation data and is frozen before test evaluation, then the test-set bootstrap can resample test rows and compute metrics at that fixed threshold.

This estimates uncertainty in final test performance for the fixed decision rule.

### 10.2 Re-selected-threshold bootstrap

If the threshold is re-selected inside each bootstrap sample, the bootstrap distribution reflects a different procedure: the combined procedure of choosing a threshold and evaluating it.

This can be useful for studying threshold-selection stability, but it should not be confused with the final evaluation of a frozen threshold.

For final reporting, the safer approach is:

```text
choose threshold before test evaluation;
freeze threshold;
bootstrap fixed-threshold test metrics.
```

---

## 11. Bootstrap for calibration curves

Calibration curves group predictions into bins and compare:

```text
average predicted probability in bin
observed event rate in bin
```

Bootstrap can quantify uncertainty around the observed event rate in each bin.

Procedure:

```text
1. Resample test rows with replacement.
2. Recompute calibration curve.
3. Store bin-level observed rates.
4. Use quantiles to produce uncertainty bands.
```

This can be useful later if the project includes probability calibration.

However, calibration curves can be unstable when bins contain few observations. Uncertainty bands help reveal this instability.

---

## 12. Bootstrap for feature importance or ablation effects

Bootstrap can also be used beyond final metrics.

Examples:

```text
feature importance uncertainty:
    bootstrap the training data, refit model, recompute importances

coefficient stability:
    bootstrap the training data, refit logistic regression, inspect coefficient spread

ablation effect:
    compute performance difference with and without a component across bootstrap samples

threshold stability:
    bootstrap validation data and see which threshold is selected
```

These uses answer different questions. They require refitting models and are more computationally expensive.

For this project, bootstrap is most immediately useful for final test metrics and paired final-model comparisons. Later, it may also be useful for feature-importance stability in tree ensembles or logistic regression.

---

## 13. What bootstrap assumes

Bootstrap is powerful, but it is not magic.

The simplest case-resampling bootstrap assumes the observed rows are representative and approximately independent.

Potential issues:

```text
time dependence:
    simple row bootstrap can break time structure

group dependence:
    multiple rows from the same customer/entity should not be resampled independently

clustered data:
    resample clusters rather than individual rows

very small sample size:
    bootstrap distribution may be unstable

rare positive class:
    some resamples may contain too few positives for PR-AUC or recall estimates
```

For the Telco churn dataset, rows are treated as independent customer observations. A simple row-level bootstrap is a reasonable practical approach, assuming no hidden grouping or temporal dependence.

---

## 14. Permutation tests for signal

A permutation test can answer:

> Does the model achieve a score that would be unlikely if the features and labels were unrelated?

The basic idea:

```text
1. Compute the original cross-validated score using the real labels.
2. Randomly permute the labels.
3. Recompute the cross-validated score with permuted labels.
4. Repeat many times.
5. Compare the original score to the null distribution of permuted-label scores.
```

If the original score is far above the permuted-label scores, this suggests the model has learned real feature-label signal.

The empirical p-value can be computed as:

\[
p
=
\frac{
1 + \#\{\text{permuted scores} \geq \text{original score}\}
}{
1 + B
},
\]

where \(B\) is the number of permutations.

Permutation tests are useful for asking whether there is predictive signal at all.

They do not directly answer whether model A is better than model B.

---

## 15. Permutation importance versus permutation test

Do not confuse:

```text
permutation test for model signal
```

with:

```text
permutation feature importance
```

Permutation test for model signal permutes the target labels and asks whether the model's score is better than chance label association.

Permutation feature importance permutes one feature at a time and measures how much model performance drops.

They answer different questions:

```text
Permutation test:
    Is there evidence of a relationship between X and y?

Permutation importance:
    How much does this fitted model rely on this feature?
```

Both may be useful, but they belong to different parts of the project.

---

## 16. McNemar's test

McNemar's test compares two classifiers on the same observations using hard predictions.

It is based on the paired disagreement table:

| | Model B correct | Model B wrong |
|---|---:|---:|
| Model A correct | \(n_{11}\) | \(n_{10}\) |
| Model A wrong | \(n_{01}\) | \(n_{00}\) |

The important counts are the disagreements:

```text
n10:
    model A correct, model B wrong

n01:
    model A wrong, model B correct
```

If the two models have the same error rate, we expect these disagreement counts to be similar.

McNemar's test focuses on:

\[
n_{10}
\quad\text{versus}\quad
n_{01}.
\]

A common large-sample test statistic with continuity correction is:

\[
\chi^2
=
\frac{
(|n_{10}-n_{01}|-1)^2
}{
n_{10}+n_{01}
}.
\]

This is compared to a chi-square distribution with one degree of freedom.

### 16.1 When McNemar is useful

McNemar's test is useful when:

```text
two classifiers are evaluated on the same test cases
hard predictions are fixed
the question is whether their error rates differ
```

### 16.2 Limitations

McNemar's test does not compare:

```text
ROC-AUC
PR-AUC
calibration
probability quality
expected cost directly
ranking quality
```

It also depends on a chosen threshold. If the threshold changes, the hard predictions and test result can change.

For this project, McNemar's test can be a useful supplementary final test for hard classification accuracy differences, but paired bootstrap differences are more flexible for the metrics we care about.

---

## 17. DeLong-style tests for ROC-AUC

ROC-AUC has a special statistical structure. It can be interpreted as the probability that a randomly selected positive observation receives a higher score than a randomly selected negative observation:

\[
\mathrm{AUC}
=
P(s(X^+) > s(X^-)).
\]

Because of this, specialized methods exist for estimating the variance of ROC-AUC and comparing correlated ROC-AUCs from two models evaluated on the same data.

DeLong's method is a common approach for comparing paired ROC-AUCs.

### 17.1 When DeLong is useful

DeLong-style testing is useful when:

```text
the main metric is ROC-AUC
two models are evaluated on the same test observations
a specialized ROC-AUC comparison is desired
```

### 17.2 Limitations

DeLong is specific to ROC-AUC. It does not directly apply to:

```text
PR-AUC
F1
precision
recall
specificity
expected cost
calibration error
```

For this project, DeLong may be mentioned or optionally used for ROC-AUC, but paired bootstrap is more general because PR-AUC and threshold-dependent metrics are central for churn.

---

## 18. Tests for PR-AUC differences

PR-AUC is often more relevant than ROC-AUC for imbalanced classification, but its statistical testing is less standardized in everyday applied workflows.

Practical options include:

```text
paired bootstrap difference intervals
permutation tests of model-score differences
repeated CV paired comparisons
```

For this project, paired bootstrap on the held-out test set is likely the most practical and consistent approach for PR-AUC differences.

Procedure:

```text
1. Keep both models' predicted scores on the same test observations.
2. Bootstrap resample rows.
3. Compute PR-AUC for both models in each resample.
4. Store PR-AUC difference.
5. Report percentile interval for the difference.
```

This works because the same resampled rows are used for both models, preserving pairing.

---

## 19. 5x2 cross-validation paired tests

The 5x2 CV paired test is a method for comparing two learning algorithms when data is limited.

The idea is:

```text
Repeat 2-fold cross-validation five times.
Each repeat splits the data into two halves.
Train/test roles are swapped within each repeat.
Compare paired differences across the resulting evaluations.
```

This method was proposed as an alternative to naive paired t-tests over k-fold CV scores, because ordinary k-fold scores are correlated due to overlapping training sets.

There are variants:

```text
5x2 CV paired t-test
5x2 CV combined F-test
```

### 19.1 When it can be useful

It can be useful when:

```text
the dataset is too small for a clean train/validation/test split
two algorithms need to be compared using repeated resampling
one wants a classical test designed for classifier comparison
```

### 19.2 Limitations for this project

This project already has a held-out test set and uses cross-validation inside the training set.

Therefore, 5x2 CV tests are useful to know about, but they may not be the primary final comparison method. A final held-out test set with bootstrap intervals and paired model differences is more aligned with the project workflow.

---

## 20. Fold-level standard deviations

A common summary is:

```text
mean CV score ± standard deviation across folds
```

This is useful descriptively. It shows how much performance varies across validation folds.

However, fold-level standard deviations should be interpreted carefully. The fold scores are not independent in a simple textbook sense because training sets overlap.

For example, in 5-fold CV, two fitted models are trained on overlapping 80 percent subsets. Their validation scores are not independent measurements from completely separate experiments.

Therefore, fold-level standard deviations are best described as:

```text
descriptive stability summaries
```

not perfect classical confidence intervals.

Repeated CV gives more fold scores and a more stable picture, but the scores are still dependent because the same observations are reused across repeats.

---

## 21. Paired fold comparisons

When two models are evaluated using the same cross-validation folds, their fold scores can be compared fold-by-fold.

For fold \(k\):

\[
d_k
=
M_A^{(k)}
-
M_B^{(k)}.
\]

Then one can summarize:

```text
mean difference across folds
standard deviation of fold differences
number of folds where A beats B
```

This is useful descriptively, but formal inference remains delicate because folds are dependent.

In this project, paired fold comparisons can be useful in the later model-comparison stage as a robustness diagnostic, especially with repeated CV.

---

## 22. Confidence intervals after model selection

A major complication is that confidence intervals are often computed after the model has already been selected.

For example:

```text
1. Try many model families and hyperparameters.
2. Select the model with the best validation PR-AUC.
3. Evaluate selected model on test set.
4. Compute a bootstrap CI on the test metric.
```

The test-set CI quantifies uncertainty in the selected final model's test performance, conditional on the selected model being fixed before test evaluation.

It does not fully describe the uncertainty from the whole prior selection process unless that selection process is included inside the resampling scheme.

This is why nested CV and final test evaluation answer different questions:

```text
Nested CV:
    estimates performance of a selection/tuning procedure.

Final test CI:
    estimates uncertainty of the final frozen model's test performance.
```

Both can be useful.

---

## 23. Bootstrap after final model selection

For the final report, the cleanest practical procedure is:

```text
1. Use training data only to choose:
       model family
       preprocessing
       hyperparameters
       threshold
       calibration method, if any

2. Fit the final model on the full training data.

3. Predict once on the untouched test set.

4. Compute final metrics.

5. Bootstrap the test set to obtain confidence intervals.
```

This produces uncertainty intervals for the final frozen model.

If comparing against a runner-up model:

```text
1. Fit both final candidate models using training data only.
2. Predict both on the same untouched test set.
3. Use paired bootstrap differences on test rows.
```

This is likely the most important final uncertainty method for the project.

---

## 24. Bootstrap for repeated CV results

Bootstrap can also be used with cross-validation results, but care is needed.

If we have repeated-CV fold scores, we might be tempted to bootstrap the fold scores. However, the fold scores are dependent because they share observations and training sets.

This can still be used as an exploratory stability tool, but it should not be presented as a perfect classical confidence interval.

A more principled approach for model-development uncertainty is:

```text
use repeated CV to describe variability;
use nested CV for procedure-level performance;
use held-out test bootstrap for final uncertainty.
```

---

## 25. Bayesian perspective on model uncertainty

A Bayesian approach would treat model parameters, predictions, and metrics probabilistically.

Examples:

```text
Bayesian logistic regression
posterior predictive intervals
Bayesian bootstrap
Bayesian model averaging
```

These can be powerful, but they are beyond the main scope of the current project.

Still, the Bayesian perspective reinforces the main idea:

```text
model performance is uncertain
model choice is uncertain
predictions can be uncertain
```

For this project, classical/bootstrap uncertainty tools are sufficient.

---

## 26. Multiple comparisons

When many models are compared, the chance of observing at least one apparently strong result by luck increases.

This is similar to multiple hypothesis testing.

Examples:

```text
trying many hyperparameter values
trying many model families
trying many thresholds
trying many feature sets
trying many metrics
```

The more comparisons are made, the more careful the interpretation should be.

In model development, we do not necessarily need formal multiple-testing corrections for every exploratory comparison. But the report should be transparent that:

```text
many models were explored;
section-level CV results are development evidence;
final claims are reserved for the final evaluation stage.
```

---

## 27. Practical recommendations for this project

### 27.1 During individual model-family sections

Use:

```text
ordinary stratified CV
transparent grids
development-stage metrics
threshold curves
careful wording
```

Avoid:

```text
claiming final superiority from small CV differences
treating selected hyperparameters as uniquely optimal
overemphasizing tiny metric gaps
```

### 27.2 During final model-family comparison

Consider:

```text
repeated CV for top models
nested CV for tuned model-family procedures
fold-level variability summaries
hyperparameter stability
metric sensitivity checks
```

### 27.3 During final test evaluation

Use:

```text
bootstrap confidence intervals for final metrics
paired bootstrap differences against runner-up models
possibly McNemar for hard classification accuracy differences
possibly DeLong for ROC-AUC if ROC-AUC is central
calibration uncertainty if calibrated probabilities are reported
```

---

## 28. Suggested final-report language

When reporting one final metric:

> The test-set PR-AUC is \(0.66\). Because this value is computed on a finite test set, it is an estimate of the model's population PR-AUC. A bootstrap confidence interval is reported to quantify test-sample uncertainty.

When comparing two models:

> Model A has a higher observed PR-AUC than Model B on the test set. Because both models are evaluated on the same observations, the comparison is paired. A paired bootstrap interval for the PR-AUC difference is used to assess whether the observed gap is large relative to test-sample variability.

When discussing CV results:

> These cross-validated scores are development-stage estimates used for tuning and model comparison within the training set. Since the same validation procedure is used to select configurations, small differences should be interpreted cautiously and final performance is deferred to the held-out test evaluation.

When discussing hyperparameters:

> The selected configuration has the highest mean cross-validated PR-AUC within the grid. Nearby configurations perform similarly, so the result is better interpreted as evidence for a stable region of the hyperparameter space than as proof that this exact setting is uniquely optimal.

---

## 29. Summary of methods and questions

| Method | Main question | Useful for | Limitations |
|---|---|---|---|
| Binomial CI | How uncertain is accuracy/recall/specificity? | Simple proportions | Not enough for PR-AUC, ROC-AUC, F1 |
| Bootstrap CI | How uncertain is one metric? | Many metrics | Assumes resampling scheme is appropriate |
| Paired bootstrap | Is model A's metric higher than model B's? | Flexible paired comparisons | Computational; depends on test sample |
| Permutation test | Is there predictive signal beyond chance label association? | Signal detection | Not a direct model A vs B comparison |
| McNemar test | Do two hard classifiers have different error rates? | Paired hard predictions | Threshold-dependent; not for AUC/PR-AUC |
| DeLong test | Do two ROC-AUCs differ? | Paired ROC-AUC comparison | Specific to ROC-AUC |
| 5x2 CV tests | Do two algorithms differ under repeated train/test splits? | Small-data classifier comparison | Less aligned with held-out-test workflow |
| Fold SD | How stable are CV scores across folds? | Descriptive stability | Not a perfect CI |
| Nested CV | How well does a tuning procedure generalize? | Comparing tuned families | More computation; not final deployment fit |

---

## 30. Final takeaway

The main message is:

> Model evaluation should report both performance and uncertainty.

For this project, the immediate role of uncertainty is interpretive: it tells us not to overstate small differences in development-stage CV results. Later, uncertainty should become operational: final test-set metrics should be accompanied by bootstrap confidence intervals, and top-model comparisons should use paired uncertainty methods.
