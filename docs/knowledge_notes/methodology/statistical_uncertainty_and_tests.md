# Statistical uncertainty, confidence intervals, and pre-final model comparison tests

This note explains statistical uncertainty in model evaluation and introduces practical methods for confidence intervals and statistical comparisons.

The central idea is:

> Model metrics are estimates computed from finite samples. Candidate model comparison should ask not only which point estimate is higher, but also how uncertain the estimate is and whether the observed difference is large enough to be meaningful. In this project, those candidate comparisons are performed before the final test set is used.

This note focuses on uncertainty and tests. It builds on the previous notes:

```text
evaluation_foundations.md
cross_validation_and_model_selection.md
```

The companion note
`final_model_selection_designs_and_candidate_comparison.md` gives the broader
framework for candidate procedures, flat repeated cross-validation, per-family
nested cross-validation, repeated nested cross-validation, bias-corrected
cross-validation, and the final single-test-set protocol. This note has the
narrower role of explaining how uncertainty should be represented and how
candidate differences should and should not be tested within those evaluation
designs.

In particular, this note distinguishes three questions that are often conflated:

```text
1. Is an observed metric estimate uncertain?
2. Is the observed difference between two candidate procedures meaningful?
3. Does a chosen final frozen model have uncertain test-set performance?
```

The appropriate resampling unit, comparison method, and interpretation depend
on which of these questions is being asked.

---

## 1. Why uncertainty matters

Suppose two candidate models are evaluated on the same validation fold, repeated-CV output, nested-CV outer fold, or another training-only evaluation sample:

```text
Model A PR-AUC = 0.657
Model B PR-AUC = 0.660
```

Model B has a higher point estimate. But the difference is only \(0.003\). Without uncertainty analysis, we do not know whether this difference is meaningful.

The observed difference may reflect:

```text
true model superiority
finite validation-sample noise
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

This is why small differences between tuned families should be interpreted cautiously and why nested CV, repeated CV, paired validation comparisons, or paired bootstrap differences before final selection can be useful.

### 2.6 Procedure-selection uncertainty and the estimand

The uncertainty discussion must specify the object being evaluated. These are
not interchangeable:

```text
Fixed configuration:
    one fully specified pipeline with fixed preprocessing, hyperparameters,
    calibration rule, threshold rule, and random-seed policy.

Tuned family procedure:
    a rule that receives training data, searches a predefined configuration
    space using an inner validation procedure, fits the selected configuration,
    and produces predictions.

Family-selection procedure:
    a rule that compares several tuned family procedures and selects one of
    them according to a predefined decision rule.

Frozen final pipeline:
    the one model, hyperparameter configuration, calibration rule, and decision
    policy selected before test evaluation and then fitted on all development
    data.
```

For example, a nested-CV outer-fold result can estimate the performance of a
tuned XGBoost procedure. It does not automatically estimate the performance of
one globally fixed XGBoost hyperparameter vector. Conversely, a bootstrap
confidence interval around a final test PR-AUC describes uncertainty in the
frozen final pipeline's test performance. It does not retrospectively quantify
every earlier model-family and hyperparameter choice.

This distinction prevents a common overstatement: a narrow interval around one
final metric does not prove that the entire preceding search and selection path
was free from uncertainty. The larger selection question must be handled using
the training-only comparison protocol.

---

## 3. Point estimates, standard errors, and confidence intervals

A point estimate is one number computed from a sample:

$$
\widehat{M}.
$$

The unknown target is the true population metric:

$$
M.
$$

A standard error estimates the typical sampling variability of \(\widehat{M}\):

$$
\operatorname{SE}(\widehat{M}).
$$

A confidence interval gives a range of plausible values for the true metric. A rough normal-approximation interval has the form:

$$
\widehat{M}
\pm
z_{1-\alpha/2}
\operatorname{SE}(\widehat{M}).
$$

For a 95 percent interval:

$$
z_{0.975}
\approx
1.96.
$$

Interpretation:

> If the same evaluation process were repeated many times and a confidence interval were computed each time, approximately 95 percent of those intervals would contain the true metric, under the assumptions of the method.

A confidence interval is not a guarantee that the true value lies inside a particular computed interval. It is a long-run procedure statement.

---

## 4. Accuracy confidence intervals

Accuracy is relatively simple because each test observation is either correct or incorrect.

For a fixed trained classifier \(h\), define:

$$
Z_i
=
\mathbf{1}\{h(x_i)=y_i\}.
$$

Then:

$$
\widehat{A}
=
\frac{1}{n}
\sum_{i=1}^{n}
Z_i.
$$

If test observations are independent and identically distributed and the model is fixed, \(Z_i\) can be treated as Bernoulli with success probability \(A\), the true accuracy.

A simple approximate standard error is:

$$
\widehat{\operatorname{SE}}(\widehat{A})
=
\sqrt{
\frac{\widehat{A}(1-\widehat{A})}{n}
}.
$$

A rough 95 percent interval is:

$$
\widehat{A}
\pm
1.96
\sqrt{
\frac{\widehat{A}(1-\widehat{A})}{n}
}.
$$

This illustrates why larger test sets give narrower confidence intervals.

### 4.1 Example

If:

```text
accuracy = 0.80
test set size = 100
```

then:

$$
\widehat{\operatorname{SE}}
=
\sqrt{\frac{0.8(0.2)}{100}}
=
0.04.
$$

A rough 95 percent interval is:

$$
0.80 \pm 1.96(0.04)
=
0.80 \pm 0.0784.
$$

So the interval is roughly:

$$
[0.722,\;0.878].
$$

That is wide. A model with observed accuracy \(0.80\) on only 100 cases is not estimated very precisely.

If the test set size were 10000 instead:

$$
\widehat{\operatorname{SE}}
=
\sqrt{\frac{0.8(0.2)}{10000}}
=
0.004.
$$

The interval becomes approximately:

$$
0.80 \pm 0.00784.
$$

The same point estimate is much more precise with a larger evaluation set.

---

## 5. Confidence intervals for recall and specificity

Recall is accuracy restricted to the positive class:

$$
\mathrm{Recall}
=
\frac{TP}{TP+FN}.
$$

If the set of actual positives is treated as the relevant evaluation sample, then recall can be viewed as a binomial proportion over positive cases:

$$
\widehat{\mathrm{Recall}}
=
\frac{TP}{n_+},
$$

where:

$$
n_+ = TP+FN.
$$

A rough standard error is:

$$
\sqrt{
\frac{
\widehat{\mathrm{Recall}}(1-\widehat{\mathrm{Recall}})
}{
n_+
}
}.
$$

Specificity is analogous over actual negatives:

$$
\mathrm{Specificity}
=
\frac{TN}{TN+FP}.
$$

A rough standard error is:

$$
\sqrt{
\frac{
\widehat{\mathrm{Specificity}}(1-\widehat{\mathrm{Specificity}})
}{
n_-
}
},
$$

where:

$$
n_- = TN+FP.
$$

These intervals are useful because recall and specificity are conditional accuracies on class-specific subsets.

In imbalanced classification, the positive class can be small. Therefore, recall may have a much wider interval than ordinary accuracy.

---

## 6. Why precision and F1 are harder

Precision is:

$$
\mathrm{Precision}
=
\frac{TP}{TP+FP}.
$$

It is conditional on predicted positives, not actual positives. The denominator \(TP+FP\) is random because it depends on the classifier's predictions.

F1 is:

$$
F_1
=
\frac{
2 \cdot \mathrm{Precision}\cdot \mathrm{Recall}
}{
\mathrm{Precision}+\mathrm{Recall}
}.
$$

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

$$
\widehat{M}^{*(1)},
\widehat{M}^{*(2)},
\ldots,
\widehat{M}^{*(B)}.
$$

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

$$
[
q_{0.025},
q_{0.975}
].
$$

where \(q_{0.025}\) and \(q_{0.975}\) are bootstrap quantiles.

### 8.1 Practical algorithm

```text
Input:
    y_eval
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

## 9. Bootstrap for paired model differences before final selection

During model development, an important question is not only:

```text
What is the uncertainty around model A?
```

but also:

```text
Is model A meaningfully better than model B before the final model is selected?
```

Because both models are evaluated on the same observations, the comparison is paired.

The paired bootstrap preserves this pairing.

For each bootstrap resample of validation or cross-validation prediction rows:

```text
1. sample row indices with replacement;
2. compute metric for model A on those rows;
3. compute metric for model B on those same rows;
4. compute the difference:
       metric_A - metric_B
```

This produces a bootstrap distribution of differences:

$$
\Delta^{*(b)}
=
\widehat{M}^{*(b)}_A
-
\widehat{M}^{*(b)}_B.
$$

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

This is one of the most useful pre-final candidate-comparison tools for this project. It should be applied to validation, repeated-CV, nested-CV, or other training-only evaluation outputs before the one final model is frozen.

### 9.1 What a paired prediction bootstrap does and does not capture

For one set of paired out-of-sample predictions, the paired bootstrap preserves
the dependence that matters for a model comparison: both candidate scores are
evaluated on the same resampled customer rows. It is therefore appropriate for
estimating uncertainty in a difference such as

$$
\Delta_{\mathrm{PR}}
=
\mathrm{PR\text{-}AUC}_{A}
-
\mathrm{PR\text{-}AUC}_{B},
$$

conditional on the observed predictions.

It does not automatically reproduce every layer of uncertainty that generated
those predictions. In particular, an ordinary row-level bootstrap of already
stored predictions does not refit models, redo hyperparameter selection, redraw
CV splits, or recreate stochastic optimizer trajectories. It therefore answers
a narrower question than a full repeated nested-CV experiment.

This distinction is especially important for repeated CV. Under repeated CV,
one customer can have multiple out-of-fold predictions, one from each repeat.
Those repeated records are not independent customer observations. The project
must not concatenate them and bootstrap them as though they were unrelated rows.
Instead, use one of the following explicitly documented designs:

```text
Single outer-CV pass:
    retain one paired outer-fold prediction per customer and bootstrap customers.

Repeated-CV aggregate:
    aggregate each customer's repeated out-of-fold scores within each candidate,
    then bootstrap customers while preserving candidate pairing.

Resampling-level comparison:
    compare paired fold or repeat summaries with a method that models their
    correlation, such as a corrected resampling or Bayesian correlated method.
```

The first two designs are prediction-level analyses. The third operates on
resampling-level summaries. They answer related but non-identical questions and
should be reported as complementary evidence rather than interchangeable tests.

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

For this project, bootstrap is most immediately useful in two distinct places: first, for paired candidate-model comparisons before final selection using validation or cross-validation outputs; and second, for confidence intervals around the single frozen final model's test metrics. Later, it may also be useful for feature-importance stability in tree ensembles or logistic regression.

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

$$
p
=
\frac{
1 + \#\{\text{permuted scores} \geq \text{original score}\}
}{
1 + B
},
$$

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

$$
n_{10}
\quad\text{versus}\quad
n_{01}.
$$

A common large-sample test statistic with continuity correction is:

$$
\chi^2
=
\frac{
(|n_{10}-n_{01}|-1)^2
}{
n_{10}+n_{01}
}.
$$

This is compared to a chi-square distribution with one degree of freedom.

### 16.1 When McNemar is useful

McNemar's test is useful when:

```text
two classifiers are evaluated on the same validation cases before final selection
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

For this project, McNemar's test may be used only as a supplementary
training-only comparison of two already fixed hard-decision rules on paired
validation predictions. It must not be used to compare candidate models on the
held-out test set. Using the test set to judge which of several alternatives is
better would turn the test set into another model-selection dataset and violate
the single-frozen-model policy. Even before final selection, paired bootstrap
differences remain more flexible because PR-AUC, calibration, and operational
threshold metrics matter more than hard-classification accuracy alone.

---

## 17. DeLong-style tests for ROC-AUC

ROC-AUC has a special statistical structure. It can be interpreted as the probability that a randomly selected positive observation receives a higher score than a randomly selected negative observation:

$$
\mathrm{AUC}
=
P(s(X^+) > s(X^-)).
$$

Because of this, specialized methods exist for estimating the variance of ROC-AUC and comparing correlated ROC-AUCs from two models evaluated on the same data.

DeLong's method is a common approach for comparing paired ROC-AUCs.

### 17.1 When DeLong is useful

DeLong-style testing is useful when:

```text
the main metric is ROC-AUC
two candidate models are evaluated on the same validation observations before final selection
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

For this project, paired bootstrap on validation or cross-validation predictions is likely the most practical and consistent approach for PR-AUC differences before final model selection.

Procedure:

```text
1. Keep both candidate models' predicted scores on the same validation or cross-validation observations.
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

Therefore, 5x2 CV tests are useful to know about, but they may not be the primary final comparison method. A final held-out test set with bootstrap intervals for one frozen final model is more aligned with the project workflow. Paired model differences belong before final selection.

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

$$
d_k
=
M_A^{(k)}
-
M_B^{(k)}.
$$

Then one can summarize:

```text
mean difference across folds
standard deviation of fold differences
number of folds where A beats B
```

This is useful descriptively, but formal inference remains delicate because folds are dependent.

In this project, paired fold comparisons can be useful in the later model-comparison stage as a robustness diagnostic, especially with repeated CV.

### 21.1 Why a naive t-test on CV fold scores is not valid

It is tempting to create one difference per fold,

$$
d_j
=
M_{A,j}
-
M_{B,j},
$$

and apply an ordinary one-sample or paired t-test to the resulting values. That
test treats the differences as independent replicates. Standard cross-validation
does not satisfy that assumption.

In ordinary $K$-fold CV, the fitted models share most of their training rows.
For example, two models trained in different folds of 5-fold CV overlap in much
of their training data. In repeated CV, the same customer can also appear in
multiple validation folds across repeats. This induces dependence among the
fold-level differences and can make a naive t-test estimate an unrealistically
small standard error. A small p-value from that calculation may therefore reflect
an invalid independence assumption rather than compelling evidence of a real
procedure-level advantage.

Thus the following distinction is mandatory:

```text
mean ± standard deviation across folds:
    useful descriptive stability summary

naive t-test across folds:
    not a primary inferential method for this project
```

### 21.2 Corrected resampling tests

Corrected resampling methods attempt to account for the dependence created by
overlapping training sets. Let $r$ be the number of repeats, $k$ the number of
folds per repeat, and $d_1,\ldots,d_{rk}$ the paired metric differences. One
commonly used Nadeau--Bengio-style variance correction has the schematic form

$$
\widehat{\operatorname{Var}}(\bar d)
\approx
\left(
\frac{1}{rk}
+
\frac{n_{\mathrm{validation}}}{n_{\mathrm{training}}}
\right)
+s_d^2,
$$

where $s_d^2$ is the empirical variance of the paired differences. Relative to
the naive variance $s_d^2/(rk)$, the additional term inflates the uncertainty to
recognize training-set overlap.

This correction is useful as a sensitivity analysis, not as an unquestionable
ground truth. Its quality depends on the resampling design and modelling
assumptions. In particular, it should not be used to manufacture a binary
"significant versus not significant" ranking for a long model library.

For this project, a corrected resampling comparison may be reported for a small
set of predeclared pairwise comparisons after the candidate library has been
evaluated. It should accompany effect sizes, practical-equivalence conclusions,
and prediction-level bootstrap intervals.

### 21.3 The 5x2-CV test as a specialized alternative

The 5x2-CV test repeats a 50/50 split five times and swaps the training and
validation halves inside each repeat. It was proposed partly to avoid the naive
independence assumption in ordinary $K$-fold comparisons.

Its strengths are a classical, explicitly paired design and a long history in
algorithm-comparison literature. Its main trade-off is that every fitted model
uses only half of the development data in each training run. That estimates a
different training-sample regime from the final model, which will be fitted on
all available development data. It is therefore valuable to document and may be
used as an optional robustness check for a focused pairwise comparison, but it
is not automatically the primary comparison design for this project.

### 21.4 Bayesian correlated comparison and practical equivalence

Frequentist hypothesis tests often focus on whether an exact null difference of
zero can be rejected. For model selection, the more useful question is usually
whether a difference is large enough to matter.

A Bayesian correlated comparison models paired resampling differences while
acknowledging their dependence. It can summarize posterior probability in three
regions:

```text
P(Delta < -epsilon):
    candidate B is meaningfully better than candidate A

P(-epsilon <= Delta <= +epsilon):
    the candidates are practically equivalent

P(Delta > +epsilon):
    candidate A is meaningfully better than candidate B
```

Here $\epsilon$ is a region of practical equivalence, often abbreviated ROPE.
For a PR-AUC comparison, it is a prespecified tolerance representing a difference
too small to justify greater model complexity, less stable training, weaker
calibration, or higher computational cost. The value of $\epsilon$ is a decision
parameter, not a universal statistical constant, and must be justified before
inspecting final candidate-comparison results.

This is particularly well aligned with the project objective. A result such as
``XGBoost has a 0.58 posterior probability of a materially higher PR-AUC, a
0.38 probability of practical equivalence, and a 0.04 probability that CatBoost
is materially higher'' is more decision-relevant than a p-value alone. It also
makes it legitimate to report an evidence-based tie rather than forcing a tiny
point-estimate difference into an artificial winner.

The Bayesian method is not magic. Its result depends on the likelihood model,
dependence approximation, prior specification, and chosen ROPE. These choices
must be recorded. It should therefore be treated as a transparent decision aid,
not an automatic oracle.

### 21.5 Multiple comparisons and the candidate library

With $m$ candidate procedures, there are $m(m-1)/2$ pairwise comparisons. A
large all-model library can therefore produce many p-values or intervals. Running
every possible test and highlighting whichever happens to look strongest creates
another selection problem.

The project should separate its outputs into three layers:

```text
Full candidate table:
    descriptive ranking, uncertainty summaries, and transparency for every
    predeclared candidate procedure.

Primary comparisons:
    a small predeclared set of scientifically or practically important pairwise
    contrasts, such as the leading boosted-tree procedure versus logistic
    regression or the two strongest practically competitive procedures.

Exploratory follow-up comparisons:
    clearly labelled as exploratory rather than used alone to justify final
    selection.
```

When a family of frequentist hypothesis tests is reported as formal evidence,
use a multiplicity adjustment such as Holm's step-down procedure. This controls
the family-wise error rate more efficiently than a simple Bonferroni correction.
Multiplicity adjustment does not replace sound design, effect-size reporting, or
practical-equivalence reasoning. It only addresses one source of false-positive
claims among many comparisons.

### 21.6 Recommended evidence hierarchy for this project

For the later candidate-comparison stage, no single test should decide the final
model. The preferred evidence hierarchy is:

```text
1. Predefined primary metric and candidate-procedure registry.
2. Same outer splits and paired outer-fold performance summaries for all
   candidate procedures.
3. Effect sizes and practical-equivalence assessment.
4. Prediction-level paired bootstrap intervals where one paired out-of-sample
   prediction per customer, or a documented customer-level aggregate, is
   available.
5. Corrected resampling and/or Bayesian correlated analysis for a limited set
   of central pairwise comparisons.
6. Calibration, threshold behaviour, seed stability, runtime, interpretability,
   and implementation complexity as explicit tie-breakers.
7. One final frozen model evaluated once on the held-out test set with bootstrap
   confidence intervals for its own metrics.
```

This hierarchy avoids two opposite mistakes: declaring a winner from a tiny
point-estimate gap, or refusing to make a justified decision simply because no
single p-value is decisive.

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

2. Freeze exactly one final modelling pipeline.

3. Fit that final model on the full training data.

4. Predict once on the untouched test set.

5. Compute final metrics for that single final model.

6. Bootstrap the test set to obtain confidence intervals for that single final model.
```

This produces uncertainty intervals for the final frozen model.

The final test set should not be used to compare multiple candidate models or to choose between a final model and additional candidate models. All candidate comparisons, paired bootstrap differences, nested-CV comparisons, repeated-CV comparisons, McNemar-style hard-classification comparisons, and DeLong-style ROC-AUC comparisons should be completed before the final model is chosen and before the test set is touched.

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
paired bootstrap differences before final selection
possibly McNemar on validation predictions before final selection
possibly DeLong on validation ROC-AUC before final selection
calibration uncertainty if calibrated probabilities are reported
```

---

## 28. Suggested final-report language

When reporting one final metric:

> The test-set PR-AUC is \(0.66\). Because this value is computed on a finite test set, it is an estimate of the model's population PR-AUC. A bootstrap confidence interval is reported to quantify test-sample uncertainty.

When comparing two candidate models before final selection:

> Model A has a higher observed PR-AUC than Model B on the same validation or cross-validation predictions. Because both models are evaluated on the same observations, the comparison is paired. A paired bootstrap interval for the PR-AUC difference is used to assess whether the observed gap is large relative to validation-sample variability.

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
| Paired bootstrap | Is model A's validation metric higher than model B's before final selection? | Flexible paired candidate comparisons | Computational; depends on validation sample |
| Permutation test | Is there predictive signal beyond chance label association? | Signal detection | Not a direct model A vs B comparison |
| McNemar test | Do two hard classifiers have different validation error rates before final selection? | Paired hard predictions | Threshold-dependent; not for AUC/PR-AUC |
| DeLong test | Do two validation ROC-AUCs differ before final selection? | Paired ROC-AUC comparison | Specific to ROC-AUC |
| 5x2 CV tests | Do two algorithms differ under repeated train/test splits? | Small-data classifier comparison | Less aligned with held-out-test workflow |
| Fold SD | How stable are CV scores across folds? | Descriptive stability | Not a perfect CI |
| Nested CV | How well does a tuning procedure generalize? | Comparing tuned families | More computation; not final deployment fit |

---

## 30. Final takeaway

The main message is:

> Model evaluation should report both performance and uncertainty.

For this project, the immediate role of uncertainty is interpretive: it tells us not to overstate small differences in development-stage CV results. Later, uncertainty should become operational: candidate comparisons should use paired uncertainty methods before final selection, and final test-set metrics should be accompanied by bootstrap confidence intervals for the single frozen final model.
