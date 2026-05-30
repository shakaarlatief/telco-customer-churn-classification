# Evaluation foundations: true performance, estimated performance, and leakage discipline

This note explains the statistical foundation behind model evaluation. The central idea is simple but extremely important:

> A reported metric is not the true quality of a model. It is an estimate of an unobserved population quantity, computed from a finite sample.

This distinction matters throughout the project. It affects how we interpret accuracy, recall, precision, ROC-AUC, PR-AUC, cross-validation scores, threshold curves, tuned hyperparameters, model comparisons, and final test-set results.

The purpose of this note is to make the evaluation language precise before moving further into more model families.

---

## 1. The data-generating distribution

In supervised learning, we usually imagine that observations are sampled from an underlying data-generating distribution.

For binary churn classification, one observation consists of features and a label:

\[
(X, Y) \sim p(x, y),
\]

where:

- \(X\) is the customer feature vector;
- \(Y \in \{0,1\}\) is the churn label;
- \(Y=1\) means churn;
- \(Y=0\) means no churn;
- \(p(x,y)\) is the population distribution that generates customer records and labels.

The dataset is not the population itself. The dataset is a finite sample from the population:

\[
\mathcal{D}
=
\{(x_i, y_i)\}_{i=1}^{n}.
\]

This distinction is the foundation of statistical model evaluation.

The model is not intended to perform well only on the observed rows. The goal is to perform well on future customers drawn from the same, or at least a sufficiently similar, distribution.

---

## 2. True metric versus sample estimate

Suppose a trained classifier is denoted by \(h\). For a customer feature vector \(x\), the classifier predicts:

\[
\hat{y}=h(x).
\]

The **true accuracy** of \(h\) is:

\[
A(h)
=
P(h(X)=Y).
\]

Equivalently,

\[
A(h)
=
\mathbb{E}_{(X,Y)\sim p}
\left[
\mathbf{1}\{h(X)=Y\}
\right].
\]

This is the probability that the classifier is correct on a newly sampled customer from the population.

However, \(A(h)\) cannot be computed directly because \(p(x,y)\) is unknown. We therefore estimate it using a finite sample:

\[
\widehat{A}(h)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\{h(x_i)=y_i\}.
\]

This is the **sample accuracy**.

The true metric is unobservable. The sample metric is observable.

This applies to all performance measures:

\[
\text{true metric}
\quad \neq \quad
\text{sample estimate}.
\]

For example:

| Population quantity | Sample estimate |
|---|---|
| True accuracy | Test-set accuracy |
| True recall | Test-set recall |
| True precision | Test-set precision |
| True specificity | Test-set specificity |
| True ROC-AUC | Test-set ROC-AUC |
| True PR-AUC | Test-set PR-AUC |
| True expected cost | Test-set estimated cost |

The sample metric is useful because it estimates the population metric. But because the sample is finite, the estimate is noisy.

---

## 3. Why metric estimates vary

Imagine evaluating the same fixed classifier on two different test samples drawn from the same population. The true accuracy of the classifier has not changed. But the observed sample accuracy can change because the samples contain different customers.

This means every reported metric contains two components:

```text
observed metric = true performance + sampling noise
```

The sampling noise gets smaller as the evaluation sample grows. This is why test-set size is not just a technical detail. A tiny test set cannot support very precise claims.

If a test set has only 100 observations, a difference such as 0.60 versus 0.59 accuracy is not meaningful by itself. It may easily be caused by the random composition of the test set. If a test set has 10000 observations, the same difference may be more informative, although this still depends on the metric and the model comparison structure.

The practical lesson is:

> Small observed differences should not automatically be interpreted as real model differences.

This will matter repeatedly in the model sections. If two neighbouring hyperparameter settings have very similar cross-validated PR-AUC, the selected setting should be interpreted as a practical development choice, not as proof that this exact hyperparameter value is uniquely optimal.

---

## 4. Accuracy as a simple statistical estimator

Accuracy is the easiest metric to analyze because each prediction is either correct or incorrect.

For a fixed classifier \(h\), define:

\[
Z_i
=
\mathbf{1}\{h(x_i)=y_i\}.
\]

Then:

\[
Z_i
=
\begin{cases}
1, & \text{if the classifier is correct on observation } i,\\
0, & \text{otherwise.}
\end{cases}
\]

If the observations are independent draws from the population, then \(Z_i\) can be viewed as a Bernoulli random variable with success probability equal to the true accuracy:

\[
Z_i \sim \mathrm{Bernoulli}(A(h)).
\]

The sample accuracy is the average:

\[
\widehat{A}(h)
=
\frac{1}{n}
\sum_{i=1}^{n} Z_i.
\]

For a Bernoulli average, the variance is:

\[
\operatorname{Var}(\widehat{A})
=
\frac{A(1-A)}{n}.
\]

The standard error is:

\[
\operatorname{SE}(\widehat{A})
=
\sqrt{
\frac{A(1-A)}{n}
}.
\]

Since the true \(A\) is unknown, it is common to plug in the estimate \(\widehat{A}\):

\[
\widehat{\operatorname{SE}}(\widehat{A})
=
\sqrt{
\frac{\widehat{A}(1-\widehat{A})}{n}
}.
\]

A rough normal-approximation 95 percent confidence interval is:

\[
\widehat{A}
\pm
1.96
\sqrt{
\frac{\widehat{A}(1-\widehat{A})}{n}
}.
\]

This formula is not perfect in every situation, especially for small samples or accuracies very close to 0 or 1, but it gives the central intuition:

```text
larger evaluation set -> smaller standard error
smaller evaluation set -> larger standard error
```

The maximum value of \(A(1-A)\) occurs at \(A=0.5\). Therefore, accuracy estimates are most uncertain when the true accuracy is near 0.5.

---

## 5. Why other metrics are harder than accuracy

Accuracy is a simple average of correctness indicators. Many classification metrics are more complicated.

For a binary classifier, the confusion matrix contains:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

From these counts:

\[
\mathrm{Recall}
=
\frac{TP}{TP+FN},
\]

\[
\mathrm{Precision}
=
\frac{TP}{TP+FP},
\]

\[
\mathrm{Specificity}
=
\frac{TN}{TN+FP},
\]

\[
F_1
=
\frac{2 \cdot \mathrm{Precision}\cdot \mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
\]

These metrics are ratios of random quantities. Their uncertainty is therefore more complicated than the uncertainty of accuracy.

ROC-AUC and PR-AUC are also more complex because they depend on rankings over prediction scores rather than one fixed set of hard predictions.

This motivates more general uncertainty methods later, especially:

```text
bootstrap confidence intervals
paired bootstrap differences
permutation tests
McNemar's test for paired hard predictions
DeLong-style tests for ROC-AUC
```

The key principle remains the same: every metric computed from a finite sample is an estimate.

---

## 6. Training performance is not generalization performance

A model can perform very well on the data it was trained on and still fail on new data. Training performance estimates how well the model fits the training sample. Generalization performance concerns future data.

A flexible model can reduce training error by capturing noise, idiosyncratic patterns, or accidental relationships in the training sample. This is overfitting.

For this reason, training metrics are not sufficient for evaluation. They are still useful for diagnosing underfitting and overfitting, but they are not the main evidence of model quality.

A useful diagnostic comparison is:

```text
training performance high, validation performance low:
    likely overfitting

training performance low, validation performance low:
    likely underfitting or optimization failure

training performance high, validation performance also high:
    better evidence of generalization, subject to validation uncertainty
```

---

## 7. Train, validation, and test roles

The clean evaluation workflow separates data into distinct roles.

### Training data

Training data is used to estimate model parameters.

Examples:

```text
logistic-regression coefficients
decision-tree splits
random-forest trees
boosting stages
neural-network weights
```

### Validation data

Validation data is used to make modelling decisions.

Examples:

```text
choose hyperparameters
choose preprocessing variants
choose feature transformations
choose feature selection strategy
choose model family
choose classification threshold
choose calibration method
```

Validation data can be a single validation split or cross-validation inside the training set.

### Test data

Test data is used only after modelling choices are fixed. It estimates final generalization performance.

The rule is:

> Use the test set once, at the end, after the model, preprocessing, hyperparameters, threshold, and calibration choices are fixed.

If the test set is repeatedly checked during development, it gradually becomes part of the model-selection process. It no longer provides an honest final estimate.

---

## 8. Why validation performance can still be optimistic

Validation data is allowed to guide modelling choices. That is its purpose. But this means validation performance can become optimistic when many choices are tried.

Suppose we try 100 hyperparameter settings and choose the one with the best validation score. The winning setting may be genuinely good, but it may also have benefited from random validation noise.

This is called selection optimism.

The issue is not that validation was used incorrectly. The issue is that the selected validation score is not a perfectly unbiased estimate of the selected model's future performance.

This distinction matters for the report:

```text
Correct:
    The selected setting achieved the highest development-stage cross-validated PR-AUC in this grid.

Too strong:
    The selected setting is definitively the best setting.
```

This is especially important when differences are small.

---

## 9. Cross-validation as development-stage evaluation

Cross-validation is used when we want to estimate validation performance while using the available training data efficiently.

In \(K\)-fold cross-validation, the training data is split into \(K\) folds. For each fold \(k\):

1. train the model on the other \(K-1\) folds;
2. evaluate on fold \(k\);
3. store the validation metric.

The cross-validation estimate is often written as:

\[
\widehat{M}_{CV}
=
\frac{1}{K}
\sum_{k=1}^{K}
\widehat{M}^{(k)},
\]

where \(\widehat{M}^{(k)}\) is the validation metric on fold \(k\).

For classification with class imbalance, stratified cross-validation is usually used so that each fold has approximately the same class distribution as the full dataset.

Cross-validation helps answer:

> How well does this modelling setup perform across different training-validation splits of the training data?

It does not replace final test evaluation. It is part of the model-development process.

---

## 10. Fold-mean metrics versus pooled out-of-fold metrics

There are two common ways to summarize cross-validated predictions.

### Fold-mean metric

Compute the metric separately on each validation fold, then average:

\[
\widehat{M}_{CV}
=
\frac{1}{K}
\sum_{k=1}^{K}
\widehat{M}^{(k)}.
\]

This is the usual formal cross-validation performance summary.

### Pooled out-of-fold metric

Collect all out-of-fold predictions into one vector, then compute the metric once on the pooled predictions.

This is useful for:

```text
confusion matrices
threshold curves
ROC curves
precision-recall curves
calibration plots
global diagnostic plots
```

However, pooled metrics and fold-mean metrics are not always identical.

For accuracy with equally sized folds, the pooled value is usually very close to the fold-size-weighted mean. For nonlinear ranking metrics such as ROC-AUC and PR-AUC, pooled out-of-fold AUC can differ from the mean fold AUC because pooled ranking compares observations whose predictions came from different fitted models.

Practical interpretation:

```text
Use fold-level metrics for formal CV summaries.
Use pooled out-of-fold predictions for curves and threshold diagnostics.
```

The project should eventually store both.

---

## 11. Data leakage

Data leakage occurs when information from validation or test data influences the training process in a way that would not be available in real deployment.

Leakage can make validation or test performance look better than it really is.

Common examples:

```text
scaling before splitting
imputing before splitting
feature selection before splitting
target encoding before splitting
oversampling before splitting
using the test set to choose a threshold
using future information in time-dependent data
splitting grouped observations incorrectly
normalizing test data using test-set statistics
```

The safest principle is:

> Anything that learns from data must be fitted only on the training part of the current split.

This is why preprocessing should be inside a pipeline. During cross-validation, the pipeline ensures that imputation, scaling, encoding, feature selection, resampling, and model fitting are learned only from the training folds.

For example, scaling should work like this:

```text
For each CV fold:
    fit scaler on training folds only
    transform training folds
    transform validation fold using training-fold scaler
```

Not:

```text
fit scaler on all data
then cross-validate
```

The second version leaks validation information.

---

## 12. Leakage and resampling

Class-imbalance methods such as oversampling, undersampling, and SMOTE require special care.

Incorrect workflow:

```text
1. Apply SMOTE to the full training dataset.
2. Run cross-validation on the resampled data.
```

This is problematic because synthetic or duplicated observations derived from validation-fold observations can influence the training folds.

Correct workflow:

```text
For each CV fold:
    split into fold-training and fold-validation
    apply oversampling or SMOTE only to the fold-training data
    fit model on resampled fold-training data
    evaluate on untouched fold-validation data
```

Resampling is part of the training procedure. It must happen inside the cross-validation loop.

In scikit-learn-style workflows, this often means using an imbalanced-learn pipeline rather than manually resampling before cross-validation.

---

## 13. Threshold selection is also model selection

Many classifiers output scores or probabilities. A threshold converts those scores into hard predictions.

For example:

\[
\hat{y}
=
\begin{cases}
1, & \hat{p}(Y=1 \mid x) \geq \tau,\\
0, & \hat{p}(Y=1 \mid x) < \tau.
\end{cases}
\]

Changing \(\tau\) changes:

```text
precision
recall
specificity
F1
predicted positive rate
expected cost
```

Therefore, threshold selection is a modelling decision.

The threshold must be selected using training or validation data, not the final test set. Once a final threshold is selected, it should be frozen before test evaluation.

In the current model-family sections, threshold curves are development diagnostics. They show how the model behaves across possible thresholds. They do not yet define the final deployment threshold.

---

## 14. Calibration is separate from discrimination

A score model can be evaluated in different ways.

### Discrimination

Discrimination asks whether the model ranks positive cases above negative cases.

Metrics:

```text
ROC-AUC
PR-AUC
ranking-based lift
```

A high ROC-AUC or PR-AUC means the model ranks examples usefully.

### Calibration

Calibration asks whether predicted probabilities are numerically reliable.

For example, among customers assigned predicted churn probability near 0.70, about 70 percent should actually churn if the model is well calibrated.

Calibration is important if probabilities are interpreted as risks or used in cost-based decision rules.

A model can have good ranking performance but poor calibration. Naive Bayes is a common example because the conditional-independence assumption can produce overconfident probabilities.

Calibration must also respect data splitting. A calibrator should be fitted using data not used to fit the base model, or by using cross-validation-based calibration procedures.

---

## 15. How this applies to the current project sections

The current model sections use cross-validation on the training set. Their purpose is:

```text
model learning
model-family understanding
development-stage comparison
hyperparameter exploration
threshold-behaviour inspection
```

They are not final performance evaluations.

Therefore, the report should use wording like:

```text
development-stage cross-validated comparison
cross-validated training-set estimate
within this tuning grid
selected by the project development criterion
small differences should be interpreted cautiously
final test-set evaluation is deferred
```

The report should avoid wording like:

```text
this model is definitively best
this hyperparameter is truly optimal
this small improvement proves superiority
final performance is ...
```

The final performance stage will happen later after all model families, hyperparameters, preprocessing choices, threshold choices, and optional calibration choices are fixed.

---

## 16. Practical interpretation rules for report writing

When comparing hyperparameter settings:

```text
If differences are large and consistent:
    describe the pattern as meaningful development evidence.

If differences are small:
    describe the selected setting as a practical choice within the grid.

If neighbouring settings perform similarly:
    prefer simpler, more stable, or more interpretable settings when reasonable.

If a more complex model only slightly improves the selected metric:
    do not overstate the improvement.

If one model improves PR-AUC but worsens recall or precision:
    discuss the metric tradeoff, not only the selected metric.
```

Example wording:

> Within the development-stage cross-validation grid, the highest PR-AUC is obtained at \(k=50\). However, several neighbouring values of \(k\) perform similarly, so this result should be interpreted as evidence that a moderately smoothed kNN model is preferable to a very local kNN model, not as strong evidence that \(k=50\) is uniquely optimal.

This is the style the report should use.

---

## 17. What will happen later in the project

After the individual model-family sections are complete, the project should add a dedicated model-comparison and uncertainty stage.

That stage can include:

```text
repeated cross-validation for more stable tuning
nested cross-validation for comparing tuned model-family procedures
bootstrap confidence intervals for final metrics
paired bootstrap comparisons between top models
calibration analysis
threshold selection using validation data only
final untouched test-set evaluation
ablation studies
```

The final test set should remain untouched until the final model-selection procedure is complete.

---

## 18. Summary

The central ideas are:

```text
1. A true metric is a population quantity.
2. A reported metric is a finite-sample estimate.
3. Finite-sample estimates have uncertainty.
4. Training performance is not generalization performance.
5. Validation and cross-validation are for development and selection.
6. Test data is for final evaluation only.
7. Hyperparameter tuning creates selection optimism.
8. Small metric differences should not be overinterpreted.
9. Leakage can invalidate evaluation.
10. Threshold selection and calibration are also modelling decisions.
```

The practical consequence for the project is:

> The model-section CV tables are useful development evidence, but they should be interpreted as development-stage estimates. Final model claims should be made only after the final model and threshold are selected and evaluated once on the untouched test set, with uncertainty quantified where possible.
