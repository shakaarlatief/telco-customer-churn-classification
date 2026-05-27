# Evaluation Metrics for Binary Classification

## Purpose

This document is a reusable methodology note for evaluating binary classification models in the Telco Customer Churn project.

The goal is to define the evaluation language once, so later modelling sections can refer back to the same concepts instead of re-explaining every metric from the beginning.

This is a living note. It should be updated if later stages introduce additional evaluation concepts such as threshold optimization, calibration curves, cost-sensitive evaluation, bootstrap uncertainty, or formal model-comparison tests.

## 1. Role in the project workflow

The project uses three different evaluation layers:

```text
Knowledge note:
    Explains evaluation concepts deeply and generally.

Notebook:
    Applies the evaluation framework to a specific modelling stage and interprets the actual outputs.

LaTeX report:
    Presents the polished explanation, selected equations, result tables, figures, and conclusions.
```

This note is the reusable reference layer.

The main practical rules are:

```text
Use training data for development.
Use cross-validation inside the training data for model comparison.
Keep the held-out test set unused until final evaluation.
Report several metrics because accuracy alone is not sufficient for churn.
```

## 2. Positive and negative class

A binary classification problem has two classes. Many metrics are defined relative to the positive class.

In this project:

```text
positive class = churn
negative class = no churn
```

Using the target encoding:

$$
Y = 1 \quad \Longleftrightarrow \quad \text{customer churned},
$$

and

$$
Y = 0 \quad \Longleftrightarrow \quad \text{customer did not churn}.
$$

This convention matters because recall, precision, false positive rate, ROC curves, PR curves, and threshold-based decisions all depend on which class is treated as positive.

The positive class is not necessarily the majority class. It is the event of interest.

## 3. Confusion matrix

A binary classifier produces predicted labels $\hat{y}_i \in \{0,1\}$.

The positive-first confusion matrix layout used in this project is:

```text
                         predicted
                    positive    negative
actual positive        TP          FN
actual negative        FP          TN
```

The four entries are:

$$
TP = \#\{y=1,\hat{y}=1\},
$$

$$
FN = \#\{y=1,\hat{y}=0\},
$$

$$
FP = \#\{y=0,\hat{y}=1\},
$$

$$
TN = \#\{y=0,\hat{y}=0\}.
$$

Interpretation for churn:

```text
TP:
    The model predicts churn, and the customer churns.

FN:
    The model predicts no churn, but the customer churns.

FP:
    The model predicts churn, but the customer does not churn.

TN:
    The model predicts no churn, and the customer does not churn.
```

False positives and false negatives have different business meanings.

A false positive may lead to unnecessary retention effort for a customer who would have stayed anyway.

A false negative means the model misses a customer who actually leaves.

Because these errors are different, no single metric should be interpreted without checking the underlying confusion-matrix counts.

## 4. Accuracy

Accuracy is the fraction of all predictions that are correct:

$$
\text{Accuracy}
=
\frac{TP + TN}{TP + FN + FP + TN}.
$$

Accuracy is intuitive, but it can be misleading when the classes are imbalanced.

In the Telco training data, no churn is the majority class. Therefore, a model that always predicts no churn can obtain high accuracy while detecting no churners at all.

This is why accuracy is reported, but never used alone.

## 5. Recall

Recall, also called true positive rate or sensitivity, measures the fraction of actual positives that are detected:

$$
\text{Recall}
=
\frac{TP}{TP+FN}.
$$

In churn terms, recall answers:

```text
Among all customers who actually churned, how many did the model flag?
```

High recall means the model misses few churners.

Low recall means the model produces many false negatives.

Recall is important when missing a positive case is costly.

## 6. Specificity

Specificity, also called true negative rate, measures the fraction of actual negatives correctly rejected:

$$
\text{Specificity}
=
\frac{TN}{TN+FP}.
$$

In churn terms, specificity answers:

```text
Among all customers who did not churn, how many did the model correctly leave unflagged?
```

High specificity means the model avoids many false churn alerts.

Low specificity means the model incorrectly flags many non-churners.

Specificity is important when false alarms are costly.

## 7. Precision

Precision measures the reliability of positive predictions:

$$
\text{Precision}
=
\frac{TP}{TP+FP}.
$$

In churn terms, precision answers:

```text
Among all customers flagged as likely churners, how many actually churned?
```

High precision means churn alerts are reliable.

Low precision means many flagged customers would not have churned.

Precision is especially important when retention actions are expensive or limited.

## 8. F1-score

The $F_1$-score combines precision and recall using the harmonic mean:

$$
F_1
=
2
\cdot
\frac{
\text{Precision}\cdot\text{Recall}
}{
\text{Precision}+\text{Recall}
}.
$$

The harmonic mean is low when either precision or recall is low. This makes $F_1$ useful when both false positives and false negatives matter.

However, $F_1$ hides the separate values of precision and recall. Therefore, it should be reported together with the underlying metrics.

## 9. Balanced accuracy

Balanced accuracy averages recall and specificity:

$$
\text{Balanced Accuracy}
=
\frac{1}{2}
\left(
\text{Recall}
+
\text{Specificity}
\right).
$$

This is useful under class imbalance because it gives equal weight to the positive and negative classes.

A classifier that always predicts one class has balanced accuracy equal to $0.5$ in a binary problem. It performs perfectly on one class and completely fails on the other.

Balanced accuracy is useful for identifying whether a model is actually learning both classes or mostly exploiting the majority class.

## 10. Predicted positive rate

The predicted positive rate is the fraction of observations classified as positive:

$$
\text{Predicted Positive Rate}
=
\frac{TP+FP}{TP+FN+FP+TN}.
$$

In churn terms, it measures the fraction of customers the model flags as likely churners.

This metric is not a performance score by itself. Instead, it helps interpret model behaviour.

For example:

```text
A very low predicted positive rate may indicate that the model almost never flags churners.

A very high predicted positive rate may indicate that the model catches many churners but creates many false positives.
```

Comparing the predicted positive rate with the observed churn rate helps reveal whether the model is too conservative or too broad.

## 11. Scores, probabilities, and thresholds

Many classifiers produce a score or probability, not only a hard label.

A probabilistic classifier estimates:

$$
\hat{p}(x)
=
\hat{P}(Y=1 \mid X=x).
$$

A threshold converts the probability into a class prediction:

$$
\hat{y}_{\tau}
=
\mathbb{1}
\{
\hat{p}(x) \geq \tau
\}.
$$

The common default threshold is:

$$
\tau = 0.5.
$$

But $0.5$ is not automatically optimal. The preferred threshold depends on the relative costs of false positives and false negatives.

In churn prediction:

```text
Lower threshold:
    more customers flagged,
    higher recall,
    more false positives,
    lower specificity.

Higher threshold:
    fewer customers flagged,
    lower recall,
    fewer false positives,
    higher specificity.
```

Threshold selection is itself a model-selection decision. It should be made using validation data or cross-validated out-of-fold predictions, not the held-out test set.

## 12. ROC curve and ROC-AUC

The ROC curve compares true positive rate and false positive rate across thresholds.

The true positive rate is recall:

$$
TPR
=
\frac{TP}{TP+FN}.
$$

The false positive rate is:

$$
FPR
=
\frac{FP}{FP+TN}
=
1 - \text{Specificity}.
$$

The ROC curve plots:

```text
x-axis: false positive rate
y-axis: true positive rate
```

ROC-AUC summarizes the curve into one number.

A useful interpretation is ranking ability:

```text
ROC-AUC measures how well the model tends to rank actual positives above actual negatives.
```

ROC-AUC is threshold-independent, but it does not directly tell us what happens at the final operating threshold.

## 13. Precision-recall curve and PR-AUC

The precision-recall curve compares precision and recall across thresholds.

It plots:

```text
x-axis: recall
y-axis: precision
```

PR-AUC summarizes how much precision the model retains as recall increases.

PR-AUC is especially useful when the positive class is relatively rare and the main interest is finding positive cases.

In this project, churn is the minority class, so PR-AUC is an important complement to ROC-AUC.

A useful baseline for PR-AUC is the positive class prevalence. If the churn rate is about $26.54\%$, then a non-informative random ranking has PR-AUC around $0.2654$.

## 14. ROC-AUC versus PR-AUC

ROC-AUC and PR-AUC often agree when one model is clearly better than another, but they can tell different stories under class imbalance.

ROC-AUC considers the tradeoff between true positive rate and false positive rate. Since false positive rate divides by the number of actual negatives, it can look acceptable even when many predicted positives are false positives.

PR-AUC directly focuses on the quality of positive predictions. This is often more informative when the positive class is rare or when false positive burden matters.

For churn modelling:

```text
ROC-AUC:
    useful for overall ranking ability.

PR-AUC:
    especially useful for judging positive-class retrieval quality.
```

Both should be reported for probability or score-based classifiers.

## 15. Calibration

A model is calibrated if predicted probabilities match empirical event frequencies.

For example, among customers assigned predicted churn probability near $0.30$, approximately $30\%$ should actually churn.

A calibrated model satisfies approximately:

$$
P(Y=1 \mid \hat{p}(X)=q) \approx q.
$$

Calibration is different from ranking.

A model can rank customers well but produce probabilities that are too high or too low. In that case, ROC-AUC may be good while probability interpretation is unreliable.

Calibration becomes important when predicted probabilities are used directly for decision-making, expected value calculations, or risk communication.

## 16. Cross-validation and out-of-fold predictions

During model development, this project uses stratified cross-validation inside the training set.

For each fold:

```text
1. Fit preprocessing on the fold-training data only.
2. Fit the model on the fold-training data only.
3. Predict the untouched validation fold.
4. Store the out-of-fold predictions.
```

Out-of-fold predictions are important because every observation is predicted by a model that was not fitted on that observation.

This avoids evaluating a model on the same data used to train it.

After cross-validation, the out-of-fold predictions can be used to compute:

```text
confusion-matrix counts
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
threshold curves
calibration checks
```

## 17. Test-set discipline

The held-out test set should not be used for:

```text
feature engineering decisions
preprocessing choices
model-family selection
hyperparameter tuning
threshold tuning
calibration decisions
resampling decisions
choosing the final metric
deciding whether results are good enough
```

The test set should be used only once at the end for final evaluation of the selected pipeline.

This protects the test set as an estimate of generalization to unseen data.

## 18. Class imbalance and evaluation

Class imbalance affects how metrics should be interpreted.

When one class is much more common, ordinary accuracy can be dominated by the majority class.

Useful metrics under imbalance include:

```text
balanced accuracy
recall
specificity
precision
F1
ROC-AUC
PR-AUC
confusion-matrix counts
predicted positive rate
```

Resampling methods such as oversampling, undersampling, and SMOTE are training interventions. They do not replace careful evaluation.

If resampling is used later, validation and test folds should remain in the natural distribution.

Correct workflow:

```text
for each cross-validation split:
    fit preprocessing on the fold-training data only
    resample the fold-training data only
    fit model on resampled fold-training data
    evaluate on untouched validation fold
```

## 19. Current project baseline interpretation

The simple baselines from section 04 establish important reference points.

The majority-class and prior-probability baselines achieve high ordinary accuracy because no churn is the majority class, but they have zero recall because they detect no churners.

The EDA-inspired rule has high recall but low specificity. It flags many churners, but it also flags many non-churners.

This creates a useful benchmark for later learned models:

```text
A useful learned model should clearly beat dummy baselines,
detect churners with non-trivial recall,
improve precision and specificity compared with the broad EDA rule,
and provide good ranking performance through ROC-AUC and PR-AUC.
```

## 20. Update policy for this note

This note should be updated when the project introduces new evaluation concepts.

Likely future additions:

```text
threshold optimization based on business costs
calibration curves
Brier score
cost-sensitive evaluation
confidence intervals for metrics
bootstrap uncertainty
formal paired model-comparison tests
final test-set reporting template
```

The note should remain a general reusable reference. Detailed results should stay in notebooks and the LaTeX report.
