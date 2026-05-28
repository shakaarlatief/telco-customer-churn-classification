# Simple Baseline Classifiers

## Purpose

This document is the knowledge note for the simple baseline classifier stage of the Telco Customer Churn classification project.

The goal is to define the baseline models that all learned models should be compared against. Baselines are not included because they are expected to be strong final models. They are included because they answer a more basic question:

```text
Does the modelling workflow learn anything beyond trivial class-frequency rules or simple manually specified patterns?
```

This note is intentionally more detailed than the notebook. The notebook contains the executable workflow, outputs, tables, and result interpretation. The LaTeX report contains the polished explanation and selected results.

## 1. Position in the modelling sequence

The simple baseline stage follows:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
```

and precedes the learned model sequence:

```text
05_linear_classification_and_logistic_regression
06_knn
07_naive_bayes
```

The baseline stage has three roles:

```text
1. establish the evaluation framework;
2. introduce simple reference models;
3. define the minimum standard for later learned models.
```

The project uses the training set only. The held-out test set remains unused.

The report section for this stage already considers five baselines: majority-class, prior-probability, stratified random, uniform random, and an EDA-inspired rule baseline. The report states that the purpose of these baselines is to establish simple reference points that later learned models must improve on, while keeping the held-out test set reserved for final evaluation. fileciteturn67file0L8-L23

## 2. Why baseline classifiers are necessary

A classification model should not only be compared with other complex models. It should first be compared with simple strategies.

Baselines answer questions such as:

```text
What performance is obtained by always predicting the majority class?
What performance is obtained by random guessing with the observed class proportions?
What performance is obtained by a simple hand-written rule from EDA?
Does a learned model improve recall, precision, specificity, and ranking quality beyond these references?
```

Without baselines, a performance score is hard to interpret.

For example, an accuracy of 0.73 might sound reasonable. But if 73% of customers do not churn, then a model can achieve 0.73 accuracy by always predicting no churn. Such a model detects zero churners and is practically useless for churn detection.

Therefore, baselines provide context.

## 3. Training-only evaluation discipline

All baseline performance should be evaluated using the training set and cross-validation, not the held-out test set.

The project uses stratified cross-validation inside the training set. This means each fold preserves approximately the same churn rate as the full training set.

For each fold:

```text
1. Fit the baseline rule using the fold-training data if the rule has fitted quantities.
2. Predict the held-out validation fold.
3. Store the out-of-fold predictions.
4. Compute metrics from all out-of-fold predictions.
```

For baselines such as majority-class and prior-probability classifiers, the class frequencies must be estimated from the fold-training data only.

This matters because even a simple baseline can leak information if it uses the full training labels to define predictions for every fold.

## 4. Positive class and class imbalance

The positive class is churn:

$$
Y=1
\quad \Longleftrightarrow \quad
\text{customer churned}.
$$

The negative class is no churn:

$$
Y=0
\quad \Longleftrightarrow \quad
\text{customer did not churn}.
$$

In the training set, the class distribution is approximately:

```text
No churn: 73.46%
Churn:    26.54%
```

This moderate imbalance is central to baseline interpretation. A model can obtain high accuracy by focusing on the majority class, so accuracy alone is not enough.

Relevant metrics include:

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
predicted positive rate
```

## 5. Majority-class baseline

The majority-class classifier always predicts the most frequent class in the training data.

For a training fold, the fitted majority class is:

$$
\widehat{c}
=
\arg\max_{c \in \{0,1\}}
\sum_{i=1}^{n}
\mathbb{1}\{y_i=c\}.
$$

The prediction rule is:

$$
\widehat{y}(x)
=
\widehat{c}.
$$

Since no churn is the majority class in this dataset, the majority-class baseline always predicts:

```text
no churn
```

This baseline uses no feature information.

### Expected behaviour

The majority-class classifier has:

```text
high ordinary accuracy if the majority class is common
zero recall for the positive churn class
specificity equal to one
balanced accuracy equal to 0.5
precision equal to zero under zero-division convention
F1 equal to zero
```

The majority-class baseline is useful because it exposes why ordinary accuracy is insufficient.

## 6. Prior-probability baseline

The prior-probability baseline estimates the class prior from the training fold:

$$
\widehat{P}(Y=c)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb{1}\{y_i=c\}.
$$

Its predicted probabilities are:

$$
\widehat{P}(Y=0)
\quad\text{and}\quad
\widehat{P}(Y=1).
$$

The hard class prediction is still the class with the highest prior probability.

In this dataset, this again means predicting no churn for every customer.

The difference from the majority-class baseline is that the prior baseline also gives probability estimates equal to the empirical class proportions. These probability estimates are not individualized. Every customer receives the same churn probability.

### Why this baseline matters

This baseline establishes the minimum reference for probability-ranking metrics.

Because every observation receives nearly the same score, ROC-AUC and PR-AUC should be at random or prevalence-level reference values. The PR-AUC should be close to the positive-class prevalence.

## 7. Stratified random baseline

The stratified random baseline predicts labels randomly according to the class distribution estimated from the training fold.

The prediction distribution is:

$$
\widehat{Y}
\sim
\operatorname{Categorical}
\left(
\widehat{P}(Y=0),
\widehat{P}(Y=1)
\right).
$$

In the Telco training data, this means predicting churn with probability approximately 0.265 and no churn with probability approximately 0.735.

This baseline uses no feature information. It only uses the target distribution.

### Expected behaviour

The stratified random baseline should have:

```text
predicted positive rate close to the observed churn rate
balanced accuracy near 0.5
ROC-AUC near 0.5
PR-AUC near the positive-class prevalence
```

It provides a reference for random guessing that respects the class imbalance.

## 8. Uniform random baseline

The uniform random baseline predicts each class with equal probability:

$$
\widehat{Y}
\sim
\operatorname{Categorical}(0.5,0.5).
$$

For binary classification, this means:

```text
P(predicted churn) = 0.5
P(predicted no churn) = 0.5
```

This baseline also uses no feature information.

### Expected behaviour

The uniform random baseline usually predicts many more positives than the observed churn rate when the positive class is the minority class.

In this project, churn prevalence is about 0.265, while the uniform baseline predicts churn about half the time.

This can raise recall compared with the majority-class baseline, but only because the classifier flags many customers. It also creates many false positives.

The uniform random baseline helps distinguish real churn detection from simply predicting churn more often.

## 9. EDA-inspired rule baseline

The EDA-inspired rule baseline uses a small number of high-risk conditions observed in the training-set exploratory analysis.

The rule assigns one risk point for each high-risk condition:

$$
s(x)
=
\sum_{j=1}^{m}
\mathbb{1}
\{
\text{condition }j\text{ is true}
\}.
$$

The predicted class is:

$$
\widehat{y}
=
\mathbb{1}
\{
s(x)\geq t
\}.
$$

In this project, the threshold is:

$$
t=2.
$$

The high-risk conditions are:

```text
month-to-month contract
electronic check payment
fiber optic internet service
no online security
no tech support
```

These conditions were chosen because the training-set EDA showed strong churn-rate differences for these categories. The rule is intentionally simple. It is not optimized to be final. It asks whether obvious EDA patterns already contain predictive signal.

The report defines this rule as a sum of high-risk conditions with threshold \(t=2\), using those five conditions. fileciteturn68file0L48-L75

## 10. What each baseline teaches

Each baseline has a different purpose.

### Majority-class baseline

Teaches:

```text
ordinary accuracy can be misleading
a high-accuracy model can have zero positive-class recall
balanced accuracy is needed under class imbalance
```

### Prior-probability baseline

Teaches:

```text
class-prior probabilities are not individualized predictions
probability metrics need a prevalence-level reference
constant scores do not provide useful ranking
```

### Stratified random baseline

Teaches:

```text
random predictions with the correct class frequency do not create useful feature-based discrimination
balanced accuracy and ROC-AUC should stay near random performance
```

### Uniform random baseline

Teaches:

```text
recall can be increased artificially by predicting the positive class more often
false positives and predicted positive rate must be inspected
```

### EDA-inspired rule baseline

Teaches:

```text
simple feature patterns already contain substantial churn signal
high recall can be achieved with broad rules
precision and specificity may remain weak
learned models should improve the tradeoff
```

## 11. Baseline results in this project

The cross-validated simple-baseline results are:

```text
EDA-inspired rule:
    Accuracy = 0.6076
    Balanced accuracy = 0.7032
    Precision = 0.3956
    Recall = 0.9070
    Specificity = 0.4994
    F1 = 0.5509
    ROC-AUC = 0.8109
    PR-AUC = 0.5435
    Predicted positive rate = 0.6084

Stratified random:
    Accuracy = 0.6140
    Balanced accuracy = 0.5065
    Precision = 0.2748
    Recall = 0.2776
    Specificity = 0.7354
    F1 = 0.2762
    ROC-AUC = 0.5065
    PR-AUC = 0.2680

Uniform random:
    Accuracy = 0.5051
    Balanced accuracy = 0.5057
    Precision = 0.2698
    Recall = 0.5070
    Specificity = 0.5045
    F1 = 0.3522
    ROC-AUC = 0.5000
    PR-AUC = 0.2654

Majority class:
    Accuracy = 0.7346
    Balanced accuracy = 0.5000
    Precision = 0.0000
    Recall = 0.0000
    Specificity = 1.0000
    F1 = 0.0000
    ROC-AUC = 0.5000
    PR-AUC = 0.2654

Prior probability:
    Accuracy = 0.7346
    Balanced accuracy = 0.5000
    Precision = 0.0000
    Recall = 0.0000
    Specificity = 1.0000
    F1 = 0.0000
    ROC-AUC = 0.4999
    PR-AUC = 0.2653
```

The corresponding confusion-matrix counts are:

```text
EDA-inspired rule:
    TP = 1356
    FN = 139
    FP = 2072
    TN = 2067

Stratified random:
    TP = 415
    FN = 1080
    FP = 1095
    TN = 3044

Uniform random:
    TP = 758
    FN = 737
    FP = 2051
    TN = 2088

Majority class:
    TP = 0
    FN = 1495
    FP = 0
    TN = 4139

Prior probability:
    TP = 0
    FN = 1495
    FP = 0
    TN = 4139
```

These values match the report's baseline metric and confusion-matrix tables. fileciteturn68file0L81-L115

## 12. Interpretation of results

The majority-class and prior-probability baselines have the highest ordinary accuracy among the simple baselines:

```text
accuracy = 0.7346
```

But this is misleading. They predict no churn for every customer. As a result, they correctly classify all non-churners but miss every churner.

Their recall is:

```text
recall = 0.0000
```

and their balanced accuracy is:

```text
balanced accuracy = 0.5000
```

This demonstrates that ordinary accuracy is not enough for churn classification.

The stratified random baseline behaves like random guessing with the observed class frequencies. Its predicted positive rate is close to the observed churn rate, and its balanced accuracy is close to 0.5.

The uniform random baseline predicts churn for about half of customers. This raises recall relative to the majority-class baseline, but it also creates many false positives.

The EDA-inspired rule is the only simple baseline using feature information. It detects most churners:

```text
recall = 0.9070
```

but it also flags many non-churners:

```text
FP = 2072
specificity = 0.4994
precision = 0.3956
```

The report interprets this as a high-recall, low-specificity baseline that confirms strong EDA patterns contain signal, but that the rule is too broad to be a final model. fileciteturn68file0L117-L127

## 13. Minimum standard for learned models

A useful learned model should improve on these baselines.

At minimum, later models should:

```text
clearly outperform dummy baselines
avoid relying only on ordinary accuracy
improve balanced accuracy relative to random baselines
detect churners with non-trivial recall
improve precision and specificity relative to the broad EDA rule
provide useful ranking performance through ROC-AUC and PR-AUC
support threshold tuning and calibration analysis later
```

The report uses exactly this logic to define the implications for later models. fileciteturn68file0L129-L142

## 14. Relationship to later sections

The simple baseline stage becomes the reference point for all learned models.

Later model sections should answer:

```text
Does the model beat the dummy baselines?
Does it beat the EDA-inspired rule in precision/specificity while retaining useful recall?
Does it improve ROC-AUC and PR-AUC?
Does it give a better threshold tradeoff?
Does it add interpretability or modelling insight?
```

Examples from later completed sections:

```text
Logistic regression:
    much better precision and specificity than the EDA rule,
    lower recall at threshold 0.5,
    stronger PR-AUC and ROC-AUC.

kNN:
    improves over default distance-based settings after tuning,
    but remains slightly weaker than logistic regression.
```

## 15. Implementation notes

In scikit-learn, dummy baselines are represented by `DummyClassifier`:

```text
strategy = "most_frequent"
strategy = "prior"
strategy = "stratified"
strategy = "uniform"
```

The EDA-inspired rule is implemented as a custom estimator so that it can be evaluated with the same cross-validation infrastructure as the other models.

This is important because every baseline should follow the same evaluation protocol as later learned models. Otherwise, comparisons become unfair.

## 16. Update policy

This note should be updated if the project introduces additional baseline types, such as:

```text
single-feature threshold baselines
cost-sensitive rule baselines
business-rule baselines
simple scorecard baselines
calibrated prior baselines
```

For now, this note documents the baseline models from section 04 and defines their role as the minimum standard for later model families.
