# Linear Classification and Logistic Regression

## Purpose

This document is a knowledge note for the linear classification and logistic regression stage of the Telco Customer Churn classification project.

The goal is to understand the mathematical foundations, modelling assumptions, preprocessing requirements, evaluation implications, and implementation plan before writing the executable notebook and the polished report section.

This note is intentionally more detailed than the final report. The report should later use the most important parts in polished standalone language and combine them with the actual model results.

In other words:

```text
Knowledge note = deep technical reference and modelling plan.
Notebook = executable workflow with code, outputs, and result interpretation.
LaTeX report = polished portfolio-ready explanation with selected results.
```

## Current project context

Completed stages:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_evaluation_and_simple_baselines
```

Important points from section 04:

- The held-out test set remains untouched.
- Development-stage evaluation uses stratified cross-validation inside the training set.
- Evaluation metrics are based on out-of-fold predictions.
- The positive class is churn: `Churn_binary = 1`.
- The target is moderately imbalanced:
  - no churn: 4139 observations, 73.46%;
  - churn: 1495 observations, 26.54%.
- Accuracy alone is not sufficient.
- Later learned models must be compared using confusion-matrix counts, recall, specificity, precision, F1, balanced accuracy, ROC-AUC, and PR-AUC.
- Resampling methods such as oversampling, undersampling, and SMOTE are not used yet. If used later, they must be applied inside cross-validation training folds only.

Simple baseline results from section 04:

```text
EDA-inspired rule:
- high recall: 0.9070
- low specificity: 0.4994
- precision: 0.3956
- balanced accuracy: 0.7032
- ROC-AUC: 0.8109
- PR-AUC: 0.5435
```

Interpretation:

- The EDA rule catches most churners but creates many false positives.
- Learned models should try to keep useful recall while improving precision and specificity.
- Majority-class and prior-probability baselines have high ordinary accuracy but zero recall.

## Why section 05 comes next

The project now moves from non-learned or weak baselines to learned linear classifiers.

Linear classifiers are a natural next step because they are:

1. mathematically simple enough to explain deeply;
2. important foundations for later models such as logistic regression, perceptron, SVMs, and neural networks;
3. useful practical baselines for tabular binary classification;
4. interpretable through coefficients when features are encoded carefully.

The section should not jump straight to `LogisticRegression()` as a black box. It should first explain the linear score and decision boundary, then compare different ways to train a linear classifier.

## Section 05 modelling goals

Section 05 should answer:

1. What is a linear classifier?
2. What does the linear decision boundary mean?
3. Why is direct classification error difficult to optimize?
4. What is least-squares classification?
5. Why can least-squares classification be used as an illustrative learned baseline?
6. Why is logistic regression better suited for binary classification?
7. How does logistic regression produce probabilities?
8. What is log loss / binary cross-entropy?
9. How does regularization control model complexity?
10. How do L1 and L2 penalties differ?
11. How should linear models be evaluated on the churn dataset?
12. What do the learned coefficients suggest about churn risk?
13. How does threshold choice affect recall, precision, and specificity?

## Proposed section structure

Recommended notebook/report structure:

```text
05.1 Purpose of this stage
05.2 Reusing the evaluation framework from section 04
05.3 Linear score functions
05.4 Linear decision boundaries
05.5 Zero-one loss and surrogate losses
05.6 Least-squares classification
05.7 Logistic regression as a probabilistic linear classifier
05.8 Log loss and maximum likelihood
05.9 Regularization: L2, L1, and optionally elastic net
05.10 Preprocessing for linear models
05.11 Cross-validated linear-model comparison
05.12 Logistic regression regularization experiment
05.13 Coefficient interpretation
05.14 Threshold behaviour
05.15 Summary and implications for later models
```

## 1. Linear score function

A linear classifier begins with a score:

$$
f(x) = w^\top x + b.
$$

Here:

- $x \in \mathbb{R}^p$ is a feature vector;
- $w \in \mathbb{R}^p$ is a vector of feature weights;
- $b \in \mathbb{R}$ is an intercept;
- $f(x)$ is a real-valued score.

The score is not yet a probability. It is simply a signed number. Large positive scores support the positive class, and large negative scores support the negative class.

For binary classification with labels encoded as $0$ and $1$, a simple decision rule is:

$$
\hat{y}
=
\begin{cases}
1, & f(x) \geq 0,\\
0, & f(x) < 0.
\end{cases}
$$

For labels encoded as $-1$ and $+1$, the prediction can be written as:

$$
\hat{y}^{\star} = \operatorname{sign}(f(x)).
$$

where:

$$
y^\star \in \{-1,+1\}.
$$

### Churn interpretation

For churn:

$$
f(x) > 0
$$

means the customer is assigned to the churn side of the linear boundary.

$$
f(x) < 0
$$

means the customer is assigned to the non-churn side.

The magnitude of $f(x)$ can be interpreted as confidence only in a rough score sense. It is not a calibrated probability unless the model is given a probabilistic interpretation, as in logistic regression.

## 2. Linear decision boundary

The decision boundary is where the classifier is exactly indifferent between the two classes:

$$
w^\top x + b = 0.
$$

This is a hyperplane in the feature space.

In two dimensions, it is a line. In three dimensions, it is a plane. In higher dimensions, it is a hyperplane.

The vector $w$ is perpendicular to the decision boundary. Moving in the direction of $w$ increases the score $f(x)$, and moving in the opposite direction decreases the score.

### Important limitation

A linear classifier can only separate the feature space with a linear boundary in the transformed feature representation. After one-hot encoding categorical variables and scaling numeric variables, the model can still represent useful additive effects. However, it does not automatically capture complex nonlinear interactions unless those interactions are explicitly represented as features.

For example, logistic regression can learn:

```text
month-to-month contract increases churn risk
fiber optic internet increases churn risk
low tenure increases churn risk
```

But without interaction features, it may not fully capture:

```text
fiber optic internet is especially risky for short-tenure customers on month-to-month contracts
```

Tree-based models and boosted models later can learn such interactions more naturally.

## 3. Zero-one loss and why surrogate losses are used

The most direct classification loss counts whether the prediction is wrong:

$$
L_{0/1}(y_i,\hat{y}_i)
=
\mathbb{1}\{y_i \neq \hat{y}_i\}.
$$

The empirical classification error is:

$$
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb{1}\{y_i \neq \hat{y}_i\}.
$$

This is exactly related to accuracy:

$$
\text{Accuracy}
=
1 -
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb{1}\{y_i \neq \hat{y}_i\}.
$$

However, zero-one loss is difficult to optimize directly because it is discontinuous and flat almost everywhere. Small changes in $w$ often do not change the predicted labels at all, and at the boundary the loss jumps.

Therefore, learned classifiers usually minimize a differentiable or more easily optimized surrogate loss. Examples include:

```text
least-squares loss    -> least-squares classifier
log loss              -> logistic regression
hinge loss            -> support vector machine
```

This is an important conceptual distinction:

```text
Training objective is not always the same as evaluation metric.
```

For example, logistic regression is trained by minimizing log loss, but we may evaluate it using recall, precision, balanced accuracy, ROC-AUC, PR-AUC, and confusion-matrix counts.

## 4. Least-squares classification

Least-squares classification treats classification as a regression-like problem.

First, binary labels are often encoded as:

$$
y_i^\star =
\begin{cases}
+1, & y_i = 1,\\
-1, & y_i = 0.
\end{cases}
$$

Then the model fits a linear score:

$$
f(x_i) = w^\top x_i + b
$$

by minimizing squared error between the score and the encoded class label:

$$
\min_{w,b}
\sum_{i=1}^{n}
\left(
w^\top x_i + b - y_i^\star
\right)^2.
$$

A class prediction is then made by thresholding the score at zero:

$$
\hat{y}_i =
\begin{cases}
1, & w^\top x_i + b \geq 0,\\
0, & w^\top x_i + b < 0.
\end{cases}
$$

### Matrix form

Let $X \in \mathbb{R}^{n \times p}$ be the feature matrix and let $y^\star \in \mathbb{R}^n$ contain labels in $\{-1,+1\}$. If the intercept is included in $X$, the least-squares objective is:

$$
\min_{\beta}
\|X\beta - y^\star\|_2^2.
$$

When $X^\top X$ is invertible, the ordinary least-squares solution is:

$$
\hat{\beta}
=
(X^\top X)^{-1}X^\top y^\star.
$$

In practice, after one-hot encoding, the design matrix can have many columns and possible collinearity. A regularized version is more stable:

$$
\min_{\beta}
\|X\beta - y^\star\|_2^2
+
\alpha \|\beta\|_2^2.
$$

This corresponds conceptually to a ridge-regularized least-squares classifier.

### Why include least-squares classification?

Least-squares classification is not usually the best classification model, but it is useful because it shows the first attempt at using a linear score for classification.

It also demonstrates why classification-specific losses are needed. Squared error treats the target labels as numeric values. A point with a very large score can dominate the loss even if it is already correctly classified. Logistic regression avoids this by modelling probabilities and optimizing log likelihood for binary outcomes.

### Implementation decision

For the project, use scikit-learn's `RidgeClassifier` as the practical implementation of a regularized least-squares linear classifier.

Reason:

- it is a standard, reliable implementation;
- it fits a linear classifier with squared-loss logic and L2 regularization;
- it handles high-dimensional one-hot encoded data more safely than unregularized least squares;
- it has a clear mathematical connection to least-squares classification.

The report can explain it as a regularized least-squares classifier.

## 5. Logistic regression

Logistic regression is a probabilistic linear classifier.

It starts with the same linear score:

$$
z_i = w^\top x_i + b.
$$

Instead of thresholding this score directly, logistic regression maps it to a probability using the sigmoid function:

$$
\sigma(z)
=
\frac{1}{1+\exp(-z)}.
$$

The predicted churn probability is:

$$
\hat{p}_i
=
P(Y_i=1 \mid X_i=x_i)
=
\sigma(w^\top x_i + b).
$$

The non-churn probability is:

$$
P(Y_i=0 \mid X_i=x_i)
=
1-\hat{p}_i.
$$

### Why sigmoid?

The linear score $w^\top x+b$ can be any real number. A probability must lie between 0 and 1. The sigmoid maps:

$$
(-\infty,\infty) \rightarrow (0,1).
$$

It has useful limits:

$$
\lim_{z\to\infty}\sigma(z)=1,
\qquad
\lim_{z\to-\infty}\sigma(z)=0.
$$

And at zero:

$$
\sigma(0)=0.5.
$$

Therefore, with the default threshold $0.5$, logistic regression predicts churn when:

$$
\sigma(w^\top x + b) \geq 0.5.
$$

Because $\sigma(0)=0.5$ and the sigmoid is monotonic, this is equivalent to:

$$
w^\top x + b \geq 0.
$$

So logistic regression has a linear decision boundary, but it also gives probability estimates.

## 6. Odds and log-odds interpretation

Logistic regression can also be written in terms of odds.

The odds of churn are:

$$
\frac{P(Y=1\mid x)}{P(Y=0\mid x)}
=
\frac{p(x)}{1-p(x)}.
$$

Logistic regression assumes the log-odds are linear in the features:

$$
\log
\left(
\frac{p(x)}{1-p(x)}
\right)
=
w^\top x + b.
$$

This means a one-unit increase in feature $x_j$, holding other features fixed, changes the log-odds by $w_j$.

Exponentiating gives the odds ratio:

$$
\exp(w_j).
$$

If:

$$
\exp(w_j) > 1,
$$

then increasing $x_j$ increases the odds of churn.

If:

$$
\exp(w_j) < 1,
$$

then increasing $x_j$ decreases the odds of churn.

### Important caveat for this project

Coefficient interpretation depends on preprocessing.

For scaled numeric features, the coefficient corresponds to a one-standard-deviation increase in the original variable, not a one-unit increase.

For one-hot encoded categorical features, coefficients are relative to an omitted or reference representation depending on encoding. If all one-hot columns are included with regularization, coefficients are still useful for directional interpretation, but they should be interpreted carefully as model parameters, not causal effects.

## 7. Bernoulli likelihood and log loss

For binary classification, assume:

$$
Y_i \mid X_i=x_i
\sim
\operatorname{Bernoulli}(\hat{p}_i).
$$

The probability of observing $y_i$ is:

$$
P(Y_i=y_i \mid x_i)
=
\hat{p}_i^{y_i}
(1-\hat{p}_i)^{1-y_i}.
$$

Assuming observations are conditionally independent given the model parameters, the likelihood is:

$$
L(w,b)
=
\prod_{i=1}^{n}
\hat{p}_i^{y_i}
(1-\hat{p}_i)^{1-y_i}.
$$

The log-likelihood is:

$$
\ell(w,b)
=
\sum_{i=1}^{n}
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
$$

Maximum likelihood chooses:

$$
(\hat{w},\hat{b})
=
\arg\max_{w,b}
\ell(w,b).
$$

Equivalently, minimize the negative log-likelihood:

$$
\mathcal{L}(w,b)
=
-\sum_{i=1}^{n}
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
$$

This is binary cross-entropy or log loss.

### Intuition

If $y_i=1$, the loss contribution is:

$$
-\log(\hat{p}_i).
$$

So the loss is small when the model gives high churn probability to an actual churner.

If $y_i=0$, the loss contribution is:

$$
-\log(1-\hat{p}_i).
$$

So the loss is small when the model gives low churn probability to a non-churner.

Confident wrong predictions are punished heavily. This is useful because a model that predicts a probability near 1 for a non-churner, or near 0 for a churner, should receive a large penalty.

## 8. Regularization

After one-hot encoding, the Telco dataset has many model features. Regularization is important because it controls coefficient magnitude and reduces overfitting.

The unregularized logistic regression objective is:

$$
\mathcal{L}(w,b)
=
-\sum_{i=1}^{n}
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
$$

### L2 regularization

L2-regularized logistic regression minimizes:

$$
\mathcal{L}_{L2}(w,b)
=
\mathcal{L}(w,b)
+
\lambda \|w\|_2^2.
$$

where:

$$
\|w\|_2^2 = \sum_{j=1}^{p}w_j^2.
$$

L2 regularization shrinks coefficients toward zero but usually does not set them exactly to zero.

It is useful when many features have small or moderate predictive contributions.

### L1 regularization

L1-regularized logistic regression minimizes:

$$
\mathcal{L}_{L1}(w,b)
=
\mathcal{L}(w,b)
+
\lambda \|w\|_1.
$$

where:

$$
\|w\|_1 = \sum_{j=1}^{p}|w_j|.
$$

L1 regularization can set some coefficients exactly to zero. This makes it useful for feature selection and sparse interpretation.

### Elastic net

Elastic net combines L1 and L2 penalties:

$$
\mathcal{L}_{EN}(w,b)
=
\mathcal{L}(w,b)
+
\lambda
\left[
\alpha \|w\|_1
+
(1-\alpha)\|w\|_2^2
\right].
$$

This can be useful when there are correlated groups of features. It is optional for this section.

### Scikit-learn convention

In scikit-learn logistic regression, the main regularization strength parameter is $C$, where:

$$
C \propto \frac{1}{\lambda}.
$$

Larger $C$ means weaker regularization.

Smaller $C$ means stronger regularization.

This is important for interpreting regularization plots.

## 9. Preprocessing for linear models

Linear models require numeric inputs. Therefore, preprocessing must:

1. impute missing values if present;
2. scale numeric features;
3. one-hot encode categorical features.

Even though the cleaned Telco training set has no missing values, imputers remain inside the pipeline for robustness and to keep the workflow production-compatible.

Numeric features should be standardized:

$$
x_{ij}^{scaled}
=
\frac{x_{ij}-\mu_j}{s_j}.
$$

This is important because regularization penalizes coefficient magnitudes. Without scaling, a feature measured in large units can receive a small coefficient and a feature measured in small units can receive a large coefficient, making the penalty uneven across features.

One-hot encoding maps categorical variables to binary indicator variables.

Example:

```text
Contract = Month-to-month
Contract = One year
Contract = Two year
```

becomes binary columns such as:

```text
Contract_Month-to-month
Contract_One year
Contract_Two year
```

For linear models, preprocessing must be inside the cross-validation pipeline so that scalers and encoders are fitted only on the training fold.

## 10. Evaluation strategy for section 05

Use the same evaluation framework as section 04:

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

For each model:

1. fit preprocessing inside the fold;
2. fit the classifier on the fold's training data;
3. produce out-of-fold predictions and scores;
4. compute:
   - TP, FN, FP, TN;
   - accuracy;
   - balanced accuracy;
   - precision;
   - recall;
   - specificity;
   - F1;
   - ROC-AUC;
   - PR-AUC;
   - predicted positive rate;
   - observed positive rate.

Important:

- Do not load or inspect the test set.
- Do not tune thresholds on the test set.
- Do not apply resampling outside the cross-validation training folds.
- For this section, probably do not use resampling yet. First understand ordinary linear classifiers under the natural training distribution.

## 11. Proposed models to fit

### A. Regularized least-squares classifier

Implementation:

```text
Pipeline(
    preprocessor=scaled_preprocessor,
    classifier=RidgeClassifier(...)
)
```

Purpose:

- learned linear classifier;
- connects to least-squares classification;
- useful reference before logistic regression.

Potential issue:

- `RidgeClassifier` gives hard predictions and a decision function, but not calibrated probabilities.
- ROC-AUC can use the decision function.
- PR-AUC can also use the decision function.
- It does not naturally output probabilities unless calibrated later.

### B. Logistic regression with weak/default L2 regularization

Implementation:

```text
LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000
)
```

Purpose:

- first proper probabilistic linear classifier;
- directly interpretable through coefficients and odds ratios;
- gives predicted probabilities.

### C. Logistic regression L2 grid

Try values such as:

```text
C = [0.001, 0.01, 0.1, 1, 10, 100]
```

Purpose:

- examine effect of regularization strength;
- compare validation metrics;
- see if stronger or weaker regularization improves generalization.

### D. Logistic regression L1 grid

Use solver:

```text
solver="liblinear"
```

or:

```text
solver="saga"
```

Try:

```text
C = [0.001, 0.01, 0.1, 1, 10]
```

Purpose:

- examine sparse coefficients;
- see which features remain important;
- compare performance with L2.

### E. Optional elastic net

Use solver:

```text
solver="saga"
```

Try a small grid over:

```text
C
l1_ratio
```

This may be useful but can be postponed if section 05 becomes too large.

## 12. Outputs to save

Recommended tables:

```text
reports/tables/linear_model_comparison.csv
reports/tables/logistic_regression_l2_regularization_results.csv
reports/tables/logistic_regression_l1_regularization_results.csv
reports/tables/logistic_regression_top_coefficients.csv
reports/tables/logistic_regression_confusion_matrices.csv
```

Recommended figures:

```text
reports/figures/logistic_regression_l2_regularization_metrics.png
reports/figures/logistic_regression_l1_regularization_metrics.png
reports/figures/logistic_regression_top_coefficients.png
reports/figures/logistic_regression_roc_curve.png
reports/figures/logistic_regression_precision_recall_curve.png
```

Optional figures:

```text
reports/figures/logistic_regression_threshold_tradeoff.png
reports/figures/logistic_regression_calibration_curve.png
```

Threshold tradeoff and calibration might be introduced here but fully developed later in the model-comparison and calibration section.

## 13. Coefficient interpretation plan

After fitting a selected logistic regression model on cross-validation or on the full training set for interpretation, extract the transformed feature names and coefficients.

Important:

- The coefficient table is for interpretation, not final test evaluation.
- Coefficients are associations learned from the training data, not causal effects.
- Numeric coefficients correspond to standardized numeric features.
- Categorical coefficients correspond to encoded category indicators.

Potential table:

```text
feature | coefficient | odds_ratio | direction | absolute_coefficient
```

where:

$$
\text{odds ratio} = \exp(w_j).
$$

Sort by absolute coefficient.

Interpretation examples:

- positive coefficient: associated with higher churn odds;
- negative coefficient: associated with lower churn odds;
- large magnitude: stronger effect in the fitted linear model.

## 14. Threshold behaviour plan

Default logistic regression uses threshold 0.5 for class predictions.

But churn decisions may prefer a different threshold.

For a range of thresholds:

$$
\tau \in \{0.05,0.10,\ldots,0.95\},
$$

compute:

```text
precision
recall
specificity
F1
predicted positive rate
```

This shows how lowering the threshold increases recall but also increases false positives.

This section can introduce threshold behaviour, but final threshold selection should happen later in the model-comparison section, after all model families have been compared.

## 15. Class imbalance handling in section 05

Do not immediately use oversampling, undersampling, or SMOTE in the first logistic regression comparison.

First fit ordinary linear models under the natural training distribution.

Then possibly include one subsection:

```text
Class-weighted logistic regression
```

with:

```text
class_weight="balanced"
```

This is simpler and less leakage-prone than resampling. It can be implemented directly in scikit-learn and used inside the pipeline.

More advanced resampling methods should be saved for a dedicated imbalance experiment after several learned models exist. If used, they must be inside an `imblearn` pipeline.

## 16. Notebook implementation plan

The notebook should:

1. load only the training data;
2. split `X` and `y`;
3. create scaled preprocessing;
4. define the linear classifiers;
5. evaluate them with out-of-fold cross-validation;
6. save metric and confusion-matrix tables;
7. run regularization grids for logistic regression;
8. choose a representative logistic regression model for interpretation;
9. fit the representative model on the full training set for coefficient extraction only;
10. save coefficient table and coefficient figure;
11. optionally plot ROC and PR curves using cross-validated out-of-fold scores.

Important distinction:

- Cross-validated metrics are for performance estimation.
- Fitting one selected model on the full training set is for interpretation and later final-model training, not for reporting in-sample performance.

## 17. Source module changes needed

Update `models.py` with model factories:

```python
def make_ridge_classifier(...)
def make_logistic_regression_classifier(...)
def make_l1_logistic_regression_classifier(...)
def make_l2_logistic_regression_classifier(...)
def make_balanced_logistic_regression_classifier(...)
def make_classifier_pipeline(...)
```

Possibly update `evaluation.py` with:

```python
get_out_of_fold_predictions(...)
make_roc_curve_dataframe(...)
make_precision_recall_curve_dataframe(...)
evaluate_threshold_grid(...)
```

Possibly update or create `features.py` with:

```python
get_feature_names_from_preprocessor(...)
extract_linear_model_coefficients(...)
```

The code should stay modular but not over-abstracted. The notebook should still clearly show the modelling logic.

## 18. Report writing plan

The LaTeX report section should include:

1. model purpose and connection to previous baselines;
2. linear score and decision boundary;
3. least-squares classification;
4. logistic regression and sigmoid;
5. Bernoulli likelihood and log loss;
6. regularization;
7. preprocessing for linear models;
8. evaluation results table;
9. regularization results;
10. coefficient interpretation;
11. threshold discussion;
12. summary of what linear models teach us.

The report should include the math, but in polished standalone language. It should not say "the lecture says." It should read like a technical project report.

## 19. What not to include yet

Do not include:

- kNN;
- Naive Bayes;
- decision trees;
- random forests;
- gradient boosting;
- SVMs;
- MLPs;
- SMOTE experiments;
- final test evaluation.

Those all have later sections.

## 20. Immediate next step

Create:

```text
notebooks/05_linear_classification_and_logistic_regression.py
```

and update:

```text
src/telco_churn/models.py
src/telco_churn/evaluation.py
src/telco_churn/features.py
```

as needed.

After the notebook runs, inspect the actual output tables and figures. Only then write:

```text
reports/latex/sections/05_linear_classification_and_logistic_regression.tex
```
