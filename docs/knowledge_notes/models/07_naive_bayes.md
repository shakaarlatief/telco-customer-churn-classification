# Naive Bayes for Churn Classification

## Purpose

This document is the knowledge note for the Naive Bayes stage of the Telco Customer Churn classification project.

The goal is to understand Naive Bayes before implementation: how it uses Bayes' rule, why it is called a generative classifier, what the conditional-independence assumption means, how categorical and numeric features can be handled, and why the model can be useful even when its assumptions are not literally true.

This note is intentionally more detailed than the notebook. The notebook should contain the executable workflow, concise explanations, result tables, figures, and interpretation. The LaTeX report should later contain the polished mathematical explanation and selected results.

## 1. Position in the modelling sequence

The project has now established:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_evaluation_and_simple_baselines
05_linear_classification_and_logistic_regression
06_knn
```

The previous two learned-model sections introduced two very different inductive biases.

Logistic regression is a discriminative linear probability model. It directly models:

$$
P(Y=1 \mid X=x).
$$

k-nearest neighbours is a non-parametric, distance-based classifier. It predicts from local neighbourhoods in the transformed feature space.

Naive Bayes introduces another modelling philosophy. It is a probabilistic generative classifier. Instead of directly modelling the posterior class probability, it models the class prior and the feature distribution within each class:

$$
P(Y=y),
\qquad
P(X=x \mid Y=y).
$$

Then it uses Bayes' rule to obtain:

$$
P(Y=y \mid X=x).
$$

This makes Naive Bayes useful as the next model because it introduces:

```text
Bayes' rule
Bayes classifier
Bayes risk and Bayes optimality
cost-sensitive Bayes decision rules
class priors
class-conditional likelihoods
generative classification
conditional independence assumptions
categorical likelihoods
Gaussian likelihoods
smoothing
log-probability computations
```

## 2. Bayes' rule for classification

For a class label \(Y\) and feature vector \(X\), Bayes' rule says:

$$
P(Y=y \mid X=x)
=
\frac{
P(X=x \mid Y=y)P(Y=y)
}{
P(X=x)
}.
$$

For classification, the denominator \(P(X=x)\) is the same for every class. Therefore, the predicted class can be obtained by comparing the unnormalized posterior scores:

$$
\widehat{y}
=
\arg\max_{y \in \{0,1\}}
P(X=x \mid Y=y)P(Y=y).
$$

For churn classification:

```text
Y = 1 means churn
Y = 0 means no churn
```

So the model compares:

$$
P(X=x \mid Y=1)P(Y=1)
$$

against

$$
P(X=x \mid Y=0)P(Y=0).
$$

The class prior \(P(Y=y)\) captures how common the class is before observing the customer's features. In this training set, churn is the minority class with prevalence about 0.265.


## 3. The Bayes classifier

Before defining Naive Bayes, it is useful to define the ideal classifier that would be used if the true data-generating distribution were known.

For a feature vector \(x\), the conditional class probabilities are:

$$
P(Y=1 \mid X=x)
\quad\text{and}\quad
P(Y=0 \mid X=x).
$$

The Bayes classifier predicts the class with the largest posterior probability:

$$
h^\star(x)
=
\arg\max_{y \in \{0,1\}}
P(Y=y \mid X=x).
$$

In binary churn classification, this is equivalent to:

$$
h^\star(x)
=
\begin{cases}
1, & \text{if } P(Y=1 \mid X=x) \geq P(Y=0 \mid X=x), \\
0, & \text{otherwise.}
\end{cases}
$$

Since the two posterior probabilities sum to one, this can also be written as:

$$
h^\star(x)
=
\mathbb{1}
\left\{
P(Y=1 \mid X=x) \geq 0.5
\right\}.
$$

This looks similar to the default threshold rule used by logistic regression, but the meaning is different. The Bayes classifier uses the true unknown posterior probability. A fitted model uses an estimated posterior probability.

The Bayes classifier is important because it is the best possible classifier under 0-1 classification loss if the true distribution is known.

## 4. Bayes risk and Bayes optimality

Under 0-1 loss, a classifier receives loss 1 when it predicts the wrong class and loss 0 when it predicts the correct class:

$$
L(h(X),Y)
=
\mathbb{1}\{h(X) \neq Y\}.
$$

The expected classification error, or risk, of a classifier \(h\) is:

$$
R(h)
=
P(h(X) \neq Y)
=
\mathbb{E}
\left[
\mathbb{1}\{h(X) \neq Y\}
\right].
$$

The Bayes classifier minimizes this risk over all possible classifiers:

$$
h^\star
=
\arg\min_h R(h).
$$

The corresponding minimum possible classification error is called the Bayes risk:

$$
R^\star
=
R(h^\star).
$$

This is the irreducible classification error implied by the overlap between the class-conditional distributions. Even a perfect learning algorithm cannot do better than the Bayes risk if the same feature information is used.

For binary classification, the Bayes risk can be written as:

$$
R^\star
=
\mathbb{E}_X
\left[
\min\{
P(Y=0 \mid X),
P(Y=1 \mid X)
\}
\right].
$$

This formula has a useful interpretation. At each feature value \(x\), the best classifier chooses the more likely class. The unavoidable local error is the probability of the less likely class. If churners and non-churners strongly overlap for a given \(x\), even the best possible classifier will sometimes be wrong.

This matters for the Telco project. The EDA and learned models show that churners and non-churners are not perfectly separable. Therefore, some classification error is expected even for strong models.

## 5. Bayes decision rule with unequal costs

The standard Bayes classifier above assumes 0-1 loss: a false positive and a false negative have equal cost.

In churn prediction, that may not be true. A false negative means missing a customer who churns. A false positive means spending retention effort on a customer who would have stayed.

Let:

```text
C_FN = cost of a false negative
C_FP = cost of a false positive
```

If predicting churn triggers a retention action, the cost-sensitive Bayes decision rule predicts churn when the expected cost of predicting churn is lower than the expected cost of predicting no churn.

Predict churn when:

$$
C_{FP}P(Y=0 \mid X=x)
\leq
C_{FN}P(Y=1 \mid X=x).
$$

Equivalently:

$$
P(Y=1 \mid X=x)
\geq
\frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

So the optimal threshold is not always \(0.5\). If missing a churner is much more costly than unnecessarily contacting a non-churner, then \(C_{FN} > C_{FP}\), and the optimal threshold is below \(0.5\).

This connects directly to the threshold analyses in the logistic regression and kNN sections. Threshold tuning is not an arbitrary trick. It is a practical approximation to cost-sensitive decision making.

## 6. From Bayes classifier to Naive Bayes

The Bayes classifier is ideal, but it requires the true posterior probability:

$$
P(Y=y \mid X=x).
$$

In real problems, this posterior is unknown. Naive Bayes approximates it by estimating the class prior and class-conditional feature distributions:

$$
P(Y=y),
\qquad
P(X=x \mid Y=y).
$$

Then it uses Bayes' rule:

$$
P(Y=y \mid X=x)
=
\frac{
P(X=x \mid Y=y)P(Y=y)
}{
P(X=x)
}.
$$

The main difficulty is estimating \(P(X=x \mid Y=y)\). Naive Bayes makes the conditional-independence assumption to make that estimation feasible. Thus, Naive Bayes can be viewed as an approximation to the Bayes classifier.

The modelling hierarchy is:

```text
Bayes classifier:
    ideal classifier using the true posterior P(Y | X)

Naive Bayes:
    practical generative model that estimates P(Y) and P(X | Y)
    using a simplifying conditional-independence assumption

Fitted Naive Bayes classifier:
    estimated version learned from finite training data
```

This distinction is important. Naive Bayes is not automatically Bayes optimal. It is only Bayes optimal if its estimated model matches the true data-generating distribution closely enough, including the conditional-independence structure. In practice, its assumptions are often false, but it can still perform well as a classifier.

## 7. Why Naive Bayes is called generative

A discriminative classifier models the decision boundary or posterior probability directly:

$$
P(Y \mid X).
$$

Logistic regression is an example.

A generative classifier models how data is generated within each class:

$$
P(X \mid Y)
\quad\text{and}\quad
P(Y).
$$

Naive Bayes is generative because it specifies a probabilistic model for the features conditional on the class.

Conceptually, a generative classifier can describe the data-generation process as:

```text
1. Draw a class label Y from the class prior P(Y).
2. Draw features X from the class-conditional distribution P(X | Y).
```

Then classification reverses this process using Bayes' rule.

For this project, the distinction matters because Naive Bayes makes assumptions about the distribution of input features inside each class, not just about the boundary between churners and non-churners.

## 8. The conditional-independence assumption

The full class-conditional feature distribution is:

$$
P(X=x \mid Y=y)
=
P(X_1=x_1,\ldots,X_p=x_p \mid Y=y).
$$

Estimating this full joint distribution is difficult because the feature vector can be high-dimensional and features can have many possible combinations.

Naive Bayes makes a simplifying assumption: conditional on the class, the features are independent.

That means:

$$
P(X=x \mid Y=y)
=
\prod_{j=1}^{p}
P(X_j=x_j \mid Y=y).
$$

The posterior score becomes:

$$
P(Y=y \mid X=x)
\propto
P(Y=y)
\prod_{j=1}^{p}
P(X_j=x_j \mid Y=y).
$$

This is the naive assumption. It is usually not literally true.

For example, in the Telco data:

```text
tenure and TotalCharges are strongly related
InternetService is related to OnlineSecurity, OnlineBackup, TechSupport, and other service add-ons
Contract is related to tenure and churn risk
```

So the conditional-independence assumption is unrealistic.

However, Naive Bayes can still work well as a classifier because accurate classification does not always require an accurate full data-generating model. The posterior ranking can still be useful even when the likelihood model is simplified.

## 9. Log-probability form

Naive Bayes multiplies many probabilities:

$$
P(Y=y)
\prod_{j=1}^{p}
P(X_j=x_j \mid Y=y).
$$

Products of many small probabilities can cause numerical underflow. The usual implementation uses log probabilities.

Taking logs gives:

$$
\log P(Y=y \mid X=x)
=
C
+
\log P(Y=y)
+
\sum_{j=1}^{p}
\log P(X_j=x_j \mid Y=y),
$$

where \(C\) does not depend on the class when comparing classes.

The classification rule becomes:

$$
\widehat{y}
=
\arg\max_{y \in \{0,1\}}
\left[
\log P(Y=y)
+
\sum_{j=1}^{p}
\log P(X_j=x_j \mid Y=y)
\right].
$$

This additive log form is also useful for interpretation: each feature contributes evidence toward each class through its class-conditional log probability.

## 10. Categorical Naive Bayes

For categorical features, Naive Bayes estimates class-conditional category probabilities.

For a categorical feature \(X_j\) with levels \(a\), the estimate is:

$$
\widehat{P}(X_j=a \mid Y=y)
=
\frac{
N_{j,a,y}
}{
N_y
},
$$

where:

```text
N_{j,a,y} = number of training observations with feature j equal to a and class y
N_y       = number of training observations with class y
```

For example, for `Contract`, the model estimates:

$$
P(\text{Contract}=\text{Month-to-month} \mid Y=1)
$$

and

$$
P(\text{Contract}=\text{Month-to-month} \mid Y=0).
$$

If month-to-month contracts are much more common among churners than non-churners, this category gives evidence toward churn.

### Smoothing

A problem occurs if a category appears in one class but not the other. Then the estimated probability can be zero. Because Naive Bayes multiplies probabilities, one zero probability can make the whole posterior score zero.

Laplace or additive smoothing prevents zero probabilities:

$$
\widehat{P}(X_j=a \mid Y=y)
=
\frac{
N_{j,a,y} + \alpha
}{
N_y + \alpha K_j
},
$$

where:

```text
alpha = smoothing strength
K_j   = number of possible categories for feature j
```

With \(\alpha=1\), this is often called Laplace smoothing. With smaller positive values, it is additive smoothing.

Smoothing is especially important when categorical levels are rare.

## 11. Bernoulli Naive Bayes

Bernoulli Naive Bayes is designed for binary indicator features.

After one-hot encoding, many categorical variables become binary indicator columns such as:

```text
Contract_Month-to-month
PaymentMethod_Electronic check
InternetService_Fiber optic
OnlineSecurity_No
```

For a binary feature \(X_j \in \{0,1\}\), Bernoulli Naive Bayes estimates:

$$
P(X_j=1 \mid Y=y)
=
\theta_{jy}.
$$

Then:

$$
P(X_j=x_j \mid Y=y)
=
\theta_{jy}^{x_j}
(1-\theta_{jy})^{1-x_j}.
$$

This means the absence of an indicator also contributes information. For example, not having `Contract_Two year` can affect the posterior through \(1-\theta_{jy}\).

Bernoulli Naive Bayes is natural for one-hot encoded categorical indicators, but it is less natural for continuous numeric variables unless they are binarized. For this project, BernoulliNB can be evaluated on one-hot encoded categorical features and optionally binned or binarized numeric features. However, a clean first version should avoid overly complex binning unless needed.

## 12. CategoricalNB versus one-hot encoding

scikit-learn has `CategoricalNB`, which models categorical variables directly as integer-coded categories. However, it expects categorical non-negative integer features and is not directly designed for continuous numeric features.

The Telco dataset contains both categorical and numeric predictors. There are several possible strategies:

```text
1. Use CategoricalNB on categorical features only.
2. Use GaussianNB on numeric features only.
3. Use BernoulliNB on one-hot encoded categorical features.
4. Use a hybrid/custom likelihood combining categorical and Gaussian components.
```

A fully custom hybrid model could be educational, but it may be too much for this stage unless we explicitly want to implement Naive Bayes ourselves.

A practical and transparent approach is to compare simple variants:

```text
GaussianNB on numeric features only
BernoulliNB on one-hot encoded categorical features only
Hybrid Gaussian-BernoulliNB on numeric plus one-hot categorical features
GaussianNB on the full one-hot encoded transformed feature matrix
```

The hybrid model is the most natural mixed-feature Naive Bayes specification. The last variant is not theoretically ideal for binary one-hot indicators, but it is commonly used as a quick baseline and can still be informative.

## 13. Gaussian Naive Bayes

Gaussian Naive Bayes assumes each numeric feature follows a normal distribution within each class:

$$
X_j \mid Y=y
\sim
\mathcal{N}(\mu_{jy}, \sigma_{jy}^{2}).
$$

The class-conditional density is:

$$
p(x_j \mid Y=y)
=
\frac{1}{\sqrt{2\pi\sigma_{jy}^{2}}}
\exp
\left(
-\frac{(x_j-\mu_{jy})^2}{2\sigma_{jy}^{2}}
\right).
$$

The model estimates \(\mu_{jy}\) and \(\sigma_{jy}^{2}\) from the training data separately for each class.

For the numeric Telco features:

```text
tenure
MonthlyCharges
TotalCharges
```

the Gaussian assumption is not perfect. EDA showed skewness and multimodality, especially for `TotalCharges` and `MonthlyCharges`.

Still, GaussianNB can serve as a useful numeric-only probabilistic baseline.

## 14. Hybrid Gaussian-Bernoulli Naive Bayes

The most natural Naive Bayes specification for this Telco dataset combines different likelihoods for different feature types.

The raw feature space contains:

```text
numeric features:
    tenure
    MonthlyCharges
    TotalCharges

categorical features:
    customer/account/service indicators such as Contract, InternetService,
    OnlineSecurity, TechSupport, PaymentMethod, and others
```

After preprocessing, the numeric features remain continuous, while the categorical features are represented as one-hot binary indicators.

A hybrid Naive Bayes model can therefore use:

```text
Gaussian likelihoods for numeric features
Bernoulli likelihoods for one-hot categorical indicators
```

Let \(x_{\mathcal{N}}\) denote the numeric part of the feature vector and \(z_{\mathcal{B}}\) denote the binary one-hot indicator part.

The hybrid class-conditional likelihood is:

$$
P(X=x, Z=z \mid Y=y)
=
\prod_{j \in \mathcal{N}}
p(x_j \mid Y=y)
\prod_{k \in \mathcal{B}}
P(z_k \mid Y=y).
$$

For numeric features:

$$
X_j \mid Y=y
\sim
\mathcal{N}(\mu_{jy}, \sigma_{jy}^{2}).
$$

For binary indicators:

$$
Z_k \mid Y=y
\sim
\operatorname{Bernoulli}(\theta_{ky}).
$$

The class log score is:

$$
\log P(Y=y)
+
\sum_{j \in \mathcal{N}}
\log p(x_j \mid Y=y)
+
\sum_{k \in \mathcal{B}}
\log P(z_k \mid Y=y).
$$

This model is more theoretically coherent than applying GaussianNB to the full one-hot encoded feature matrix, because binary indicators are not Gaussian continuous variables.

The hybrid model is therefore the preferred Naive Bayes specification to include before interpreting the section. It is also educational because it shows that Naive Bayes is not one single fixed algorithm: it is a modelling framework where each feature or feature group can have a suitable class-conditional likelihood.

## 15. Multinomial Naive Bayes

Multinomial Naive Bayes is often used for count data, especially text classification. It assumes features represent counts or frequencies, such as word counts in a document.

The Telco dataset is not naturally count-based. Numeric variables such as `MonthlyCharges` and `TotalCharges` are continuous amounts, not counts. One-hot indicators are binary categories, not word-count vectors.

Therefore, MultinomialNB is not the most natural model for this project. It can be mentioned conceptually, but it does not need to be a main experiment unless there is a clear transformed representation that justifies non-negative count-like features.

## 16. Naive Bayes and feature scaling

Unlike kNN, Naive Bayes is not distance-based. Standardization is not required for the same reason.

For GaussianNB, scaling a numeric variable changes the estimated mean and variance, but the Gaussian density adjusts accordingly. In exact arithmetic, a simple linear rescaling should not change the classifier's information content if applied consistently.

For BernoulliNB and CategoricalNB, scaling is inappropriate because the features represent categories or binary events.

Therefore, the preprocessing choice should match the Naive Bayes variant:

```text
GaussianNB:
    numeric features, usually no scaling required

CategoricalNB:
    categorical features encoded as integer categories

BernoulliNB:
    one-hot encoded or binary features

MultinomialNB:
    non-negative count-like features
```

This is different from logistic regression and kNN, where a general scaled one-hot pipeline was appropriate.

## 17. Naive Bayes and correlated features

The major weakness of Naive Bayes is the conditional-independence assumption.

In this dataset, several features are clearly dependent:

```text
tenure and TotalCharges
InternetService and service add-on variables
StreamingTV and StreamingMovies
Contract and tenure
```

When correlated features all point in the same direction, Naive Bayes may effectively double-count evidence.

For example, if several internet-service add-on indicators all encode similar information, the product of their likelihoods can make the model overconfident. This is why Naive Bayes probabilities are often poorly calibrated even when classification performance is reasonable.

This matters for threshold analysis. Naive Bayes may rank customers usefully but produce probabilities that should not be treated as calibrated without checking.

## 18. Evaluation plan

The Naive Bayes section should use the same training-only evaluation framework as previous sections:

```text
stratified cross-validation on the training set
no held-out test set use
out-of-fold predictions
confusion matrix
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
threshold table
ROC curve
precision-recall curve
```

Because Naive Bayes is probabilistic, ROC and PR curves are important.

## 19. Candidate experiments

A useful first experiment set is:

```text
1. GaussianNB on numeric features only
2. BernoulliNB on one-hot encoded categorical features only
3. Hybrid Gaussian-BernoulliNB on numeric plus one-hot categorical features
4. GaussianNB on one-hot encoded full feature matrix
```

This separates the contribution of numeric and categorical information while also including the theoretically cleaner mixed-feature Naive Bayes model.

## 20. Expected behaviour on Telco churn

Expected behaviour:

```text
Numeric-only GaussianNB may perform moderately because tenure and MonthlyCharges contain signal.
Categorical-only BernoulliNB may perform well because contract, internet service, support/security services, payment method, and billing contain strong churn patterns.
Hybrid Gaussian-BernoulliNB should be more theoretically coherent than full GaussianNB because it uses suitable likelihoods for the two feature types.
Full GaussianNB may be affected by unrealistic Gaussian assumptions for one-hot indicators.
Naive Bayes may have useful ROC-AUC and PR-AUC but may be less calibrated than logistic regression.
Naive Bayes may be more sensitive to correlated groups of features than logistic regression.
```

It is not obvious that Naive Bayes will beat logistic regression. The main value of this section is to learn probabilistic generative classification and compare its assumptions against discriminative and distance-based models.

## 21. Implementation plan for notebook 07

The notebook should include:

```text
1. Load training data only.
2. Split features and target.
3. Build model-specific preprocessing pipelines.
4. Evaluate numeric-only GaussianNB.
5. Evaluate categorical-only BernoulliNB.
6. Evaluate hybrid Gaussian-BernoulliNB.
7. Evaluate full transformed GaussianNB.
8. Compare models using cross-validated metrics.
9. Select a representative Naive Bayes model by PR-AUC.
10. Produce out-of-fold predicted probabilities.
11. Save model comparison table.
12. Save confusion-matrix table.
13. Save threshold table and threshold plot.
14. Save ROC and precision-recall curves.
15. Interpret performance versus logistic regression and kNN.
```

## 22. Report plan for section 07

The report section should include:

```text
- Bayes' rule for classification
- Bayes classifier and Bayes optimality
- Bayes risk under 0-1 loss
- cost-sensitive Bayes decision rule
- generative versus discriminative classification
- Naive Bayes conditional-independence assumption
- categorical / Bernoulli / Gaussian likelihoods
- hybrid Gaussian-Bernoulli Naive Bayes for mixed tabular features
- smoothing for categorical or Bernoulli probabilities
- model variants used in this project
- cross-validated results
- threshold behaviour
- ROC and PR curves
- comparison with logistic regression and kNN
- limitations of independence assumptions and probability calibration
```

## 23. Update policy

This note should be updated if the implementation introduces additional details, such as:

```text
direct CategoricalNB with ordinal/category encoders
custom hybrid Naive Bayes
probability calibration experiments
feature-group-specific Naive Bayes models
smoothing sensitivity analysis
```

For now, the section should stay focused on core Naive Bayes ideas and a transparent comparison of simple Naive Bayes variants.
