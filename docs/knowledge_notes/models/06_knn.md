# k-Nearest Neighbours for Churn Classification

## Purpose

This document is the knowledge note for the k-nearest neighbours stage of the Telco Customer Churn classification project.

The goal is to understand kNN before implementation: what the model does, why it is called a lazy method, how distances define the prediction rule, why feature scaling matters, how the number of neighbours controls the bias-variance tradeoff, and why high-dimensional one-hot encoded tabular data can make distance-based methods difficult.

This note is intentionally more detailed than the notebook. The notebook should contain the executable workflow, concise explanations, result tables, figures, and interpretation. The LaTeX report should later contain the polished mathematical explanation and selected results.

## 1. Position in the modelling sequence

The project has now established:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_evaluation_and_simple_baselines
05_linear_classification_and_logistic_regression
```

Logistic regression is a learned parametric model. It estimates coefficients and produces a linear score that can be converted into predicted probabilities.

k-nearest neighbours is different. It is a non-parametric, instance-based, distance-based method. It does not learn a small set of coefficients during training. Instead, it stores the training observations and uses them directly when a new observation must be classified.

This makes kNN useful as the next model because it introduces several new ideas:

```text
lazy learning
local classification
distance metrics
feature scaling
nonlinear decision boundaries
bias-variance control through k
high-dimensional distance problems
```

## 2. Basic idea

Let the training set be

$$
\mathcal{D}_{train}
=
\{(x_i,y_i)\}_{i=1}^{n},
$$

where

$$
x_i \in \mathbb{R}^p,
\qquad
y_i \in \{0,1\}.
$$

For a new customer with feature vector $x$, kNN finds the $k$ training observations closest to $x$ under a chosen distance function.

Let

$$
\mathcal{N}_k(x)
$$

denote the index set of the $k$ nearest training observations to $x$.

The unweighted kNN predicted churn probability is

$$
\hat{p}(Y=1 \mid X=x)
=
\frac{1}{k}
\sum_{i \in \mathcal{N}_k(x)}
y_i.
$$

The default class prediction at threshold $\tau=0.5$ is

$$
\hat{y}
=
\mathbb{1}
\left\{
\hat{p}(Y=1 \mid X=x) \geq 0.5
\right\}.
$$

So, kNN classifies a customer by local majority vote among similar customers.

## 3. Lazy learning

kNN is called a lazy learning method because the training stage is minimal.

In a linear model, training estimates parameters such as coefficients. In a decision tree, training chooses splits. In kNN, training mainly stores the training data.

Most of the work happens at prediction time:

```text
Given a new point:
    compute distances to training points
    find the nearest k points
    aggregate their labels
    output class or probability
```

This has an important computational implication. Prediction can be expensive when the training set is large, because the model may need to compare the new point with many stored observations.

For the Telco dataset this is not a serious computational problem because the training set is small enough. But the principle matters for larger datasets.

## 4. Distance functions

The model depends completely on the definition of distance.

### Euclidean distance

The most common distance is Euclidean distance:

$$
d_2(x,x')
=
\sqrt{
\sum_{j=1}^{p}
(x_j-x'_j)^2
}.
$$

Euclidean distance measures straight-line distance in feature space.

### Manhattan distance

Another common choice is Manhattan distance:

$$
d_1(x,x')
=
\sum_{j=1}^{p}
|x_j-x'_j|.
$$

Manhattan distance adds absolute coordinate differences. It can behave differently in high-dimensional spaces and with sparse one-hot encoded features.

### Minkowski distance

Both Euclidean and Manhattan distance are special cases of Minkowski distance:

$$
d_p(x,x')
=
\left(
\sum_{j=1}^{p}
|x_j-x'_j|^p
\right)^{1/p}.
$$

When $p=1$, this is Manhattan distance. When $p=2$, this is Euclidean distance.

In scikit-learn's `KNeighborsClassifier`, the common setting is:

```text
metric = "minkowski"
p = 2
```

which corresponds to Euclidean distance.

## 5. Why scaling is essential

kNN is scale-sensitive because distances are computed directly from feature values.

Suppose one feature is measured in months and another in euros. A one-unit difference in one feature may not mean the same thing as a one-unit difference in another feature. If features are left on their raw scales, large-scale variables can dominate the distance calculation.

For example, in the Telco dataset:

```text
tenure ranges from 0 to 72
MonthlyCharges ranges roughly from 18 to 119
TotalCharges ranges from 0 to 8684.8
```

Without scaling, `TotalCharges` can dominate Euclidean distances simply because it has a much larger numeric range.

Standardization transforms numeric variables as

$$
z_j
=
\frac{x_j-\mu_j}{\sigma_j},
$$

where $\mu_j$ and $\sigma_j$ are estimated from the training data.

In cross-validation, this must happen inside the pipeline:

```text
for each fold:
    fit scaler on fold-training data only
    transform fold-training and validation data
    fit kNN on fold-training transformed data
    evaluate on validation transformed data
```

The validation fold must not influence the fitted scaling parameters.

## 6. Categorical variables and one-hot encoding

kNN requires a numeric feature representation. For this project, categorical features are one-hot encoded.

For example, the variable `Contract` becomes indicators such as:

```text
Contract_Month-to-month
Contract_One year
Contract_Two year
```

One-hot encoding allows kNN to operate on mixed tabular data, but it changes the geometry.

Two customers with different categories differ in the corresponding indicator dimensions. With many categorical variables, the feature space becomes higher-dimensional and partly sparse.

This matters because kNN's notion of similarity is geometric. After one-hot encoding, "nearest" means nearest in the constructed numeric indicator space, not necessarily nearest in a human intuitive sense.

## 7. Weighted voting

The basic kNN probability is an unweighted average:

$$
\hat{p}(Y=1 \mid X=x)
=
\frac{1}{k}
\sum_{i \in \mathcal{N}_k(x)}
y_i.
$$

A weighted version gives closer neighbours more influence:

$$
\hat{p}(Y=1 \mid X=x)
=
\frac{
\sum_{i \in \mathcal{N}_k(x)} w_i(x)y_i
}{
\sum_{i \in \mathcal{N}_k(x)} w_i(x)
}.
$$

A common distance-based weight is approximately

$$
w_i(x)
=
\frac{1}{d(x,x_i)+\epsilon},
$$

where $\epsilon>0$ prevents division by zero.

Weighted voting can help when closer neighbours are much more relevant than farther neighbours. However, it can also increase variance if individual very-close observations strongly influence predictions.

In scikit-learn, the main options are:

```text
weights = "uniform"
weights = "distance"
```

## 8. The role of k

The most important kNN hyperparameter is $k$, the number of neighbours.

### Small k

When $k$ is small, the model is very local.

For $k=1$:

$$
\hat{y}(x)
=
y_{i^\star},
\qquad
i^\star
=
\arg\min_i d(x,x_i).
$$

The prediction copies the label of the single nearest training observation.

This can create flexible and highly irregular decision boundaries. It may fit local structure, but it can also be very sensitive to noise.

Small $k$ generally means:

```text
low bias
high variance
greater sensitivity to noise
more irregular decision boundaries
```

### Large k

When $k$ is large, predictions average over many neighbours.

Large $k$ generally means:

```text
higher bias
lower variance
smoother decision boundaries
less sensitivity to individual noisy observations
```

If $k$ becomes too large, predictions move toward the global class distribution and the model may underfit.

Thus, $k$ controls a bias-variance tradeoff:

```text
small k -> low bias, high variance
large k -> high bias, low variance
```

This is why $k$ should be tuned using cross-validation inside the training set.

## 9. Decision boundaries

kNN can learn nonlinear decision boundaries because it does not impose a linear score function.

The decision boundary is determined by regions of feature space where the local majority class changes.

For small $k$, the boundary can be highly jagged.

For larger $k$, the boundary becomes smoother.

This is an important contrast with logistic regression:

```text
logistic regression:
    global linear boundary in transformed feature space

kNN:
    local neighbourhood-based boundary
```

However, flexibility is not automatically better. If the distance metric is not meaningful, the local neighbourhoods may not correspond to meaningful customer similarity.

## 10. Probability estimates

kNN can produce class probabilities by using the proportion of neighbours in each class.

For binary churn classification:

$$
\hat{p}(Y=1 \mid X=x)
=
\frac{
\#\{i \in \mathcal{N}_k(x): y_i=1\}
}{k}.
$$

With uniform weights, the possible probability values are discrete:

$$
0,\frac{1}{k},\frac{2}{k},\ldots,1.
$$

This means probability estimates can be coarse when $k$ is small.

Distance-weighted kNN can produce less discrete probabilities, but these are still local empirical estimates rather than probabilities from a parametric likelihood.

kNN probabilities are not guaranteed to be well calibrated. Calibration should be checked later if probabilities are used directly for decisions.

## 11. Curse of dimensionality

kNN can struggle in high-dimensional spaces.

As the number of dimensions increases, distances between points can become less informative. Points may all become relatively far away from each other, and the difference between "near" and "far" can shrink.

This is often called the curse of dimensionality.

The Telco dataset is not huge, but one-hot encoding expands the feature space. This can make kNN less competitive than models that learn feature weights or tree splits.

Important consequences:

```text
irrelevant features can hurt kNN
high-dimensional one-hot spaces can weaken distance meaning
feature scaling is necessary but not sufficient
feature selection may help but should be tuned carefully
```

Because this project is educational, kNN is still useful even if it does not outperform logistic regression.

## 12. kNN and class imbalance

kNN is affected by class imbalance because local neighbourhoods may contain more majority-class observations.

If the positive churn class is less frequent, many neighbourhoods may be dominated by non-churners unless churners form local clusters.

Possible responses include:

```text
tuning k
using distance weighting
changing the classification threshold
resampling the training folds
using class-sensitive metrics
```

For this first kNN section, we should not immediately introduce SMOTE or oversampling. We can first evaluate ordinary kNN and distance-weighted kNN under the natural training distribution.

If resampling is tested later, it must occur inside the fold-training data only.

## 13. Hyperparameters for this project

The main kNN hyperparameters are:

```text
n_neighbors:
    number of neighbours k

weights:
    "uniform" or "distance"

metric / p:
    distance definition, e.g. Manhattan p=1 or Euclidean p=2
```

A sensible first grid:

```text
n_neighbors = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
weights = ["uniform", "distance"]
p = [1, 2]
```

This is still small enough for grid search.

Important:

```text
Do not use Optuna yet.
Do not use the held-out test set.
Use stratified cross-validation inside the training set.
Fit preprocessing inside each fold.
```

## 14. Model comparison goals

The kNN section should answer:

```text
Does kNN improve over simple baselines?
Does kNN improve over logistic regression?
How sensitive is performance to k?
Do uniform and distance weighting behave differently?
Does Manhattan or Euclidean distance work better?
Does the selected kNN model produce useful ROC-AUC and PR-AUC?
What recall/precision/specificity tradeoff does kNN produce?
```

Because kNN is a probability-scoring classifier, we should evaluate:

```text
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
confusion matrix counts
threshold behaviour
```

## 15. Expected behaviour on Telco churn

Based on the dataset and previous results, expected behaviour is:

```text
kNN may perform reasonably because there are clear customer segments.
kNN may struggle because the feature space contains many one-hot encoded categorical indicators.
kNN may not beat logistic regression because logistic regression can learn global feature weights.
Small k may overfit and produce unstable probability estimates.
Large k may underfit and move predictions toward the global churn prevalence.
Distance weighting may improve performance for moderate k.
```

This is why kNN is important for learning but not necessarily expected to become the final best model.

## 16. Implementation plan for notebook 06

The notebook should include:

```text
1. Load training data only.
2. Build scaled preprocessing pipeline.
3. Define kNN model factory if needed.
4. Evaluate a baseline kNN model.
5. Grid-search k, weights, and p using training-set cross-validation.
6. Save the full grid results.
7. Plot performance versus k.
8. Select a representative kNN model from the grid.
9. Compute out-of-fold probabilities for the selected model.
10. Save confusion matrix and metric tables.
11. Save ROC and precision-recall curves.
12. Save threshold tradeoff table and plot.
13. Interpret whether kNN adds value relative to logistic regression.
```

The notebook should not contain all the theory in this knowledge note. It should give enough context to understand the workflow and outputs.

## 17. Report plan for section 06

The report section should include:

```text
- why kNN is a lazy, distance-based classifier;
- the neighbour set and voting formula;
- the role of distance functions;
- why scaling is required;
- how k controls bias and variance;
- the hyperparameter grid;
- cross-validated results;
- selected kNN model interpretation;
- comparison with logistic regression;
- limitations in high-dimensional one-hot encoded tabular data.
```

## 18. Update policy

This note should be updated if the kNN implementation introduces additional ideas, such as:

```text
feature selection for kNN
dimensionality reduction before kNN
resampling inside kNN pipelines
calibration of kNN probabilities
approximate nearest-neighbour search
```

For now, the section should stay focused on the core kNN model and its practical behaviour on the Telco training data.
