# 09. Bagging and Random Forests

## 1. Purpose of this model-family note

This note prepares the next modelling section of the Telco Customer Churn classification project: bagging and random forests.

The previous decision-tree section introduced a single tree as a nonlinear, rule-based classifier. A single tree partitions the feature space into leaves, estimates a local churn proportion in each leaf, and then uses those leaf proportions for hard classification and ranking. The executed decision-tree workflow showed an important pattern: the unrestricted tree overfit badly, while a regularized pre-pruned tree was much stronger. The selected single tree reached development-stage PR-AUC around 0.628 and ROC-AUC around 0.824, which was useful but still below logistic regression.

Bagging and random forests are the natural next step. They keep the basic decision-tree building block but replace one fitted tree by a collection of fitted trees. The central question becomes:

```text
Can averaging many high-variance trees reduce instability and improve churn ranking performance?
```

This section should remain focused on bagging and random forests only. Boosting is deliberately left for the next model-family section. Bagging trains many base learners independently, usually in parallel, and combines their predictions. Boosting trains learners sequentially, with later learners focusing on mistakes or residual structure from earlier learners. That conceptual distinction should be preserved in the project structure.

## 2. From a single tree to an ensemble

A decision tree is an unstable learner. Small changes in the training data can change which split is selected near the top of the tree. Because upper splits determine which observations are available to lower nodes, early split changes can produce a substantially different tree. This is especially true for deep trees, where small leaves fit local training-set irregularities.

Instability is not always bad. A high-variance model can capture complex structure, but its fitted form depends strongly on the particular sample. The idea behind bagging is to exploit this instability constructively. Instead of trying to find one best tree, we fit many trees on perturbed versions of the training data and average their predictions.

For classification, an ensemble consists of base classifiers

$$
h_1, h_2, \ldots, h_B,
$$

where $B$ is the number of fitted base models. A hard-voting ensemble predicts the class that receives the most votes:

$$
\hat{y}(x)
=
\operatorname{mode}\{h_1(x), h_2(x), \ldots, h_B(x)\}.
$$

For binary classification with labels in $\{0,1\}$, the ensemble can also estimate a churn probability by averaging the individual tree probabilities:

$$
\hat{p}_{\mathrm{ens}}(Y=1 \mid X=x)
=
\frac{1}{B}\sum_{b=1}^{B}\hat{p}_b(Y=1 \mid X=x).
$$

If each tree only contributes a hard class vote, then the vote fraction is

$$
\hat{p}_{\mathrm{vote}}(Y=1 \mid X=x)
=
\frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\{h_b(x)=1\}.
$$

In scikit-learn tree ensembles, predicted probabilities are typically averages of the predicted class probabilities from the individual trees. Since each tree probability is a leaf class proportion, the ensemble probability is an average of many leaf-level estimates.

The key practical implication is that the ensemble can produce a smoother ranking than a single tree. A single tree gives all observations in the same leaf exactly the same score. A forest averages many tree scores, so two customers that are tied in one tree may be separated by other trees.

## 3. Bootstrap resampling

Bagging relies on bootstrap resampling. Given a training set

$$
D = \{(x_i,y_i)\}_{i=1}^{n},
$$

a bootstrap sample $D^{*(b)}$ is created by sampling $n$ observations from $D$ with replacement. Some original observations appear more than once, and some are left out.

For a particular observation $i$, the probability that it is not selected in one draw is

$$
1 - \frac{1}{n}.
$$

Because a bootstrap sample has $n$ draws, the probability that observation $i$ is never selected is

$$
\left(1 - \frac{1}{n}\right)^n.
$$

As $n$ becomes large,

$$
\left(1 - \frac{1}{n}\right)^n \rightarrow e^{-1} \approx 0.368.
$$

Therefore, each bootstrap sample contains about

$$
1 - e^{-1} \approx 0.632
$$

of the distinct original observations, with the remaining roughly 36.8 percent left out of that bootstrap sample. The included observations are not all unique, because some are duplicated.

This is important for two reasons:

```text
1. Each base learner sees a different perturbed version of the training set.
2. The observations left out of a tree's bootstrap sample can be used for out-of-bag evaluation.
```

The bootstrap can be interpreted as sampling from the empirical distribution of the observed data. As the dataset grows, the empirical distribution becomes a better approximation to the unknown data-generating distribution. In bagging, this gives a practical way to create multiple plausible training samples from one observed dataset.

## 4. Bootstrap aggregation, or bagging

Bootstrap aggregation, abbreviated as bagging, follows this general algorithm.

Given a base learning algorithm $\mathcal{A}$, a training set $D$, and a number of ensemble members $B$:

```text
For b = 1, ..., B:
    1. Draw a bootstrap sample D^{*(b)} from D.
    2. Fit a base model h_b = A(D^{*(b)}).

For prediction:
    regression: average predictions;
    classification: vote or average predicted probabilities.
```

For binary probability prediction, the bagged prediction is

$$
\hat{p}_{\mathrm{bag}}(x)
=
\frac{1}{B}\sum_{b=1}^{B}\hat{p}_b(x).
$$

The corresponding default hard prediction at threshold $\tau=0.5$ is

$$
\hat{y}_{\tau}(x)
=
\mathbf{1}\{\hat{p}_{\mathrm{bag}}(x) \geq \tau\}.
$$

In this project, threshold 0.5 is only a default diagnostic threshold. The final operating threshold should still be selected later using training-only validation evidence.

Bagging does not change the base learner's objective. If the base learner is a decision tree, each tree is still built by recursive partitioning and impurity reduction. Bagging changes the training data supplied to each base learner and changes the prediction rule by averaging over fitted models.

## 5. Why bagging reduces variance

Bagging is most helpful for high-variance, unstable learners. A single deep tree can vary strongly across training samples. Averaging many such trees can stabilize the prediction.

A simplified variance calculation shows the idea. Suppose the prediction errors of $B$ base models have equal variance $\sigma^2$ and pairwise correlation $\rho$. The variance of their average is

$$
\operatorname{Var}\left(\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)\right)
=
\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.
$$

This expression has two important parts.

First, as $B$ increases, the second term decreases:

$$
\frac{1-\rho}{B}\sigma^2 \rightarrow 0.
$$

Second, the first term remains:

$$
\rho\sigma^2.
$$

Therefore, averaging many models is most effective when the base models are not too correlated. If all trees make very similar errors, averaging cannot remove much variance. If the trees are diverse and their errors are weakly correlated, averaging can reduce variance substantially.

This explains why random forests add feature subsampling on top of bootstrap sampling. Bootstrap samples perturb the rows. Feature subsampling perturbs the columns considered at each split. Both mechanisms reduce correlation between trees.

## 6. Bagging with decision trees

A bagged tree ensemble fits many decision trees on bootstrap samples and averages their predictions. In scikit-learn, this can be implemented with `BaggingClassifier` using a `DecisionTreeClassifier` as the base estimator.

For trees, bagging is especially natural because decision trees are unstable. A deep tree can fit the data very closely and has low bias but high variance. Bagging many such trees can reduce variance while preserving much of the tree's ability to capture nonlinearities and interactions.

A bagged-tree model can learn effects that a single shallow tree cannot capture, because each individual tree can be flexible. At the same time, the average across trees is less sensitive to one particular tree's split sequence.

The main bagging hyperparameters are:

```text
n_estimators:
    number of trees in the ensemble.

max_samples:
    number or fraction of rows sampled for each base tree.

bootstrap:
    whether rows are sampled with replacement.

base tree complexity:
    max_depth, min_samples_leaf, min_samples_split, and related tree controls.
```

For classical bagging, `bootstrap=True`. The usual bootstrap sample size is the same as the original training size, corresponding to `max_samples=1.0` in scikit-learn.

Bagged trees usually benefit from relatively flexible base trees. If each base tree is too shallow, the ensemble may have high bias. If each base tree is very deep, bagging can still reduce variance, but computation increases and probability estimates can still be imperfect. In practice, a small grid over base-tree complexity can be useful.

## 7. Random forests

A random forest is bagging specialized for decision trees, with an additional source of randomness. Each tree is trained on a bootstrap sample of rows, and at each split, the algorithm considers only a random subset of features as candidate split variables.

The general random forest procedure is:

```text
For b = 1, ..., B:
    1. Draw a bootstrap sample of rows.
    2. Grow a decision tree.
    3. At each split, randomly select a subset of features.
    4. Choose the best split only among that subset.

For prediction:
    average tree probabilities or take a majority vote.
```

The feature subsampling step is the main difference between bagged trees and random forests. In a bagged tree ensemble, every split can consider all features. In a random forest, each split considers only a random subset.

The random feature subset size is controlled by `max_features`. For classification, common choices include:

```text
sqrt:
    consider approximately sqrt(p) features at each split.

log2:
    consider approximately log2(p) features at each split.

None:
    consider all features, making the model closer to ordinary bagged trees.

float values:
    consider a fixed fraction of features at each split.
```

Here $p$ is the number of transformed features after preprocessing. Because the Telco pipeline one-hot encodes categorical variables, $p$ is larger than the original 19 input columns. This matters because `max_features="sqrt"` is computed in the transformed feature space.

## 8. Why feature subsampling helps

In a tabular dataset, some features can be highly predictive. In the Telco churn data, contract type, tenure, internet service, payment method, and several service add-ons have strong relationships with churn. If every tree can always use all features, many trees may choose the same high-signal variables near the top. The trees then become strongly correlated.

Feature subsampling forces different trees, and even different nodes inside a tree, to consider different candidate variables. Some trees may split first on contract type, while others may split on tenure, internet service, payment method, or other features. This decorrelates the trees.

From the variance formula above, reducing the correlation $\rho$ is valuable because the limiting variance of the average is $\rho\sigma^2$. Adding more trees reduces only the part divided by $B$; reducing tree correlation reduces the part that remains even when $B$ is large.

This is why random forests are often stronger than plain bagged trees: they combine row perturbation with feature perturbation.

## 9. What random forests do and do not fix

Random forests mainly reduce variance. They do not primarily reduce bias. If the base learner cannot represent the true signal, averaging many versions of it will not automatically solve that limitation.

For deep decision trees, the base learner is flexible and usually low-bias but high-variance. This is the regime where bagging and random forests are useful.

For very simple trees, such as stumps, the base learner may have high bias. Bagging many stumps can reduce variance, but the ensemble may still underfit because every base model is too weak. Boosting is the later method family that more directly addresses bias by combining weak learners sequentially.

The distinction for the project is:

```text
Bagging/random forests:
    independent or parallel trees;
    variance reduction;
    averaging/voting;
    less sensitive to one fitted tree.

Boosting:
    sequential trees;
    each later learner depends on previous learner errors or gradients;
    often stronger bias reduction;
    more sensitive to learning rate, number of estimators, and overfitting control.
```

This distinction should be stated in the report, but boosting experiments should not be added to the bagging/random-forest section.

## 10. Outputs: hard classes, probabilities, and rankings

Bagged trees and random forests output both hard classes and probabilities. In binary classification, the predicted probability is typically an average over tree-level probabilities:

$$
\hat{p}(Y=1 \mid X=x)
=
\frac{1}{B}\sum_{b=1}^{B}\hat{p}_b(Y=1 \mid X=x).
$$

This probability is also a ranking score. Customers with larger predicted probabilities are ranked as higher churn risk.

Compared with a single tree, the ensemble ranking is usually smoother. A single tree with $M$ leaves can produce at most $M$ distinct probability values. A forest averages probability values across many trees, which can produce many more possible score levels.

This matters for ROC-AUC and PR-AUC. The random forest is not directly trained to optimize ROC-AUC or PR-AUC. It is trained by fitting many impurity-based trees and averaging them. Therefore, ranking quality must be evaluated explicitly using cross-validated predicted probabilities.

The same caution about calibration applies. A random forest probability is an average of empirical leaf proportions. It may rank customers well without being perfectly calibrated. Calibration should be considered later if final probabilities are interpreted as literal churn risks or used in cost-sensitive decision rules.

## 11. Out-of-bag observations and out-of-bag evaluation

Because each bootstrap sample excludes roughly 36.8 percent of the distinct training observations, each tree has a natural set of out-of-bag observations. An observation is out-of-bag for tree $b$ if it was not selected in that tree's bootstrap sample.

For observation $i$, define

$$
\mathcal{B}_{\mathrm{OOB}}(i)
=
\{b : (x_i,y_i) \notin D^{*(b)}\}.
$$

The out-of-bag prediction for observation $i$ can be formed by averaging only the trees for which $i$ was out-of-bag:

$$
\hat{p}_{\mathrm{OOB}}(x_i)
=
\frac{1}{|\mathcal{B}_{\mathrm{OOB}}(i)|}
\sum_{b\in \mathcal{B}_{\mathrm{OOB}}(i)}\hat{p}_b(Y=1 \mid X=x_i).
$$

This gives a training-only performance estimate without an explicit validation split. It is conceptually similar to cross-validation in the sense that each out-of-bag prediction is made by trees that did not train on that observation.

However, out-of-bag evaluation should not be confused with the final held-out test set. It is still part of model development. It can be useful for diagnostics and hyperparameter screening, but the project should continue to use consistent stratified cross-validation for section-level comparisons unless there is a clear reason to switch.

In the notebook, out-of-bag score can be included as an additional diagnostic for random forests, but the main comparison should remain based on the same training-set cross-validation framework used in earlier sections.

## 12. Hyperparameters for bagging and random forests

The main hyperparameters for this section are listed below.

### Number of estimators

`n_estimators` controls the number of trees.

More trees usually reduce Monte Carlo variability in the ensemble prediction. Performance often improves quickly at first and then plateaus. More trees increase runtime and model size. Unlike boosting, adding more trees to a bagged or random-forest ensemble does not usually cause severe overfitting in the same way, because the ensemble average stabilizes as more trees are added. Nevertheless, runtime and diminishing returns matter.

A reasonable educational grid might compare values such as:

```text
n_estimators in {100, 300, 500}
```

If runtime is acceptable, a final representative forest can use 500 trees. If runtime is high, 300 trees may be enough for the section.

### Maximum features

`max_features` controls how many transformed features are considered at each split. Smaller values decorrelate trees more strongly but can increase bias if important features are often unavailable.

Candidate values:

```text
max_features in {"sqrt", "log2", 0.5, None}
```

`None` lets each split consider all transformed features and is therefore closer to bagging.

### Tree depth and leaf size

`max_depth`, `min_samples_leaf`, and `min_samples_split` control base-tree complexity. Random forests often use deep trees, but leaf-size constraints can improve probability stability and reduce noise.

Candidate values:

```text
max_depth in {None, 6, 10, 14}
min_samples_leaf in {1, 5, 10, 25}
min_samples_split in {2, 10, 25}
```

The grid should not become too large. A practical section can evaluate a few transparent configurations rather than an exhaustive search.

### Bootstrap and sample size

For bagging and random forests, `bootstrap=True` is the classical setting. `max_samples` can control the bootstrap sample size. Smaller bootstrap samples can increase diversity, while full-size bootstrap samples are the standard choice.

Candidate values:

```text
bootstrap = True
max_samples in {None, 0.8}
```

In scikit-learn, `max_samples=None` with `bootstrap=True` means each tree receives a bootstrap sample of size equal to the original training set.

### Class weighting

`class_weight="balanced"` can make trees more sensitive to the minority churn class. This may increase recall but can also increase false positives. Because threshold tuning already controls precision-recall tradeoffs, class weighting should be treated carefully.

The first random-forest section can include class weighting as an optional variant rather than making it the main experiment.

## 13. Preprocessing implications

Tree ensembles do not require numeric feature scaling. Split rules depend on feature order, not Euclidean distance or coefficient penalty scale. Therefore, the unscaled preprocessing pipeline is appropriate.

The pipeline still needs categorical encoding because scikit-learn's standard tree ensembles require numeric inputs. The current project uses one-hot encoding. This means:

```text
Original categorical variable:
    Contract with levels Month-to-month, One year, Two year.

Transformed features:
    Contract_Month-to-month, Contract_One year, Contract_Two year.
```

A tree split on a one-hot feature tests membership in a category or group encoded by that indicator. This differs from tree implementations that support native categorical splits, such as CatBoost or some gradient boosting libraries. For this project, the one-hot approach remains consistent with earlier scikit-learn sections.

A random forest with one-hot features can still learn interactions. For example, one branch may split on contract type, then another branch may split on tenure or internet service. The ensemble averages many such interaction structures.

## 14. Feature importance

Random forests provide impurity-based feature importances through `feature_importances_`. For feature $j$, the importance aggregates the impurity reductions produced by splits on that feature across all trees.

Impurity-based feature importance is useful but limited. It can be influenced by:

```text
feature cardinality;
correlated predictors;
one-hot encoding choices;
the impurity criterion;
the fitted forest structure.
```

In the Telco dataset, correlated predictors are important. For example, tenure and TotalCharges are strongly related, and several internet-service add-ons are structurally related to whether the customer has internet service. Importance can be split among correlated features or assigned more strongly to one of them depending on the fitted trees.

Therefore, impurity-based importance should be interpreted as a model-usage diagnostic:

```text
Which transformed features did the forest use to reduce impurity?
```

It should not be interpreted as:

```text
Which features causally determine churn?
```

Permutation importance is a useful later alternative because it measures performance degradation after a feature is shuffled. However, permutation importance also has caveats under correlated features. For section 09, impurity-based importance plus careful wording is sufficient, and permutation importance can be left for a later interpretability or final-comparison stage unless needed.

## 15. Evaluation plan for the Telco project

The section should use the same development-stage evaluation discipline as previous model sections.

The held-out test set remains unused. All section-level results should use `data/processed/train.csv` and stratified cross-validation.

The main metrics should include:

```text
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
predicted positive rate
observed positive rate
positive-first confusion-matrix counts
```

The primary selection metric should remain PR-AUC, with balanced accuracy and F1 as secondary diagnostics. PR-AUC is appropriate because churn is the minority class and the project cares about identifying likely churners.

The section should compare:

```text
1. Selected single decision tree from section 08, as the reference tree baseline.
2. Bagged tree ensemble.
3. Random forest default or baseline configuration.
4. Tuned random forest configuration from a small transparent grid.
5. Optional class-weighted random forest variant if useful.
```

The section should save:

```text
reports/tables/bagging_random_forest_model_comparison.csv
reports/tables/bagging_random_forest_confusion_matrices.csv
reports/tables/random_forest_grid_results.csv
reports/tables/random_forest_threshold_results.csv
reports/tables/random_forest_feature_importance.csv

reports/figures/random_forest_grid_pr_auc.png
reports/figures/random_forest_grid_balanced_accuracy.png
reports/figures/random_forest_threshold_tradeoff.png
reports/figures/random_forest_roc_curve.png
reports/figures/random_forest_precision_recall_curve.png
reports/figures/random_forest_feature_importance.png
```

If a bagged-tree grid is included separately, it can save similarly named bagging grid results. To keep the section manageable, the main grid can focus on random forests while including one or a few bagged-tree reference models.

## 16. Comparison with previous sections

The report should compare the selected random-forest model against previous development-stage models:

```text
Logistic regression:
    ROC-AUC about 0.846
    PR-AUC about 0.658

Selected kNN:
    ROC-AUC about 0.836
    PR-AUC about 0.628

Selected hybrid Naive Bayes:
    ROC-AUC about 0.822
    PR-AUC about 0.615

Selected single decision tree:
    ROC-AUC about 0.824
    PR-AUC about 0.628
```

The important question is whether the random forest improves over the single tree and whether it can approach or improve on logistic regression.

A likely qualitative expectation is:

```text
Random forests should improve over a single tree because averaging reduces tree instability.
Whether they beat logistic regression is empirical and must be determined from the executed notebook results.
```

The report should not assume that random forests will dominate. On many tabular problems they perform strongly, but a well-regularized logistic regression can remain competitive when the signal is mostly additive and well captured by one-hot encoded categories and numeric features.

## 17. Selection optimism and nested validation caveat

The random-forest section will try multiple hyperparameter settings. Therefore, the selected configuration should be described as selected within a development grid.

The selected cross-validated score is useful development evidence, but it is not an unbiased estimate of the full tune-and-select procedure. This is the same methodological point established earlier for logistic regression, kNN, Naive Bayes, and decision trees.

For stricter model-family comparison, a later final comparison stage can use repeated cross-validation or nested cross-validation for serious candidates. In nested cross-validation, the inner loop would choose random-forest hyperparameters, and the outer loop would estimate the performance of that hyperparameter-selection procedure.

For section 09, ordinary stratified cross-validation is appropriate because the goal is educational model-family development, not final performance certification.

Recommended language:

```text
The selected random forest is the strongest configuration in the tried development grid according to PR-AUC, with balanced accuracy and F1 used as secondary diagnostics. The result is a development-stage estimate, not a final test-set claim. Final model-family comparison and final test-set evaluation remain deferred.
```

Avoid language such as:

```text
The random forest is definitively the best model.
The selected hyperparameters are uniquely optimal.
This CV score is final performance.
```

## 18. Expected notebook structure

The notebook should follow the established model-section workflow.

### Step 1: imports and paths

Load project utilities, scikit-learn ensemble estimators, plotting tools, and output paths.

### Step 2: load training data only

Use `data/processed/train.csv`. Do not load or inspect the held-out test set.

### Step 3: define reusable estimators

Create pipelines using the unscaled preprocessor:

```text
BaggingClassifier + DecisionTreeClassifier
RandomForestClassifier
```

Use fixed `random_state=42` where available.

### Step 4: evaluate reference models

Evaluate:

```text
selected single decision tree reference
bagged trees
baseline random forest
```

The single-tree reference can be reconstructed from the selected section 08 configuration:

```text
criterion = "gini"
max_depth = 6
min_samples_split = 25
min_samples_leaf = 10
ccp_alpha = 0
```

### Step 5: tune random forest grid

Use a moderate grid over:

```text
n_estimators
max_features
max_depth
min_samples_leaf
class_weight if included
```

Keep the grid small enough to run locally without making the notebook too slow.

### Step 6: select representative random forest

Sort by:

```text
PR-AUC descending
balanced accuracy descending
F1 descending
```

Then reconstruct and evaluate the selected model with out-of-fold predictions.

### Step 7: threshold diagnostics

Use out-of-fold predicted probabilities for threshold curves.

### Step 8: ROC and PR curves

Save ROC and precision-recall curves for the selected forest.

### Step 9: feature importance

Fit the selected forest on the full training set only for interpretation and save top impurity-based feature importances. This full-training refit should not be used for performance reporting.

### Step 10: interpretation placeholders and later update

The first notebook version can contain clear placeholders for actual results. After the user runs it locally and sends the outputs, the interpretation should be updated with observed numbers before the report section is written.

## 19. Expected report structure

The LaTeX report section should be written only after the executed notebook outputs are available. It should probably contain:

```text
1. Purpose of bagging and random forests.
2. Bootstrap sampling and bagging.
3. Variance reduction and tree correlation.
4. Random forests and feature subsampling.
5. Experimental design.
6. Random-forest grid results.
7. Model comparison against single tree and earlier models.
8. Threshold behaviour.
9. ROC and precision-recall curves.
10. Feature importance.
11. Summary and transition to boosting.
```

The section should explicitly state that bagging and random forests are tree ensembles built from independently trained trees, whereas boosting is handled later as a sequential ensemble method.

## 20. Practical expectations for section 09

The section should teach the following lessons:

```text
1. Single decision trees are unstable.
2. Bootstrap samples create different training sets from the empirical distribution.
3. Bagging averages many unstable learners to reduce variance.
4. Random forests add feature subsampling to decorrelate trees further.
5. More trees usually stabilize performance but have diminishing returns.
6. Random forests often improve over a single tree, but they do not automatically solve every problem.
7. Random forests mainly reduce variance, while boosting is introduced later as a sequential approach that can reduce bias more directly.
8. Feature importance from random forests is useful but not causal.
9. Section-level CV results remain development-stage estimates.
```

