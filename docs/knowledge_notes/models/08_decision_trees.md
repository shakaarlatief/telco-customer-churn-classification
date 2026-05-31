# Decision Trees for Churn Classification

## Purpose

This document is the knowledge note for the decision-tree stage of the Telco Customer Churn classification project.

The goal is to understand decision trees before implementation: how a tree recursively partitions the feature space, how split rules are selected, how leaf predictions are formed, why tree depth and leaf size control overfitting, how impurity criteria such as entropy and Gini impurity work, and why single trees are useful both as interpretable models and as the foundation for later tree ensembles.

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
07_naive_bayes
```

The previous learned-model sections introduced three distinct modelling philosophies:

```text
Logistic regression:
    discriminative, parametric, global linear score model

k-nearest neighbours:
    non-parametric, instance-based, local distance model

Naive Bayes:
    probabilistic generative model with conditional-independence assumptions
```

Decision trees introduce a fourth modelling idea: **recursive rule-based partitioning**. A tree does not fit a single global linear boundary, does not classify by distance to stored observations, and does not require an explicit probabilistic likelihood for all features. Instead, it repeatedly asks simple questions about feature values and uses the answers to route an observation to a terminal leaf.

This makes decision trees valuable at this point in the project because they introduce:

```text
recursive partitioning
axis-aligned decision rules
leaf-level class distributions
impurity criteria
information gain
Gini impurity
hierarchical interactions
nonlinear decision boundaries
overfitting through deep trees
regularization through depth and leaf constraints
cost-complexity pruning
decision stumps
```

A single decision tree may not be the strongest final model on tabular data, but it is important because bagging, random forests, AdaBoost, and gradient boosting are all easiest to understand after the single-tree model is clear.

## 2. Basic idea

A decision tree represents a classifier as a sequence of questions.

For a customer with feature vector $x$, the model starts at the root node. Each internal node applies a split rule, such as:

```text
Contract_Month-to-month <= 0.5

tenure <= 12.5

MonthlyCharges <= 70.25
```

The answer sends the customer to the left or right child node. This process continues until the customer reaches a terminal leaf. The leaf stores information about the training observations that reached it, such as the number of churners and non-churners.

For binary churn classification, a terminal leaf $m$ contains a subset of training observations:

$$
R_m
\subseteq
\{1,\ldots,n\}.
$$

The empirical churn rate in that leaf is:

$$
\widehat{p}_m
=
\frac{1}{|R_m|}
\sum_{i \in R_m} y_i,
$$

where $y_i=1$ means churn and $y_i=0$ means no churn.

The default hard class prediction in that leaf is:

$$
\widehat{y}_m
=
\mathbb{1}\{\widehat{p}_m \geq 0.5\}.
$$

Thus, the tree predicts by assigning each customer to a leaf and then using the class distribution of the training customers in that leaf.

## 3. Feature-space partitioning

A decision tree partitions the feature space into regions. Each terminal leaf corresponds to one region.

For numeric features, a split usually has the form:

$$
x_j \leq t,
$$

where $x_j$ is feature $j$ and $t$ is a threshold. One child receives observations satisfying the condition, and the other receives observations that do not.

Because each split checks one feature at a time, standard decision-tree boundaries are **axis-aligned** in the transformed feature space. With two numeric features, a tree creates rectangular regions. With more features, it creates hyperrectangular regions.

This is a major contrast with logistic regression. Logistic regression creates one global linear decision boundary in the transformed feature space. A decision tree creates many local regions, each with its own class prediction.

### Churn interpretation

A tree can represent hierarchical decision logic such as:

```text
if Contract is month-to-month:
    if tenure is low:
        high churn risk
    else:
        moderate churn risk
else:
    lower churn risk
```

This kind of hierarchy naturally represents interactions. For example, low tenure may matter differently for month-to-month customers than for two-year-contract customers. A linear logistic regression model only captures this directly if interaction features are explicitly included, while a tree can learn such conditional structure through successive splits.

## 4. Internal nodes, branches, and leaves

A decision tree has three basic components.

```text
Root node:
    the first split applied to all observations

Internal node:
    a non-terminal node that applies a split rule

Branch:
    the path taken after a split condition is evaluated

Terminal leaf:
    a final node that stores a prediction or class distribution
```

For classification, each leaf usually stores class counts:

$$
(n_{m0}, n_{m1}),
$$

where $n_{m0}$ is the number of non-churners in leaf $m$ and $n_{m1}$ is the number of churners.

The estimated leaf probability is:

$$
\widehat{p}_m
=
\frac{n_{m1}}{n_{m0}+n_{m1}}.
$$

This probability is a local training-set proportion. It is useful as a score for ROC and precision-recall analysis, but it should not automatically be interpreted as a well-calibrated probability. Deep trees can create small leaves, and small-leaf proportions can be noisy.

## 5. Greedy top-down tree construction

The space of possible trees is enormous. A practical decision-tree algorithm does not search globally over every possible tree. Instead, it uses a greedy top-down strategy.

A simplified training algorithm is:

```text
Start with one root node containing all training observations.

For each current leaf:
    If a stopping condition is met:
        make the node terminal and assign a leaf prediction.
    Otherwise:
        search over candidate splits.
        choose the split that gives the largest impurity reduction.
        create child nodes.

Repeat until all leaves are terminal.
```

The algorithm is greedy because it chooses the best split at the current node according to a local criterion. Once a split is chosen, the standard algorithm does not backtrack and reconsider earlier choices.

This matters for interpretation. A fitted tree is not guaranteed to be the globally optimal tree of a given size. It is the result of a sequence of locally preferred split decisions.

## 6. Node impurity

A split should create child nodes whose class distributions are less mixed than the parent node. To define this precisely, decision trees use an impurity function.

For binary classification, let $p$ denote the proportion of churners in a node:

$$
p
=
P(Y=1 \mid \text{node}).
$$

A pure node has $p=0$ or $p=1$. A maximally mixed binary node has $p=0.5$.

An impurity function should be:

```text
low for pure nodes
high for mixed nodes
largest near a balanced class distribution
```

Two common impurity criteria are entropy and Gini impurity.

## 7. Entropy

For a node with class probabilities $(p_0,p_1)$, the entropy is:

$$
H(p_0,p_1)
=
-
\sum_{c \in \{0,1\}}
p_c \log_2 p_c.
$$

For binary classification, this can be written as:

$$
H(p)
=
-
 p\log_2 p
-
(1-p)\log_2(1-p),
$$

with the convention that $0\log 0=0$.

Entropy measures class uncertainty. If a node is pure, entropy is zero. If a binary node contains a 50/50 mixture of churners and non-churners, entropy is maximal.

For churn classification:

```text
low entropy leaf:
    mostly churners or mostly non-churners

high entropy leaf:
    a mixed group where the class label remains uncertain
```

## 8. Information gain and impurity reduction

Suppose a parent node contains observation index set $S$. A candidate split divides it into child nodes:

$$
S_L
\quad\text{and}\quad
S_R.
$$

More generally, a split could create children $S_1,\ldots,S_K$, although scikit-learn decision trees use binary splits.

The weighted post-split impurity is:

$$
I_{\text{children}}
=
\sum_{k=1}^{K}
\frac{|S_k|}{|S|}
I(S_k),
$$

where $I(\cdot)$ is the chosen impurity function.

The impurity reduction is:

$$
\Delta I
=
I(S)
-
\sum_{k=1}^{K}
\frac{|S_k|}{|S|}
I(S_k).
$$

When entropy is the impurity function, this quantity is often called **information gain**:

$$
\operatorname{Gain}(S,\text{split})
=
H(S)
-
\sum_{k=1}^{K}
\frac{|S_k|}{|S|}
H(S_k).
$$

The tree chooses the candidate split with the largest impurity reduction.

The weighting by child size is important. A split that creates one tiny pure child and one large mixed child should not automatically be considered excellent. The criterion rewards purity improvements in proportion to the number of observations affected.

## 9. Gini impurity

Another common impurity criterion is Gini impurity.

For a node with class probabilities $(p_0,p_1)$, the Gini impurity is:

$$
G(p_0,p_1)
=
1
-
\sum_{c \in \{0,1\}}
p_c^2.
$$

For binary classification, this becomes:

$$
G(p)
=
1-p^2-(1-p)^2
=
2p(1-p).
$$

Gini impurity is zero for pure nodes and largest at $p=0.5$.

A useful interpretation is that Gini impurity is the probability of misclassification if a class label were randomly assigned according to the node's empirical class distribution. If a node is pure, this random assignment would always be correct. If the node is evenly mixed, random assignment is highly uncertain.

Both entropy and Gini impurity generally prefer splits that make child nodes more class-pure. They often produce similar trees, although not always identical ones.

## 10. Candidate splits for numeric and encoded categorical features

### Numeric features

For a numeric feature $x_j$, the tree considers threshold splits of the form:

$$
x_j \leq t.
$$

The threshold $t$ is chosen from candidate values between observed feature values. The algorithm evaluates the impurity reduction for candidate thresholds and selects the best one for the current node.

For this project, examples include:

```text
tenure <= 12.5
MonthlyCharges <= 70.25
TotalCharges <= 250.0
```

The same numeric feature may be used multiple times in different parts of the tree, or even along the same path, with different thresholds. This allows a tree to create interval-like regions.

### One-hot encoded categorical features

Scikit-learn's standard `DecisionTreeClassifier` expects numeric input. In this project, categorical features are represented through one-hot encoding inside the preprocessing pipeline.

A category such as `Contract = Month-to-month` becomes an indicator feature. A split can then check whether that indicator is below or above a threshold such as $0.5$:

```text
Contract_Month-to-month <= 0.5
```

This is equivalent to separating customers who are not in that category from customers who are in that category.

One-hot encoding is practical and compatible with the existing preprocessing workflow, but it changes the interpretation of categorical splits. A single split usually tests one category indicator at a time rather than splitting a raw categorical feature into all of its levels at once.

## 11. Stopping rules and regularization

If a tree is allowed to grow until every leaf is pure or nearly pure, it can become very deep and highly specific to the training data. Regularization controls this complexity.

Common stopping and regularization parameters include:

```text
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
```

### Maximum depth

`max_depth` limits the number of split levels from the root to a leaf.

```text
small max_depth:
    simpler tree
    higher bias
    lower variance
    easier interpretation

large max_depth:
    more flexible tree
    lower training error
    higher variance
    greater overfitting risk
```

A depth-one tree is called a decision stump.

### Minimum samples split

`min_samples_split` controls the minimum number of observations required at a node before the algorithm is allowed to split it.

A larger value prevents the tree from continuing to split very small groups of customers.

### Minimum samples leaf

`min_samples_leaf` controls the minimum number of observations required in each terminal leaf.

This is especially important for probability estimates. If a leaf contains only a few observations, its churn proportion can be very noisy. Larger leaves produce smoother and more stable predicted probabilities.

### Maximum leaf nodes

`max_leaf_nodes` directly limits the number of terminal regions. This can be useful when interpretability is important because a tree with fewer leaves is easier to inspect.

### Minimum impurity decrease

`min_impurity_decrease` requires a split to improve impurity by at least a specified amount. This prevents the tree from adding splits with negligible gain.

## 12. Overfitting in decision trees

Decision trees are highly flexible. A deep tree can isolate small groups of observations and eventually memorize idiosyncratic training patterns.

Overfitting appears when:

```text
training performance keeps improving
validation performance stops improving or worsens
leaves become very small
rules become too specific
predicted probabilities become extreme and unstable
```

For churn classification, a very deep tree might learn rules that describe accidental details of the training set, such as a small combination of service indicators and charges that happens to contain many churners in one fold. Such a rule may not generalize to new customers.

This creates the usual bias-variance tradeoff:

```text
shallow tree:
    higher bias
    lower variance
    more interpretable
    may underfit

deep tree:
    lower bias
    higher variance
    less stable
    may overfit
```

The decision-tree section should therefore compare an intentionally simple stump, an unconstrained or default tree, and regularized/pruned alternatives.

## 13. Cost-complexity pruning

An alternative to stopping early is to grow a larger tree and then prune it back. This is usually called **post-pruning**. It differs operationally from pre-pruning controls such as `max_depth`, `min_samples_leaf`, and `max_leaf_nodes`, because the tree is first allowed to become relatively large and is then simplified afterward.

Cost-complexity pruning defines a tradeoff between training impurity and tree size. Let $T$ be a tree and let $|T|$ denote the number of terminal leaves. A simplified cost-complexity objective is:

$$
R_\alpha(T)
=
R(T)
+
\alpha |T|,
$$

where:

```text
R(T)      = empirical impurity or error-like criterion for the tree
|T|       = number of leaves
alpha     = complexity penalty
```

When $\alpha=0$, the penalty for tree size disappears and a larger tree may be preferred. As $\alpha$ increases, smaller trees are preferred. In scikit-learn, the relevant parameter is `ccp_alpha`. Larger `ccp_alpha` values prune the tree more aggressively.

From a model-selection perspective, `ccp_alpha` is a hyperparameter. It is not fundamentally different from `max_depth`, `min_samples_leaf`, or `max_leaf_nodes`. The main difference is procedural:

```text
pre-pruning hyperparameters:
    restrict tree growth while the tree is being fitted

post-pruning hyperparameters:
    grow a larger tree first, then select a smaller subtree afterward
```

The validation logic is the same in both cases. The pruning strength should be chosen with validation evidence or cross-validation, not from the held-out test set.

The project can study cost-complexity pruning by:

```text
1. Fit a preliminary tree inside the development workflow.
2. Obtain candidate ccp_alpha values from the pruning path.
3. Evaluate selected ccp_alpha values by cross-validation inside the training set.
4. Select a representative pruned tree using development-stage metrics.
```

The key reason validation is needed is that training performance usually improves as the tree becomes larger. If pruning were chosen by training error alone, the largest tree would often look best. Validation performance can instead reveal the bias-variance tradeoff:

```text
too much pruning:
    tree is too small and underfits

moderate pruning:
    tree keeps useful structure while removing noise

too little pruning:
    tree is too large and overfits training irregularities
```

## 14. Decision stumps

A decision stump is a tree with depth one. It consists of exactly one split and two terminal leaves.

A stump is simple:

```text
one question
one left leaf
one right leaf
```

A stump is usually too weak to be the best standalone classifier, but it is useful for three reasons.

First, it is highly interpretable. It reveals the single split that most improves impurity at the root.

Second, it provides a simple learned rule baseline, more flexible than a manually designed EDA rule but much simpler than a full tree.

Third, stumps are important for later boosting methods. AdaBoost is often explained as building an ensemble of weak learners such as decision stumps. Therefore, including a stump in this section prepares the project for the later boosting section.

## 15. Preprocessing implications

Decision trees have different preprocessing needs from logistic regression and kNN.

### Scaling is not required

Trees split by thresholds and impurity improvements. A monotone rescaling of a numeric variable does not change the ordering of observations, so it usually does not change the possible split structure in the same way it would affect distance-based models or regularized linear models.

Therefore, decision trees do not require standardization of numeric features.

### Numeric imputation is still needed

Scikit-learn's standard `DecisionTreeClassifier` does not accept missing values in the usual pipeline setup. The existing project data should already be clean, but preprocessing should still be pipeline-based and robust.

Numeric missing values can be imputed with the median, and categorical missing values can be imputed with the most frequent category if needed.

### Categorical encoding is still needed

The raw Telco data contains categorical variables. Scikit-learn's standard tree implementation requires numeric input, so categorical variables should still be encoded.

The natural project pipeline is:

```text
numeric features:
    impute if needed
    no scaling required

categorical features:
    impute if needed
    one-hot encode
```

This should be done inside cross-validation folds through a scikit-learn pipeline to avoid leakage and to remain consistent with the rest of the project.

## 16. Outputs of a decision tree

A fitted classification tree can produce several outputs.

### Hard labels

The hard prediction is the majority class in the reached leaf:

$$
\widehat{y}(x)
=
\mathbb{1}\{\widehat{p}_{m(x)} \geq 0.5\},
$$

where $m(x)$ is the leaf reached by observation $x$.

### Predicted probabilities

The predicted churn probability is the empirical churn proportion in the reached leaf:

$$
\widehat{P}(Y=1 \mid X=x)
=
\widehat{p}_{m(x)}.
$$

These scores can be used for ROC and precision-recall curves. However, because leaf proportions can be noisy, especially in small leaves, tree probabilities may be poorly calibrated.

### Ranking with decision trees

A decision tree ranks observations by the predicted churn probability of the leaf they reach. If customer $a$ reaches a leaf with churn proportion $0.70$ and customer $b$ reaches a leaf with churn proportion $0.20$, the tree ranks customer $a$ as a higher churn risk than customer $b$.

This ranking mechanism has a special property: all observations in the same leaf receive exactly the same score. Therefore, a tree produces a **stepwise ranking** with ties. The number of distinct score values is at most the number of leaves.

This has several implications for ROC-AUC and PR-AUC:

```text
small tree:
    few leaves
    few possible scores
    coarse ranking with many ties
    smoother but less flexible

deep tree:
    many leaves
    many possible scores
    more detailed ranking
    higher risk of noisy leaf proportions
```

This connects directly to regularization. A very deep tree may create many small leaves and therefore many score levels, but the score in each small leaf can be unstable. A shallow or pruned tree gives a coarser ranking, but each leaf probability is estimated from more observations and may generalize better.

It is also important that decision trees are not trained to optimize ROC-AUC or PR-AUC directly. Standard classification trees choose splits by local impurity reduction, such as Gini impurity reduction or information gain. A split that improves local purity often helps classification and ranking, but the training criterion is not the same as the final evaluation metric.

For this project, tree ranking quality should therefore be evaluated explicitly with ROC-AUC and PR-AUC from out-of-fold predictions, not inferred only from impurity reductions or tree depth.

### Feature importance

Decision trees can report impurity-based feature importances. A feature receives high importance if it is used in splits that reduce impurity substantially.

However, impurity-based importance should be interpreted cautiously. It can be biased toward features with many possible split points or high cardinality, and in one-hot encoded data importance can be spread across multiple indicators belonging to the same original categorical variable.

For the decision-tree section, feature importance can be useful as a diagnostic, but it should not be treated as a definitive explanation of churn drivers.

## 17. Pruning, hyperparameter tuning, and validation layers

The validation issue around pruning is an example of a more general model-selection principle. Any data used to choose a model, tune a hyperparameter, choose a pruning level, select features, choose a threshold, calibrate probabilities, or compare model families has influenced the modelling process. It is no longer an independent evaluation set.

This matters because decision-tree pruning can be presented in two different validation setups.

### 17.1 Simple development cross-validation

For the section-level decision-tree notebook, the practical workflow can treat `ccp_alpha` like any other tree-complexity hyperparameter. It can be included in a normal cross-validation grid together with parameters such as:

```text
criterion
max_depth
min_samples_leaf
min_samples_split
max_leaf_nodes
ccp_alpha
```

This is appropriate for development-stage model selection. The result should be described carefully:

```text
selected within the tried grid
development-stage cross-validated estimate
not an independent final estimate of the full search procedure
final test evaluation deferred
```

In other words, normal cross-validation is enough to choose a useful pruning value for the current model-family section. The selected cross-validated score is useful development evidence, but it should not be treated as final performance.

### 17.2 Nested validation logic

A stricter setup is needed when one validation set is reserved for higher-level model comparison. Suppose the available labelled data is split into:

```text
training block
outer validation block
test block
```

If the outer validation block is meant to compare model families, such as kNN versus a decision tree, then it should simulate the future test set. In that case, the same outer validation block should not also be used to choose the decision-tree pruning level. The pruning choice should happen inside the training block, using an inner validation split or inner cross-validation.

The logic is:

```text
inner training data:
    fit candidate trees

inner validation data:
    choose pruning level or other tree hyperparameters

outer validation data:
    compare the already-selected tree procedure with other model families

test data:
    final evaluation only after all choices are fixed
```

In cross-validation form, this becomes nested cross-validation:

```text
inner CV:
    tune pruning and other hyperparameters

outer CV:
    estimate or compare the performance of the tuned procedure
```

Nested validation is not required every time a pruning parameter is tuned. It is required for the stricter goal of estimating the performance of the full tune-and-select procedure with less selection optimism.

### 17.3 Practical implication for this project

The Telco project uses ordinary training-set cross-validation inside each model-family section to learn model behaviour and select representative candidates. Therefore, section 08 can tune pruning by normal cross-validation, while explicitly stating that the result is development-stage evidence.

Later, when serious finalist models are compared across model families, the project can use repeated cross-validation or nested-style validation to obtain a more stable comparison of model-selection procedures. The held-out test set remains untouched until one final model, threshold rule, calibration decision, and preprocessing strategy have been fixed.

## 18. Evaluation implications for this project

The decision-tree section should use the same evaluation discipline as the previous model sections.

```text
Use only the training set.
Use stratified cross-validation.
Keep preprocessing inside the pipeline.
Use out-of-fold predictions for metrics and curves.
Do not touch the held-out test set.
Treat section-level results as development-stage estimates.
```

The core metrics remain:

```text
accuracy
balanced accuracy
precision
recall
specificity
F1
ROC-AUC
PR-AUC
positive-first confusion-matrix counts
predicted positive rate
observed positive rate
```

PR-AUC remains important because churn is the minority class. ROC-AUC remains useful for overall ranking. Confusion-matrix counts remain essential because the business meaning of false negatives and false positives is different.

## 19. Planned model variants for section 08

The notebook should evaluate a transparent sequence of decision-tree variants.

### 18.1 Decision stump

Purpose:

```text
simple learned rule
high interpretability
baseline for tree-based learning
bridge to boosting
```

Likely configuration:

```text
DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
```

### 18.2 Default decision tree

Purpose:

```text
show what happens when a tree is allowed to grow with default settings
illustrate flexibility and overfitting risk
provide contrast with regularized trees
```

The default tree is not expected to be the final choice. It is useful because it reveals the high-variance behaviour of unconstrained tree learning.

### 18.3 Tuned regularized tree

Purpose:

```text
study tree-complexity controls
select a representative regularized tree by cross-validation
compare depth and leaf-size effects
```

Possible grid:

```text
criterion in {"gini", "entropy"}
max_depth in {2, 3, 4, 5, 6, 8, 10, None}
min_samples_leaf in {1, 5, 10, 25, 50, 100}
min_samples_split in {2, 10, 25, 50, 100}
ccp_alpha in a small set of values from a pruning path or manual grid
```

The grid should stay transparent and not become excessively large. The goal is to learn how tree regularization behaves, not to exhaustively tune every possible tree. Cost-complexity pruning can be evaluated either as part of the same hyperparameter grid or as a separate pruning-path diagnostic.

### 18.4 Cost-complexity-pruned tree

Purpose:

```text
study pruning as an alternative regularization route
connect tree size to validation performance
compare pruned tree with pre-pruned tuned tree
```

The pruning experiment can be included if it remains manageable. If it becomes too large, it can be simplified to a representative pruning path and a small evaluated alpha grid.

## 20. Selection logic

The section should define a clear development-stage selection rule before inspecting final section outputs.

A reasonable rule is:

```text
Primary metric:
    cross-validated PR-AUC

Secondary metrics:
    ROC-AUC
    balanced accuracy
    F1

Interpretation constraints:
    prefer simpler trees when metrics are very close
    do not overinterpret tiny metric differences
    treat the selected tree as a representative candidate within the tried grid
```

The simplicity preference matters because decision trees can become large and unstable. If two trees have practically similar ranking performance, the smaller tree may be more useful as a single-tree model.

## 21. Expected comparison with earlier models

A single decision tree may improve interpretability and capture nonlinear interactions, but it may also be less stable than logistic regression.

The expected comparison should be framed cautiously:

```text
Logistic regression:
    strong global linear benchmark
    stable ranking performance
    interpretable through coefficients

kNN:
    local similarity model
    sensitive to scaling and distance geometry

Naive Bayes:
    generative probabilistic model
    useful recall but limited by conditional independence

Decision tree:
    rule-based nonlinear model
    interpretable when small
    unstable and prone to overfitting when deep
```

If the tree performs below logistic regression, that does not make the section a failure. The tree teaches a different modelling principle and prepares for ensembles. If a regularized tree performs competitively, it may become a useful candidate, but final model-family claims remain deferred.

## 22. Limitations of single decision trees

Single decision trees have several important limitations.

### Instability

Small changes in the training data can change the selected root split or later splits. Because the algorithm is greedy, an early change can alter the entire downstream tree.

### Axis-aligned splits

Standard trees split one feature at a time. This creates rectangular regions. Some relationships may be represented inefficiently if the true boundary is oblique or smooth.

### Overfitting

Deep trees can memorize training irregularities, especially with small leaves.

### Probability calibration

Leaf proportions may not be calibrated probabilities. Small leaves can produce extreme probabilities such as 0 or 1.

### One-hot feature interpretation

In one-hot encoded data, the tree may split on individual category indicators. This can make original categorical-feature interpretation less direct unless indicator-level splits are mapped back to original variables.

### Bias toward high-opportunity features

Features with more possible thresholds or more encoded indicators may have more opportunities to reduce impurity by chance. This affects impurity-based feature importance and split interpretation.

## 23. Link to later ensemble sections

Single trees are the foundation for later tree ensembles.

```text
Bagging:
    train many high-variance trees on bootstrap samples
    average their predictions to reduce variance

Random forests:
    bagged trees plus feature subsampling
    decorrelate trees to improve variance reduction

Boosting:
    train trees sequentially
    each new tree focuses on errors or gradients from previous trees
```

The current section should not implement bagging, random forests, or boosting. It should prepare for them by making the behaviour of one tree clear.

## 24. Implementation plan

The executable notebook should follow the established project style.

Recommended structure:

```text
08.1 Purpose and methodological discipline
08.2 Import project utilities
08.3 Define output paths
08.4 Load training data only
08.5 Brief decision-tree theory needed for the notebook
08.6 Cross-validation and unscaled preprocessing
08.7 Helper functions for tree experiments
08.8 Evaluate decision stump
08.9 Evaluate default decision tree
08.10 Evaluate regularized tree grid
08.11 Optional cost-complexity pruning experiment
08.12 Select representative decision-tree model
08.13 Confusion-matrix and metric comparison
08.14 Tree complexity diagnostics
08.15 Threshold behaviour for selected tree
08.16 ROC and precision-recall curves
08.17 Optional feature-importance or tree-rule inspection
08.18 Save tables and figures
08.19 Summary and implications for ensembles
```

Suggested saved artifacts:

```text
decision_tree_model_comparison.csv
decision_tree_confusion_matrices.csv
decision_tree_grid_results.csv
decision_tree_pruning_results.csv
decision_tree_threshold_results.csv

decision_tree_pr_auc_by_depth.png
decision_tree_pr_auc_by_leaf_size.png
decision_tree_pruning_curve.png
decision_tree_threshold_tradeoff.png
decision_tree_roc_curve.png
decision_tree_precision_recall_curve.png
decision_tree_feature_importance.png
```

The exact artifact list can be adjusted when the notebook is written, but the section should save enough material for the report.

## 25. Report plan

The polished report section should not repeat every detail from this note. It should include the most important mathematics and the observed results.

Recommended report structure:

```text
1. Introduce decision trees as recursive partitioning models.
2. Explain leaf predictions and class proportions.
3. Explain entropy/Gini impurity and impurity reduction.
4. Explain overfitting and regularization.
5. Describe the evaluated variants: stump, default tree, tuned tree, pruned tree if included.
6. Present cross-validated model comparison table.
7. Present confusion-matrix table.
8. Present selected diagnostic figures.
9. Discuss threshold behaviour and ranking curves.
10. Compare cautiously with logistic regression, kNN, and Naive Bayes.
11. Explain why single trees motivate bagging, random forests, and boosting.
```

The report should use development-stage language. It should not claim final performance, and it should not use the held-out test set.

## 26. Summary

Decision trees classify by recursively partitioning the feature space into leaf regions and assigning predictions based on the class distribution within each leaf. They are intuitive and can capture nonlinear interactions, but they are also prone to overfitting when allowed to grow too deep.

The main technical mechanism is impurity reduction. At each node, the algorithm searches over candidate splits and chooses the split that most reduces class impurity, measured by criteria such as entropy or Gini impurity. Because the algorithm is greedy, a fitted tree is the result of local split decisions rather than a guaranteed globally optimal tree.

For the Telco churn project, decision trees should be evaluated as a sequence from simple to flexible: a decision stump, a default tree, a tuned regularized tree, and possibly a cost-complexity-pruned tree. The section should emphasize both predictive behaviour and interpretability. A single tree may not outperform logistic regression, but it is a crucial modelling stage because it introduces the building block for later tree ensembles.
