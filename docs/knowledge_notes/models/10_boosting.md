# 10. Boosting, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost

## 1. Purpose of this model-family note

This note prepares the boosting section of the Telco Customer Churn classification project.

The previous tree sections established three important ideas:

```text
1. A single decision tree is interpretable, nonlinear, and rule based, but unstable.
2. Bagging and random forests reduce tree instability by averaging many independently trained trees.
3. The next natural question is whether trees can be combined sequentially rather than independently.
```

Boosting answers the third question. Instead of fitting many trees in parallel and averaging them, boosting builds an additive model one component at a time. Each new component is trained after seeing what the current ensemble does poorly. The general idea is therefore:

```text
Current ensemble makes imperfect predictions.
A new weak learner is trained to correct, reweight, or follow the remaining error signal.
The new learner is added to the ensemble with a controlled contribution.
The process repeats.
```

The goal of this note is to explain boosting deeply enough that the later notebook and report can use it confidently. The note covers both the classical material and the modern tabular-ML ecosystem:

```text
1. The hypothesis boosting idea.
2. The difference between bagging and boosting.
3. Weighted training and AdaBoost.
4. Exponential loss, model weights, and sample-weight updates.
5. Gradient boosting as forward stagewise additive modelling.
6. Residuals, negative-gradient pseudo-residuals, and binary log-loss gradients.
7. Regularization in boosting: learning rate, number of trees, tree depth, subsampling, early stopping, and penalties.
8. scikit-learn AdaBoost, GradientBoostingClassifier, and HistGradientBoostingClassifier.
9. XGBoost as regularized second-order tree boosting.
10. LightGBM as histogram-based, leaf-wise, efficiency-oriented gradient boosting.
11. CatBoost as gradient boosting with native categorical-feature handling and ordered boosting.
12. How the notebook should evaluate these models in the Telco project.
```

The held-out test set remains unused. This note is for model-family understanding and training-set development only.

## 2. Where boosting fits after bagging and random forests

Bagging and random forests train many trees independently. Each tree sees a perturbed version of the data, and the final prediction is an average or vote. This primarily reduces variance.

Boosting trains models sequentially. The model added at iteration $t$ depends on the ensemble built at iterations $1, \ldots, t-1$. This dependency is the main conceptual difference.

A useful contrast is:

```text
Bagging and random forests:
    independent or parallel base learners;
    bootstrap row sampling and sometimes feature sampling;
    prediction by averaging or voting;
    strongest when the base learner has low bias but high variance;
    mainly variance reduction.

Boosting:
    sequential base learners;
    each learner focuses on previous mistakes, residuals, or gradients;
    prediction by an additive weighted sum;
    can turn weak learners into a strong learner;
    often reduces bias and can also reduce variance if regularized carefully.
```

The key object in boosting is an additive ensemble score. For binary classification, a boosted model often constructs a real-valued score

$$
F_T(x) = \sum_{t=1}^{T} \nu_t h_t(x),
$$

where $h_t$ is the base learner added at stage $t$, and $\nu_t$ is its contribution. Depending on the algorithm, $\nu_t$ may be a learned model weight, a fixed learning-rate multiplier, or a product of both.

For classification, the final score can be used in multiple ways:

```text
hard class:
    predict churn if F_T(x) is above a threshold;

probability:
    transform F_T(x) through a sigmoid or algorithm-specific probability mapping;

ranking:
    rank customers by F_T(x) or by the predicted churn probability.
```

This is important for the Telco project because PR-AUC and ROC-AUC evaluate ranking quality, while threshold metrics such as precision, recall, specificity, and F1 depend on the chosen threshold.

## 3. The hypothesis boosting question

The classical motivation for boosting is sometimes called the hypothesis boosting question.

Suppose we have a family of weak learners. A weak learner is only slightly better than random guessing. Examples include:

```text
1. Decision stumps, which are depth-one decision trees.
2. Very shallow trees.
3. Simple linear rules.
```

A single weak learner has high bias because it cannot represent a complex decision boundary. The boosting question is:

```text
Can many weak learners be combined to form a strong learner?
```

The answer is yes, if each new weak learner is trained in a way that complements the current ensemble. The important word is complements. If we fit many identical weak learners independently, the ensemble may not become much stronger. Boosting makes later learners different because it changes the training emphasis after each step.

For churn classification, this means a first stump might split on contract type, a second stump might focus on customers the contract split handled poorly and split on tenure, a third might focus on remaining mistakes and split on payment method, and so on. The final boosted model can represent a much richer rule surface than any individual stump.

## 4. Weighted training

AdaBoost starts from a simple mechanism: give each training observation a weight.

Let the training set be

$$
D = \{(x_i, y_i)\}_{i=1}^{n}.
$$

In binary AdaBoost it is convenient to encode labels as

$$
y_i \in \{-1, +1\}.
$$

At boosting iteration $t$, each observation has a nonnegative weight $w_i^{(t)}$. A weighted empirical loss has the form

$$
L_t(\theta) = \sum_{i=1}^{n} w_i^{(t)} \ell(y_i, f_\theta(x_i)).
$$

The weights decide which observations matter most to the next learner. If the current ensemble classifies observation $i$ poorly, its weight increases. If the current ensemble classifies it correctly and confidently, its weight decreases.

Some algorithms fit weighted models directly. For example, decision trees can use sample weights when computing split criteria. If a learning algorithm does not support weights directly, a weighted dataset can be approximated by resampling observations with probabilities proportional to their weights.

In the Telco setting, weighted training means that later learners may pay more attention to customers that earlier learners could not classify well. These difficult cases might include customers with mixed signals, such as month-to-month contract but long tenure, or low monthly charges but risky service combinations.

## 5. AdaBoost: additive classification with exponential loss

AdaBoost, short for adaptive boosting, is the classical boosting algorithm for classification. It gives a principled way to choose both:

```text
1. the sample weights used to train the next weak learner;
2. the contribution weight assigned to that weak learner in the ensemble.
```

For binary labels $y_i \in \{-1,+1\}$, let each weak learner output

$$
h_t(x) \in \{-1,+1\}.
$$

The boosted ensemble score after $t$ learners is

$$
F_t(x) = \sum_{s=1}^{t} \alpha_s h_s(x),
$$

where $\alpha_s$ is the weight of weak learner $s$. The hard prediction is

$$
\hat{y}(x) = \operatorname{sign}(F_t(x)).
$$

AdaBoost can be derived by minimizing the exponential loss

$$
L(F) = \sum_{i=1}^{n} \exp(-y_i F(x_i)).
$$

The product $y_i F(x_i)$ is called the margin. If $y_i F(x_i)$ is large and positive, the ensemble predicts the correct class confidently. If it is negative, the ensemble predicts the wrong class. The exponential loss penalizes negative margins heavily.

At iteration $t$, suppose the previous ensemble is $F_{t-1}$. The new ensemble is

$$
F_t(x) = F_{t-1}(x) + \alpha_t h_t(x).
$$

The exponential loss contribution of observation $i$ becomes

$$
\exp(-y_i F_t(x_i))
= \exp(-y_i F_{t-1}(x_i)) \exp(-\alpha_t y_i h_t(x_i)).
$$

Define the current observation weight as

$$
w_i^{(t)} = \exp(-y_i F_{t-1}(x_i)).
$$

Then the part of the loss affected by the new learner is

$$
\sum_{i=1}^{n} w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i)).
$$

If $h_t$ classifies observation $i$ correctly, then $y_i h_t(x_i)=1$, so the multiplicative factor is $\exp(-\alpha_t)$. If it misclassifies the observation, then $y_i h_t(x_i)=-1$, so the multiplicative factor is $\exp(\alpha_t)$.

Therefore, for a fixed $\alpha_t$, choosing the new weak learner reduces to minimizing the weighted misclassification error:

$$
\varepsilon_t
=
\frac{\sum_{i=1}^{n} w_i^{(t)} \mathbb{1}\{h_t(x_i) \neq y_i\}}
{\sum_{i=1}^{n} w_i^{(t)}}.
$$

The optimal model weight for the binary AdaBoost derivation is

$$
\alpha_t = \frac{1}{2}\log\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right).
$$

This formula has an intuitive interpretation:

```text
If epsilon_t is small:
    the weak learner is good;
    (1 - epsilon_t) / epsilon_t is large;
    alpha_t is large;
    the learner receives a large vote.

If epsilon_t is close to 0.5:
    the weak learner is barely better than random;
    alpha_t is close to zero;
    the learner contributes little.

If epsilon_t is above 0.5:
    the learner is worse than random under the current weights;
    standard binary AdaBoost cannot use it directly without flipping or stopping.
```

After selecting $h_t$ and $\alpha_t$, the weights are updated as

$$
w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i)).
$$

Correctly classified observations are multiplied by $\exp(-\alpha_t)$, so their weights shrink. Incorrectly classified observations are multiplied by $\exp(\alpha_t)$, so their weights grow. The weights are then normalized so that they sum to one.

This produces the central AdaBoost loop:

```text
Initialize all sample weights equally.
For t = 1, ..., T:
    Fit a weak learner using the current sample weights.
    Compute its weighted error epsilon_t.
    Compute its learner weight alpha_t.
    Increase weights on misclassified observations.
    Decrease weights on correctly classified observations.
Return the weighted ensemble sign(sum alpha_t h_t(x)).
```

## 6. AdaBoost with decision stumps and shallow trees

Decision stumps are common AdaBoost base learners because they are weak but interpretable. A stump can only split on one feature once. Alone, it is underpowered. In a sequence, however, many stumps can form a flexible additive model.

In the Telco churn dataset, a boosted stump ensemble might learn a sequence like:

```text
1. Split on Contract_Month-to-month.
2. Increase attention to mistakes.
3. Split on tenure.
4. Increase attention to remaining mistakes.
5. Split on InternetService_Fiber optic.
6. Continue with payment method, tech support, monthly charges, and interactions implied by the additive sequence.
```

Each stump is simple, but the weighted sum of many stumps can approximate a complex boundary.

Shallow trees with depth 2 or 3 are also common. They let each base learner capture a small interaction, such as contract type combined with tenure, or internet service combined with technical support. This can improve performance, but it also increases the risk that each base learner is too strong, which can make the boosted ensemble more prone to overfitting.

Important AdaBoost hyperparameters are:

```text
n_estimators:
    number of boosting rounds;
    too few can underfit;
    too many can overfit or focus too much on noisy observations.

learning_rate:
    shrinks each learner contribution;
    smaller values usually require more estimators;
    smaller values often improve generalization but increase runtime.

base estimator depth:
    depth 1 gives stumps;
    depth 2 or 3 permits simple interactions;
    deeper trees can make AdaBoost less stable.
```

AdaBoost is often sensitive to mislabeled observations and outliers because misclassified cases receive increasing weight. If some labels are noisy, AdaBoost may repeatedly focus on impossible or unrepresentative cases. For churn, this means we should interpret AdaBoost carefully if performance does not improve smoothly.

## 7. Gradient boosting: from residuals to functional gradient descent

AdaBoost uses reweighting and exponential loss. Gradient boosting generalizes the boosting idea to arbitrary differentiable loss functions.

The core form is an additive model:

$$
F_T(x) = F_0(x) + \sum_{t=1}^{T} \nu h_t(x),
$$

where $F_0$ is an initial model, $h_t$ is the learner added at stage $t$, and $\nu$ is the learning rate.

For squared-error regression, the idea is intuitive. Suppose the current model is $F_{t-1}$. The residual for observation $i$ is

$$
r_i^{(t)} = y_i - F_{t-1}(x_i).
$$

If we can fit a model $h_t$ that predicts these residuals, then adding it to the ensemble improves the prediction:

$$
F_t(x) = F_{t-1}(x) + \nu h_t(x).
$$

This is the residual-fitting interpretation.

But the deeper reason is gradient descent in function space. We are not only optimizing a finite parameter vector. We are optimizing the predictions $F(x_i)$ themselves. For squared-error loss

$$
\ell(y_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2,
$$

the derivative with respect to the current prediction is

$$
\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} = F(x_i) - y_i.
$$

The negative gradient is

$$
-\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} = y_i - F(x_i),
$$

which is exactly the residual.

So for squared error, fitting residuals is the same as fitting negative gradients.

For other losses, the residual is replaced by the negative gradient, often called a pseudo-residual:

$$
r_i^{(t)} = -\left[\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{t-1}}.
$$

The new learner $h_t$ is trained to predict $r_i^{(t)}$ from $x_i$. This learner approximates the direction in output space that would most reduce the loss.

The generic gradient boosting algorithm is:

```text
Initialize F_0(x) as a constant model.
For t = 1, ..., T:
    Compute pseudo-residuals r_i^(t) = negative gradient of the loss.
    Fit a base learner h_t(x) to predict these pseudo-residuals.
    Choose a step size or leaf values, often by line search or Newton updates.
    Update F_t(x) = F_{t-1}(x) + learning_rate * h_t(x).
Return F_T(x).
```

This is why gradient boosting can be used for regression, binary classification, multiclass classification, ranking, survival modelling, and custom differentiable objectives.

## 8. Binary classification with log loss

For binary classification, gradient boosting often uses the log-loss objective.

Let the model produce a real-valued score $F(x)$. Convert it to a probability through the sigmoid function:

$$
p(x) = \sigma(F(x)) = \frac{1}{1 + \exp(-F(x))}.
$$

For labels $y_i \in \{0,1\}$, the binary log loss is

$$
\ell(y_i, F(x_i))
=
-y_i \log(p_i) - (1-y_i)\log(1-p_i),
$$

where $p_i = \sigma(F(x_i))$.

The derivative of log loss with respect to the score $F(x_i)$ is

$$
\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} = p_i - y_i.
$$

Therefore, the negative gradient is

$$
y_i - p_i.
$$

This quantity is the classification pseudo-residual. It has a very intuitive interpretation:

```text
If y_i = 1 and p_i is too small:
    y_i - p_i is positive;
    the next tree should increase the score for this observation.

If y_i = 0 and p_i is too large:
    y_i - p_i is negative;
    the next tree should decrease the score for this observation.

If p_i is already close to y_i:
    the pseudo-residual is near zero;
    the next tree does not need to change the prediction much.
```

This is the direct link between the informal statement "later trees focus on mistakes" and the mathematical statement "later trees fit negative gradients." Mistakes produce large gradients. Well-handled observations produce small gradients.

## 9. Stagewise tree fitting and leaf values

In gradient-boosted decision trees, the base learner is usually a regression tree, even for classification. The tree partitions feature space into leaves. Within each leaf, the algorithm chooses a constant update value.

At stage $t$, a tree defines regions

$$
R_{1t}, R_{2t}, \ldots, R_{J_t t}.
$$

The tree update can be written as

$$
h_t(x) = \sum_{j=1}^{J_t} \gamma_{jt} \mathbb{1}\{x \in R_{jt}\},
$$

where $\gamma_{jt}$ is the leaf value for region $j$. The updated model is

$$
F_t(x) = F_{t-1}(x) + \nu h_t(x).
$$

For squared error, the leaf value is often the mean residual in the leaf. For log loss and other objectives, the leaf value can be chosen by line search, Newton approximation, or a library-specific formula using gradients and Hessians.

The important concept is that each tree adds a piecewise-constant correction to the current score function. Early trees learn large global corrections. Later trees learn smaller refinements.

Because the package sections below use similar notation, it is useful to separate gradients, Hessians, observation weights, and leaf outputs:

```text
g_i:
    gradient of the loss for observation i with respect to the current model score.

h_i:
    Hessian, or second derivative, of the loss for observation i with respect to the current model score.

gamma_j or w_j:
    output value of leaf j;
    the numeric correction added to the current model score for observations in that leaf.

w_i:
    optional sample weight for observation i.
```

For a leaf region $R_j$, the summed gradient and Hessian are

$$
G_j = \sum_{i \in R_j} g_i,
\quad
H_j = \sum_{i \in R_j} h_i.
$$

The notation warning matters: $w_i$ and $w_j$ are different quantities. The term $w_i$ is an observation or sample weight. The term $w_j$ is a leaf output, also called a leaf weight in some implementations, and it is the score correction assigned to leaf $j$.

## 10. Regularization in boosting

Boosting can be very powerful, so regularization is essential. Important controls include:

```text
learning_rate:
    Shrinks each tree's contribution.
    Smaller learning rates usually require more trees.
    This is one of the most important boosting hyperparameters.

n_estimators / iterations / max_iter:
    Number of boosting rounds.
    More rounds increase flexibility.
    With a small learning rate, more rounds can improve performance.
    Too many rounds can overfit if not stopped or regularized.

tree depth / max_leaf_nodes / num_leaves:
    Controls interaction complexity of each tree.
    Stumps learn additive main effects.
    Depth 2 or 3 trees learn low-order interactions.
    Larger trees can fit complex interactions but increase overfitting risk.

min_samples_leaf / min_child_weight:
    Prevents leaves from being based on too little data.
    Helps probability stability and generalization.

subsample:
    Uses only a fraction of rows for each boosting round.
    Creates stochastic gradient boosting.
    Can reduce variance and improve robustness.

colsample / max_features / feature_fraction:
    Uses only a subset of features for each tree or split.
    Decorrelates trees and reduces overfitting.

L1 and L2 penalties:
    Penalize leaf weights or model complexity in some implementations.

early stopping:
    Stops adding trees once validation performance stops improving.
```

The learning-rate and number-of-trees tradeoff is central:

```text
Large learning rate + few trees:
    fast but can be coarse and unstable.

Small learning rate + many trees:
    slower but often better generalization.
```

For this project, the boosting notebook should use transparent development grids rather than very large automatic searches. However, because boosting has many consequential hyperparameters, the section can include modern libraries and still remain methodologically careful by documenting search spaces and interpreting results as development-stage evidence.

## 11. scikit-learn AdaBoostClassifier

`AdaBoostClassifier` is the most direct implementation of classical boosting in scikit-learn. It fits a sequence of classifiers where later classifiers focus more on observations that previous classifiers handled incorrectly. The default base estimator is a depth-one decision tree, which corresponds to a decision stump.

For the Telco project, useful AdaBoost variants include:

```text
AdaBoost with decision stumps:
    estimator = DecisionTreeClassifier(max_depth=1)
    interpretable weak learners
    classical AdaBoost setting

AdaBoost with shallow trees:
    estimator = DecisionTreeClassifier(max_depth=2 or 3)
    allows simple interactions
    may improve performance but can overfit more easily
```

Important parameters:

```text
n_estimators:
    number of boosting rounds.

learning_rate:
    shrinks each estimator's contribution.

estimator:
    base classifier, usually a shallow decision tree.
```

AdaBoost should be evaluated with the same PR-AUC primary metric and threshold diagnostics as earlier sections. It may or may not beat random forests or logistic regression. The important lesson is how sequential reweighting changes the ensemble compared with bagging.

## 12. scikit-learn GradientBoostingClassifier

`GradientBoostingClassifier` implements stagewise additive gradient boosting for classification. In binary classification, it fits one regression tree per stage to the negative gradient of the loss. With `loss="log_loss"`, the objective is binomial deviance, the same probabilistic classification loss underlying logistic regression.

The model starts from an initial constant prediction. For binary log loss, this constant is related to the training-set class prior on the log-odds scale.

At each boosting stage, the current model produces probabilities $p_i$. For binary log loss, the negative-gradient or pseudo-residual signal is

$$
r_i = y_i - p_i.
$$

A regression tree is fitted to these pseudo-residual targets. Therefore, the split structure is built by regression-tree impurity reduction on $r_i$, roughly variance or squared-error reduction of the pseudo-residuals. It is not built by Gini impurity or entropy on the original class labels.

After the terminal regions are formed, the leaf output is a loss-specific correction. For binary log loss, the terminal-region update is Newton-style:

$$
\gamma_j
\approx
\frac{\sum_{i \in R_j} w_i (y_i - p_i)}
{\sum_{i \in R_j} w_i p_i(1-p_i)}
$$

Equivalently, using gradient and Hessian notation,

$$
\gamma_j
\approx
-\frac{G_j}{H_j}.
$$

The numerator is the summed negative gradient in the leaf, and the denominator is the summed Hessian or curvature term for binary log loss. If no sample weights are supplied, $w_i$ can be treated as 1. The important implementation detail is that `GradientBoostingClassifier` fits the tree structure from pseudo-residuals, then uses a loss-specific Newton-style terminal-region update. It does not simply average pseudo-residuals for binary log loss.

`GradientBoostingClassifier` does not expose the same explicit XGBoost-style $\lambda$ and $\gamma$ regularized tree objective as its central formulation. Its regularization is mainly controlled by `learning_rate`, `n_estimators`, `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `subsample`, `max_features`, `ccp_alpha`, and early stopping parameters.

Important parameters:

```text
loss:
    usually "log_loss" for probabilistic binary classification.

learning_rate:
    shrinkage applied to each tree.

n_estimators:
    number of boosting stages.

max_depth or max_leaf_nodes:
    complexity of each regression tree.

min_samples_leaf:
    minimum observations in a leaf.

subsample:
    if below 1.0, uses stochastic gradient boosting.

validation_fraction, n_iter_no_change, tol:
    early-stopping controls.
```

This implementation is useful pedagogically because it closely follows the gradient boosting theory. It is slower than histogram-based variants for larger datasets, but the Telco dataset is small enough that it should still run.

## 13. scikit-learn HistGradientBoostingClassifier

`HistGradientBoostingClassifier` is scikit-learn's faster histogram-based gradient boosting implementation. Instead of searching exact split thresholds over all continuous feature values, it bins feature values and builds trees using histograms. This can greatly speed up training and reduce memory usage.

The model starts from an initial constant prediction, again related to the class prior for binary log loss. Features are binned into histograms, and candidate splits are evaluated using accumulated gradient and Hessian statistics over bins.

An L2-regularized second-order split-gain expression has the form

$$
\operatorname{Gain}
=
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G_P^2}{H_P+\lambda}
\right].
$$

Here $P$ denotes the parent node, while $L$ and $R$ denote the proposed left and right child nodes. The formula measures the approximate reduction in the second-order objective from splitting the parent into two children. Unlike the XGBoost gain formula below, this expression does not include a separate $-\gamma$ split-penalty term.

The leaf output uses the Newton-style L2-regularized value

$$
w_j
=
-\frac{\sum_{i \in R_j} g_i}
{\sum_{i \in R_j} h_i + \lambda}
=
-\frac{G_j}{H_j+\lambda}.
$$

Here $\lambda$ corresponds to `l2_regularization`. It shrinks leaf outputs, especially when a leaf has weak Hessian support. Tree complexity is also controlled by `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `max_bins`, `max_features`, and early stopping.

It is important not to overstate the distinction: `GradientBoostingClassifier` also uses Newton-style terminal-region updates for log loss. The difference is that `HistGradientBoostingClassifier` uses histogram-based gradient/Hessian statistics more directly in split construction and exposes explicit L2 leaf regularization.

Important parameters:

```text
max_iter:
    number of boosting iterations.

learning_rate:
    shrinkage.

max_leaf_nodes:
    maximum leaves per tree.

min_samples_leaf:
    regularization through leaf size.

l2_regularization:
    L2 penalty on leaf values.

early_stopping:
    whether to stop when validation performance stops improving.

class_weight:
    optional class weighting.
```

This model is useful as a bridge between standard gradient boosting and modern histogram-based libraries such as LightGBM and XGBoost.

## 14. XGBoost: regularized second-order tree boosting

XGBoost stands for Extreme Gradient Boosting. It is a widely used implementation of gradient-boosted decision trees. Its conceptual contribution is not only speed, but also a very explicit regularized objective for tree boosting.

XGBoost starts from `base_score`, also interpretable as a global bias. In simple binary examples this can be understood as an initial constant probability or log-odds, but it should not be stated as always exactly 0.5 because modern XGBoost versions may estimate an intercept-like base score automatically depending on the objective and settings.

A boosted tree ensemble can be written as

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \quad f_k \in \mathcal{F},
$$

where each $f_k$ is a CART-like tree that maps an observation to a leaf score.

XGBoost writes the objective as

$$
\operatorname{obj}
=
\sum_{i=1}^{n} \ell(y_i, \hat{y}_i)
+
\sum_{k=1}^{K} \Omega(f_k),
$$

where the first term is training loss and the second term penalizes tree complexity.

At boosting step $t$, the prediction becomes

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i).
$$

XGBoost approximates the change in objective using a second-order Taylor expansion around the current predictions:

$$
\operatorname{obj}^{(t)}
\approx
\sum_{i=1}^{n}
\left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right]
+ \Omega(f_t),
$$

where

$$
g_i = \frac{\partial \ell(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}},
\quad
h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}.
$$

Thus, XGBoost uses both gradients and Hessians. This is why it is often described as second-order gradient boosting or Newton-style boosting.

XGBoost evaluates candidate splits using this second-order Taylor approximation of the regularized objective. This is conceptually related to the `HistGradientBoostingClassifier` formulas above, but XGBoost presents the regularized objective and split-gain calculation as central parts of the implementation.

A common tree complexity penalty is

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2,
$$

where $T$ is the number of leaves, $w_j$ is the score in leaf $j$, $\gamma$ penalizes adding leaves, and $\lambda$ is L2 regularization on leaf scores.

For a fixed tree structure, define

$$
G_j = \sum_{i \in I_j} g_i,
\quad
H_j = \sum_{i \in I_j} h_i,
$$

where $I_j$ is the set of observations in leaf $j$. The optimal leaf weight is

$$
w_j^* = -\frac{G_j}{H_j + \lambda}.
$$

A split is evaluated by its gain:

$$
\operatorname{Gain}
=
\frac{1}{2}
\left[
\frac{G_L^2}{H_L + \lambda}
+
\frac{G_R^2}{H_R + \lambda}
-
\frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right]
-
\gamma.
$$

This formula shows why XGBoost is not just ordinary decision-tree impurity reduction. It evaluates splits through the loss gradients, Hessians, and regularization terms.

In this expression, $G_L,H_L$ are the gradient and Hessian sums in the left child, and $G_R,H_R$ are the gradient and Hessian sums in the right child. The terms $G_L+G_R$ and $H_L+H_R$ are the parent gradient and Hessian sums. The parameter $\gamma$ is a split hurdle, so a split must improve the approximate regularized objective enough to justify adding the split.

The regularization controls have direct objective-level interpretations. The parameter `reg_lambda`, written as $\lambda$, is L2 leaf regularization. The parameter `reg_alpha`, written as $\alpha$, is L1 leaf regularization. The parameter `gamma` is the minimum required split improvement, while `min_child_weight` controls the minimum Hessian mass required in a child node. Tree depth, row subsampling, feature subsampling, and shrinkage are controlled through `max_depth`, `subsample`, `colsample_bytree`, and `learning_rate`.

The distinction is not that XGBoost uniquely uses gradients, Hessians, or regularization. `HistGradientBoostingClassifier` also uses second-order gradient/Hessian logic. XGBoost's distinguishing feature is that it exposes a very explicit regularized objective and split-gain formulation, with $\gamma$, $\lambda$, $\alpha$, and `min_child_weight` as central controls.

Important XGBoost parameters:

```text
n_estimators:
    number of boosting rounds.

learning_rate / eta:
    shrinkage applied to each tree.

max_depth:
    maximum tree depth.

min_child_weight:
    minimum summed Hessian needed in a child node;
    larger values make the model more conservative.

gamma:
    minimum loss reduction needed for a split;
    larger values prune more aggressively.

subsample:
    row subsampling per tree.

colsample_bytree:
    feature subsampling per tree.

reg_lambda:
    L2 regularization on leaf weights.

reg_alpha:
    L1 regularization on leaf weights.

scale_pos_weight:
    class-imbalance weighting, often roughly negative_count / positive_count.

eval_metric:
    validation metric such as aucpr, auc, or logloss.
```

For this project, XGBoost is valuable because it is one of the most recognized tabular classification models. It should be included in the executable boosting notebook if the package is installed in the project environment.

## 15. LightGBM: histogram-based and leaf-wise gradient boosting

LightGBM is another modern gradient-boosted decision-tree framework. It is designed for speed, memory efficiency, and large-scale training.

LightGBM is also a second-order histogram-based GBDT implementation. It bins feature values, accumulates gradient and Hessian statistics in bins, and evaluates candidate splits using gain from aggregated gradient/Hessian statistics.

An L2-regularized split-gain expression can be written as

$$
\operatorname{Gain}
=
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda_2}
+
\frac{G_R^2}{H_R+\lambda_2}
-
\frac{G_P^2}{H_P+\lambda_2}
\right].
$$

Here $P$ is the parent node, $L$ and $R$ are the proposed child nodes, and $\lambda_2$ is L2 leaf regularization. LightGBM also has `min_gain_to_split`, also called `min_split_gain`, which acts as a minimum gain required to perform a split.

The L2-regularized Newton-style leaf output is

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda_2}.
$$

With both L1 and L2 regularization, the leaf output can be written as

$$
w_j^*
=
-\frac{\operatorname{sgn}(G_j)\max(|G_j|-\lambda_1,0)}
{H_j+\lambda_2}.
$$

Here $\lambda_1$ corresponds to L1 leaf regularization and $\lambda_2$ corresponds to L2 leaf regularization. The $\max(|G_j|-\lambda_1,0)$ term is soft-thresholding of the gradient sum. This is regularized GBDT logic and should not be presented as the only thing that makes LightGBM unique.

Important regularization and tree-control parameters include `num_leaves`, `max_depth`, `min_child_samples`, `min_child_weight`, `min_gain_to_split`, `lambda_l1`, `lambda_l2`, `feature_fraction`, `bagging_fraction`, and early stopping. LightGBM should therefore not be read as just "fast XGBoost." It shares the core second-order GBDT split and leaf logic with modern boosted-tree methods, but differs especially through histogram split finding, leaf-wise or best-first tree growth, GOSS, EFB, and optional native categorical split handling.

Important LightGBM ideas include:

```text
1. Histogram-based split finding.
2. Leaf-wise, best-first tree growth.
3. Efficient handling of sparse features.
4. Native categorical split support.
5. Gradient-Based One-Side Sampling, often abbreviated GOSS.
6. Exclusive Feature Bundling, often abbreviated EFB.
```

### 15.1 Histogram-based split finding

Instead of searching exact split points over all continuous feature values, LightGBM bins continuous values into discrete buckets. Split gain can then be computed from histograms.

This reduces computation because the algorithm searches over bins rather than all distinct values. It also reduces memory usage because binned values can be stored compactly.

This idea is also why scikit-learn's `HistGradientBoostingClassifier` belongs naturally in the same conceptual family.

### 15.2 Leaf-wise tree growth

Many tree algorithms grow trees level-wise. That means they expand all leaves at the same depth before moving deeper.

LightGBM grows trees leaf-wise, also called best-first growth. At each step, it chooses the leaf whose split gives the largest loss reduction.

This can improve accuracy because the algorithm spends tree complexity where it most reduces the objective. However, leaf-wise growth can also overfit if not controlled. Important controls include:

```text
num_leaves:
    maximum number of leaves;
    one of the most important LightGBM complexity parameters.

max_depth:
    optional hard depth limit.

min_child_samples:
    minimum observations in a leaf.

min_child_weight:
    minimum Hessian sum in a leaf.

lambda_l1, lambda_l2:
    leaf-weight regularization.
```

### 15.3 GOSS

Gradient-Based One-Side Sampling uses the observation that examples with large gradients are currently poorly fitted and therefore informative. LightGBM can keep all large-gradient observations while sampling from small-gradient observations. This reduces training cost while preserving much of the important gradient information.

Conceptually:

```text
large gradient:
    model is currently wrong or uncertain;
    keep the observation.

small gradient:
    model already handles the observation well;
    sample a subset to save computation.
```

This is different from ordinary random subsampling because sampling depends on the gradient magnitude.

### 15.4 EFB

Exclusive Feature Bundling is useful in sparse high-dimensional feature spaces. One-hot encoded variables often produce mutually exclusive columns. For example, a customer cannot simultaneously have `Contract_One year` and `Contract_Two year` equal to one.

EFB bundles features that are rarely nonzero at the same time, reducing the effective number of features with little information loss. This is particularly relevant for datasets with many one-hot encoded categorical variables.

### 15.5 Native categorical features

LightGBM can handle categorical features without ordinary one-hot encoding by searching splits that partition categories into two groups. This differs from scikit-learn's basic tree pipelines in this project, where categories are one-hot encoded before modelling.

For the Telco project, we have two possible LightGBM workflows:

```text
Option A: one-hot pipeline
    consistent with earlier model sections;
    easy to compare with scikit-learn and XGBoost pipelines;
    does not use LightGBM's native categorical advantages.

Option B: native categorical LightGBM
    uses raw categorical columns encoded as categorical dtype or integer category codes;
    can exploit LightGBM categorical split logic;
    requires a separate preprocessing branch and careful validation.
```

For a full modern boosting section, it is reasonable to include at least one native-categorical model because categorical handling is one of the practical differences among modern GBDT libraries.

Important LightGBM parameters:

```text
n_estimators:
    number of boosting iterations.

learning_rate:
    shrinkage.

num_leaves:
    maximum leaves per tree.

max_depth:
    depth cap.

min_child_samples:
    minimum observations in a leaf.

subsample / bagging_fraction:
    row subsampling.

colsample_bytree / feature_fraction:
    feature subsampling.

reg_alpha / lambda_l1:
    L1 regularization.

reg_lambda / lambda_l2:
    L2 regularization.

class_weight / scale_pos_weight / is_unbalance:
    class-imbalance controls.
```

## 16. CatBoost: categorical boosting and ordered target statistics

CatBoost is a gradient-boosted decision-tree library designed especially for datasets with categorical variables. This is important for Telco churn because most original features are categorical.

CatBoost is still a gradient-boosted tree method. At each boosting step, the next tree is chosen to improve the current ensemble by approximating the negative-gradient signal of the loss. If the current ensemble score is $F_{t-1}(x_i)$, define the ordinary gradient as

$$
g_i
=
\left[
\frac{\partial \ell(a,y_i)}{\partial a}
\right]_{a=F_{t-1}(x_i)}.
$$

The negative-gradient signal is therefore $-g_i$. The candidate tree produces values $a_i=f_t(x_i)$. CatBoost uses score functions to evaluate how well a candidate tree approximates the negative-gradient signal.

Implementation lens: split scoring and leaf values.

CatBoost forms candidate feature-split pairs and chooses splits greedily using score functions. For the first-order L2 score, the quality of a candidate tree can be written as

$$
S_{L2}(a,g)
=
-\sum_i w_i(a_i + g_i)^2.
$$

Here $a_i$ is the candidate tree output for observation $i$, $g_i$ is the ordinary gradient for observation $i$, and $w_i$ is an optional sample weight. Maximizing this score means choosing a tree whose outputs approximate $-g_i$ well.

For a fixed leaf region $R_j$, the first-order optimal leaf value under this L2 score is the weighted average negative gradient in that leaf:

$$
a_j^*
=
-\frac{\sum_{i \in R_j} w_i g_i}
{\sum_{i \in R_j} w_i}.
$$

CatBoost also supports Newton-style score functions such as `NewtonL2` and `NewtonCosine`, which use second derivatives. If $h_i$ denotes the Hessian or curvature term for observation $i$, then a second-order regularized leaf update has the familiar Newton-style form

$$
a_j^*
=
-\frac{\sum_{i \in R_j} w_i g_i}
{\sum_{i \in R_j} w_i h_i + \lambda}.
$$

This is the same sign convention used earlier: $g_i$ is the ordinary gradient, so the tree update points in the negative-gradient direction. If using ordinary gradient notation $G_j=\sum_{i\in R_j}\partial \ell/\partial F$, the same update would be written as $-G_j/(H_j+\lambda)$. First-order CatBoost scoring can be understood as fitting leaf outputs to negative gradients, while Newton-style scoring uses Hessian or curvature information as well. The parameter $\lambda$ corresponds conceptually to L2 leaf-value regularization, exposed through `l2_leaf_reg`. The choice of score function controls whether first-order or second-order information is used during split scoring.

CatBoost commonly uses symmetric, also called oblivious, trees. In a symmetric tree, all nodes at the same depth use the same split condition. Therefore, split scoring is not simply choosing the best independent split for each individual leaf. At each depth, CatBoost searches for a split rule that is applied across the current level of the tree. This constrains the tree structure, makes prediction fast, and acts as a structural regularizer.

For an ordinary asymmetric tree, different branches can choose different split rules at the same depth. For a symmetric CatBoost tree, the same split is used across all nodes at a given depth. Therefore, a candidate split must be evaluated by its combined effect across the affected leaves, not only by its effect on one isolated leaf.

Important CatBoost regularization and tree-control parameters include `depth`, `iterations`, `learning_rate`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `border_count`, and class-weighting parameters such as `class_weights`, `auto_class_weights`, or `scale_pos_weight`.

CatBoost should not be presented as just XGBoost with categorical preprocessing. It belongs to the gradient-boosted tree family, but its distinctive contributions are ordered target statistics for categorical variables, ordered boosting to reduce prediction shift, symmetric or oblivious tree structure, and strong native categorical-feature handling.

CatBoost's main practical ideas include:

```text
1. Native categorical-feature handling.
2. Ordered target statistics for categorical encodings.
3. Ordered boosting to reduce prediction shift and target leakage.
4. Symmetric, also called oblivious, trees in the default setting.
5. Strong default performance with relatively little preprocessing.
```

### 16.1 Why categorical handling matters

A common approach is one-hot encoding. This works well for low-cardinality features, but it can become inefficient or weak for high-cardinality categories. Another common approach is target encoding, where a category is replaced by the mean target value for that category.

Naive target encoding can leak target information. If the target mean for a category is computed using all rows, then the encoded value for a row uses that row's own label. This creates target leakage and overfitting.

CatBoost uses ordered target statistics. The data are randomly permuted, and the target statistic for a row is computed using only earlier rows in the permutation. This mimics the idea that the current row's label should not be used to encode its own features.

A simplified target statistic for a category value can be written as

$$
\operatorname{CTR}(x_i)
=
\frac{\text{count of positive labels among previous rows with same category} + \text{prior}}
{\text{count of previous rows with same category} + 1}.
$$

The important word is previous. By using a permutation order, CatBoost avoids using the current label to compute the current categorical encoding.

### 16.2 Ordered boosting

CatBoost's ordered boosting is designed to reduce a prediction shift that can occur when gradients are computed using models that have already been trained on the same observations. The high-level idea is similar to the ordered categorical statistics: when estimating quantities used for training an observation, avoid using that observation's own target information in a way that creates leakage-like bias.

The technical implementation is more complex than a simple cross-validation encoding, but the conceptual goal is:

```text
reduce target leakage and prediction shift inside the boosting process,
especially for categorical-feature transformations and gradient estimates.
```

### 16.3 Symmetric or oblivious trees

CatBoost often uses symmetric trees. In a symmetric tree, all nodes at the same depth use the same split rule. This gives a balanced tree structure.

For example:

```text
Depth 1:
    split on feature A.

Depth 2:
    both child nodes split on feature B.

Depth 3:
    all four nodes split on feature C.
```

This is less flexible than arbitrary trees, but it has advantages:

```text
fast prediction;
regularization through constrained structure;
stable implementation;
compact model representation.
```

Important CatBoost parameters:

```text
iterations:
    number of boosting rounds.

learning_rate:
    shrinkage.

depth:
    tree depth.

l2_leaf_reg:
    L2 regularization on leaf values.

random_strength:
    randomness in split scoring.

bagging_temperature:
    controls Bayesian bootstrap intensity.

border_count:
    number of bins for numerical features.

loss_function:
    for binary classification, usually Logloss.

eval_metric:
    metric monitored for validation, such as AUC or PRAUC.

auto_class_weights / class_weights / scale_pos_weight:
    class-imbalance handling.

one_hot_max_size:
    threshold for one-hot encoding low-cardinality categorical features internally.
```

For the Telco project, CatBoost is especially interesting because it can be run on raw categorical features rather than the one-hot encoded pipeline. That gives a meaningful comparison:

```text
scikit-learn / XGBoost with one-hot encoded features
versus
CatBoost with native categorical features.
```

This comparison should be interpreted carefully because it changes both the model and the preprocessing representation. Still, it is educational and portfolio-relevant.

## 17. How XGBoost, LightGBM, and CatBoost differ

All three are gradient-boosted tree libraries, but they emphasize different technical choices.

Modern boosted-tree implementations can be compared through the same implementation lens: how split candidates are scored, how leaf values are computed, how split creation and leaf magnitudes are regularized, and how the package handles computation, tree growth, categorical variables, and leakage or prediction-shift issues. XGBoost, `HistGradientBoostingClassifier`, and LightGBM all use second-order gradient/Hessian logic in some form, while their differences lie in objective formulation, histogram machinery, tree-growth strategy, regularization controls, and categorical-feature handling.

```text
XGBoost:
    regularized second-order tree boosting;
    explicit objective with gradients, Hessians, and tree-complexity penalties;
    strong control over regularization;
    widely recognized in tabular ML and competitions;
    supports exact, approximate, and histogram tree methods.

LightGBM:
    histogram-based training by design;
    leaf-wise best-first tree growth;
    strong speed and memory focus;
    GOSS and EFB for efficiency;
    native categorical split support;
    often very fast on large tabular datasets.

CatBoost:
    gradient boosting with strong categorical-feature handling;
    ordered target statistics;
    ordered boosting to reduce prediction shift;
    symmetric trees by default;
    strong defaults and useful when many categorical features are present.
```

A practical summary for the Telco project:

```text
XGBoost asks:
    How much do we gain from regularized second-order boosted trees on one-hot features?

LightGBM asks:
    How much do we gain from fast histogram boosting and possibly native categorical splits?

CatBoost asks:
    How much do we gain from a boosting library designed around categorical features?
```

## 18. Probability calibration caveat

Boosting models often produce strong rankings. However, strong ROC-AUC or PR-AUC does not guarantee well-calibrated probabilities.

Reasons include:

```text
AdaBoost with exponential loss can produce aggressive margins.
Gradient boosting can become overconfident if too many trees are fitted.
Regularization and early stopping affect score scale.
Class weighting changes the interpretation of predicted probabilities.
Native categorical encodings can improve ranking without guaranteeing calibration.
```

Therefore, in this project:

```text
Use predicted scores and probabilities for ROC-AUC, PR-AUC, and threshold diagnostics.
Do not treat boosted probabilities as final calibrated churn probabilities yet.
Postpone calibration analysis until the later final model-comparison stage.
```

If a boosted model becomes a serious final candidate, calibration methods such as Platt scaling, isotonic regression, or calibration curves can be evaluated using training-only validation procedures.

## 19. Class imbalance in boosting

The churn positive class is about 26.54 percent of the training set. This is not extreme, but it is important enough that PR-AUC, recall, precision, specificity, and balanced accuracy should remain central.

Boosting libraries offer class-imbalance controls:

```text
AdaBoost / scikit-learn:
    can use sample weights through the base estimator and algorithm.

GradientBoostingClassifier:
    can use sample weights in fitting, but class-weight handling is less direct than some other models.

HistGradientBoostingClassifier:
    supports class_weight in newer scikit-learn versions.

XGBoost:
    scale_pos_weight is commonly used.

LightGBM:
    class_weight, is_unbalance, or scale_pos_weight may be used.

CatBoost:
    class_weights, auto_class_weights, or scale_pos_weight may be used.
```

However, class weighting changes the fitted score distribution and can trade precision for recall. Since the project already studies threshold curves, the boosting notebook should probably start with unweighted models and optionally include class-weighted variants as explicit sensitivity checks rather than silently mixing them into the main comparison.

## 20. Early stopping and validation discipline

Boosting is sequential. Each additional tree changes the model. Therefore, choosing the number of trees is a model-selection decision.

Early stopping chooses the iteration where validation performance stops improving. This is useful, but it must follow the same leakage rules as all other tuning:

```text
Inside cross-validation:
    early stopping must use only the training portion of the fold,
    usually by splitting that fold-training data into inner training and inner validation.

Outside final testing:
    never use the held-out test set for early stopping.
```

For the model-family notebook, there are two safe strategies:

```text
Strategy A: fixed grids without early stopping
    simple and transparent;
    all models evaluated by the same outer cross-validation procedure;
    number of estimators is just another hyperparameter.

Strategy B: fold-internal early stopping
    more realistic for modern GBDT libraries;
    more complex to implement cleanly;
    requires each cross-validation fit to create its own validation split from the fold-training data.
```

For the first boosting section, the cleanest approach is a transparent fixed grid. Early stopping can still be discussed and maybe used in library-specific reference fits, but the main section should avoid accidental validation leakage.

## 21. Proposed Telco boosting notebook scope

Because the project aims to be portfolio-ready and technically rich, the boosting notebook should include the full modern boosting ecosystem.

Recommended models:

```text
References from earlier sections:
    selected logistic regression;
    selected single decision tree;
    selected bagged trees;
    selected random forest.

Core boosting models:
    AdaBoostClassifier with decision stumps and shallow trees;
    GradientBoostingClassifier;
    HistGradientBoostingClassifier.

Modern GBDT libraries:
    XGBoost XGBClassifier;
    LightGBM LGBMClassifier;
    CatBoost CatBoostClassifier.
```

The notebook can use two preprocessing branches:

```text
One-hot encoded branch:
    scikit-learn AdaBoost;
    scikit-learn GradientBoostingClassifier;
    scikit-learn HistGradientBoostingClassifier;
    XGBoost;
    possibly LightGBM in one-hot mode for direct comparability.

Native categorical branch:
    LightGBM native categorical model, if implemented cleanly;
    CatBoost native categorical model.
```

It is acceptable that native categorical models are not perfectly identical preprocessing comparisons. They answer a different and useful question: how do modern GBDT libraries perform when allowed to use their intended categorical handling?

## 22. Suggested development grids

The grids should be large enough to teach and compare, but not so large that the section becomes an uncontrolled optimization contest.

### 22.1 AdaBoost grid

```text
base_depth in {1, 2}
n_estimators in {50, 100, 200}
learning_rate in {0.03, 0.1, 0.3, 1.0}
```

This gives 24 configurations.

### 22.2 GradientBoostingClassifier grid

```text
n_estimators in {100, 200}
learning_rate in {0.03, 0.1}
max_depth in {2, 3}
min_samples_leaf in {10, 25}
subsample in {0.8, 1.0}
```

This gives 32 configurations.

### 22.3 HistGradientBoostingClassifier grid

```text
max_iter in {100, 200}
learning_rate in {0.03, 0.1}
max_leaf_nodes in {15, 31}
min_samples_leaf in {10, 25}
l2_regularization in {0.0, 1.0}
```

This gives 32 configurations.

### 22.4 XGBoost grid

```text
n_estimators in {100, 200}
learning_rate in {0.03, 0.1}
max_depth in {2, 3, 4}
min_child_weight in {1, 5}
subsample in {0.8, 1.0}
colsample_bytree in {0.8, 1.0}
reg_lambda in {1.0}
```

This gives 48 configurations.

### 22.5 LightGBM grid

```text
n_estimators in {100, 200}
learning_rate in {0.03, 0.1}
num_leaves in {15, 31}
min_child_samples in {10, 25}
subsample in {0.8, 1.0}
colsample_bytree in {0.8, 1.0}
reg_lambda in {0.0, 1.0}
```

This gives 64 configurations. If runtime is high, reduce by fixing `reg_lambda=1.0` or using fewer subsampling combinations.

### 22.6 CatBoost grid

```text
iterations in {100, 200}
learning_rate in {0.03, 0.1}
depth in {3, 4, 6}
l2_leaf_reg in {3, 10}
```

This gives 24 configurations.

CatBoost should be run with `verbose=False` or similar settings to keep notebook output clean.

## 23. Evaluation plan

Use the same model-section discipline as before:

```text
1. Load train.csv only.
2. Use stratified 5-fold cross-validation.
3. Use PR-AUC as the primary selection metric.
4. Use ROC-AUC, balanced accuracy, F1, precision, recall, and specificity as secondary diagnostics.
5. Use pooled out-of-fold predictions for confusion matrices, threshold curves, ROC curves, and PR curves.
6. Keep threshold curves diagnostic only.
7. Do not use the held-out test set.
8. Avoid claiming that close CV differences are statistically significant.
```

The section should compare models at three levels:

```text
1. Within-family tuning:
    Which settings work well for each boosting implementation?

2. Boosting-family comparison:
    AdaBoost versus gradient boosting versus histogram boosting versus modern GBDT libraries.

3. Comparison against earlier candidates:
    Does boosting improve over logistic regression, single trees, bagging, and random forests?
```

Because several strong models may be close, the report should use cautious language:

```text
selected within the development grid;
strongest observed configuration in this section;
small differences should not be overinterpreted;
final model-family comparison remains deferred;
held-out test evaluation remains unused.
```

## 24. Expected artifacts

The notebook should save tables such as:

```text
reports/tables/boosting_model_comparison.csv
reports/tables/boosting_confusion_matrices.csv
reports/tables/adaboost_grid_results.csv
reports/tables/gradient_boosting_grid_results.csv
reports/tables/hist_gradient_boosting_grid_results.csv
reports/tables/xgboost_grid_results.csv
reports/tables/lightgbm_grid_results.csv
reports/tables/catboost_grid_results.csv
reports/tables/boosting_selection_summary.csv
reports/tables/boosting_threshold_results.csv
reports/tables/boosting_feature_importance.csv
```

And figures such as:

```text
reports/figures/adaboost_pr_auc_grid.png
reports/figures/gradient_boosting_pr_auc_grid.png
reports/figures/hist_gradient_boosting_pr_auc_grid.png
reports/figures/xgboost_pr_auc_grid.png
reports/figures/lightgbm_pr_auc_grid.png
reports/figures/catboost_pr_auc_grid.png
reports/figures/boosting_model_comparison_pr_auc.png
reports/figures/boosting_threshold_tradeoff.png
reports/figures/boosting_roc_curve.png
reports/figures/boosting_precision_recall_curve.png
reports/figures/boosting_feature_importance.png
```

If too many plots are generated, the report should include only the most informative ones and keep the rest as saved artifacts.

## 25. Expected report structure

A polished report section could use this structure:

```text
10.1 From bagging to boosting
10.2 AdaBoost and weighted mistakes
10.3 Gradient boosting and pseudo-residuals
10.4 Regularization and early stopping
10.5 Modern gradient-boosted tree libraries
10.6 Experimental design
10.7 AdaBoost results
10.8 Gradient boosting results
10.9 XGBoost, LightGBM, and CatBoost results
10.10 Model comparison against previous candidates
10.11 Threshold behaviour
10.12 ROC and precision-recall curves
10.13 Feature importance and interpretation caveats
10.14 Summary
```

The report should not become a package advertisement. It should explain what each model adds technically:

```text
AdaBoost:
    reweighted mistakes and exponential loss.

Gradient boosting:
    negative-gradient fitting and additive modelling.

HistGradientBoosting:
    histogram-based acceleration.

XGBoost:
    second-order gradients and explicit tree regularization.

LightGBM:
    histogram training, leaf-wise growth, GOSS, EFB, and categorical splits.

CatBoost:
    ordered categorical statistics, ordered boosting, and symmetric trees.
```

## 26. Practical expectations for the Telco dataset

Boosting often performs very well on tabular data. However, the Telco dataset is moderate in size and has strong categorical predictors that logistic regression already captures well after one-hot encoding. Therefore, boosting may improve performance, but it should not be assumed to dominate.

Possible outcomes:

```text
1. Boosting clearly improves PR-AUC over logistic regression and random forests.
2. Boosting is close to logistic regression and random forests, suggesting the signal is already well captured.
3. Some modern libraries improve ranking but only by small margins.
4. CatBoost may benefit from native categorical handling, but the difference may be small because the original categorical features have low to moderate cardinality.
```

The most important learning result is not only the final ranking. The section should explain why boosting works, when it helps, why it can overfit, and how modern GBDT implementations differ.

## 27. Recommended language for interpretation

Use language like:

```text
The selected boosted model is the strongest configuration within the tried development grid.
The result is a training-set cross-validated development estimate, not a final test-set claim.
The difference between close boosted models should be interpreted cautiously.
The main conclusion is whether boosting provides meaningful development-stage improvement over earlier model families.
Final model-family comparison, threshold selection, calibration, and statistical uncertainty analysis remain deferred to later stages.
```

Avoid language like:

```text
XGBoost is definitively the best model.
CatBoost proves native categorical handling is superior.
The selected boosting hyperparameters are globally optimal.
The CV score is final performance.
The test set confirms this result.
```

## 28. External technical references used for this note

```text
scikit-learn AdaBoostClassifier documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

scikit-learn GradientBoostingClassifier documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

XGBoost boosted-tree tutorial:
https://xgboost.readthedocs.io/en/stable/tutorials/model.html

LightGBM features documentation:
https://lightgbm.readthedocs.io/en/stable/Features.html

CatBoost training overview:
https://catboost.ai/docs/en/concepts/algorithm-main-stages

CatBoost categorical-feature transformation documentation:
https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic

CatBoost parameter tuning documentation:
https://catboost.ai/docs/en/concepts/parameter-tuning

CatBoost paper:
CatBoost: unbiased boosting with categorical features
https://arxiv.org/abs/1706.09516

XGBoost paper:
XGBoost: A Scalable Tree Boosting System
https://arxiv.org/abs/1603.02754
```
