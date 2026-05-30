# Cross-validation, model selection, and hyperparameter tuning

This note explains how validation, cross-validation, repeated cross-validation, nested cross-validation, and hyperparameter search fit together.

The central idea is:

> Cross-validation is not one single method with one single interpretation. It can estimate the performance of a fixed modelling setup, guide hyperparameter tuning, compare model families, or evaluate a complete model-selection procedure. The interpretation depends on how it is used.

This note is part of the evaluation methodology module for the Telco Customer Churn classification project.

---

## 1. Why validation is needed

A supervised learning model is trained to minimize a loss or fit a relationship on training data. But the goal is not training performance. The goal is generalization to new data.

The simplest clean split is:

```text
training set:
    fit model parameters

validation set:
    choose hyperparameters, preprocessing choices, feature choices,
    thresholds, calibration choices, and model family

test set:
    estimate final performance once after all choices are fixed
```

If validation data is not separated from training data, then model choices can overfit the training sample. If test data is used for validation, then the test set becomes part of model development and no longer gives an honest final estimate.

Validation data is therefore the controlled place where model development happens.

---

## 2. Hold-out validation

The simplest validation strategy is a single hold-out split.

For example:

```text
training portion:
    80 percent of the available development data

validation portion:
    20 percent of the available development data
```

The workflow is:

```text
1. Fit the model on the training portion.
2. Predict on the validation portion.
3. Compute validation metrics.
4. Use the validation result to choose modelling decisions.
```

This is simple and easy to understand.

The main disadvantage is that the validation result depends on one particular split. If the validation subset is slightly easier or harder than average, the estimate can be lucky or unlucky.

Another disadvantage is that part of the available development data is not used for fitting each candidate model. This can be inefficient when the dataset is not very large.

---

## 3. k-fold cross-validation

k-fold cross-validation reduces dependence on a single validation split.

The data is split into \(K\) folds:

$$
\mathcal{D}
=
\mathcal{D}_1
\cup
\mathcal{D}_2
\cup
\cdots
\cup
\mathcal{D}_K.
$$

For each fold \(k\):

```text
validation fold:
    D_k

training folds:
    all folds except D_k
```

The model is trained on the training folds and evaluated on the validation fold. This is repeated for all \(K\) folds.

If \(\widehat{M}^{(k)}\) is the metric on fold \(k\), the cross-validation estimate is:

$$
\widehat{M}_{CV}
=
\frac{1}{K}
\sum_{k=1}^{K}
\widehat{M}^{(k)}.
$$

Common values are:

```text
K = 5
K = 10
```

For classification with class imbalance, stratified k-fold cross-validation is usually preferred. Stratification keeps the class distribution approximately similar in each fold.

In this project, the positive class is churn. Because churn is the minority class, stratification is important. It helps ensure that each validation fold contains a representative proportion of churners.

---

## 4. What ordinary cross-validation estimates

For a **fixed modelling setup**, cross-validation estimates development-stage generalization performance.

A fixed modelling setup means that the following are already specified:

```text
model family
preprocessing
features
hyperparameters
class weighting or resampling choice
training algorithm
threshold rule, if hard predictions are being evaluated
metric
```

Example fixed setup:

```text
model:
    logistic regression

preprocessing:
    median-imputed numeric features, standardized numeric features,
    one-hot encoded categorical features

hyperparameters:
    L2 penalty
    C = 1.0
    class_weight = None

metric:
    PR-AUC
```

For this fixed setup, 5-fold CV estimates how well the setup performs when trained on approximately 80 percent of the development data and evaluated on the remaining 20 percent, averaged over the five validation folds.

This is useful, but it is not the same as final test performance.

---

## 5. Cross-validation for hyperparameter tuning

Cross-validation is often used not only to evaluate a fixed setup, but also to choose hyperparameters.

Suppose the model family is k-nearest neighbours. Important hyperparameters include:

```text
number of neighbours k
distance metric
uniform versus distance weighting
```

A tuning grid could be:

```text
k in {3, 5, 11, 21, 51, 101}
distance in {euclidean, manhattan}
weights in {uniform, distance}
```

For each hyperparameter combination:

```text
1. Run cross-validation.
2. Compute the mean validation metric.
3. Store the score.
```

Then select the hyperparameter combination with the best mean validation score.

This is normal and valid as a model-development procedure.

However, the interpretation changes. The selected score is no longer just the score of one fixed model. It is the best score among several tried configurations.

---

## 6. Same folds for fair hyperparameter comparison

When comparing hyperparameter settings, it is usually best to use the same fold splits for every setting.

If setting A and setting B are evaluated on different random fold splits, their difference includes:

```text
difference in model behaviour
+
difference caused by different validation splits
```

Using the same folds makes the comparison more controlled:

```text
setting A and setting B are evaluated on the same validation observations
```

This is a paired comparison structure. The same easy and hard validation cases are used for all settings.

In practice, this means defining the cross-validation splitter once and reusing it across the grid.

---

## 7. Selection optimism

Hyperparameter tuning creates selection optimism.

Suppose we evaluate 100 hyperparameter settings. Each setting has a true validation performance, but each cross-validation estimate is noisy. Some settings will get lucky and score higher than their true performance. Others will get unlucky.

When we choose the setting with the highest estimated score, we are more likely to choose a setting that is good and lucky.

Therefore:

```text
the best observed CV score
```

can be somewhat optimistic for the true performance of the selected setting.

This does not mean the tuning procedure is wrong. It means the selected CV score should be interpreted carefully.

Correct interpretation:

```text
Within this development-stage grid, this setting achieved the highest
cross-validated score according to the chosen metric.
```

Too strong:

```text
This setting is definitely the best possible setting.
```

The problem becomes larger when:

```text
many hyperparameter settings are tried
the validation set is small
fold-level variability is high
many model families are compared
the final report emphasizes only the single winning score
```

---

## 8. Regression toward the mean

Selection optimism is closely related to regression toward the mean.

An observed score contains true performance plus random variation:

```text
observed score = true performance + validation noise
```

The winning score is often high partly because of favourable noise. If the selected configuration is evaluated again on fresh data, its score may move downward toward its true performance.

This does not imply the model was bad. It means the first estimate was partly lucky.

This is why very small differences should not be overinterpreted. If two configurations differ by 0.001 in PR-AUC, that difference may be much smaller than the evaluation noise.

---

## 9. Repeated cross-validation

Repeated cross-validation repeats the cross-validation process with different random fold splits.

Example:

```text
5-fold CV with 10 repeats:
    10 different stratified 5-fold partitions
    50 validation scores in total
```

Repeated CV estimates are usually more stable than a single CV split because the average is less dependent on one particular random partition.

Repeated CV is useful for:

```text
more stable hyperparameter tuning
estimating sensitivity to fold construction
describing score variability across folds and repeats
reducing the chance that one unlucky split determines the chosen hyperparameter
```

However, repeated CV does not fully solve selection optimism. If many hyperparameter settings are evaluated and the best repeated-CV score is selected, selection over noisy estimates still occurs. The estimates are less noisy than single-CV estimates, but they are still estimates.

The distinction is:

```text
Repeated CV:
    improves stability of performance estimates and tuning choices.

Nested CV:
    evaluates a tuning or model-selection procedure more honestly.
```

---

## 10. When repeated CV is worth using

Repeated CV is most useful when:

```text
the dataset is not very large
single-CV results are unstable
hyperparameter choices change a lot across random splits
model rankings are close
the model family is important for final selection
compute cost is acceptable
```

Repeated CV may be unnecessary when:

```text
the dataset is very large
scores are extremely stable
the model section is mainly educational
the model family is not a serious final candidate
the model is computationally expensive and early-stage
```

For this project, ordinary stratified 5-fold CV is reasonable for model-family learning sections. Later, repeated CV can be used for serious candidate models before final selection.

---

## 11. Nested cross-validation

Nested cross-validation separates hyperparameter tuning from performance estimation.

It has two loops:

```text
outer loop:
    evaluate the full tuning procedure

inner loop:
    choose hyperparameters using only the outer-training data
```

For each outer fold:

```text
1. Split the development data into outer-training and outer-validation parts.

2. Inside the outer-training part:
       run inner cross-validation over the hyperparameter grid;
       choose the best hyperparameter setting.

3. Fit the model with the chosen hyperparameters on the full outer-training part.

4. Evaluate the fitted model once on the outer-validation fold.
```

The outer-validation fold is not used for tuning. It only evaluates the result of the tuning procedure.

The final nested-CV estimate is the average of the outer validation scores.

---

## 12. What nested CV estimates

Nested CV estimates the performance of a **procedure**, not one fixed hyperparameter setting.

For kNN, the evaluated procedure might be:

```text
Given a training sample:
    use inner CV to choose k, distance metric, and weighting rule;
    fit kNN with the chosen settings on the full training sample;
    use the fitted model for prediction.
```

Different outer folds may choose different \(k\) values. That is not a problem. It reflects the fact that the procedure can choose different hyperparameters depending on the training sample.

This is why nested CV is especially useful for comparing tuned model families.

---

## 13. Nested CV for comparing tuned model families

Suppose we want to compare:

```text
tuned logistic regression
tuned kNN
tuned Naive Bayes
tuned decision tree
tuned random forest
tuned boosting
tuned SVM
tuned MLP
```

A non-nested comparison could tune each family with CV and compare the best CV scores. This is useful during development, but the best scores may contain selection optimism.

Nested CV gives a cleaner comparison.

For each outer fold, each model family gets its own inner tuning procedure:

```text
Outer fold k:

    Logistic regression:
        inner CV chooses penalty and C
        fit selected logistic model on outer-training data
        evaluate on outer-validation data

    kNN:
        inner CV chooses k, distance, and weights
        fit selected kNN on outer-training data
        evaluate on outer-validation data

    Random forest:
        inner CV chooses tree and forest hyperparameters
        fit selected forest on outer-training data
        evaluate on outer-validation data
```

The outer-fold scores compare procedures under the same outer validation data.

This answers:

> If each family is allowed to tune itself using only training data, which family generalizes better?

It does not directly answer:

> What is the final hyperparameter setting I should deploy?

For final deployment, hyperparameters are usually selected using a tuning procedure on the full training set, then the final model is fitted on the full training set.

---

## 14. Nested CV versus final train/test evaluation

Nested CV is useful for estimating the performance of a tuning procedure on the development data. A held-out test set is still the clean final evaluation if available.

A strong project can use both:

```text
nested CV:
    compare model-family procedures before touching the test set

final test evaluation:
    evaluate the chosen final model once after all choices are fixed
```

For this project, a practical roadmap is:

```text
Individual model sections:
    ordinary stratified CV for learning and development

Later comparison stage:
    repeated CV and/or nested CV for serious candidate families

Final stage:
    train final model on full training data
    evaluate once on the untouched test set
```

---

## 15. Fold-level metrics versus pooled out-of-fold metrics in model selection

When using CV for model selection, there are two different summaries:

```text
mean fold metric:
    compute the metric separately on each validation fold;
    average the fold metrics

pooled out-of-fold metric:
    collect all out-of-fold predictions;
    compute one metric on the pooled predictions
```

For formal CV model selection, mean fold metrics are usually the cleaner summary.

Pooled out-of-fold predictions are still extremely useful for:

```text
threshold curves
confusion-matrix summaries
ROC curve plots
precision-recall curve plots
calibration plots
visual diagnostics
```

However, for nonlinear metrics such as ROC-AUC and PR-AUC, pooled OOF AUC and mean fold AUC may differ.

A strong evaluation utility should ideally store both:

```text
fold_mean_pr_auc
fold_std_pr_auc
pooled_oof_pr_auc

fold_mean_roc_auc
fold_std_roc_auc
pooled_oof_roc_auc
```

This project can add that later when building the rigorous comparison stage.

---

## 16. Hyperparameter search strategies

Hyperparameter search is the process of choosing the candidate settings that will be evaluated.

Common strategies:

```text
manual search
grid search
random search
Bayesian optimization
successive halving / early stopping search
Optuna-style adaptive search
```

Each has different strengths and weaknesses.

---

## 17. Manual tuning

Manual tuning means the analyst chooses settings based on model knowledge, diagnostic plots, and previous results.

Advantages:

```text
uses domain and modelling understanding
efficient when the analyst knows which parameters matter
easy to explain
good for educational projects
```

Disadvantages:

```text
subjective
hard to reproduce exactly
can favour models the analyst understands better
can give unequal tuning effort across model families
can accidentally overfit validation data through repeated trial and error
```

Manual tuning is acceptable when it is transparent and disciplined. The report should state what was tried and why.

---

## 18. Grid search

Grid search defines a finite set of values for each hyperparameter and evaluates every combination.

Example:

```text
max_depth in {2, 3, 4, 5, None}
min_samples_leaf in {1, 5, 10, 25}
criterion in {gini, entropy}
```

Grid search is simple and reproducible.

Advantages:

```text
easy to understand
systematic
good when few hyperparameters matter
works well for small grids
```

Disadvantages:

```text
expensive as dimensionality grows
wastes trials on unimportant dimensions
requires manually chosen value grids
can miss good values between grid points
```

For parameters that operate by orders of magnitude, grids should often be logarithmic rather than linear.

Examples:

```text
C in {0.001, 0.01, 0.1, 1, 10, 100}
learning_rate in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
alpha in {0.001, 0.01, 0.1, 1, 10}
```

---

## 19. Random search

Random search samples hyperparameter combinations randomly from specified distributions.

The important insight is that not all hyperparameters matter equally. If only one or two dimensions are important, grid search may waste many trials repeating the same values of the important dimensions while varying unimportant dimensions.

Random search often explores more distinct values of the important hyperparameters for the same number of trials.

Advantages:

```text
often more efficient than grid search in high-dimensional spaces
easy to parallelize
works well when only a few hyperparameters matter strongly
can sample from continuous distributions
```

Disadvantages:

```text
results vary by random seed
may miss important regions if the number of trials is small
less exhaustive than grid search
requires choosing sampling distributions
```

Random search is often a strong default for larger hyperparameter spaces.

---

## 20. Bayesian optimization and Optuna-style search

Bayesian optimization uses previous trial results to decide which hyperparameters to try next.

The general idea is:

```text
1. Try some initial configurations.
2. Fit a surrogate model of validation performance over hyperparameter space.
3. Choose new configurations that balance exploration and exploitation.
4. Repeat.
```

Optuna is a practical hyperparameter optimization framework that can implement adaptive search and pruning. Pruning stops unpromising trials early when intermediate results suggest they are unlikely to become competitive.

Advantages:

```text
can be more sample-efficient than grid or random search
useful for expensive models
can handle complex search spaces
supports early stopping / pruning
```

Disadvantages:

```text
more complex
less transparent than simple grids
can overfit validation performance if used excessively
requires careful search-space design
adds another layer of randomness and procedure choice
```

For this project, Optuna is useful later for serious candidate models, but it is not necessary for every educational model-family section.

---

## 21. Fair tuning effort across model families

A model comparison can be unfair if some model families receive much more tuning effort than others.

Example:

```text
logistic regression:
    only default C = 1

boosting:
    500 Optuna trials over many parameters
```

If boosting wins, part of the gain may come from greater search effort rather than model-family superiority.

Fair comparison principles:

```text
use the same training/validation/test split
use the same primary selection metric
use comparable preprocessing discipline
use comparable search effort where feasible
report search spaces
report tuning strategy
avoid tuning the favourite model much more than baselines
prefer automatic search for final comparison if manual effort would be unfair
```

This does not mean every model must receive exactly identical compute. Some models have more important hyperparameters than others. But the report should be transparent about tuning effort.

---

## 22. Simplicity versus performance

The highest validation score is not always the best modelling choice.

A very complex model or highly specific hyperparameter configuration may be less robust than a simpler model with nearly identical validation performance.

For example:

```text
Model A:
    PR-AUC = 0.657
    simple, stable, interpretable

Model B:
    PR-AUC = 0.659
    complex, unstable, requires many tuned tricks
```

Model B has a slightly higher observed score, but the difference may be within validation noise. Model A may be preferable if interpretability, robustness, and simplicity matter.

Practical rule:

> When performance differences are small, prefer the simpler or more stable modelling choice unless there is a clear reason not to.

This is especially important for portfolio work because the goal is not only to maximize a leaderboard score. The goal is to demonstrate sound modelling judgement.

---

## 23. Metric choice is part of model selection

The chosen selection metric shapes the selected model.

Examples:

```text
accuracy:
    may favour majority-class behaviour under imbalance

recall:
    may favour aggressive positive prediction

precision:
    may favour conservative positive prediction

F1:
    balances precision and recall at a fixed threshold

ROC-AUC:
    evaluates ranking across thresholds, less sensitive to prevalence than PR-AUC

PR-AUC:
    focuses more on positive-class retrieval under class imbalance

expected cost:
    directly incorporates business costs if costs are known
```

For churn prediction, PR-AUC is useful because churn is the positive minority class and positive-class retrieval matters. But PR-AUC is not the only relevant quantity.

The project should continue to report multiple metrics and interpret tradeoffs.

---

## 24. Threshold tuning as model selection

For probabilistic classifiers, hyperparameters are not the only tuning choices. The classification threshold is also a tuning choice.

If a threshold is chosen to maximize F1, minimize expected cost, or achieve a target recall, that threshold was selected using validation data.

Therefore, threshold selection must obey the same data discipline as hyperparameter tuning:

```text
validation data:
    choose threshold

test data:
    evaluate chosen threshold once
```

Threshold curves in the model sections are diagnostic. They show possible operating points. They do not yet define the final deployment threshold.

---

## 25. Calibration as model selection

Probability calibration is another layer of modelling.

A classifier can be transformed by calibration methods such as:

```text
Platt scaling / sigmoid calibration
isotonic regression
temperature scaling
```

Calibration should be selected and fitted without leaking test information.

Possible workflow:

```text
training data:
    fit base model

calibration/validation data:
    fit calibrator

test data:
    evaluate calibrated probabilities once
```

or cross-validation-based calibration methods.

Calibration changes probability estimates and can affect threshold-based decisions. It should therefore be treated as part of the model-selection procedure.

---

## 26. How this affects sections 05, 06, and 07

The already completed model-family sections should be interpreted as development-stage comparisons.

For logistic regression, kNN, and Naive Bayes:

```text
The selected configuration is the best within the tried development grid
according to the chosen selection criterion.

The cross-validated score is useful development evidence.

Small differences between similar settings should be interpreted cautiously.

Final model-family claims are deferred until the later comparison and
test-evaluation stages.
```

This wording is important because many neighbouring hyperparameter settings can perform similarly.

Example for kNN:

```text
Rather than saying:
    k = 51 is definitively the best kNN model.

Say:
    Within the development grid, k = 51 gives the strongest PR-AUC.
    Nearby values perform similarly, so the result mainly suggests that
    moderate smoothing is preferable to very local neighbourhoods.
```

Example for Naive Bayes:

```text
Rather than saying:
    Hybrid Naive Bayes is proven better than full GaussianNB.

Say:
    Within the development comparison, the hybrid model has the strongest
    PR-AUC and is more theoretically appropriate for the mixed feature space.
    The observed performance gap is useful evidence but should not be
    treated as a final statistical superiority claim.
```

---

## 27. Recommended project strategy

The project should use different evaluation strategies at different stages.

### Stage 1: model-family learning sections

Use:

```text
ordinary stratified 5-fold CV
transparent tuning grids
development-stage metrics
threshold diagnostics
model-specific interpretation
```

Purpose:

```text
learn each model family
understand assumptions
compare broad behaviour
build project narrative
```

### Stage 2: rigorous model-family comparison

After all model families are implemented, consider:

```text
repeated CV for stable tuning of top candidates
nested CV for comparing tuned model-family procedures
fold-level variability summaries
hyperparameter stability analysis
metric sensitivity analysis
```

Purpose:

```text
compare serious candidates more carefully
avoid overclaiming based on small development differences
evaluate selection procedures
```

### Stage 3: final model and threshold selection

Use training data only to choose:

```text
final model family
final preprocessing
final hyperparameters
final threshold
optional calibration
```

### Stage 4: final test evaluation

Use the untouched test set once.

Report:

```text
point estimates for the single frozen final model
confidence intervals for the single frozen final model
calibration diagnostics for the single frozen final model if probabilities matter
```

---

## 28. Summary

The main lessons are:

```text
1. Validation and cross-validation are tools for model development.
2. Hyperparameter tuning changes the interpretation of CV scores.
3. The best CV score after a search can be optimistic.
4. Repeated CV improves stability but does not fully remove selection optimism.
5. Nested CV evaluates a tuning or model-selection procedure.
6. Nested CV is especially useful for comparing tuned model families.
7. Fair model comparison requires comparable tuning effort and transparent search spaces.
8. Small differences between tuned configurations should be interpreted cautiously.
9. Simplicity and robustness matter when performance differences are small.
10. Final performance must be evaluated once on the untouched test set for exactly one frozen final model.
```

For the current project, the individual model sections should continue to use ordinary stratified CV for clear learning and development. The more advanced repeated-CV, nested-CV, statistical-testing, and final test-set uncertainty methods should be introduced later in a dedicated comparison and final evaluation stage.
