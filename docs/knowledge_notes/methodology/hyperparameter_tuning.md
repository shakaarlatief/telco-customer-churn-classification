# Hyperparameter Tuning Methodology

## Purpose

This document is a reusable project methodology note for hyperparameter tuning.

The goal is not only to obtain strong validation scores. The goal is to tune models in a way that is statistically honest, reproducible, professionally defensible, and consistent with the train/validation/test discipline of the project.

This is a living note. It should be updated when later model families require additional tuning methods, for example gradient boosting, support vector machines, neural networks, or cost-sensitive threshold optimization.

## 1. Parameters versus hyperparameters

A model parameter is learned directly from the training data during fitting.

Examples:

```text
linear regression coefficients
logistic regression coefficients
tree split thresholds
neural-network weights
```

A hyperparameter is chosen outside the ordinary fitting process. It controls the learning algorithm, model class, model complexity, or training procedure.

Examples:

```text
regularization strength C in logistic regression
number of neighbours k in kNN
maximum tree depth
minimum samples per leaf
number of trees in a random forest
learning rate in gradient boosting or neural networks
number of hidden layers
number of hidden units
dropout rate
batch size
number of epochs
kernel type and gamma in SVMs
```

Parameters are fitted. Hyperparameters are selected.

## 2. Why hyperparameter tuning is dangerous

Hyperparameter tuning is itself a learning process. If many hyperparameter values are tried and the best validation score is selected, the validation set has influenced the final model choice.

This is useful and necessary, but it can overfit the validation data.

The risk becomes stronger when:

```text
the validation set is small
many hyperparameter values are tried
many model families are compared
many metrics are inspected
many plots are used informally to choose a model
the same validation results are revisited many times
```

This is a form of multiple testing. If many alternatives are tested, one setting may look good because it got lucky on the validation split.

Therefore, hyperparameter tuning must be separated from final test evaluation.

## 3. Test-set discipline

The held-out test set is not used for:

```text
preprocessing decisions
feature engineering decisions
model-family selection
hyperparameter selection
threshold tuning
calibration decisions
resampling decisions
choosing between metrics
choosing between plots
deciding whether a result is good enough
```

The test set is used once at the end for the final selected modelling pipeline.

In this project:

```text
data/processed/train.csv = development data
data/processed/test.csv  = untouched final test data
```

Within the development data, cross-validation is used to estimate performance and tune hyperparameters.

## 4. Cross-validation for hyperparameter selection

For a fixed hyperparameter setting, cross-validation works as follows:

```text
split the development data into K folds

for each fold:
    fit preprocessing on the K-1 training folds only
    fit the model on the K-1 training folds only
    evaluate on the held-out validation fold

average the validation scores across folds
```

For hyperparameter tuning:

```text
for each hyperparameter setting:
    run the cross-validation procedure
    compute the mean validation score

choose the setting with the best validation score according to a preselected metric
```

Important:

```text
The validation fold must remain untouched.
All preprocessing must be inside the pipeline.
Any resampling must be inside the fold-training part only.
```

## 5. Final model after cross-validation

Cross-validation answers two questions:

```text
Which hyperparameter setting seems best?
What validation performance can we expect approximately?
```

However, cross-validation does not directly give one final fitted model, because each fold fits a different model.

After choosing the hyperparameters, the final development-stage model is usually fitted on the whole training/development set:

```text
selected hyperparameters
+ full training data
= final fitted model
```

The held-out test set is then used once later to estimate final generalization.

This resolves a common question:

```text
"I used 5-fold cross-validation. Which model is the model?"
```

The answer is:

```text
The cross-validation models are for selection and estimation.
The final model is refit on the full available training data after selection.
```

## 6. Grid search

Grid search tries every combination in a manually specified grid.

Example for logistic regression:

```text
C = [0.001, 0.01, 0.1, 1, 10, 100]
penalty = ["l2"]
```

Grid search is simple, transparent, reproducible, and easy to report.

It is useful when:

```text
there are few hyperparameters
the grid is small
a sensible scale is known
the model is cheap to fit
the purpose is partly educational
```

Grid search becomes inefficient when there are many hyperparameters. A grid with six values for each of five hyperparameters requires:

```text
6^5 = 7776 configurations
```

before even accounting for cross-validation folds. This grows too quickly.

## 7. Random search

Random search samples hyperparameter combinations from distributions.

Example:

```text
C sampled log-uniformly between 0.001 and 100
max_depth sampled from integer values
learning_rate sampled log-uniformly
```

Random search is often preferable when:

```text
there are many hyperparameters
only some hyperparameters matter strongly
the tuning budget is limited
continuous scales are involved
a full grid would be too expensive
```

A practical rule:

```text
Use grid search for small, simple, one-dimensional or two-dimensional experiments.
Use random search for larger model families.
```

Random search also makes the computation budget explicit through the number of sampled configurations.

## 8. Successive halving and multi-fidelity search

Successive halving methods allocate a small amount of resources to many configurations, then keep only the better-performing configurations and allocate more resources to them.

The idea:

```text
start many configurations with a small budget
discard weak configurations early
spend more budget on promising configurations
```

The resource can be:

```text
number of training samples
number of estimators
number of epochs
number of iterations
```

This can be much faster than fully training every configuration.

Successive halving is useful when:

```text
many configurations are possible
partial training gives a useful signal
model training is moderately expensive
a clear resource parameter exists
```

It is less useful when:

```text
partial training is not predictive of final performance
the model is very cheap to fit
the resource parameter is unclear
```

For this project, successive halving can be considered later for ensembles or neural networks, but it is unnecessary for logistic regression.

## 9. Coarse-to-fine tuning

A practical tuning strategy is coarse-to-fine search.

The idea:

```text
1. Start with a broad search range.
2. Use an appropriate scale, often logarithmic.
3. Identify a promising region.
4. Search more finely within that region if needed.
```

This is especially important for hyperparameters such as:

```text
regularization strength
learning rate
SVM gamma
neural-network weight decay
tree/boosting learning rates
```

These often act multiplicatively rather than additively. The difference between 0.001 and 0.01 can matter as much as the difference between 1 and 10. Therefore, a logarithmic scale is often more sensible than a linear scale.

For logistic regression, a good first grid is:

```text
C = [0.001, 0.01, 0.1, 1, 10, 100]
```

because scikit-learn uses `C` as inverse regularization strength:

```text
smaller C = stronger regularization
larger C  = weaker regularization
```

If the best value is around `C = 0.1`, a second finer grid could be:

```text
C = [0.03, 0.05, 0.1, 0.2, 0.3]
```

However, too much repeated refinement can overfit validation data and distract from the main modelling objective. Fine search should be used only when it is likely to change the modelling conclusion.

## 10. Bayesian optimization

Bayesian optimization treats hyperparameter tuning as an optimization problem where the objective function is expensive to evaluate.

The objective function is usually something like:

```text
f(hyperparameters) = cross-validated validation loss
```

or:

```text
f(hyperparameters) = - cross-validated validation score
```

The true function is unknown and expensive because every evaluation requires training and validating models.

Bayesian optimization builds a surrogate model of the objective function. The surrogate estimates both expected performance and uncertainty about that performance.

An acquisition rule then chooses which hyperparameter configuration to evaluate next. The method balances exploitation and exploration:

```text
exploitation:
    try configurations expected to perform well

exploration:
    try uncertain configurations that might reveal better regions
```

Bayesian optimization is useful when:

```text
model training is expensive
the search space is continuous or mixed
the number of evaluations is limited
hyperparameters interact
manual grids become inefficient
```

It is usually unnecessary for small, transparent tuning tasks such as logistic regression with one main regularization parameter.

## 11. Optuna

Optuna is a Python framework for automatic hyperparameter optimization.

The main concepts are:

```text
study:
    one optimization run for an objective function

trial:
    one evaluation of a hyperparameter configuration

objective function:
    function that chooses hyperparameters, trains/evaluates a model, and returns a score
```

Optuna is useful because it supports:

```text
dynamic Python-defined search spaces
sequential model-based search methods such as TPE
random search
trial pruning for early stopping of weak configurations
parallel execution
visualization of optimization history and hyperparameter importance
integration with many machine-learning and deep-learning workflows
```

A simplified Optuna objective looks like:

```python
def objective(trial):
    C = trial.suggest_float("C", 1e-3, 100, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    pipeline = make_pipeline_for_this_trial(C=C, penalty=penalty)

    score = cross_validated_score(
        pipeline,
        X_train,
        y_train,
        scoring="average_precision",
    )

    return score
```

Then the study runs many trials:

```python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### When Optuna is useful

Optuna is useful when:

```text
there are many hyperparameters
the search space is large
some hyperparameters are conditional
hyperparameters interact
training is expensive enough that efficient search matters
manual grids become too large
pruning weak trials saves time
```

Good candidates in this project:

```text
gradient boosting
random forests
SVM with RBF kernel
MLP / neural networks
possibly kNN if the search becomes larger
```

### When Optuna is unnecessary

Optuna is usually unnecessary when:

```text
there is only one important hyperparameter
a small log-scale grid is clearer
the purpose is educational
the model is cheap to fit
the search space is easy to inspect manually
```

For logistic regression in section 05, a small transparent grid is better than Optuna.

### Optuna does not remove validation discipline

Optuna does not automatically prevent leakage.

The objective function must still implement the correct validation procedure:

```text
for each Optuna trial:
    choose hyperparameters
    run cross-validation on the training set
    fit preprocessing inside each fold
    evaluate on untouched validation folds
    return mean validation score
```

Wrong:

```text
Use Optuna to optimize performance on the test set.
```

Also wrong:

```text
Apply SMOTE to the full training data before Optuna cross-validation.
```

If resampling is used with Optuna, the sampler must be inside the fold-training pipeline.

## 12. Other professional tuning tools

Several professional tools exist. The project does not need all of them, but it is useful to know where they fit.

### scikit-learn search tools

For classical tabular workflows, scikit-learn's built-in search tools are usually enough:

```text
GridSearchCV
RandomizedSearchCV
HalvingGridSearchCV
HalvingRandomSearchCV
```

They integrate naturally with:

```text
Pipeline
ColumnTransformer
cross-validation splitters
custom scoring functions
multi-metric evaluation
```

These are the default tools for simple and medium-sized scikit-learn experiments.

### Optuna

Optuna is a strong general-purpose choice when the search space becomes larger or more flexible than a simple grid.

It is especially useful for:

```text
gradient boosting
SVMs
neural networks
conditional search spaces
mixed categorical/integer/continuous hyperparameters
```

### Ray Tune

Ray Tune is useful for larger-scale distributed tuning. It is most relevant when many trials must run in parallel across multiple CPUs, GPUs, or machines.

It is more infrastructure-heavy than what this Telco project currently needs.

### KerasTuner

KerasTuner is focused on tuning Keras/TensorFlow models. It can be useful for neural-network architecture search, for example tuning:

```text
number of layers
number of units
dropout
learning rate
activation functions
```

For this project, KerasTuner is only relevant if the neural-network section uses TensorFlow/Keras. If the neural-network section uses scikit-learn's MLPClassifier or PyTorch, it may not be needed.

### MLflow, Weights & Biases, and experiment tracking

Experiment tracking tools are not hyperparameter optimizers by themselves, but they are professionally important for recording:

```text
hyperparameters
metrics
artifacts
models
plots
run metadata
```

For this portfolio project, simple CSV tables and saved figures are enough for now. If the project becomes larger, adding MLflow or Weights & Biases could improve experiment management.

## 13. Appropriate scales

A key part of tuning is choosing the right scale.

Use logarithmic scales for hyperparameters where multiplicative changes matter:

```text
regularization strength
learning rate
SVM gamma
SVM C
neural-network weight decay
tree boosting learning rate
```

Use integer ranges for count-like hyperparameters:

```text
number of neighbours
max depth
min samples per leaf
number of hidden units
number of estimators
```

Use categorical choices for discrete algorithm options:

```text
penalty type
solver
kernel type
activation function
criterion
```

Bad grid:

```text
C = [1, 2, 3, 4, 5]
```

Better first grid:

```text
C = [0.001, 0.01, 0.1, 1, 10, 100]
```

## 14. Parallelization

Many hyperparameter evaluations are independent.

For example:

```text
C = 0.001
C = 0.01
C = 0.1
...
```

can be evaluated separately. Cross-validation folds can also often be evaluated in parallel.

In scikit-learn, many tools support:

```text
n_jobs=-1
```

This uses available CPU cores.

However, parallelization should be used carefully when:

```text
models already use internal parallelism
memory usage is high
the machine becomes unresponsive
```

For this project, use parallelization when it is safe, but keep reproducibility and clarity more important than speed.

## 15. Choosing the tuning metric

Before tuning, decide what metric selects the hyperparameters.

For churn classification, accuracy is not suitable by itself because the majority-class baseline already has high accuracy and zero recall.

Possible tuning metrics:

```text
balanced_accuracy
f1
roc_auc
pr_auc
recall subject to minimum precision
precision subject to minimum recall
custom cost-based score
```

For early model-family exploration, reasonable default choices are:

```text
primary selection metric: pr_auc or roc_auc for probability/ranking models
secondary metrics: balanced_accuracy, recall, precision, specificity, f1
```

The exact choice depends on the project objective.

In this project, because churn is the minority class and the quality of positive predictions matters, PR-AUC is especially informative for ranking models. Balanced accuracy is also useful because it treats churners and non-churners more symmetrically.

A clean approach for section 05:

```text
Use PR-AUC or ROC-AUC to tune probability/ranking models.
Report the full metric table.
Do not tune the classification threshold yet.
Use threshold 0.5 for default class predictions.
Discuss threshold tuning separately.
```

## 16. Threshold tuning is also hyperparameter tuning

For probability models, the classification threshold is a hyperparameter.

Example:

```text
predict churn if p(churn) >= threshold
```

The default threshold is usually:

```text
threshold = 0.5
```

But churn prediction may prefer lower thresholds if missing churners is costly.

Important:

```text
Do not tune the threshold on the test set.
```

Threshold tuning should be performed using validation data or cross-validated out-of-fold predictions.

Good workflow:

```text
1. Tune model hyperparameters using cross-validation.
2. Generate out-of-fold predicted probabilities for the selected model.
3. Examine threshold tradeoffs on those out-of-fold predictions.
4. Choose a threshold based on a predefined rule or business cost.
5. Apply the chosen threshold once to the final test predictions.
```

For section 05, threshold curves can be shown as educational analysis, but final threshold selection should probably wait until later model comparison.

## 17. Resampling and tuning

Resampling methods include:

```text
random oversampling
random undersampling
SMOTE
data augmentation
```

These are training interventions. They modify the training distribution.

Correct cross-validation workflow with resampling:

```text
for each cross-validation split:
    fit preprocessing on the fold-training data
    apply resampling to the fold-training data only
    fit the model on the resampled fold-training data
    evaluate on the untouched validation fold
```

Incorrect workflow:

```text
resample the entire training dataset first
then cross-validate
```

This leaks information and gives overly optimistic validation results.

In Python, use an imbalanced-learn pipeline when resampling is needed:

```text
preprocessing -> sampler -> classifier
```

For section 05, do not immediately use SMOTE. We can compare ordinary logistic regression and possibly class-weighted logistic regression. More advanced resampling can be a later dedicated imbalance section.

## 18. Early stopping as tuning

Early stopping is a form of hyperparameter-controlled training.

It monitors validation performance during training and stops when validation error stops improving.

It is especially relevant for:

```text
neural networks
gradient boosting
iterative optimization algorithms
```

The important point is that early stopping uses validation data. Therefore, if early stopping is used while also comparing model families or other hyperparameters, the validation structure must be carefully designed.

In simple scikit-learn workflows, early stopping can be handled inside cross-validation if the model internally splits its training fold or receives a validation fold correctly. We should document the exact procedure when we reach neural networks or boosting.

## 19. Bias and variance perspective

Hyperparameter tuning often controls the bias-variance tradeoff.

High bias:

```text
training performance poor
validation performance poor
model is too simple or undertrained
```

Possible responses:

```text
increase model complexity
reduce regularization
train longer
add useful features
```

High variance:

```text
training performance very good
validation performance much worse
model overfits
```

Possible responses:

```text
increase regularization
simplify the model
use more data
use early stopping
use dropout for neural networks
use data augmentation when appropriate
```

For logistic regression:

```text
small C  -> stronger regularization -> more bias, less variance
large C  -> weaker regularization   -> less bias, more variance
```

For trees:

```text
deeper tree    -> less bias, more variance
shallower tree -> more bias, less variance
```

For kNN:

```text
small k -> low bias, high variance
large k -> high bias, low variance
```

## 20. Reporting tuning honestly

In the report, explicitly state:

```text
The test set was not used for hyperparameter tuning.
Hyperparameters were selected using cross-validation inside the training set.
Preprocessing was fitted inside each cross-validation fold.
The final model was refit on the full training set only after hyperparameter selection.
The test set was used only for the final evaluation.
```

For each tuned model family, report:

```text
hyperparameter grid or search distribution
selection metric
cross-validation procedure
best selected hyperparameters
full metric table
```

Avoid reporting only the best number without explaining how it was selected.

## 21. Practical tuning policy for this project

Use increasingly flexible tuning methods as model complexity increases.

### Section 05: linear classification and logistic regression

Use small, interpretable grids:

```text
RidgeClassifier alpha: [0.01, 0.1, 1, 10, 100]
LogisticRegression C: [0.001, 0.01, 0.1, 1, 10, 100]
L1 LogisticRegression C: [0.001, 0.01, 0.1, 1, 10]
```

No SMOTE yet.

Possibly include:

```text
class_weight=None
class_weight="balanced"
```

as a small controlled comparison.

Use grid search because it is transparent and the search space is small.

### kNN section

Tune:

```text
n_neighbors
weights
distance metric
```

Use scaling. Use a sensible range, not an enormous grid.

Grid search is usually enough.

### Trees section

Tune:

```text
max_depth
min_samples_leaf
min_samples_split
criterion
```

Use validation/CV. Discuss pruning and complexity control.

Grid search or random search can both be appropriate.

### Ensembles section

Tune:

```text
n_estimators
max_depth
learning_rate
subsample
max_features
min_samples_leaf
```

Consider random search or Optuna rather than a full grid.

### SVM section

Tune:

```text
C
gamma
kernel
degree for polynomial kernel
```

Use log-scale grids. Scaling is required.

For a larger kernel SVM experiment, Optuna or random search may be more efficient than a large grid.

### MLP section

Tune carefully:

```text
hidden_layer_sizes
alpha
learning_rate_init
batch_size
activation
early_stopping
```

Use a limited budget. Neural networks have many hyperparameters, so coarse-to-fine search, random search, Optuna, or a framework-specific tuner can be more suitable than a large grid.

## 22. Immediate decision for section 05

For logistic regression, use a simple log-scale grid search first.

Recommended section 05 approach:

```text
1. Fit RidgeClassifier as a regularized least-squares classifier.
2. Fit default L2 logistic regression.
3. Tune L2 logistic regression over C values.
4. Tune L1 logistic regression over C values.
5. Compare class_weight=None versus class_weight="balanced" as a small controlled experiment.
6. Use cross-validated metrics and out-of-fold predictions.
7. Extract coefficients from a selected model fitted on the full training set for interpretation.
8. Do not use the test set.
9. Do not use SMOTE yet.
10. Do not use Optuna yet.
```

This keeps the section rigorous, educational, and not unnecessarily complicated.

## 23. Update policy for this note

This note should be updated when:

```text
a new tuning method is introduced
a new model family needs a different tuning strategy
threshold optimization becomes part of the workflow
class imbalance experiments begin
distributed or expensive experiments become relevant
experiment tracking is added
```

The note should remain a general methodology reference. Actual model results belong in notebooks and the LaTeX report.
