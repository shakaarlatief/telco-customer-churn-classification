# Telco Customer Churn Classification: Model Inventory and Roadmap

## Purpose

This document is the strategic modelling roadmap for the Telco Customer Churn classification project.

The project is not only about finding the best churn model. It is also a reusable classification reference project. The goal is to apply, preserve, and explain machine-learning knowledge through a simple-to-complex modelling sequence on one coherent dataset.

The polished report should read as a standalone technical report. Knowledge notes may preserve deeper theory and modelling plans, but report prose should not depend on lecture references.

## Current project state

Latest confirmed GitHub commit:

```text
b85658e4bcfddfe7f2255f9f4f42324209a09227
Revise naive Bayes section with hybrid model
```

Completed and committed modelling/report stages through that commit:

```text
01_raw_data_audit
02_cleaning_and_splitting
03_training_set_eda
04_preprocessing_and_simple_baselines
05_linear_classification_and_logistic_regression
06_k_nearest_neighbours
07_naive_bayes
```

Prepared after that commit and intended to be added before continuing:

```text
statistical evaluation methodology knowledge notes
report methodology rewrite
documentation cleanup and handoff update
```

The next modelling stage after cleanup is:

```text
08_decision_trees
```

## Important data state

Clean modelling dataset:

```text
7043 observations
```

Training set:

```text
5634 observations
about 26.54% churn
```

Held-out test set:

```text
1409 observations
about 26.54% churn
```

Target:

```text
Churn_binary
```

Positive class:

```text
Churn_binary = 1 means churn
```

The held-out test set must remain unused until final evaluation.

## Feature groups

Numeric features:

```text
tenure
MonthlyCharges
TotalCharges
```

Categorical features:

```text
SeniorCitizen
gender
Partner
Dependents
PhoneService
PaperlessBilling
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaymentMethod
```

`customerID` is excluded from modelling because it is a unique identifier.

## High-level modelling philosophy

Each model section should answer:

```text
1. What kind of model is this?
2. What mathematical function or rule does it use?
3. What loss function, impurity criterion, likelihood, or search criterion does it use?
4. What assumptions does it make?
5. What preprocessing does it need?
6. What does it output: hard classes, scores, probabilities, or rankings?
7. How should its output be evaluated?
8. What does it teach us on this dataset?
9. What are its limitations?
10. How does it compare with earlier models?
```

The project should include multiple model families even if some are not ultimately best. A weaker model can still be valuable if it teaches a different modelling principle.

## Evaluation philosophy

All model sections before the final stage use training-set cross-validation only.

Section-level results are:

```text
development-stage cross-validated estimates
```

They are not:

```text
final performance claims
```

This matters especially when comparing close hyperparameter settings or close model families.

Use language such as:

```text
selected within the development grid
representative strong candidate
development-stage evidence
small differences should be interpreted cautiously
final test evaluation is deferred
```

Final model-family comparison and final test-set evaluation are separate later stages.

## Completed model-family sections

### 04. Preprocessing, evaluation, and simple baselines

Role:

```text
establishes evaluation language, preprocessing infrastructure, positive-first
confusion matrix, simple dummy baselines, and the EDA-inspired rule baseline
```

Simple baselines:

```text
majority-class baseline
prior-probability baseline
stratified random baseline
uniform random baseline
EDA-inspired rule baseline
```

Main lesson:

```text
accuracy alone is misleading under churn imbalance
the EDA rule detects many churners but creates many false positives
learned models should improve precision/specificity while retaining useful recall
```

### 05. Linear classification and logistic regression

Role:

```text
first learned model family and first interpretable benchmark
```

Models:

```text
RidgeClassifier as regularized least-squares classification
L2 logistic regression
L1 logistic regression
class-weighted L2 logistic regression
```

Main development results:

```text
L1 and L2 logistic regression are effectively tied in PR-AUC and ROC-AUC
standard logistic regression gives ROC-AUC about 0.846 and PR-AUC about 0.658
class weighting strongly increases recall but creates many more false positives
```

Main lesson:

```text
logistic regression is a strong, stable, interpretable benchmark
regularization strength matters mainly when it is extremely strong
the exact best C should not be overinterpreted
```

### 06. k-nearest neighbours

Role:

```text
first non-parametric local similarity model
```

Grid:

```text
n_neighbors = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
weights = ["uniform", "distance"]
p = [1, 2]
```

Selected development configuration:

```text
k = 101
uniform weights
Manhattan distance
```

Main development results:

```text
ROC-AUC about 0.836
PR-AUC about 0.628
```

Main lesson:

```text
small k is too noisy
larger smoother neighbourhoods work better
kNN improves strongly over default k=5
logistic regression remains stronger by ranking metrics
```

### 07. Naive Bayes

Role:

```text
first explicitly generative model family
```

Models:

```text
GaussianNB numeric only
BernoulliNB categorical only
Hybrid Gaussian-BernoulliNB
GaussianNB full transformed
```

Important source-code addition:

```text
HybridGaussianBernoulliNB in src/telco_churn/models.py
```

Selected development model:

```text
Hybrid Gaussian-BernoulliNB alpha=1
```

Main development results:

```text
ROC-AUC about 0.822
PR-AUC about 0.615
recall about 0.809
```

Main lesson:

```text
hybrid NB is theoretically cleaner for mixed numeric and one-hot features
Naive Bayes has useful recall and ranking ability
conditional independence remains a major limitation
logistic regression remains strongest so far by PR-AUC and ROC-AUC
```

## Methodology knowledge module

The project now has or should add these methodology notes:

```text
docs/knowledge_notes/methodology/evaluation_foundations.md
docs/knowledge_notes/methodology/cross_validation_and_model_selection.md
docs/knowledge_notes/methodology/statistical_uncertainty_and_tests.md
docs/knowledge_notes/methodology/final_model_comparison_plan.md
docs/knowledge_notes/methodology/hyperparameter_tuning.md
```

Together these cover:

```text
true metric versus sample metric
sampling uncertainty
train/validation/test discipline
leakage
cross-validation
repeated CV
nested CV
hyperparameter tuning
selection optimism
fair tuning effort
bootstrap confidence intervals
paired model comparisons
statistical tests
final model comparison plan
```

## Remaining model-family roadmap

### 08. Decision trees

Planned next.

Topics:

```text
recursive partitioning
leaf predictions
Gini impurity
entropy
information gain / impurity reduction
max depth
minimum samples split
minimum samples leaf
cost-complexity pruning
overfitting and interpretability
decision stumps
```

Expected experiments:

```text
simple stump
default decision tree
small grid over depth and leaf size
possibly cost-complexity pruning path
threshold and ranking diagnostics
comparison against logistic regression, kNN, and Naive Bayes
```

### 09. Bagging and random forests

Topics:

```text
bootstrap aggregation
variance reduction
decorrelated trees
feature subsampling
out-of-bag intuition
random forest feature importance
```

Expected experiments:

```text
bagged trees
random forest
small or moderate hyperparameter search
comparison with single decision tree
```

### 10. Boosting

Topics:

```text
boosting as sequential error correction
AdaBoost
weak learners and decision stumps
sample weights
gradient boosting
learning rate
number of estimators
tree depth
early stopping
```

Expected experiments:

```text
AdaBoost
gradient boosting
possibly histogram gradient boosting
possibly XGBoost or LightGBM only if added deliberately
```

### 11. Support vector machines

Topics:

```text
linear margin classifiers
hinge loss
soft margin
C
kernel trick
RBF kernel
gamma
decision scores
calibration if probabilities are needed
```

Expected experiments:

```text
linear SVM
RBF SVM
scaled preprocessing
careful tuning because SVMs are sensitive to C and gamma
```

### 12. Multilayer perceptron

Topics:

```text
feedforward network
hidden layers
activation functions
binary cross-entropy
optimization
regularization
early stopping
dropout or batch normalization only if justified
train/eval mode if using PyTorch
```

Expected experiments:

```text
small MLP for tabular classification
compare against classical models
avoid overcomplicating the project
```

## Later methodology and final comparison roadmap

After the individual model-family sections, create a dedicated comparison stage.

Topics:

```text
candidate shortlist
repeated CV for serious candidates
possibly nested CV for tuned model-family procedures
metric sensitivity
threshold selection
calibration analysis if probabilities matter
bootstrap confidence intervals
paired bootstrap differences against runner-up models
final test-set evaluation
ablation studies
interpretability
```

The final test set should be used once after all choices are fixed.

## Methods saved for later projects

Not central for this Telco tabular binary classification project:

```text
CNNs
GANs
sequence models
transformers
matrix factorization / recommender systems
reinforcement learning
```

They can be mentioned as future-project topics but should not distract from the classification roadmap.

## Immediate next stage

Before starting decision trees:

```text
1. Add documentation cleanup files.
2. Add methodology knowledge notes.
3. Add report methodology rewrite.
4. Compile and check the PDF.
5. Commit the documentation/methodology update.
6. Start decision tree knowledge note and section 08.
```
