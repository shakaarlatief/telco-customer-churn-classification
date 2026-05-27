# Telco Customer Churn Classification: Model Inventory and Roadmap

## Purpose of this document

This document is a project roadmap for the Telco Customer Churn classification project.

The purpose is to slow down before modelling, go through the Machine Learning course material systematically, and decide which models should be used, where they appear in the course material, what mathematics should be explained, and where each model belongs in the project.

This project is not only about finding the best churn model. It is also a practical reference project for applying and preserving classification knowledge through a structured, simple-to-complex modelling sequence. The formal report should read as a standalone technical report, while this roadmap may still track where concepts enter the modelling plan.

The intended workflow is:

1. Go through models roughly in chronological slide order.
2. Keep a simple-to-complex modelling progression.
3. For each model, understand the mathematical definition.
4. Decide what preprocessing the model needs.
5. Apply the model to the Telco churn dataset when suitable.
6. Write professional report sections that explain the model without referring to lectures directly.
7. Save models that are not suitable for tabular binary churn classification for later projects.

## Current project state

Completed workflow stages:

1. `01_raw_data_audit`
2. `02_cleaning_and_splitting`
3. `03_training_set_eda`
4. `04_preprocessing_evaluation_and_simple_baselines`

Important data state:

- The clean modelling dataset has 7043 observations.
- The training set has 5634 observations.
- The held-out test set has 1409 observations.
- The target is `Churn_binary`.
- The positive class is churn: `Churn_binary = 1`.
- The target is moderately imbalanced: about 26.54% churn and 73.46% non-churn.
- The test set must remain unused until final evaluation.

Feature groups:

Numeric features:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Categorical features:

- `SeniorCitizen`
- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `PaperlessBilling`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaymentMethod`

Important EDA patterns:

- Churn is strongly associated with lower tenure.
- Churn is somewhat associated with higher monthly charges.
- Total charges are lower among churners, largely because churners tend to have shorter tenure.
- Strong categorical churn-rate differences occur for contract type, internet service type, payment method, online security, tech support, paperless billing, and senior-citizen status.
- These patterns are associations, not causal effects.

## High-level modelling philosophy

The project should use multiple model families, even if some are not ultimately best, because this is a reference project.

Each model section should answer:

1. What kind of model is this?
2. What mathematical function does it use?
3. What loss function or search criterion does it use?
4. What assumptions does it make?
5. What preprocessing does it need?
6. What does it predict: hard classes, scores, probabilities, or rankings?
7. How should its output be evaluated?
8. What does it teach us on this dataset?
9. What are its limitations?
10. How does it compare with earlier models?

## Model inventory by chronological slide order

### 1. Introduction: supervised learning, abstract tasks, and first examples

Source file:

- `11.Introduction.annotated.pdf`

Main concepts:

- Machine learning as learning from examples.
- Offline supervised learning.
- Instances, features, and targets.
- Classification as predicting a discrete label.
- Regression as predicting a numeric value.
- k-nearest neighbours and decision trees introduced as simple examples.
- Model capacity and overfitting intuition appear early.

Models or model families mentioned:

1. Linear models
2. k-nearest neighbours
3. Decision trees
4. Regression trees
5. kNN regression

Telco relevance:

- Classification framing is directly relevant.
- kNN is relevant for tabular binary classification.
- Decision trees are relevant for tabular binary classification.
- Regression tree and kNN regression are not target models here because churn is binary, but they help explain the classification/regression distinction.

Mathematics to explain later:

```math
D = \{(x_i, y_i)\}_{i=1}^n,
\qquad
x_i \in \mathcal{X},
\qquad
y_i \in \{0,1\}.
```

```math
h: \mathcal{X} \rightarrow \{0,1\}.
```

```math
\hat{p}(Y=1 \mid X=x) \in [0,1].
```

```math
\hat{y} =
\begin{cases}
1, & \hat{p}(Y=1 \mid X=x) \geq \tau,\\
0, & \hat{p}(Y=1 \mid X=x) < \tau.
\end{cases}
```

Project placement:

- Introductory report sections.
- Baseline and evaluation sections.
- kNN and tree sections later.

### 2. Linear Models 1: linear models, loss functions, gradient descent, and least-squares classification

Source file:

- `12.LinearModels1.annotated.pdf`

Main concepts:

- Linear functions.
- Model space versus feature space.
- Loss functions.
- Mean squared error.
- Gradient descent.
- Linear decision boundary.
- Classification with a linear hyperplane.
- Least-squares classification.

Models or model families:

1. Linear regression
2. Linear classifier
3. Least-squares classifier

Telco relevance:

- Linear regression is not a churn classifier, but its mathematics helps explain linear models and least-squares loss.
- A linear classifier is directly relevant as the foundation for logistic regression, linear SVM, and perceptron.
- Least-squares classification should be included as an illustrative learned linear classifier, not as a serious final model.

Mathematics to explain:

```math
f(x) = w^\top x + b.
```

```math
\hat{y} =
\begin{cases}
1, & w^\top x + b \geq 0,\\
0, & w^\top x + b < 0.
\end{cases}
```

```math
y_i^\star \in \{-1,+1\}.
```

```math
\min_{w,b}
\sum_{i=1}^{n}
\left(w^\top x_i + b - y_i^\star\right)^2.
```

```math
w^\top x + b = 0.
```

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t).
```

Important interpretation:

- Least-squares classification uses regression machinery for classification.
- It is smooth and easy to optimize.
- It may behave poorly because points far from the decision boundary can dominate the squared loss.
- It is useful pedagogically because it motivates better classification losses.

Project placement:

- After simple baselines, before logistic regression or inside the linear models section.
- Possible section name: `05_linear_classification_and_logistic_regression`.

### 3. Methodology 1: evaluation, train/validation/test discipline, overfitting, and metrics

Source file:

- `21.Methodology1.annotated.pdf`

Main concepts:

- Training error versus validation error.
- Generalization.
- Overfitting and underfitting.
- Hyperparameter choice.
- Confusion matrix.
- Precision and recall.
- True positive rate and false positive rate.
- ROC and AUC.
- PR curves.

Models or model families:

- Not primarily a model lecture.
- Essential for every modelling section.

Telco relevance:

- Directly relevant.
- The churn target is moderately imbalanced, so accuracy alone is incomplete.
- Evaluation must distinguish false positives and false negatives.

Mathematics to explain:

```math
TP = \#\{\hat{y}=1, y=1\},
\quad
FP = \#\{\hat{y}=1, y=0\},
```

```math
TN = \#\{\hat{y}=0, y=0\},
\quad
FN = \#\{\hat{y}=0, y=1\}.
```

```math
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}.
```

```math
\text{Precision} = \frac{TP}{TP + FP}.
```

```math
\text{Recall} = \text{TPR} = \frac{TP}{TP + FN}.
```

```math
\text{Specificity} = \frac{TN}{TN + FP}.
```

```math
\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}.
```

```math
F_1 =
2 \cdot
\frac{\text{Precision}\cdot \text{Recall}}
{\text{Precision}+\text{Recall}}.
```

```math
\text{Balanced Accuracy}
=
\frac{1}{2}
(\text{Recall}+\text{Specificity}).
```

Project placement:

- Before or inside `04_preprocessing_and_baselines`.
- Reused in every model section.

### 4. Methodology 2: preprocessing, outliers, class imbalance, and feature design

Source file:

- `22.Methodology2.annotated.pdf`

Main concepts:

- Missing values.
- Outliers.
- Natural versus corrupted extreme values.
- Class imbalance.
- Resampling training data.
- Validation and test sets should represent the production distribution.
- Feature design.

Models or model families:

- Not primarily a model lecture.
- Important for model-specific preprocessing.

Telco relevance:

- Directly relevant.
- We already used the missing-values and outlier logic in raw audit and EDA.
- Later, class imbalance handling may include class weights or resampling, but only inside training folds.

Mathematics and methodology to explain:

```math
\hat{p}(Y=1) = \frac{1}{n}\sum_{i=1}^{n} y_i.
```

```math
\hat{p}(Y=1) \approx 0.2654.
```

Resampling principle:

- Validation and test data should keep the natural class distribution.
- Training folds may be resampled, but resampling must happen inside the training part of each validation split.

Project placement:

- `04_preprocessing_and_baselines`.
- Later class imbalance experiment.
- Possibly a dedicated section: `class_imbalance_and_resampling`.

### 5. Probabilistic Models 1: maximum likelihood, Bayes classifiers, Naive Bayes, and logistic regression

Source file:

- `31.ProbabilisticModels1.annotated.pdf`

Main concepts:

- Likelihood and log-likelihood.
- Negative log-likelihood as a loss.
- Probabilistic classifiers.
- Discriminative versus generative classifiers.
- Bayes classifier.
- Naive Bayes.
- Laplace smoothing.
- Continuous Naive Bayes.
- Logistic regression.
- Sigmoid.
- Log loss / cross-entropy.

Models or model families:

1. Bayes optimal classifier, conceptual only
2. Bayes classifier
3. Naive Bayes classifier
4. Gaussian Naive Bayes
5. Bernoulli Naive Bayes or categorical Naive Bayes
6. Logistic regression
7. Softmax regression as multiclass extension

Telco relevance:

- Bayes optimal classifier is conceptual, not implemented.
- Bayes classifier can be explained conceptually.
- Naive Bayes variants can be implemented.
- Logistic regression is a core model and should receive a deep section.
- Softmax regression is mainly context, because Telco churn is binary.

Mathematics to explain:

```math
\hat{\theta}
=
\arg\max_{\theta}
p(D \mid \theta).
```

```math
\hat{\theta}
=
\arg\max_{\theta}
\sum_{i=1}^{n}
\log p(x_i \mid \theta).
```

```math
L(\theta)
=
-\sum_{i=1}^{n}
\log p(x_i \mid \theta).
```

```math
p(y \mid x)
=
\frac{p(x \mid y)p(y)}
{\sum_{c} p(x \mid c)p(c)}.
```

```math
p(x \mid y)
=
\prod_{j=1}^{p}
p(x_j \mid y).
```

```math
p(y \mid x)
=
\frac{p(y)\prod_{j=1}^{p}p(x_j \mid y)}
{\sum_{c}p(c)\prod_{j=1}^{p}p(x_j \mid c)}.
```

```math
x_j \mid Y=c
\sim
\mathcal{N}(\mu_{jc}, \sigma_{jc}^2).
```

```math
p(x_j \mid Y=c)
=
\frac{1}{\sqrt{2\pi\sigma_{jc}^2}}
\exp\left(
-\frac{(x_j-\mu_{jc})^2}{2\sigma_{jc}^2}
\right).
```

```math
p(x_j \mid Y=c)
=
\theta_{jc}^{x_j}(1-\theta_{jc})^{1-x_j},
\qquad
x_j \in \{0,1\}.
```

```math
\hat{p}(X_j=a \mid Y=c)
=
\frac{N_{j,a,c}+\alpha}
{N_c+\alpha V_j}.
```

```math
z_i = w^\top x_i + b.
```

```math
\sigma(z)
=
\frac{1}{1+\exp(-z)}.
```

```math
\hat{p}_i
=
p(Y_i=1 \mid x_i)
=
\sigma(w^\top x_i + b).
```

```math
L(w,b)
=
-\sum_{i=1}^{n}
\left[
y_i \log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
```

```math
L_{\text{L2}}(w,b)
=
L(w,b)
+
\lambda \lVert w \rVert_2^2.
```

```math
L_{\text{L1}}(w,b)
=
L(w,b)
+
\lambda \lVert w \rVert_1.
```

Project placement:

- Naive Bayes likely after logistic regression or before it, depending on chronological strictness.
- Logistic regression should receive a major section.
- Regularization can be inside logistic regression section or a follow-up subsection.

### 6. Linear Models 2: perceptron, feedforward networks, backpropagation, SVMs, hinge loss, and kernels

Source file:

- `32.LinearModels2.annotated.pdf`

Main concepts:

- Perceptron.
- Nonlinear activation functions.
- Feedforward neural networks.
- Backpropagation.
- Maximum margin classifier.
- Soft-margin SVM.
- Hinge loss.
- Lagrange optimization.
- Kernel trick.
- RBF kernel.

Models or model families:

1. Perceptron
2. Feedforward neural network
3. Multilayer perceptron
4. Linear SVM
5. Soft-margin SVM
6. Kernel SVM
7. RBF-kernel SVM

Telco relevance:

- Perceptron is historically and conceptually relevant.
- MLP is relevant for a tabular neural-network comparison.
- Linear SVM is relevant.
- RBF-kernel SVM is relevant.
- Deep networks beyond simple MLP are not central for this tabular dataset.

Mathematics to explain:

```math
f(x) = w^\top x + b.
```

```math
\hat{y} = \operatorname{sign}(w^\top x + b).
```

Perceptron mistake update:

```math
w \leftarrow w + \eta y_i x_i,
\qquad
b \leftarrow b + \eta y_i,
```

when

```math
y_i(w^\top x_i + b) \leq 0.
```

Feedforward neural network:

```math
a^{(0)} = x.
```

```math
z^{(\ell)} = W^{(\ell)}a^{(\ell-1)} + b^{(\ell)}.
```

```math
a^{(\ell)} = g(z^{(\ell)}).
```

```math
\hat{p}
=
\sigma(z^{(L)}).
```

Hard-margin SVM:

```math
\min_{w,b}
\frac{1}{2}\lVert w\rVert_2^2
\quad
\text{subject to}
\quad
y_i(w^\top x_i+b) \geq 1.
```

Soft-margin / hinge-loss SVM:

```math
\min_{w,b}
\frac{1}{2}\lVert w\rVert_2^2
+
C\sum_{i=1}^{n}
\max(0, 1-y_i(w^\top x_i+b)).
```

Kernel decision function:

```math
f(x)
=
\sum_{i=1}^{n}
\alpha_i y_i K(x_i,x) + b.
```

RBF kernel:

```math
K(x,z)
=
\exp(-\gamma \lVert x-z\rVert_2^2).
```

Project placement:

- Perceptron can be included in the linear classification section or as a historical conceptual baseline.
- SVM should have its own section.
- MLP can come later after classical models.

### 7. Deep Learning 1 and Deep Learning 2: neural-network training, output distributions, CNNs, GANs

Source files:

- `41.DeepLearning1.annotated.pdf`
- `51.Deep Learning2.annotated.pdf`, if present in project sources

Main concepts:

- Deep networks.
- Backpropagation.
- Loss functions as negative log-likelihoods.
- Bernoulli output with binary cross-entropy.
- Categorical output with categorical cross-entropy.
- Normal output with MSE.
- CNNs.
- GANs.

Models or model families:

1. Deep feedforward neural networks
2. CNNs
3. Generative models / GANs

Telco relevance:

- MLP / feedforward neural network is relevant.
- CNNs are not natural for this tabular churn dataset.
- GANs are not central for this supervised tabular classification project.
- Deep-learning probability-output ideas are useful for explaining why binary classification often uses sigmoid plus binary cross-entropy.

Mathematics to explain for Telco:

```math
Y_i \mid x_i \sim \text{Bernoulli}(\hat{p}_i).
```

```math
\hat{p}_i = f_\theta(x_i).
```

```math
-\log p(y_i \mid x_i, \theta)
=
-\left[
y_i \log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
```

Project placement:

- MLP near the end of the tabular modelling sequence.
- CNNs and GANs saved for future image or generative projects.

### 8. Trees and ensembles: decision trees, stumps, bagging, random forests, boosting, AdaBoost, and gradient boosting

Source file:

- `52.Trees.annotated.pdf`

Main concepts:

- Decision trees.
- Recursive partitioning.
- Tree depth and overfitting.
- Decision stumps.
- Bagging.
- Random forests.
- Boosting.
- AdaBoost.
- Gradient boosting.

Models or model families:

1. Decision stump
2. Decision tree
3. Bagging ensemble
4. Random forest
5. Boosting
6. AdaBoost
7. Gradient boosting

Telco relevance:

- Highly relevant for tabular classification.
- Decision trees are interpretable and handle nonlinear interactions.
- Random forests and gradient boosting are strong tabular baselines.
- Decision stumps are useful as weak learners and simple interpretable models.

Mathematics to explain:

A tree partitions the feature space into regions:

```math
\mathcal{R}_1,\ldots,\mathcal{R}_M.
```

Leaf probability:

```math
\hat{p}_m
=
\frac{1}{|\mathcal{R}_m|}
\sum_{i:x_i\in \mathcal{R}_m}
y_i.
```

Class prediction:

```math
\hat{y}
=
\mathbb{1}\{\hat{p}_m \geq \tau\}.
```

Gini impurity:

```math
G(m)
=
1-\sum_{c=1}^{K} p_{mc}^2.
```

Entropy:

```math
H(m)
=
-\sum_{c=1}^{K}p_{mc}\log p_{mc}.
```

Impurity reduction:

```math
\Delta I
=
I(parent)
-
\frac{n_L}{n}I(left)
-
\frac{n_R}{n}I(right).
```

Bagging probability average:

```math
\hat{p}(Y=1\mid x)
=
\frac{1}{B}
\sum_{b=1}^{B}
\hat{p}_b(Y=1\mid x).
```

AdaBoost ensemble:

```math
F_T(x)
=
\sum_{t=1}^{T}
\alpha_t h_t(x).
```

```math
\hat{y}
=
\operatorname{sign}(F_T(x)).
```

Weighted error:

```math
\epsilon_t
=
\frac{\sum_i w_i^{(t)}\mathbb{1}\{h_t(x_i)\neq y_i\}}
{\sum_i w_i^{(t)}}.
```

Learner weight:

```math
\alpha_t
=
\frac{1}{2}
\log
\left(
\frac{1-\epsilon_t}{\epsilon_t}
\right).
```

Gradient boosting pseudo-residuals:

```math
r_i^{(m)}
=
-\left[
\frac{\partial L(y_i, F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}.
```

Gradient boosting update:

```math
F_m(x)
=
F_{m-1}(x)
+
\eta h_m(x).
```

Project placement:

- Decision stumps and decision trees after kNN or after linear/probabilistic models.
- Bagging, random forest, AdaBoost, and gradient boosting after decision trees.
- Gradient boosting likely one of the strongest practical models for this dataset.

### 9. Matrices: matrix factorization and recommender systems

Source file:

- `62.Matrices.annotated.pdf`

Main concepts:

- Matrix factorization.
- Embeddings.
- Alternating least squares.
- Gradient descent for recommendation models.

Models or model families:

1. Matrix factorization
2. Alternating least squares
3. Embedding-based recommender model

Telco relevance:

- Not central for this tabular churn classification project.
- Useful for future recommender-system projects.
- Not part of the main modelling roadmap here.

Mathematics to know but not use now:

```math
R \approx U^\top M.
```

```math
L(U,M)
=
\sum_{(i,j)\in \Omega}
(R_{ij}-u_i^\top m_j)^2.
```

Project placement:

- Not in this Telco project except maybe a future-project note.

### 10. Sequential models and Transformers

Source files:

- `61.SequentialModels.annotated.pdf`
- `Transformers.annotated.pdf`

Main concepts:

- Sequence modelling.
- RNNs.
- LSTMs or gated recurrent models if covered.
- Attention.
- Self-attention.
- Transformers.

Models or model families:

1. RNN
2. LSTM / gated RNN
3. Sequence-to-sequence model
4. Attention model
5. Transformer

Telco relevance:

- Not natural for the current static tabular churn dataset.
- Relevant for future text, time-series, or event-sequence projects.
- Could become relevant if we had customer activity sequences over time, but this dataset is not structured that way.

Project placement:

- Exclude from main Telco modelling.
- Mention in roadmap as future projects.

### 11. Reinforcement learning

Source file:

- `71.Reinforcement Learning.annotated.pdf`

Main concepts:

- Agent.
- Environment.
- State.
- Action.
- Reward.
- Policy.
- Value function.
- Exploration versus exploitation.

Telco relevance:

- Not a supervised classification model.
- Not suitable for the current static churn prediction setup.
- Could be relevant in a future decision-making project, for example optimizing retention actions over time.

Project placement:

- Exclude from main Telco modelling.
- Mention as future project type.

## Recommended project section order

This order balances chronological course order with simple-to-complex applied modelling.

### 04. Preprocessing and simple baselines

Purpose:

- Establish reusable modelling infrastructure.
- Use training data only.
- Define feature preprocessing.
- Define validation strategy.
- Define evaluation metrics.
- Fit simple baselines.

Models:

1. Majority-class baseline
2. Prior-probability baseline
3. Stratified random baseline
4. Uniform random baseline
5. EDA-inspired rule baseline

Mathematics:

```math
\hat{y}
=
\arg\max_{c\in\{0,1\}}
\sum_{i=1}^{n}\mathbb{1}\{y_i=c\}.
```

```math
\hat{p}(Y=c)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb{1}\{y_i=c\}.
```

```math
\hat{Y}
\sim
\operatorname{Categorical}
(\hat{p}(Y=0),\hat{p}(Y=1)).
```

```math
\hat{Y}
\sim
\operatorname{Categorical}(0.5,0.5).
```

```math
s(x)
=
\sum_{j=1}^{m}
\mathbb{1}\{\text{high-risk condition }j\text{ is true}\}.
```

```math
\hat{y}
=
\mathbb{1}\{s(x)\geq t\}.
```

Important:

- Logistic regression should not be described as a simple baseline.
- It can be introduced as the first learned reference model, or moved to section 05.
- Preferred decision: section 04 includes only simple baselines and infrastructure.

### 05. Linear classification and logistic regression

Models:

1. Linear classifier
2. Perceptron, maybe conceptual or implemented
3. Least-squares classifier
4. Logistic regression
5. Regularized logistic regression
6. Softmax regression, brief multiclass context only

Main learning goals:

- Linear decision boundary.
- Score versus probability.
- Loss functions for classification.
- Why least squares is not ideal for classification.
- Why sigmoid plus log loss is better.
- Regularization and coefficient interpretation.

### 06. k-nearest neighbours

Models:

1. kNN classifier

Main learning goals:

- Lazy learning.
- Distance metrics.
- Role of `k`.
- Bias-variance tradeoff.
- Importance of scaling.
- Problems with high-dimensional one-hot encoded spaces.

### 07. Probabilistic classifiers and Naive Bayes

Models:

1. Bayes classifier, conceptual
2. Naive Bayes
3. Gaussian Naive Bayes
4. Bernoulli Naive Bayes
5. Categorical or Multinomial Naive Bayes, if useful

Main learning goals:

- Generative versus discriminative classification.
- Conditional independence.
- Likelihood per class.
- Priors.
- Smoothing.
- Why Naive Bayes can work despite unrealistic assumptions.

### 08. Decision trees

Models:

1. Decision stump
2. Decision tree

Main learning goals:

- Recursive partitioning.
- Impurity criteria.
- Interpretability.
- Tree depth and overfitting.
- Interactions and nonlinear structure.
- Minimal preprocessing compared with distance-based models.

### 09. Ensembles

Models:

1. Bagging
2. Random forest
3. Boosting
4. AdaBoost
5. Gradient boosting

Main learning goals:

- Variance reduction.
- Bias reduction.
- Bootstrap sampling.
- Feature subsampling.
- Sequential correction of mistakes.
- Pseudo-residuals and loss optimization.
- Why tree ensembles are strong for tabular data.

### 10. Support vector machines

Models:

1. Linear SVM
2. Soft-margin SVM
3. Kernel SVM
4. RBF-kernel SVM

Main learning goals:

- Margins.
- Hinge loss.
- Regularization.
- Support vectors.
- Kernel trick.
- RBF similarity.
- Scaling and hyperparameter sensitivity.

### 11. Neural networks for tabular classification

Models:

1. Feedforward neural network
2. MLP classifier

Main learning goals:

- Layers and nonlinear transformations.
- Learned representations.
- Backpropagation.
- Binary cross-entropy.
- Regularization.
- Why MLPs are flexible but not automatically best for small tabular data.

### 12. Model comparison, threshold tuning, and calibration

Main concepts:

1. Cross-validated comparison.
2. Confusion matrix comparison.
3. ROC-AUC.
4. PR-AUC.
5. Threshold tuning.
6. Calibration curves.
7. Probability calibration if needed.
8. Final model selection without test-set leakage.

Mathematics:

```math
\hat{y}_\tau
=
\mathbb{1}\{\hat{p}(Y=1\mid x)\geq \tau\}.
```

```math
\text{Expected Cost}(\tau)
=
C_{FP}FP(\tau)
+
C_{FN}FN(\tau).
```

```math
P(Y=1 \mid \hat{p}=q) \approx q.
```

### 13. Final test evaluation

Purpose:

- Use the held-out test set once.
- Evaluate the final chosen model and threshold.
- Report uncertainty and limitations.
- Do not tune after seeing test performance.

## Models suitable for this Telco project

Use directly:

- Majority-class baseline
- Prior-probability baseline
- Stratified random baseline
- Uniform random baseline
- Rule-based baseline
- Linear classifier
- Perceptron, at least conceptually
- Least-squares classifier
- Logistic regression
- Regularized logistic regression
- kNN
- Naive Bayes variants
- Decision stump
- Decision tree
- Bagging
- Random forest
- AdaBoost
- Gradient boosting
- Linear SVM
- RBF-kernel SVM
- MLP

Mention but do not make central:

- Softmax regression, because the target is binary
- Bayes optimal classifier, because it is conceptual
- Full Bayes classifier with multivariate covariance, because it is less practical for mixed high-dimensional one-hot data

Save for later projects:

- CNN
- RNN
- LSTM
- Transformer
- GAN
- Reinforcement learning
- Matrix factorization

## Model-specific preprocessing expectations

### Models that need or strongly benefit from numeric scaling

- Logistic regression with regularization
- Least-squares classifier
- Perceptron
- kNN
- Linear SVM
- RBF SVM
- MLP

Reason:

- Coefficients, distances, margins, gradients, or regularization penalties are scale-sensitive.

### Models that usually do not require numeric scaling

- Decision tree
- Decision stump
- Random forest
- Bagging of trees
- AdaBoost with tree stumps
- Gradient boosting trees

Reason:

- Tree splits depend on feature ordering, not Euclidean geometry or coefficient penalties.

### Models needing special feature representation

Naive Bayes:

- Gaussian NB: numeric features can be modelled as continuous Gaussian variables.
- Bernoulli NB: needs binary features.
- Categorical NB: needs integer-coded categorical variables with appropriate encoding.
- Multinomial NB: mostly natural for count data, less natural for Telco unless features are constructed carefully.

kNN and SVM:

- One-hot encoding can create high-dimensional sparse features.
- Scaling numeric variables is necessary.
- Distance and kernel behaviour should be interpreted carefully.

## Next immediate action

Before coding more models:

1. Review this inventory.
2. Decide whether section 04 should include only simple baselines or also a first learned reference model.
3. Preferred decision: section 04 includes only simple baselines and infrastructure.
4. Start section 05 with learned linear classifiers, including least-squares classification and logistic regression.
5. Build the report so each model section contains both mathematics and practical results.

## Current recommended next file

Create:

```text
docs/model_inventory_and_roadmap.md
```

Then continue with:

```text
notebooks/04_preprocessing_and_simple_baselines.py
```

instead of the earlier broader name:

```text
notebooks/04_preprocessing_and_baselines.py
```

Potential reason for renaming:

- It makes clear that section 04 is about preprocessing and simple baselines, not all learned baseline models.
- Logistic regression then naturally starts section 05.
