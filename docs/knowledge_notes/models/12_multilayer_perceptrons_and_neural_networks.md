# Multilayer Perceptrons and Feed-Forward Neural Networks

## Purpose and project placement

This note is the reusable model-family reference for multilayer perceptrons (MLPs) and fully connected feed-forward neural networks in the Telco Customer Churn classification project. It is written before the MLP notebook so that the architecture, loss, preprocessing, optimization, regularization, probability interpretation, and evaluation design are understood before experiments are run.

The project studies MLPs as a distinct modelling family, not because a neural network is assumed to outperform every earlier model. The dataset is a relatively small mixed tabular classification problem with 5,634 development observations after the held-out test split. Earlier development-stage work found boosted-tree methods to form a strong group. The MLP workflow asks a narrower question:

> After fold-safe scaling and one-hot encoding, does a learned nonlinear representation add useful signal beyond the already established linear and tree-based candidates?

The held-out test set remains completely untouched during this workflow. Architecture choices, optimization choices, regularization, class weighting, calibration diagnostics, and threshold diagnostics must be based on the training partition only.

```text
Allowed in this workflow:
    training-only cross-validation;
    preprocessing fitted separately inside every training fold;
    transparent, limited candidate grids;
    validation and pooled out-of-fold diagnostics;
    learning-curve and convergence diagnostics;
    training-only threshold and calibration diagnostics.

Not allowed:
    checking the held-out test set;
    choosing width, depth, alpha, early stopping, or a threshold from test results;
    treating close cross-validation scores as proof that one architecture is superior;
    presenting an MLP as the final production model.
```

The scope is deliberately limited to tabular, supervised binary classification with feed-forward networks. Convolutional networks, recurrent networks, attention mechanisms, transformers, self-supervised pretraining, and large language models use related optimization ideas but impose different architectural assumptions and are not part of this workflow.

---

## 1. The central idea: learn the feature transformation

A regularized logistic-regression model maps an input vector x directly to a linear score:

$$
z = w^T x + b,
$$

then converts that score to a churn probability with the sigmoid function:

$$
p = σ(z) = 1 / (1 + exp(-z)).
$$

The model can express a linear boundary in the transformed feature space supplied to it. One-hot encoding expands categorical variables into indicators, and manually designed interaction or nonlinear features can further expand that space. However, logistic regression does not learn its own hidden representation of the inputs.

An MLP adds one or more hidden layers. Each hidden layer computes a learned affine transformation followed by a nonlinearity. The hidden units can therefore act as learned features. A final linear output layer combines those learned features into a binary logit.

For one hidden layer, the model is:

$$
z^(1) = W^(1) x + b^(1),
$$

$$
h = g(z^(1)),
$$

$$
z^(2) = v^T h + c,
$$

$$
p = σ(z^(2)).
$$

Here:

```text
x:
    input feature vector

W^(1), b^(1):
    weights and biases of the hidden layer

g:
    hidden-layer activation function

h:
    learned hidden representation

v, c:
    output-layer weights and bias

z^(2):
    output logit, also called the raw score

p:
    predicted probability of the positive class
```

The key difference from manual feature engineering is that the representation h is learned jointly with the output classifier. The parameters of both layers are adjusted to improve the same predictive objective.

---

## 2. Why hidden nonlinearities are necessary

A stack of affine transformations without activation functions is still only an affine transformation. Suppose a model has two layers:

$$
a = W^(1) x + b^(1),
$$

$$
z = W^(2) a + b^(2).
$$

Substituting the first equation into the second gives:

$$
z = W^(2)(W^(1)x + b^(1)) + b^(2)
  = (W^(2)W^(1))x + (W^(2)b^(1) + b^(2)).
$$

This has exactly the same form as one linear model. Adding more linear layers changes parameterization but does not create a richer class of functions.

A nonlinear activation function breaks that collapse:

$$
h = g(W^(1)x + b^(1)).
$$

The composition

$$
W^(2) g(W^(1)x + b^(1)) + b^(2)
$$

is generally not reducible to a single affine transformation. This is why MLPs can learn curved, interacting, and piecewise nonlinear relations from tabular features.

A neural network is not nonlinear merely because it has many layers. It is nonlinear because at least one non-affine activation function is placed between learned affine transformations.

---

## 3. Feed-forward architecture and notation

### 3.1 Layers and acyclic information flow

A feed-forward network is a directed acyclic graph. Inputs flow from the feature layer through one or more hidden layers to the output layer. There are no loops and no feedback from later layers to earlier layers during the forward computation.

For a network with L learned layers:

$$
A^(0) = X,
$$

$$
Z^(l) = A^(l-1) W^(l) + 1 (b^(l))^T,
$$

$$
A^(l) = g^(l)(Z^(l)),  for l = 1, ..., L - 1.
$$

For binary classification, the final layer typically produces one logit:

$$
z = A^(L-1) w^(L) + b^(L),
$$

$$
p = σ(z).
$$

The notation uses a mini-batch representation:

```text
X:
    matrix with B rows, one row per observation in the current mini-batch

A^(l):
    activations emitted by layer l

Z^(l):
    pre-activation values, before applying g^(l)

W^(l):
    matrix of weights connecting layer l - 1 to layer l

b^(l):
    bias vector for layer l

B:
    mini-batch size
```

Using matrices instead of scalar loops matters in practice. Dense matrix multiplication is efficient on modern CPUs and especially GPUs. More importantly, it gives one consistent representation for the forward pass, loss calculation, and backpropagation.

### 3.2 Fully connected layers

In a fully connected, or dense, layer, every input unit connects to every output unit. If the previous layer has m units and the new layer has q units, the layer contains:

$$
m q + q
$$

parameters: m q weights plus q biases.

For an input with d transformed columns, one hidden layer with h units, and one scalar output, the total number of trainable parameters is:

$$
(dh + h) + (h + 1) = dh + 2h + 1.
$$

After one-hot encoding, the transformed feature count d can be appreciably larger than the number of raw columns. Parameter count therefore increases directly with hidden-layer width.

### 3.3 Width, depth, and capacity

Width is the number of units in a hidden layer. Depth is the number of hidden layers. Increasing either usually increases representational capacity, but it also increases the number of parameters, the flexibility of the fitted function, optimization difficulty, and overfitting risk.

For this project, the first MLP comparison should distinguish between:

```text
shallow networks:
    one hidden layer, such as (16,), (32,), or (64,)

moderately deep networks:
    two hidden layers, such as (32, 16) or (64, 32)
```

The purpose is not to search every possible architecture. It is to test whether a small learned nonlinear representation is helpful on this specific tabular problem and whether additional depth appears justified by training-only evidence.

---

## 4. Activation functions

### 4.1 Hidden-layer activations

The activation function is applied elementwise to a layer's pre-activation values. Common choices are:

| Activation | Definition | Main role and limitation |
|---|---|---|
| Identity | g(a) = a | Leaves the model linear when used everywhere |
| Sigmoid | g(a) = 1 / (1 + exp(-a)) | Bounded and probabilistically interpretable, but can saturate |
| Hyperbolic tangent | g(a) = tanh(a) | Zero-centered bounded activation, but can also saturate |
| ReLU | g(a) = max(0, a) | Simple piecewise-linear default with efficient gradients for positive values |
| Leaky ReLU | g(a) = max(αa, a), α > 0 small | Reduces the risk that a unit remains inactive for all observed inputs |

The standard initial hidden-layer choice for tabular MLPs is usually ReLU. It is computationally simple and avoids the severe positive-and-negative saturation behavior of sigmoid or tanh over much of its active region.

For ReLU:

$$
g(a) = max(0, a),
$$

and away from the point a = 0,

$$
g'(a) =
\begin{cases}
1, & a > 0, \\
0, & a < 0.
\end{cases}
$$

A ReLU unit with consistently negative pre-activations emits zero and has zero local derivative for those observations. This is often called a dying ReLU. It is a practical warning, not a reason to avoid ReLU by default. Careful initialization, reasonable learning rates, and inspection of training behavior address the issue in most small tabular experiments. A leaky activation is a reasonable contingency if an implementation supports it and evidence suggests widespread inactive units.

### 4.2 Sigmoid for the binary output

The final output is different from a hidden layer. For binary classification, the model should output a probability in [0, 1]. The sigmoid function maps any real-valued logit z into that interval:

$$
σ(z) = 1 / (1 + exp(-z)).
$$

It is useful to distinguish:

```text
logit:
    an unrestricted real-valued score z

probability:
    σ(z), a number between 0 and 1

hard prediction:
    a class assigned after comparing the probability or logit with a chosen threshold
```

The sigmoid function is appropriate for the final binary output because it defines a Bernoulli probability model. It is not normally the preferred default hidden activation for a modern MLP because its derivative becomes small in strongly positive or negative regions:

$$
σ'(z) = σ(z)(1 - σ(z)).
$$

Small derivatives can slow optimization through many layers. This is one historical contributor to vanishing-gradient problems.

### 4.3 Why activation choice is a modelling choice

Activation functions contribute to the model's inductive bias. ReLU networks are piecewise linear in the input space, but their composition can create a large number of regions with different linear behavior. Tanh and sigmoid layers create smooth bounded transformations. The activation should not be viewed as a cosmetic hyperparameter. It determines which functions are easy or difficult for the network to represent and optimize.

For the initial Telco workflow:

```text
primary hidden activation:
    ReLU

one controlled alternative, if useful:
    tanh

output activation:
    sigmoid through the binary-loss formulation
```

A large activation sweep would add multiple-testing noise without answering a proportionately important modelling question.

---

## 5. Binary probabilistic model, logits, and cross-entropy

### 5.1 Bernoulli likelihood

Let y_i be the churn label for observation i, where y_i is 1 for churn and 0 otherwise. Let p_i be the MLP's estimated probability of churn. The Bernoulli likelihood for one observation is:

$$
P(y_i | x_i) = p_i^(y_i) (1 - p_i)^(1 - y_i).
$$

Assuming conditional independence of observations given the model parameters, the likelihood over a development sample is the product of those terms. Taking a negative average log-likelihood gives binary cross-entropy:

$$
L_BCE = - (1 / n) Σ_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)].
$$

Minimizing binary cross-entropy is equivalent to maximum-likelihood estimation for the Bernoulli model, subject to the chosen network architecture and regularization.

### 5.2 Why logits are numerically preferable

Computing a sigmoid probability and then separately taking logarithms can be numerically unstable for very large positive or negative logits. A practical implementation should use a stable combined binary-cross-entropy-with-logits routine when it is available.

Conceptually, this is still the same model:

$$
p_i = σ(z_i).
$$

The numerical implementation simply evaluates the loss from z_i in a way that avoids overflow or underflow. In PyTorch, this is the purpose of `BCEWithLogitsLoss`. In scikit-learn's `MLPClassifier`, the corresponding output and loss handling are internal to the estimator.

### 5.3 A useful gradient identity

For one observation, with sigmoid output and binary cross-entropy loss, differentiation with respect to the output logit gives:

$$
∂L / ∂z = p - y.
$$

This is an important simplification. The sigmoid derivative and the cross-entropy derivative combine cleanly, leaving predicted probability minus observed label.

Interpretation:

```text
p close to 1 when y = 0:
    large positive gradient signal, pushing the logit downward

p close to 0 when y = 1:
    large negative gradient signal, pushing the logit upward

p close to y:
    small gradient signal
```

This does not imply that every parameter receives the same update. Backpropagation distributes this output error through the hidden layers according to the chain rule.

### 5.4 Probability estimates are not automatically calibrated

A sigmoid output has the mathematical form of a probability, but this alone does not guarantee empirical calibration. A well-calibrated model would satisfy an approximate statement such as:

> Among cases assigned a churn probability around 0.30, roughly 30 percent should actually churn.

Regularization, class imbalance handling, early stopping, architecture capacity, and finite-sample variation can all affect calibration. Ranking quality and calibration are therefore separate questions:

```text
PR-AUC and ROC-AUC:
    assess ordering or ranking quality

calibration:
    assesses whether numerical probabilities correspond to observed event frequencies

decision threshold:
    converts a score or probability into an action
```

Calibration and final threshold selection remain later training-only finalist-stage questions unless the MLP emerges as a serious candidate.

---

## 6. Class imbalance and weighted binary cross-entropy

The positive churn class represents about 26.54 percent of the training data. This is not an extreme rarity, but it is imbalanced enough that accuracy alone is not an adequate primary metric.

A weighted binary cross-entropy can assign a larger penalty to errors on the positive class:

$$
L_weighted = - (1 / n) Σ_i [
w_+ y_i log(p_i) +
w_- (1 - y_i) log(1 - p_i)
].
$$

The weights change the training objective. They do not merely alter how the fitted model is reported.

Expected practical effect:

```text
larger positive-class weight:
    tends to make the model more responsive to churn observations;
    may improve recall at a default boundary;
    may reduce precision and specificity;
    may change raw probability calibration.
```

In an idealized population setting, a weighted probability target q can be related to an unweighted probability p through:

$$
p = [w_- q] / [w_+ (1 - q) + w_- q].
$$

This identity is useful for understanding why class weighting can alter probability interpretation. It should not be treated as a substitute for empirical calibration assessment. Real finite-sample models, early stopping, regularization, and distribution shift complicate the picture.

The initial MLP workflow should first establish an unweighted baseline. A class-weighted or sample-weighted variant can be added only as a clearly labeled training-only comparison. The local scikit-learn version must be checked before relying on sample-weight support in `MLPClassifier`.

---

## 7. Backpropagation

### 7.1 The learning problem

The network parameters are collectively denoted by θ. Training means finding parameter values that minimize a regularized loss:

$$
J(θ) = L_BCE(θ) + R(θ).
$$

For L2 regularization:

$$
R(θ) = (λ / 2) Σ_l ||W^(l)||_F^2,
$$

where ||.||_F is the Frobenius norm of a weight matrix. Biases are often excluded from the regularization term.

The loss depends on every layer. Changing one early-layer weight changes hidden activations, later-layer logits, probabilities, and ultimately the loss. Backpropagation calculates the derivative of the loss with respect to each parameter efficiently by reusing intermediate derivatives.

### 7.2 Chain rule view

If a scalar loss L depends on an intermediate value u, which depends on v, the chain rule is:

$$
dL / dv = (dL / du)(du / dv).
$$

Backpropagation applies this systematically from the output back toward the input. Each layer receives an upstream gradient, multiplies it by its local derivative, and passes a resulting gradient to the preceding layer.

The forward pass computes and stores:

```text
pre-activations Z^(l)
activations A^(l)
final logits z
probabilities p
loss value
```

The backward pass uses those stored values to compute gradients.

### 7.3 Matrix form for a dense network

For a batch, define the final output error:

$$
Δ^(L) = p - y.
$$

For the final output weights:

$$
∇_(w^(L)) J = (A^(L-1))^T Δ^(L) / B + λ w^(L),
$$

$$
∇_(b^(L)) J = mean(Δ^(L)).
$$

For a hidden layer l, the error is propagated backward as:

$$
Δ^(l) = [Δ^(l+1) (W^(l+1))^T] ⊙ g'^(l)(Z^(l)),
$$

where ⊙ means elementwise multiplication.

The gradients for that hidden layer are:

$$
∇_(W^(l)) J = (A^(l-1))^T Δ^(l) / B + λ W^(l),
$$

$$
∇_(b^(l)) J = mean_rows(Δ^(l)).
$$

This matrix formulation is the scalable version of differentiating one weight at a time. Automatic-differentiation frameworks compute equivalent gradients from a computation graph, but understanding these equations prevents the process from becoming a black box.

### 7.4 Vanishing and exploding gradients

Backpropagation repeatedly multiplies gradients by weight matrices and activation derivatives. If these factors are often smaller than one in magnitude, gradients can shrink toward zero as they reach early layers. If they are too large, gradients can grow unstably.

This motivates:

```text
activation choices that preserve usable derivatives;
initialization schemes matched to the activation;
moderate depth for small tabular datasets;
adaptive optimization;
reasonable learning rates;
regularization and monitoring of learning curves.
```

A small MLP with one or two hidden layers is unlikely to encounter the same severity of gradient problems as a very deep network, but the underlying mechanisms still matter.

---

## 8. Optimization

### 8.1 Batch gradient descent, stochastic gradient descent, and mini-batches

A full-batch gradient update uses all training observations to estimate each gradient:

$$
θ_(t+1) = θ_t - η ∇J(θ_t).
$$

This can be stable but expensive on large datasets. Stochastic gradient descent uses one observation at a time. It is cheap per update but noisy.

Mini-batch training is the standard compromise. Each update uses a batch of B observations:

$$
θ_(t+1) = θ_t - η g_t,
$$

where g_t is the gradient estimated from the current mini-batch.

Terminology:

```text
batch size:
    number of observations in one parameter update

iteration:
    one mini-batch update

epoch:
    one complete pass through the training data

learning rate η:
    step size used by the optimizer
```

Mini-batch noise can help optimization avoid overly rigid deterministic trajectories, but it also makes results sensitive to random initialization, batch ordering, and chosen seed.

### 8.2 Learning rate

The learning rate is usually the most consequential optimization hyperparameter.

```text
too large:
    loss may oscillate, diverge, or produce unstable validation behavior

too small:
    learning is slow and may stop before reaching a useful region

well chosen:
    training loss descends meaningfully while validation performance remains stable
```

The appropriate learning rate depends on input scaling, activation, optimizer, batch size, initialization, architecture, and regularization. It should be searched within a limited, theory-informed range rather than tuned endlessly.

### 8.3 Momentum

Classical momentum maintains a velocity vector:

$$
v_t = μ v_(t-1) + g_t,
$$

$$
θ_(t+1) = θ_t - η v_t.
$$

The velocity averages gradients across iterations. This can accelerate movement along persistent descent directions and reduce oscillation across narrow directions in the loss landscape.

Momentum is useful to understand because it explains the idea of optimizer state. It need not be the main optimizer in the initial Telco MLP grid.

### 8.4 Adam and AdamW

Adam keeps exponentially decaying estimates of first and second gradient moments:

$$
m_t = β_1 m_(t-1) + (1 - β_1) g_t,
$$

$$
v_t = β_2 v_(t-1) + (1 - β_2) g_t^2.
$$

After bias correction, the update is approximately:

$$
θ_(t+1) = θ_t - η m_hat_t / (sqrt(v_hat_t) + ε).
$$

Adam adapts the effective step size separately for parameters based on their recent gradient scale. It is a practical default for many small neural-network experiments.

Weight decay deserves careful terminology. In plain gradient descent, L2 regularization and multiplicative weight decay can be equivalent under a suitable parameterization. For adaptive optimizers such as Adam, they are not generally identical. AdamW decouples weight decay from the adaptive gradient step.

For the first MLP workflow:

```text
initial optimizer:
    Adam or the scikit-learn MLPClassifier solver="adam"

regularization interpretation:
    document the implementation-specific parameter clearly;
    do not casually equate every alpha parameter with AdamW-style decoupled weight decay.

secondary optimizer comparison:
    optional and limited, only after the main Adam baseline is understood.
```

### 8.5 Convergence warnings and finite iteration budgets

An optimizer can stop because it reaches a tolerance, fails to improve for a patience window, or simply reaches its maximum iteration count. A maximum-iteration warning is diagnostic information, not automatic proof that the model is unusable.

A notebook should record:

```text
number of iterations or epochs completed;
training loss trajectory;
internal validation trajectory when available;
whether convergence warnings occurred;
whether probabilities are finite and non-degenerate;
whether repeated runs are materially unstable.
```

Increasing maximum iterations is justified only when the learning curves indicate that the model is still making useful progress. It should not become an automatic response to every warning.

---

## 9. Initialization

All hidden units in one layer must not start with identical weights. If they do, they receive identical gradients and remain identical, wasting the intended representational capacity.

Weights are therefore initialized randomly but with a controlled scale. The scale should depend on the number of incoming and outgoing connections.

Two important families are:

```text
Glorot or Xavier initialization:
    designed to maintain activation and gradient variance for symmetric activations
    such as tanh or, in some settings, sigmoid.

He initialization:
    designed for rectifier activations such as ReLU.
```

Typical variance heuristics are:

$$
Var(W) ≈ 2 / (fan_in + fan_out)
$$

for Glorot-style initialization, and

$$
Var(W) ≈ 2 / fan_in
$$

for He-style ReLU initialization.

The initial Telco workflow should use estimator defaults rather than custom initialization code. The theory remains important because it explains why neural-network fitting can behave differently across random seeds and why initialization is not an unimportant implementation detail.

---

## 10. Regularization and capacity control

### 10.1 Capacity is the first regularizer

A network with too many units or layers can memorize idiosyncratic structure in the training folds. The simplest capacity controls are:

```text
number of hidden layers;
units per hidden layer;
activation choice;
maximum number of optimization iterations;
early stopping.
```

For this dataset size, shallow and moderately deep architectures are more defensible than a large deep network. The project should not treat many hidden layers as inherently more advanced or more appropriate.

### 10.2 L2 regularization

L2 regularization penalizes large weights:

$$
J(θ) = L_BCE(θ) + (λ / 2) Σ_l ||W^(l)||_F^2.
$$

It encourages smoother, less extreme parameter values. In scikit-learn's `MLPClassifier`, the parameter named `alpha` controls an L2 penalty, with implementation details documented by scikit-learn. The note and notebook should state the exact estimator semantics rather than importing terminology from a different library.

A limited alpha grid should be logarithmic because the practical effect of regularization often changes by orders of magnitude rather than by small arithmetic increments.

### 10.3 Early stopping

Early stopping interrupts optimization when performance on a validation subset no longer improves. It can control overfitting and reduce unnecessary training time.

However, early stopping has two distinct roles that should not be conflated:

```text
internal optimization control:
    stops one fitted neural network before it overfits its internal validation subset

outer model evaluation:
    estimates the candidate procedure on cross-validation validation folds
```

For scikit-learn's `MLPClassifier`, built-in early stopping uses an internal validation fraction and monitors validation accuracy. This is convenient but does not directly monitor the project primary metric, PR-AUC. It is still a legitimate training-only regularization device, but the notebook should state the mismatch explicitly and use outer cross-validation PR-AUC for candidate comparison.

Early stopping must not use the held-out project test set.

### 10.4 Dropout

Dropout randomly sets a fraction of hidden activations to zero during training. At evaluation time, dropout is disabled and the full network is used with the corresponding scaling convention.

Conceptually, dropout reduces co-adaptation: a unit cannot rely too strongly on the simultaneous presence of particular other units because random subsets are removed during training.

Dropout is useful to know but should not be forced into the initial MLP workflow:

```text
scikit-learn MLPClassifier:
    does not provide a standard dropout layer

PyTorch or Keras:
    can implement dropout directly

project decision:
    start with capacity control, L2 regularization, and early stopping;
    introduce dropout only if a later framework-based MLP extension is justified.
```

### 10.5 Batch normalization

Batch normalization normalizes intermediate activations using mini-batch statistics during training and typically maintains running statistics for evaluation. It can improve optimization stability in larger neural networks, but it adds train-versus-evaluation behavior, extra parameters, and another modelling decision.

Batch normalization is outside the first scikit-learn MLP workflow. It is a possible later extension in a lower-level framework, not a default requirement for a small tabular baseline.

---

## 11. Preprocessing for this tabular MLP

### 11.1 Scaling numeric variables

MLPs optimize by gradient-based methods. If numeric inputs have very different scales, the loss surface can become poorly conditioned. A shared learning rate then produces updates that are too large for some directions and too small for others.

The Telco numeric features should therefore be:

```text
imputed with statistics learned from the training fold only;
standardized with training-fold mean and standard deviation;
passed to the MLP as scaled numeric columns.
```

The project already has `make_scaled_preprocessor()` for models that benefit from standardized numeric variables. It performs median numeric imputation, standardization, categorical mode imputation, and one-hot encoding inside a `ColumnTransformer`.

### 11.2 One-hot encoded categorical variables

The categorical predictors should remain categorical and be transformed with one-hot encoding. The network receives numeric indicator columns after preprocessing.

One-hot indicators are already in a comparable 0/1 range. They generally do not require separate standardization. Centering them can obscure the simple interpretation of absence and presence without creating an obvious benefit for this first workflow.

A fully connected MLP does not natively understand category labels as unordered symbolic values. Passing arbitrary integer category codes to a dense network would incorrectly suggest an ordinal geometry. One-hot encoding avoids that problem.

### 11.3 Dense model input

Many neural-network implementations expect dense floating-point arrays. The project has a dense preprocessing factory, `make_dense_preprocessor(scale_numeric=True)`, which retains fold-safe imputation and one-hot encoding but requests dense output.

The MLP pipeline should use a dense, scaled preprocessor. It may be appropriate to add a clearly named convenience factory such as `make_dense_scaled_preprocessor()` when the executable workflow is implemented. No global preprocessing matrix should be fitted before cross-validation.

### 11.4 Leakage boundary

The preprocessing pipeline must be part of the estimator passed to cross-validation. For each fold:

```text
1. fit imputers, scaler, and encoder on that fold's training partition;
2. transform the fold's training partition;
3. transform the fold's validation partition using only fitted training-fold statistics;
4. fit the MLP on the transformed training partition;
5. score the transformed validation partition.
```

Fitting the scaler, imputer, or encoder on the full training table before cross-validation would leak validation-fold distribution information into each fitted candidate. This is prohibited even though the held-out test set remains untouched.

---

## 12. Evaluation design for stochastic neural networks

### 12.1 Stochasticity and reproducibility

MLP results can vary because of:

```text
random initial weights;
mini-batch ordering;
internal early-stopping validation split;
optimization trajectory;
potential non-determinism in some numerical backends.
```

A fixed random seed is useful for a first transparent grid because it makes comparisons reproducible. It does not establish that a selected configuration is robust to initialization.

If an MLP becomes a serious candidate, later training-only work should assess seed sensitivity. Reasonable options include:

```text
repeat a small set of shortlisted configurations across several seeds;
repeat cross-validation with different stratified splits;
summarize mean and spread of the primary metric;
compare procedures rather than one lucky random run.
```

This is especially important when observed PR-AUC differences are small.

### 12.2 Primary and secondary metrics

The project uses average precision, commonly reported as PR-AUC, as the primary development metric because churn is the minority positive class and the practical question focuses on identifying likely churners.

For the MLP workflow, record:

```text
primary:
    mean fold PR-AUC and pooled out-of-fold PR-AUC

secondary ranking:
    ROC-AUC

default-boundary diagnostics:
    balanced accuracy;
    precision;
    recall;
    specificity;
    F1;
    predicted positive rate;
    positive-first confusion matrix.
```

Training loss is an optimization diagnostic, not the sole model-selection metric. Internal early-stopping validation accuracy is also a training diagnostic, not a replacement for outer-fold PR-AUC.

### 12.3 Out-of-fold probabilities

For a probabilistic MLP, the proper out-of-fold score is the predicted positive-class probability from the model fitted on the corresponding training fold. Concatenating these predictions yields one score for every development observation, each generated by a model that did not train on that observation.

Pooled out-of-fold probabilities support:

```text
precision-recall and ROC curves;
calibration plots and Brier-type diagnostics if later justified;
threshold-tradeoff tables;
positive-first confusion matrices at documented thresholds.
```

They do not turn repeated experimentation into final validation. They remain development-stage evidence.

### 12.4 Learning curves

Learning curves should be interpreted with care:

```text
training loss decreasing and validation metric improving:
    evidence that optimization is progressing

training loss decreasing while validation performance deteriorates:
    possible overfitting

both training and validation performance weak:
    possible underfitting, poor optimization, or weak representation

highly erratic curves:
    possible overly large learning rate, small validation subset, or high stochastic variation
```

A learning curve should trigger a diagnostic question, not an automatic hyperparameter reaction. The answer depends on the pattern and should be supported by repeated evidence.

### 12.5 Thresholds and calibration

The default binary threshold 0.50 is a convention, not automatically the correct business decision threshold. Its appropriateness depends on retention value, contact capacity, false-positive cost, false-negative cost, and calibration.

The MLP notebook may show threshold diagnostics using pooled out-of-fold probabilities. It must describe those as exploratory, training-only operating-point evidence. The project will decide calibration method and final decision threshold only after a limited finalist set is defined.

---

## 13. MLPs compared with earlier model families

### 13.1 Logistic regression

Both logistic regression and a binary MLP commonly end in a sigmoid probability model with cross-entropy loss. The difference is representation.

```text
logistic regression:
    one linear logit in the supplied transformed feature space

MLP:
    one or more learned nonlinear hidden transformations before the final logit
```

A strong logistic-regression result means that much of the useful signal may already be linearly accessible after the existing preprocessing. An MLP needs to demonstrate added value rather than receive credit merely for being more flexible.

### 13.2 k-nearest neighbours

kNN uses local neighborhoods directly. An MLP maps observations through a learned global representation and then applies smooth nonlinear transformations. Both can react to geometry, but their inductive biases are different.

### 13.3 Decision trees and tree ensembles

Trees build piecewise-constant rules through axis-aligned splits. Random forests average many such rules. Gradient boosting adds trees sequentially to reduce loss.

An MLP instead learns distributed hidden features and produces a smooth probability surface after sigmoid transformation. It may capture interactions differently, but it does not automatically have an advantage on tabular data. Empirical benchmark literature often finds gradient-boosted trees highly competitive on medium-sized tabular datasets, so boosted trees remain a serious reference group for this project.

### 13.4 Support vector machines

Linear SVMs and MLPs can both create a linear separator after a representation. The SVM uses a maximum-margin objective and normally exposes decision scores rather than calibrated probabilities. A binary MLP uses a probabilistic cross-entropy objective and naturally outputs sigmoid probabilities, subject to calibration assessment.

### 13.5 Interpretability

MLP weights are not interpretable in the direct way that logistic-regression coefficients are. A hidden unit depends on many input features, and later units depend on combinations of earlier hidden units. Weight inspection may be useful for debugging but should not be presented as a simple effect-size analysis.

If the MLP becomes a finalist, model-agnostic explanations or feature-ablation analysis could be considered later. Such work should be clearly separated from causal interpretation.

---

## 14. Framework and implementation strategy

### 14.1 Initial implementation: scikit-learn MLPClassifier

The first MLP workflow should use `sklearn.neural_network.MLPClassifier` because it fits the existing project architecture:

```text
existing pattern:
    reusable preprocessing factories;
    scikit-learn Pipeline objects;
    cross-validation utilities;
    project-wide evaluation helpers;
    consistent artifact generation.

MLPClassifier provides:
    fully connected hidden layers;
    ReLU, tanh, logistic, or identity hidden activations;
    Adam, SGD, or L-BFGS solvers;
    L2 regularization through alpha;
    mini-batch optimization;
    early stopping;
    predict_proba outputs for binary classification.
```

This is a practical baseline, not a claim that scikit-learn is the complete neural-network ecosystem.

Important limitations:

```text
no standard dropout layer;
no standard batch-normalization layer;
built-in early stopping monitors validation accuracy rather than PR-AUC;
some optional features, including sample-weight support, depend on the installed
scikit-learn version;
the estimator is designed for relatively small feed-forward networks rather than
large or specialized deep-learning architectures.
```

The local package version should be inspected before the notebook is finalized.

### 14.2 PyTorch and Keras as later extensions

PyTorch provides a flexible lower-level framework for defining custom tensor computations, losses, optimizers, dropout, batch normalization, and bespoke training loops. Keras provides a higher-level interface for building and training neural networks. Both are useful later when the project has a specific methodological question that cannot be answered cleanly with `MLPClassifier`.

The initial Telco MLP workflow should not create a framework competition. Introducing PyTorch or Keras is justified only if it adds a clearly defined capability, such as PR-AUC-monitored early stopping, weighted loss control, dropout, batch normalization, or a reproducibility analysis requiring a custom loop.

### 14.3 Reusable factory plan

When implementation begins, reusable factory functions should be considered only after the notebook design is fixed. Likely additions are:

```text
src/telco_churn/preprocessing.py:
    make_dense_scaled_preprocessor()

src/telco_churn/models.py:
    make_mlp_classifier(...)
    make_mlp_pipeline(...)
```

The factories should return unfitted objects. They should validate architecture and optimizer arguments, document random-state behavior, and preserve fold-safe preprocessing through the pipeline.

---

## 15. Proposed training-only notebook design

The following is a design plan, not a frozen experimental grid. Exact candidate values should be adjusted only when there is a documented reason.

### Step 1: State the modelling question

Describe the MLP as a test of learned nonlinear feature representation for the existing tabular preprocessing. Restate that all results are training-only development estimates and the held-out test set is not loaded for selection.

### Step 2: Verify the modelling input

Use the training partition only. Check:

```text
target coding and positive-class convention;
absence of customerID from predictors;
numeric and categorical feature lists;
dense transformed shape within a fold-safe pipeline;
finite transformed values;
probability output shape and positive-class column.
```

Do not fit a global preprocessor merely to inspect transformed shape. Use a local diagnostic fit on one training subset and label it as a diagnostic.

### Step 3: Establish a deterministic sanity candidate

Start with one small ReLU MLP, for example:

```text
hidden_layer_sizes:
    (32,)

solver:
    adam

activation:
    relu

alpha:
    a modest L2 value

early stopping:
    enabled

random_state:
    fixed
```

The purpose is to confirm the pipeline, dense transformation, convergence behavior, `predict_proba` output, and metric functions before comparing models.

### Step 4: Screen shallow capacity and L2 regularization

A transparent initial screen can vary:

```text
hidden-layer width:
    16, 32, 64

alpha:
    0.0001, 0.001, 0.01
```

Keep activation, optimizer, initial learning rate, batch-size policy, and early-stopping settings fixed. This identifies whether the model appears under-capacity, over-capacity, or relatively insensitive within a small justified grid.

### Step 5: Test limited added depth

Shortlist one or two strong shallow settings, then compare them with a limited number of two-layer alternatives, such as:

```text
(32, 16)
(64, 32)
```

Do not infer that a deeper configuration is better because its point estimate is marginally higher. Compare fold-level variation, pooled out-of-fold diagnostics, convergence, and seed sensitivity where useful.

### Step 6: Consider one controlled activation comparison

If the ReLU candidates show unstable or clearly unsatisfactory learning behavior, compare a limited tanh alternative at a comparable architecture. This is a diagnostic comparison, not a broad activation sweep.

### Step 7: Consider imbalance handling only after the unweighted baseline

If the unweighted MLP has poor positive-class recall at its natural threshold, evaluate a clearly documented class-sensitive alternative only if supported by the local library version. Keep the comparison training-only and distinguish ranking changes from threshold changes.

### Step 8: Selected-candidate diagnostics

For the representative selected MLP, create:

```text
cross-validation summary table;
pooled OOF PR curve;
pooled OOF ROC curve;
positive-first confusion matrix at 0.50;
threshold diagnostic table;
probability distribution by observed class;
calibration diagnostic, clearly labeled development-stage;
loss curve and internal validation curve if exposed;
convergence-warning summary;
architecture and parameter-count table.
```

### Step 9: Compare with established representatives

Compare against the prior representative logistic regression, kNN, decision tree, bagging or random forest, strongest boosting representative, and linear SVM. Use shared training-only evaluation conventions. The comparison is contextual, not a final tournament.

### Step 10: Preserve the evaluation boundary

Conclude with an explicit statement that the held-out test set has not been touched. Report the MLP as a development-stage candidate or as an educational model-family result, depending on observed evidence.

---

## 16. Hyperparameter rationale

The following parameters have distinct meanings and should not be tuned indiscriminately.

| Parameter | Main meaning | Initial project treatment |
|---|---|---|
| hidden_layer_sizes | capacity through width and depth | small transparent grid |
| activation | shape and derivative of hidden transformation | ReLU default, limited tanh check only if useful |
| alpha | L2 regularization strength in scikit-learn | logarithmic small grid |
| solver | optimizer family | Adam initial default |
| learning_rate_init | base optimizer step size | fixed initially, limited search only if diagnostics require it |
| batch_size | gradient-noise and update-frequency control | use a defensible fixed value or documented auto behavior |
| max_iter | training budget | sufficiently high to observe learning, not automatically increased |
| early_stopping | internal optimization regularizer | enabled for Adam candidate, with accuracy-monitoring caveat |
| validation_fraction | internal holdout share for early stopping | fixed and documented |
| n_iter_no_change | patience before early stopping | fixed and documented |
| random_state | initialization and stochastic reproducibility | fixed for first grid, varied later for shortlisted candidates |

The number of candidate procedures must remain modest. Searching architecture, alpha, activation, learning rate, batch size, optimizer, class weighting, early-stopping settings, and seeds all at once would create a large multiple-comparison problem and obscure the learning goal.

---

## 17. Interpretation language

Use language such as:

```text
selected within the tried training-only development grid;
representative MLP candidate;
development-stage cross-validated estimate;
observed learning-curve behavior;
small differences should be interpreted cautiously;
the held-out test evaluation is deferred.
```

Avoid language such as:

```text
the neural network is definitively best;
the MLP proves nonlinear effects are present;
the observed probability is automatically calibrated;
the selected threshold is final;
the MLP has final production performance.
```

A predictive improvement would support the statement that the MLP captured useful statistical structure under the evaluated procedure. It would not by itself identify causal effects, prove a unique architecture, or establish deployment value.

---

## 18. Implementation and reporting plan

### Implementation plan

```text
1. Inspect current preprocessing, model-factory, evaluation, visualization, and notebook conventions.
2. Add a dense scaled preprocessing helper only if it improves clarity and remains fold-safe.
3. Add unfitted MLP factory functions only after the notebook design is fixed.
4. Write the training-only MLP notebook source.
5. Run the notebook locally and return executed outputs, tables, figures, warnings, and package-version information.
6. Update result-specific notebook interpretation only from observed outputs.
7. Add a focused smoke test if reusable factories or plot helpers are added.
```

### Report plan

The later LaTeX section should be written only after executed results are available. It should include:

```text
the purpose of learned nonlinear representations for this tabular problem;
the feed-forward architecture and binary cross-entropy objective;
the role of scaling, one-hot encoding, and fold-safe preprocessing;
capacity control, L2 regularization, early stopping, and stochasticity;
the transparent architecture and regularization screen;
observed training-only cross-validation results;
careful comparison with prior model families;
limitations, calibration caveats, and final-test deferral.
```

The report should explain the mathematics and modelling decisions but should not reproduce every experimental diagnostic or every implementation detail from the notebook.

---

## 19. Summary

A multilayer perceptron is a feed-forward model that alternates learned affine transformations and nonlinear activation functions. The nonlinearities allow the model to learn hidden feature representations that cannot be collapsed into one linear score.

For binary churn classification, the MLP produces a logit z and a sigmoid probability p. Binary cross-entropy is the negative Bernoulli log-likelihood, and the output-layer gradient simplifies to p - y. Backpropagation carries that error through the hidden layers using the chain rule, while a mini-batch optimizer updates every weight and bias.

For the Telco workflow, the practical priorities are:

```text
dense, scaled, one-hot encoded input built inside fold-safe pipelines;
small shallow and moderately deep architectures;
ReLU plus a constrained activation comparison only when justified;
Adam-based optimization with monitored convergence;
capacity control, L2 regularization, and training-only early stopping;
PR-AUC as the primary outer evaluation metric;
probability, calibration, and threshold interpretation kept separate;
seed sensitivity considered before any MLP is treated as a serious finalist;
the held-out test set left untouched until a final full pipeline is frozen.
```

The immediate next step is an executable, training-only notebook that tests the planned MLP candidates and returns observed artifacts for interpretation.

---

## References and implementation materials

- Feed-forward neural-network, backpropagation, and maximum-likelihood materials used in the project’s machine-learning reference slides.
- Deep-learning systems and practical optimization materials used in the project’s deep-learning reference slides.
- Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
- He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving deep into rectifiers.
- Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization.
- Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting.
- Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift.
- scikit-learn documentation for `MLPClassifier`, including estimator parameters, early stopping, and probability prediction behavior.
- PyTorch documentation for binary cross-entropy with logits, AdamW, dropout, and batch normalization.
- Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data?
- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. (2021). Revisiting deep learning models for tabular data.
