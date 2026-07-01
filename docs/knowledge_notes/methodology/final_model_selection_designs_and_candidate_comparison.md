# Final model selection designs, broad candidate comparison, and pre-test decision protocol

## Purpose and status

This note is a reusable technical reference for selecting one final binary-classification model from a broad library of candidate model families while preserving a genuinely untouched final test set.

It is written for the Telco Customer Churn project, but the conceptual framework applies to many tabular supervised-learning problems. The project deliberately studies many model families, not only the current highest-scoring one. The later final-selection stage must therefore distinguish carefully between:

1. **learning about model families**;
2. **tuning a model family**;
3. **comparing candidate families**;
4. **choosing one deployable configuration**;
5. **choosing calibration and an operational decision rule**;
6. **estimating final performance on data that did not influence any of those choices**.

This note does **not** freeze the final empirical protocol yet. It explains the credible alternatives, their estimands, strengths, limitations, computational implications, and how they can be used to choose a final model. After the design is reviewed, the project-specific `final_model_comparison_plan.md` should be updated with the exact selected protocol before any final comparison results are examined.

The note intentionally treats the following as separate questions:

```text
Which tuned model family should be selected?

Which exact hyperparameter configuration should be deployed inside that family?

How uncertain is the difference between candidates?

Which calibrated probability and threshold policy should be used operationally?

How well does the one frozen final system perform on untouched test data?
```

Confusing those questions is one of the main reasons that model-selection workflows become difficult to explain or accidentally optimistic.

---

## 1. Project setting and notation

Let the original cleaned dataset be

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n},
$$

where $x_i$ is a customer feature vector and

$$
y_i \in \{0,1\},
$$

with $y_i=1$ denoting churn.

The project already has a held-out test set. Write:

$$
\mathcal{D}
=
\mathcal{D}_{\mathrm{dev}}
\cup
\mathcal{D}_{\mathrm{test}},
\qquad
\mathcal{D}_{\mathrm{dev}}
\cap
\mathcal{D}_{\mathrm{test}}
=
\varnothing.
$$

Here:

```text
Development data:
    all model development, feature work, resampling, tuning,
    candidate comparison, calibration selection, threshold selection,
    ablations, and uncertainty analysis.

Test data:
    one final evaluation of one frozen pipeline only.
```

For the Telco project:

```text
Development set:
    5,634 rows

Held-out test set:
    1,409 rows

Positive class:
    churn
```

The candidate library contains model families indexed by

$$
j \in \{1,\ldots,J\}.
$$

A model family is not one fitted estimator. For example, "XGBoost" is a family that can contain many configurations. Let

$$
\lambda \in \Lambda_j
$$

denote a configuration from family $j$, where $\Lambda_j$ is the predefined search space.

A configuration may include:

```text
preprocessing variant
feature engineering option
feature selection option
resampling or class-weighting policy
model hyperparameters
random-state policy
early-stopping policy
calibration option, if calibration is part of the candidate
threshold option, if a threshold is part of the candidate
```

A fitted pipeline trained on dataset $S$ with candidate family $j$ and configuration $\lambda$ is written:

$$
\widehat{f}_{j,\lambda,S}.
$$

A performance metric is written $M(\widehat{f}, V)$, where $V$ is an evaluation set. For this project, the primary ranking metric will likely remain average precision or a precisely defined PR-AUC implementation. The exact metric implementation must be declared because average precision and trapezoidal area under a precision-recall curve are related but not numerically identical summaries.

---

## 2. The objects that can be selected or evaluated

A rigorous workflow becomes easier when the selected object is named explicitly.

### 2.1 A fitted model

A fitted model has learned parameters from one training sample:

```text
a logistic-regression coefficient vector
a collection of random-forest trees
an XGBoost booster
a trained neural network
```

It is specific to its training sample.

### 2.2 A configuration

A configuration is a fixed modelling specification:

```text
XGBoost:
    max_depth = 3
    learning_rate = 0.05
    n_estimators = 500
    subsample = 0.8
    colsample_bytree = 0.8
    reg_lambda = 5
```

A fixed configuration can be evaluated with ordinary or repeated cross-validation. The evaluation estimates performance of that configuration when trained on folds smaller than the full development set.

### 2.3 A tuned model-family procedure

A tuned family procedure is a rule:

```text
Given a training sample:
    search the predefined XGBoost configuration space by inner cross-validation;
    select the configuration with the strongest primary metric;
    refit that configuration on all available training data;
    predict new cases.
```

Formally, let the tuning rule be

$$
\widehat{\lambda}_j(S)
=
\arg\max_{\lambda \in \Lambda_j}
\widehat{M}_{\mathrm{inner}}(j,\lambda;S).
$$

The tuned procedure for family $j$ is:

$$
\mathcal{P}_j(S)
=
\widehat{f}_{j,\widehat{\lambda}_j(S),S}.
$$

This object can choose different hyperparameters when given a different training sample. That is expected. It is the object evaluated by per-family nested cross-validation.

### 2.4 A candidate procedure

For final comparison, a candidate should be a complete, reproducible procedure, not merely a model name.

```text
Candidate procedure:
    preprocessing pipeline
    feature representation
    optional feature selection
    optional resampling or class weighting
    model family
    search space
    search method and budget
    selection metric
    random-state policy
    early-stopping protocol where relevant
```

Thus, a candidate can be described as:

```text
"XGBoost procedure A"
```

rather than only:

```text
"XGBoost"
```

This matters because two different preprocessing or resampling rules can make the same classifier family into different candidate procedures.

### 2.5 A final deployment specification

The final deployment specification must be one frozen object:

```text
data version
feature set
preprocessing
model family
exact hyperparameters
random-state policy
calibration method
threshold or top-k policy
all fitted artifacts
```

Only after this specification is frozen may it be fitted once on the full development set and evaluated once on the held-out test set.

---

## 3. Candidate library versus individual hyperparameter configurations

The project is intentionally extensive. It is reasonable to compare a broad candidate library rather than only a very small shortlist.

A plausible initial library can include:

```text
1. L2 logistic regression
2. k-nearest neighbours
3. Hybrid Gaussian-Bernoulli Naive Bayes
4. Regularized decision tree
5. Bagged trees
6. Random forest
7. GradientBoostingClassifier
8. XGBoost
9. CatBoost
10. Linear SVM
11. RBF SVM
12. Multilayer perceptron
```

The exact candidate library should eventually be frozen in a registry before the final comparative results are examined.

The key distinction is:

```text
Candidate library:
    model procedures that will be compared.

Configuration search:
    settings explored inside each candidate procedure.
```

For example:

```text
Candidate:
    tuned XGBoost procedure

Its internal configurations:
    60 possible combinations of tree depth, learning rate,
    subsampling, regularization, and number of estimators
```

The 60 configurations are not 60 separate final candidate families. They are alternatives used by the XGBoost procedure to tune itself.

### 3.1 Cross-cutting variants

Some choices can either be treated as internal configurations or distinct candidate procedures:

```text
unweighted versus class-weighted learning
no resampling versus SMOTE
full feature set versus selected feature set
uncalibrated versus calibrated probabilities
different threshold policies
```

The correct treatment depends on the scientific question.

A useful rule is:

```text
Treat a choice as an internal configuration when it is a normal tuning
dimension of the same predictive procedure.

Treat a choice as a separate candidate procedure when it changes the
substantive modelling strategy, operational interpretation, or data
processing path enough that it deserves a separate comparison.
```

For example, a cost-sensitive class-weighted XGBoost model may be a distinct procedure if the project wants to study how weighting changes the recall-precision operating profile. A minor change in `min_child_weight` is normally only an internal configuration.

The final comparison should avoid an uncontrolled Cartesian product of every family, every resampling rule, every feature-selection rule, every calibrator, and every threshold. That design can become computationally huge and can itself create a large selection problem. A structured factor design is preferable:

```text
Stage A:
    compare core model-family procedures under a common baseline pipeline.

Stage B:
    investigate cross-cutting additions, such as feature selection,
    resampling, or calibration, for all or a justified subset of procedures.

Stage C:
    define final candidate procedures that include only additions
    supported by the training-only evidence.
```

This is still comprehensive. It is simply organized rather than combinatorially uncontrolled.

---

## 4. What the final-selection stage must accomplish

The final-selection stage has five distinct outputs.

```text
Output 1:
    a transparent ranking or equivalence grouping of candidate procedures.

Output 2:
    a selected model family or a practical tie set.

Output 3:
    one final hyperparameter configuration inside the selected family.

Output 4:
    a calibration and decision-policy specification when probabilities
    will be interpreted operationally.

Output 5:
    one frozen pipeline for a single test-set evaluation.
```

The stage should also produce a detailed evidence record:

```text
candidate registry
search spaces
search budgets
splitter definitions and seeds
fold-level metrics
pooled out-of-fold diagnostics where appropriate
runtime and convergence records
selected configurations
stability summaries
pairwise comparisons
tie-breaking rationale
final frozen specification
```

This record is important for reproducibility, reporting, and future learning. It also prevents the final choice from becoming an undocumented sequence of informal tweaks.

---

## 5. Evaluation design A: internal validation holdout

The simplest design reserves part of the development data as an internal validation set.

```text
Development training portion:
    tune all model families.

Internal validation portion:
    compare the selected family/configuration from each procedure.

Final test set:
    evaluate one frozen final model.
```

### 5.1 Procedure

1. Split $\mathcal{D}_{\mathrm{dev}}$ into $\mathcal{D}_{\mathrm{train}}$ and $\mathcal{D}_{\mathrm{val}}$.
2. Tune every candidate family using only $\mathcal{D}_{\mathrm{train}}$, perhaps with cross-validation.
3. Fit each selected candidate on all of $\mathcal{D}_{\mathrm{train}}$.
4. Compare them once on $\mathcal{D}_{\mathrm{val}}$.
5. Choose one family and one configuration.
6. Refit the chosen full pipeline on all of $\mathcal{D}_{\mathrm{dev}}$.
7. Evaluate once on $\mathcal{D}_{\mathrm{test}}$.

### 5.2 What it estimates

It provides a clean internal comparison because the validation set did not influence the candidate-specific tuning. It estimates the performance of tuned candidates trained on a reduced fraction of the development data.

### 5.3 Advantages

```text
very easy to explain
clear separation between tuning and candidate comparison
simple to implement
allows a single paired set of validation predictions for comparisons
```

### 5.4 Limitations

```text
less data are available for tuning and fitting
the decision can depend strongly on one random validation split
the internal validation sample may be modest for stable PR-AUC comparison
the selected candidate can still be chosen partly because of validation noise
```

For the Telco project, this is a useful reference design and possible sensitivity analysis. It is not automatically the strongest primary design because the development set is valuable and the project wants to make extensive use of it.

---

## 6. Evaluation design B: flat repeated cross-validation

Flat repeated cross-validation is the direct approach described in the project discussion.

### 6.1 Core algorithm

For each candidate family $j$:

```text
1. Define a search space Lambda_j.

2. Reuse the same repeated stratified CV splits for all configurations
   of that candidate and, ideally, for all candidate procedures.

3. Evaluate each configuration on the repeated-CV splits.

4. Select the configuration with the strongest predefined summary,
   such as mean fold average precision.

5. Store all fold-level scores, configuration results, and out-of-fold
   predictions where appropriate.
```

After candidate-specific tuning:

```text
6. Compare the best selected repeated-CV summary of every family.

7. Apply the predeclared selection and tie-breaking rule.

8. Choose the final family and its selected configuration.

9. Fit that selected full pipeline on all development data.

10. Evaluate once on the untouched test data.
```

### 6.2 What this design directly chooses

Flat repeated CV can directly choose both:

```text
a model family
an exact hyperparameter configuration
```

For example:

```text
Random forest:
    best repeated-CV configuration gives mean AP = 0.661

XGBoost:
    best repeated-CV configuration gives mean AP = 0.670

Decision:
    choose the selected XGBoost configuration
```

No further tuning run is intrinsically required. The selected configuration has already been chosen using the full development set through repeated cross-validation. The final operational action is to fit that fixed configuration on all development rows.

A rerun is only necessary when the initial search was intentionally preliminary, when the final search budget is expanded by a predeclared rule, or when the selected candidate procedure includes a final calibration or threshold component that was not part of the initial comparison.

### 6.3 What it estimates

For a fixed configuration, repeated CV estimates development-stage performance more stably than one random fold partition.

For the winner selected after looking across many configurations and families, the observed best repeated-CV score is not a clean unbiased estimate of future performance. It has been selected from noisy estimates. This is selection optimism, sometimes described informally as a winner's-curse effect.

### 6.4 Why repeated CV still has value

Repeated CV improves stability. It can reveal whether:

```text
a configuration wins only because of one fold partition
neighbouring hyperparameter settings are effectively tied
rankings are sensitive to split construction
randomized algorithms are unstable
```

It does **not** eliminate selection optimism. Repetition reduces the variance of the estimates but does not change the fact that the winner is selected after many estimated scores have been inspected.

### 6.5 Strengths

```text
directly returns a family and configuration
uses the entire development dataset across the resampling design
simple to report
often much less expensive than nested CV
well suited to a broad candidate library
```

### 6.6 Limitations

```text
the displayed winner score is selected from many estimates
candidate families with broader or more favorable search spaces can
benefit more from search opportunity
stronger performance-estimation claims require additional caution
```

Flat repeated CV is not invalid. It is a common practical design. Its output should simply be interpreted correctly.

---

## 7. Evaluation design C: per-family nested cross-validation

Per-family nested CV is the central alternative to flat repeated CV when the goal is a more defensible comparison of tuned model families.

### 7.1 Core algorithm

For each outer split $r,k$:

```text
1. Reserve one outer-validation fold.

2. For each candidate family j:
       a. Use only the outer-training data.
       b. Run the candidate's inner tuning procedure.
       c. Select its best inner-CV configuration.
       d. Refit that selected configuration on all outer-training data.
       e. Predict the untouched outer-validation fold.

3. Store:
       outer-fold metric for every candidate family;
       selected inner configuration for every candidate family;
       predictions for every candidate on the outer-validation fold.
```

Repeat over outer folds and optionally outer repeats.

For each candidate family $j$, summarize its outer-fold metrics:

$$
\widehat{M}^{\mathrm{nested}}_j
=
\frac{1}{R_{\mathrm{outer}}K_{\mathrm{outer}}}
\sum_{r=1}^{R_{\mathrm{outer}}}
\sum_{k=1}^{K_{\mathrm{outer}}}
\widehat{M}_{j,r,k}^{\mathrm{outer}}.
$$

Then compare the candidate families based on their paired outer-fold results.

### 7.2 What nested CV evaluates

Nested CV evaluates the tuned procedure:

```text
Given a training sample:
    tune this family using its predefined inner design;
    fit the selected configuration;
    predict unseen data.
```

It does **not** provide one unique final configuration. Different outer folds may select different configurations:

```text
Outer fold 1:
    XGBoost depth 3, learning rate 0.05

Outer fold 2:
    XGBoost depth 4, learning rate 0.03

Outer fold 3:
    XGBoost depth 3, learning rate 0.10
```

That is not a defect. It describes hyperparameter-selection stability. If neighbouring configurations alternate frequently, it may indicate that the family has a broad practically equivalent region rather than one uniquely preferred point.

### 7.3 How it chooses a final model

Per-family nested CV still supports the ultimate practical goal:

```text
1. Compare nested outer-fold summaries of all candidate families.

2. Select the strongest family, or a practical equivalence set,
   using the predefined primary metric and tie rule.

3. On all development data, run the final tuning procedure for that
   chosen family only.

4. Select one final exact configuration.

5. Fit it on all development data.

6. Freeze calibration and threshold policy.

7. Evaluate once on test data.
```

Thus:

```text
Nested CV:
    chooses the winning model family more rigorously.

A final tuning run on all development data:
    chooses the exact final configuration inside that winning family.
```

The final tuning run is allowed to use all development data because the held-out test set remains untouched. It is not intended to provide another unbiased performance estimate. It is intended to make the best use of available training data after the family decision has been made.

### 7.4 Strengths

```text
separates candidate-specific tuning from outer performance estimation
reduces the direct effect of hyperparameter-selection optimism on
family-level comparison
provides fold-by-fold evidence for every candidate
reveals hyperparameter stability across outer samples
supports paired outer-fold comparisons because candidates share
the same outer validation folds
```

### 7.5 Limitations

```text
computationally expensive
more complex to implement and explain
outer models train on less than all development data
does not itself produce one final deployable configuration
does not erase adaptivity from earlier exploratory work on the same
development dataset
```

The last point is important. Earlier model-family notebooks have already informed the project. A final nested stage is much stronger evidence for a predefined candidate library and search protocol, but it cannot make all prior human decisions independent of the development dataset. The untouched test set remains the final independent performance check.

---

## 8. Evaluation design D: repeated nested cross-validation

Repeated nested CV repeats the outer fold construction several times.

```text
Outer design:
    repeated stratified K-fold CV

Within every outer split:
    candidate-specific inner tuning
```

For example:

```text
outer loop:
    5 folds x 3 repeats = 15 outer validation evaluations

inner loop:
    4-fold stratified CV
```

### 8.1 Why repeat the outer loop?

One 5-fold nested-CV run can still be influenced by one partition of the development dataset. Repeating outer partitions provides a richer picture of:

```text
candidate ranking stability
split sensitivity
selection stability
randomized-model sensitivity
fold-level metric variation
```

### 8.2 Computational cost

Suppose candidate $j$ evaluates $T_j$ hyperparameter trials inside $K_{\mathrm{inner}}$ folds. Ignoring early stopping and shared computation, approximate fit count is:

$$
N_{\mathrm{fits}}
\approx
\sum_{j=1}^{J}
R_{\mathrm{outer}}
K_{\mathrm{outer}}
\left(
T_j K_{\mathrm{inner}} + 1
\right).
$$

For a broad library, this becomes large quickly. With:

```text
12 candidate families
5 outer folds
3 outer repeats
20 trials per family on average
4 inner folds
```

the approximate number of inner fits is:

$$
12 \times 5 \times 3 \times 20 \times 4
=
14{,}400,
$$

before outer refits, calibration variants, threshold analyses, or ablations.

The actual cost is uneven. Logistic regression may be cheap; CatBoost, XGBoost, RBF SVM, and MLP trials can be much more expensive. Therefore, a feasibility profiling step is needed before fixing the design.

### 8.3 What repeated nested CV does not solve

Repeated outer folds create more correlated resampling results. They should not be treated as a set of completely independent datasets. Repetition improves descriptive stability and supports correlation-aware comparison methods, but it does not make naive independent-sample inference valid.

---

## 9. Evaluation design E: repeated random holdout or Monte Carlo cross-validation

Repeated random holdout draws multiple training-validation splits rather than using disjoint K-fold partitions.

For repeat $r$:

```text
randomly select a stratified training subset
use the remaining data as validation
fit, tune, and evaluate
repeat with a new split
```

### 9.1 Advantages

```text
training and validation proportions can be selected directly
easy to repeat many times
useful for sensitivity analyses
can be convenient when a fixed validation fraction is desired
```

### 9.2 Limitations

```text
some observations may appear in validation many times
some may appear rarely or not at all
validation sets overlap
results remain dependent
less tidy out-of-fold coverage than ordinary K-fold CV
```

For this project, repeated random holdout is a useful alternative reference design, but repeated stratified K-fold CV is likely easier to present because it ensures systematic fold coverage within each repeat.

---

## 10. Evaluation design F: bootstrap approaches

Bootstrap procedures resample observations with replacement. They can be used for performance estimation, confidence intervals, candidate differences, or bias correction.

### 10.1 Ordinary bootstrap confidence intervals

For one fixed final model evaluated on the untouched test set:

```text
1. Keep the test labels and predictions fixed.
2. Resample test rows with replacement.
3. Recompute the metric.
4. Repeat many times.
5. Use quantiles of the bootstrap distribution.
```

This is suitable for final uncertainty intervals around one frozen model's test metrics.

It is not a model-selection method by itself.

### 10.2 The .632 bootstrap

The .632 bootstrap was historically proposed to combine in-sample and out-of-bag performance estimates when sample sizes are limited. It can be useful to know, but it is not the natural main design here because the project already has a development/test split and wants transparent cross-validation-based tuning and comparison.

### 10.3 Paired bootstrap for candidate differences

When two candidates produce predictions for the same evaluation observations, a paired bootstrap resamples observation indices and recomputes both metrics on the same resample.

For a metric $M$, define:

$$
\Delta
=
M_A - M_B.
$$

For bootstrap draw $b$:

$$
\Delta^{*(b)}
=
M_A^{*(b)}
-
M_B^{*(b)}.
$$

The resulting distribution can provide a confidence interval for the performance difference.

This is especially useful for PR-AUC or average-precision differences, where a standard analytic test is less universally used than DeLong's ROC-AUC comparison.

### 10.4 Important repeated-CV caution

A simple row-level paired bootstrap assumes one paired prediction per evaluation observation.

With repeated CV, the same customer may have multiple out-of-fold predictions from different repeats. Those repeated predictions are not independent new customers. Therefore, do **not** simply concatenate all repeated predictions into one huge row-level bootstrap sample and claim that it represents $n \times R$ independent observations.

Better options include:

```text
perform a paired bootstrap within each repeat separately and summarize
the repeat-level results;

use one predeclared complete OOF partition for prediction-level bootstrap
diagnostics;

or use an explicitly designed hierarchical or correlation-aware method.
```

This distinction matters especially when a report tries to make formal uncertainty claims.

---

## 11. Evaluation design G: bias-corrected flat CV

Flat CV is attractive because it directly selects a family and configuration. Its drawback is the optimism of the selected winner's CV score. Several methods try to estimate or correct that selection bias without the full computational cost of nested CV.

### 11.1 Tibshirani and Tibshirani bias correction

Tibshirani and Tibshirani proposed a correction for the optimism of the minimum cross-validation error after tuning. The correction uses fold-level information already generated during cross-validation and therefore has little additional computational cost.

In the error-minimization setting, if $\widehat{\mathrm{Err}}_{\lambda}$ is the average CV error for configuration $\lambda$, then the selected configuration is:

$$
\widehat{\lambda}
=
\arg\min_{\lambda \in \Lambda}
\widehat{\mathrm{Err}}_{\lambda}.
$$

The basic problem is that:

$$
\widehat{\mathrm{Err}}_{\widehat{\lambda}}
$$

tends to be optimistic for the true generalization error of the selected configuration. Their method estimates a correction from the way fold-level winners vary.

The method is useful to understand because it highlights that selection bias can be studied with information already generated by CV. Its exact applicability depends on the metric, selection rule, and details of the implementation.

### 11.2 Bootstrap Bias Corrected Cross-Validation, BBC-CV

Bootstrap Bias Corrected Cross-Validation uses out-of-sample predictions already produced during a flat CV search.

Suppose the CV search creates an out-of-fold prediction matrix:

$$
P_{i,c},
$$

where $P_{i,c}$ is the out-of-fold score for observation $i$ under candidate configuration $c$.

For each bootstrap draw:

```text
1. Resample observation indices.

2. On the bootstrap sample:
       choose the apparent best configuration.

3. On observations not selected in that bootstrap sample:
       evaluate that chosen configuration.

4. Repeat.

5. Average the out-of-bootstrap evaluations.
```

The key insight is that the bootstrap reruns the **selection** step using the stored out-of-fold predictions. It attempts to estimate how much performance is inflated by selecting the apparent best configuration.

### 11.3 Why BBC-CV is interesting here

A broad all-model candidate library can make repeated nested CV expensive. BBC-CV is attractive because, after the original out-of-fold prediction matrix exists, it can evaluate many bootstrap selection replicates without retraining every model inside new nested loops.

### 11.4 Why BBC-CV is not automatically the default

BBC-CV is a sophisticated alternative, not a free universal replacement for nested CV.

Important cautions:

```text
it depends on stored out-of-fold predictions for every candidate;
the candidate library and selection rule must be clearly defined;
it estimates performance of selection among the supplied candidates,
not of arbitrary future human exploration;
complex pipelines with fold-specific preprocessing and calibration require
careful prediction bookkeeping;
it should be validated against a simpler reference design in this project.
```

For this project, BBC-CV is worth researching and potentially implementing as a secondary design or computationally efficient sensitivity analysis. It should not be adopted merely because it is more advanced.

---

## 12. Hyperparameter-search strategies are not evaluation designs

The resampling design answers:

```text
How are candidates tuned and compared?
```

The search strategy answers:

```text
Which configurations are tried inside a candidate procedure?
```

These are different layers.

### 12.1 Grid search

Grid search evaluates every predefined combination.

Strengths:

```text
transparent
reproducible
easy to explain
useful when the meaningful space is compact
```

Limitations:

```text
grows exponentially with dimensionality
may waste budget on unimportant dimensions
can miss useful values between grid points
```

### 12.2 Random search

Random search draws configurations from specified distributions.

Strengths:

```text
often more efficient in high-dimensional spaces
can sample continuous and log-scale values
easy to set a fixed trial budget
```

Limitations:

```text
depends on random seed
requires well-chosen distributions
may underexplore an important region when the budget is small
```

### 12.3 Bayesian optimization and Optuna-style adaptive search

Adaptive search uses previous trials to decide where to try next.

Strengths:

```text
can use expensive trials more efficiently
useful for flexible boosting models and MLPs
can combine with early stopping or pruning
```

Limitations:

```text
more complex
less transparent
requires fixed search policy and budget for fair comparison
must be rerun inside each inner loop of nested CV
adds another source of randomness and adaptivity
```

### 12.4 Successive halving and Hyperband-style search

These approaches allocate limited resources to many configurations and stop weak configurations early.

They can be useful for models with natural partial-training budgets:

```text
number of trees
number of epochs
training data fraction
```

However, early stopping rules become part of the candidate procedure and must be applied consistently inside the comparison design.

### 12.5 Search fairness across model families

Literal equality of trial counts is not always fair. A logistic regression has fewer consequential hyperparameters than XGBoost. But the process should avoid giving one favorite family a dramatically more generous opportunity to discover a lucky configuration.

A practical fairness record should state:

```text
candidate family
search method
search-space rationale
trial budget or grid size
inner metric
inner splitter
random seed policy
early-stopping policy
runtime cap, if any
```

One useful design is a tiered budget:

```text
simple families:
    compact complete grids

complex families:
    fixed random-search or adaptive-search trial budgets

all families:
    explicit rationale and logged compute cost
```

---

## 13. Leakage-safe treatment of preprocessing and feature selection

Everything that learns from data must be fitted only on the relevant training portion of the resampling design.

This includes:

```text
missing-value imputation
scaling
one-hot encoding when category learning is data-dependent
rare-category grouping
feature engineering based on observed distributions
feature selection
dimensionality reduction
outlier rules estimated from data
resampling such as SMOTE
probability calibration
threshold selection when a threshold is tuned
```

### 13.1 Flat repeated CV

For each fold:

```text
fit preprocessing on the fold-training partition
transform fold-training and fold-validation partitions
fit candidate model on transformed fold-training partition
predict fold-validation partition
```

### 13.2 Nested CV

For each outer fold:

```text
outer-validation data:
    never informs any tuning, preprocessing, feature selection,
    resampling, calibration selection, or threshold selection.

inner folds:
    all learned components must be fitted within each inner-training portion.
```

A feature selector outside the inner loop is leakage. A scaler fitted once on all outer-training data before inner CV can also be leakage for the inner tuning comparison if it uses statistics from inner-validation rows.

Pipelines and project-level reusable factories are therefore not merely software conveniences. They encode the statistical order of operations.

---

## 14. Calibration and threshold selection

Ranking, calibration, and decision policies are different.

### 14.1 Ranking

Ranking metrics answer whether churners tend to receive larger scores than non-churners.

```text
average precision or PR-AUC
ROC-AUC
precision-recall curve
```

### 14.2 Calibration

Calibration asks whether a predicted probability of $0.30$ corresponds approximately to a 30 percent observed churn frequency in comparable cases.

Relevant summaries include:

```text
Brier score
log loss
calibration curve
calibration intercept
calibration slope
```

### 14.3 Decision policy

A threshold or capacity rule converts risk scores into action.

```text
contact customers with probability >= tau
contact the top 10 percent highest-risk customers
choose tau to target 75 percent recall
choose tau to satisfy a campaign-capacity constraint
choose tau to maximize expected net value
```

### 14.4 Where calibration belongs

If calibration is merely an operational post-selection question and the core candidate comparison uses a ranking metric, it can be studied after the family is chosen.

If the final decision criterion depends on calibrated probability quality, then calibration method is part of the candidate procedure and should be selected using training data only.

A strict calibration procedure has its own data separation:

```text
base-model training data:
    fit the base learner.

calibration data:
    fit the calibrator on predictions from a base learner not trained
    on those calibration rows.

final test data:
    evaluate the already-frozen calibrated system once.
```

Cross-fitting can support this without wasting large portions of development data.

### 14.5 Where threshold selection belongs

A threshold is a model-selection decision because it changes precision, recall, specificity, F1, predicted positive rate, and expected cost.

For final evaluation:

```text
choose threshold using development-only evidence
freeze threshold
evaluate fixed threshold on test data
```

If business costs or capacity are unknown, no universally optimal threshold exists. The final report can then present several predeclared operating points rather than falsely presenting one threshold as objectively optimal.

---

## 15. Primary metric, secondary metrics, and metric implementation

### 15.1 Primary metric

For churn as a minority positive class, the project can use a precision-recall-based ranking metric as the primary model-family selection criterion.

Before implementation, define precisely whether the project uses:

```text
average precision
trapezoidal PR-AUC
another explicitly named precision-recall summary
```

The final stage must use the same definition across all candidates and all designs.

### 15.2 Secondary ranking metric

ROC-AUC remains useful because it summarizes pairwise ranking discrimination across classes. It should be interpreted alongside PR-focused metrics, not as a replacement for them.

### 15.3 Probability metrics

When scores will be interpreted as risks:

```text
Brier score
log loss
calibration intercept and slope
calibration curve
```

become important secondary criteria.

### 15.4 Threshold-dependent metrics

At a predeclared policy or threshold:

```text
precision
recall
specificity
balanced accuracy
F1
predicted positive rate
expected value or cost
```

The final stage should avoid selecting a model by a threshold-dependent metric at a threshold that was opportunistically optimized separately for every model unless that threshold optimization itself is included in the candidate procedure.

---

## 16. Fold-mean metrics and pooled out-of-fold predictions

Cross-validation can produce two different summaries.

### 16.1 Fold-mean metric

For fold metrics $M^{(k)}$:

$$
\overline{M}
=
\frac{1}{K}
\sum_{k=1}^{K} M^{(k)}.
$$

With repeated CV, average over the repeat-fold metrics.

This is usually the cleanest summary for candidate selection because each score is evaluated on one fold under the same resampling design.

### 16.2 Pooled out-of-fold metric

Collect out-of-fold predictions and compute the metric once:

$$
M_{\mathrm{pooled}}
=
M\left(
\{y_i,\widehat{s}_i^{\mathrm{OOF}}\}_{i=1}^{n}
\right).
$$

Pooled predictions are valuable for:

```text
ROC curves
precision-recall curves
calibration curves
threshold curves
confusion matrices
probability-distribution plots
paired prediction-level comparisons
```

For nonlinear metrics such as ROC-AUC and average precision, a pooled metric may differ from the mean of fold-level metrics. This is not automatically an error. The summaries answer slightly different questions.

For the final comparison, predeclare which summary is primary and retain both for diagnostics.

---

## 17. Why ordinary t-tests on cross-validation folds are not enough

A common mistake is:

```text
run repeated 5-fold CV
obtain 25 or 50 fold scores per model
perform an ordinary paired t-test
treat all fold differences as independent observations
```

This is generally not justified.

The fitted models across folds use overlapping training data. In repeated CV, the same observations are reused across repeats. Therefore, fold-level scores and differences are correlated.

Consequences:

```text
naive standard errors can be too small
p-values can be misleading
confidence intervals can be too narrow
a small difference may appear more certain than it really is
```

This does not mean repeated-CV scores are useless. They are highly informative descriptive evidence. It means formal inference must account for the resampling design or be described as approximate sensitivity analysis.

---

## 18. Statistical comparison tools for one dataset

The Telco project has one dataset, not a collection of independent benchmark datasets. This affects which tests are appropriate.

### 18.1 Descriptive paired outer-fold comparison

For two candidates $A$ and $B$, on outer split $r,k$, compute:

$$
d_{r,k}
=
M_{A,r,k}
-
M_{B,r,k}.
$$

Report:

```text
mean difference
median difference
standard deviation of differences
quantiles
number of outer splits where A exceeds B
runtime difference
configuration-selection stability
```

This is essential descriptive evidence. It does not automatically create a valid classical confidence interval because the differences are dependent, but it shows the empirical pattern clearly.

### 18.2 Corrected resampled t-test

Nadeau and Bengio proposed a variance correction for repeated holdout and repeated cross-validation comparisons.

Let $m=RK$ be the number of score differences, $\bar d$ the mean difference, and $s_d^2$ the sample variance of the differences. A commonly used corrected standard error has the form:

$$
\widehat{\mathrm{SE}}_{\mathrm{corr}}
=
\sqrt{
\left(
\frac{1}{m}
+
\frac{n_{\mathrm{val}}}{n_{\mathrm{train}}}
\right)
s_d^2
}.
$$

The corresponding statistic is:

$$
t_{\mathrm{corr}}
=
\frac{\bar d}
{
\widehat{\mathrm{SE}}_{\mathrm{corr}}
}.
$$

For K-fold CV, the ratio is approximately:

$$
\frac{n_{\mathrm{val}}}{n_{\mathrm{train}}}
\approx
\frac{1}{K-1}.
$$

This correction is useful as a conventional sensitivity analysis, not as unquestionable ground truth. Its assumptions are approximate, and it does not resolve every dependence issue.

### 18.3 Dietterich's 5x2 CV test

The 5x2 design repeats a random two-fold split five times and swaps training and validation roles within each repeat.

Its purpose is to reduce the dependence problem of ordinary K-fold testing. It is a recognized comparison method for two algorithms.

Limitations for this project:

```text
models train on only half the development data in each evaluation
the design is less aligned with the project's desire to exploit
the available development set efficiently
it compares only two algorithms at a time
it is less natural as the main analysis for a large candidate library
```

It can be reported as a methodological alternative and perhaps a sensitivity analysis for a final close pair, but it is not the leading candidate for the main final workflow.

### 18.4 Bayesian correlated t-test and ROPE

A Bayesian correlated t-test models repeated-CV score differences while incorporating an assumed resampling correlation.

Conceptually, it estimates a posterior distribution for the mean difference $\mu_d$. Instead of asking only whether:

$$
\mu_d = 0,
$$

it can quantify the posterior probability that the difference lies in three regions:

$$
\mu_d < -\delta,
\qquad
-\delta \le \mu_d \le \delta,
\qquad
\mu_d > \delta,
$$

where $\delta$ is a region of practical equivalence, often called a ROPE.

For example:

```text
P(XGBoost meaningfully improves AP over random forest) = 0.31
P(the difference is practically negligible) = 0.63
P(random forest meaningfully improves AP) = 0.06
```

This representation is often more decision-relevant than a binary significant/not-significant statement.

The ROPE must be chosen before examining the final comparison results. It should represent a difference too small to justify added model complexity or operational burden. For this project, a PR-based ROPE should be justified through the scale of observed fold variation, operational relevance, and model-complexity trade-offs. The numerical value should not be selected after seeing which candidates are close.

Bayesian correlated methods are model-based. Their conclusions depend on assumptions such as the correlation parameter and prior choice. They are powerful tools for transparent decision support, not automatic truth machines.

### 18.5 Paired bootstrap difference intervals

When candidates have paired predictions on the same validation observations, a paired bootstrap interval can be generated for a metric difference such as average precision.

This is particularly useful for PR-based metrics.

Interpretation:

```text
interval entirely above zero:
    evidence in favor of A on the evaluated prediction set

interval includes zero:
    the data do not clearly separate the candidates on this metric

interval entirely positive but very small:
    possible statistical difference, but practical relevance remains a
    separate question
```

For repeated CV, obey the dependence caution from Section 10.4.

### 18.6 DeLong-style ROC-AUC comparison

DeLong's approach compares correlated ROC-AUCs computed from two score vectors on the same observations.

It is appropriate only when:

```text
ROC-AUC is the metric of interest
two score vectors are available on the same evaluation observations
the goal is a paired ROC-AUC comparison
```

It does not directly test:

```text
average precision
PR-AUC
F1
calibration
expected value
threshold-specific precision or recall
```

Because the Telco project prioritizes precision-recall behavior, DeLong should be secondary at most.

### 18.7 McNemar's test

McNemar's test compares hard prediction errors for two models on the same observations at fixed thresholds.

It can be informative for one explicitly fixed operating policy. It does not evaluate ranking, calibration, probability quality, or threshold-free PR performance. It is therefore supplementary, not a primary candidate-selection test.

It must not be used on the final held-out test set to compare several candidates and then choose a winner. That would turn the test set into a validation dataset.

### 18.8 Permutation tests

A permutation test can assess whether a model performs better than expected under random label association. That tests predictive signal.

A different paired randomization test may compare two score vectors under a null exchangeability assumption. These procedures can be valuable but require careful design and should not be used mechanically. They are optional for the final project rather than a default primary tool.

---

## 19. Multiple comparisons and a broad candidate library

A library of 12 candidates has:

$$
\binom{12}{2}
=
66
$$

pairwise comparisons.

Running 66 unadjusted hypothesis tests and highlighting whichever p-values are below 0.05 would be poor practice. Some apparently significant results would arise by chance.

### 19.1 A better reporting hierarchy

```text
Layer 1:
    report all candidates in a complete descriptive comparison table.

Layer 2:
    use the predefined primary metric and practical tie rule to form
    a performance ordering or practical-equivalence group.

Layer 3:
    report targeted pairwise uncertainty analyses for comparisons that
    are decision-relevant, such as the leading candidate versus each
    close competitor or an interpretable baseline.

Layer 4:
    apply a multiple-testing correction, such as Holm's procedure,
    when frequentist pairwise claims are reported as a family.
```

This does not hide models. Every candidate is reported. It prevents the statistical narrative from becoming an uncontrolled collection of p-values.

### 19.2 Why Friedman and Nemenyi are not the natural primary tests here

Friedman and Nemenyi procedures are commonly used to compare many classifiers across **multiple independent datasets**. Cross-validation folds from one dataset are not equivalent to independent benchmark datasets.

Therefore:

```text
do not treat 15 repeated outer folds as 15 unrelated datasets
do not use Friedman/Nemenyi as the main all-model inference method
for this single Telco dataset
```

They can still be documented as methods for multi-dataset benchmark studies, which is a different research design.

### 19.3 Practical equivalence is more useful than only testing zero

A broad library should not force a false total order.

Suppose:

```text
CatBoost mean AP: 0.671
XGBoost mean AP:  0.670
Random forest:    0.665
Logistic model:   0.662
```

A responsible interpretation might be:

```text
CatBoost and XGBoost are practically tied under the specified design.
Random forest remains competitive but appears somewhat lower.
The logistic model offers a potentially useful accuracy-interpretability
trade-off.
```

The final decision then can consider:

```text
runtime
retraining simplicity
software dependencies
probability calibration
stability
interpretability
deployment constraints
```

This is not abandoning statistical rigor. It is recognizing that a tiny performance gap need not justify a substantially more complex system.

---

## 20. Selection rules beyond "highest mean score"

The project should define its decision rule before the final comparison results are reviewed.

### 20.1 Highest primary metric

```text
Select the candidate with the highest mean outer-fold primary metric.
```

Simple and transparent. It can be unstable when scores are extremely close.

### 20.2 One-standard-error rule

Choose the simplest candidate whose performance is within one standard error of the strongest candidate. The exact meaning of "standard error" must be treated carefully under resampling dependence, but the underlying principle is valuable:

```text
Do not accept substantial complexity for a tiny uncertain gain.
```

### 20.3 ROPE-based practical-equivalence rule

```text
If the posterior or uncertainty analysis indicates that a candidate is
practically equivalent to the apparent leader, treat them as tied.

Among tied candidates, choose according to predeclared secondary criteria.
```

### 20.4 Lexicographic rule

A disciplined hierarchy could be:

```text
1. primary ranking metric
2. practical-equivalence rule
3. calibration quality, if probabilities are operational
4. stability across splits and seeds
5. threshold-policy behavior
6. simplicity and interpretability
7. runtime and dependency burden
```

This avoids arbitrary post-hoc decision making.

### 20.5 Multi-objective or Pareto selection

A candidate is Pareto-dominated if another candidate is at least as good on every relevant criterion and strictly better on at least one.

Potential criteria:

```text
average precision
ROC-AUC
Brier score
runtime
model size
interpretability
stability
expected utility under a fixed policy
```

The Pareto frontier can show that several candidates represent different legitimate trade-offs. One final model must still be chosen, but the report can explain why the selected point was preferred.

---

## 21. The two core workflows for choosing one final model

This section states the two core practical workflows directly.

### 21.1 Workflow A: flat repeated-CV selection

```text
For every candidate family:
    tune hyperparameters with repeated CV on all development data.

Compare:
    each family's selected repeated-CV result.

Choose:
    the strongest family and exact configuration by a predefined rule.

Fit:
    that fixed complete pipeline on all development data.

Evaluate:
    once on test data.
```

**Output:**

```text
selected family
selected exact configuration
development-stage repeated-CV comparison
```

**Main strength:**

```text
directly chooses the final configuration.
```

**Main limitation:**

```text
the winner's repeated-CV score is selected from many estimates and can
be optimistic as an estimate of post-selection performance.
```

### 21.2 Workflow B: per-family nested-CV selection

```text
For every outer split:
    tune every family only inside its outer-training subset;
    evaluate each tuned family on the outer-validation subset.

Compare:
    outer-fold performance of tuned family procedures.

Choose:
    the strongest family or practical tie set.

Then:
    tune only the selected family on all development data;
    choose one final exact configuration;
    fit it on all development data.

Evaluate:
    once on test data.
```

**Output before final tuning:**

```text
comparative evidence for tuned family procedures
selected winning family
hyperparameter-stability information
```

**Output after final tuning:**

```text
one exact deployable configuration
```

**Main strength:**

```text
a stronger family-level comparison because outer validation data did not
influence that family's configuration choice.
```

**Main limitation:**

```text
requires a separate final tuning step for the winning family and costs
substantially more computation.
```

### 21.3 What is not a core model-selection workflow here

An outer loop that chooses one family inside each outer training set evaluates the generalization performance of an **automated family-selection policy**.

That can be an interesting research question. It does not directly return the one final configuration needed for deployment. It is therefore not the main decision design in this project. The project may discuss it as a distinct estimand, but should not confuse it with the two workflows above.

---

## 22. Research evidence and how to interpret it

The literature does not imply that one design is always mandatory.

### 22.1 Why nested CV exists

Varma and Simon demonstrate that using the same cross-validation process both to tune and to report selected model performance can create substantial bias in certain settings. Cawley and Talbot emphasize that model-selection overfitting can materially distort comparisons between learning algorithms.

The key lesson is:

```text
Do not interpret the best observed tuning score as an unbiased estimate
of the selected procedure's future performance.
```

### 22.2 Why flat CV remains common

Flat CV is computationally simpler and directly selects a final configuration. Wainer and Cawley report that flat CV often selected algorithms of practically similar quality to nested CV in their benchmark study, particularly when candidate algorithms had relatively few hyperparameters.

The key lesson is:

```text
Nested CV is stronger for performance estimation after tuning,
but its additional cost is not automatically justified in every
applied selection task.
```

### 22.3 Why the Telco project deserves a stronger comparison stage

This project differs from a minimal applied workflow:

```text
broad candidate library
flexible boosted-tree and neural-network candidates
extensive documentation objective
desire for uncertainty analysis
need to explain not merely what won, but why the selection method is credible
```

Those features make nested-CV and bias-corrected alternatives especially worthwhile to study and possibly compare empirically.

The correct conclusion is not:

```text
Nested CV is always best.
```

It is:

```text
The evaluation design must match the decision being made, the amount of
tuning, the breadth of candidate exploration, the available computation,
and the strength of evidence the report intends to claim.
```

---

## 23. Recommended research plan for this project

Before choosing the final empirical protocol, perform the following documentation and design work.

### 23.1 Freeze a candidate registry template

For every candidate procedure, record:

```text
candidate identifier
model family
preprocessing factory
feature representation
search strategy
search-space definition
trial budget
primary metric
secondary metrics
inner splitter
outer splitter
random-state policy
runtime cap
early-stopping rule
calibration status
threshold status
reason for inclusion
```

### 23.2 Define the comparisons to be studied

The project can study several design families without confusing them:

```text
Design A:
    flat repeated CV across the broad candidate library.

Design B:
    per-family repeated nested CV across the broad candidate library.

Design C:
    bias-corrected flat CV, such as BBC-CV, if the stored OOF structure
    supports it.

Design D:
    an internal validation-holdout design as a transparent reference.
```

The purpose is not to run every possible design indefinitely. It is to understand how the final selected model changes, or does not change, across defensible designs.

### 23.3 Profile computational feasibility before full execution

Before committing to repeated nested CV across all candidates:

```text
run a deterministic smoke test
run a small representative profiling experiment
measure fit time and memory by candidate family
estimate full design runtime from fit counts
set documented resource budgets
```

The profiling workflow must remain training-only and should use the same reusable factories planned for the final stage.

### 23.4 Predefine comparison rules

Before looking at full final comparison results, define:

```text
primary metric
metric implementation
fold summary statistic
candidate registry
search budgets
outer and inner splitters
random seeds
practical-equivalence rule
secondary tie-breakers
comparison methods
reporting format
```

### 23.5 Keep a decision log

The final report should be able to answer:

```text
Why was this candidate library chosen?
Why did each family receive this search budget?
Why was this evaluation design chosen?
How were close differences interpreted?
Why was the selected final model preferred?
What decisions were frozen before test evaluation?
```

---

## 24. Provisional implementation architecture

No final-selection code is created by this note. When coding begins, reusable logic should be implemented through the project's shared modules rather than copied independently into a large notebook.

A likely architecture is:

```text
src/telco_churn/
    preprocessing.py
        reusable preprocessing factories

    models.py
        reusable candidate factories and parameter-space definitions

    evaluation.py
        repeated-CV and nested-CV utilities
        fold result storage
        OOF prediction storage
        metric computation

    comparison.py
        candidate registry
        paired comparison helpers
        bootstrap difference utilities
        practical-equivalence summaries

    visualization.py
        comparison tables and plots
        performance distributions
        equivalence diagrams
        calibration and threshold figures

scripts/
    smoke_test_final_model_selection.py
        deterministic training-only smoke test of registry,
        splitters, factories, metric paths, stored artifacts,
        and representative comparisons

notebooks/
    final_candidate_comparison.py
        transparent orchestration and result interpretation
```

The exact modules can be adapted to the repository's existing structure after inspection. The required workflow is:

```text
inspect current src/ modules and earlier notebooks
decide what is reusable
extend src/ first
create the matching smoke test
run smoke test locally
run the complete notebook
inspect observed outputs
add result-specific notebook interpretation
write the LaTeX report section only after observed results are known
```

---

## 25. Suggested report structure for the eventual final-selection chapter

The final report chapter can be extensive without becoming disorganized.

```text
1. Purpose of final model selection
2. Candidate library and what counts as a candidate procedure
3. Primary metric and secondary decision criteria
4. Why model-selection scores can be optimistic
5. Alternative validation designs
       - internal holdout
       - flat repeated CV
       - per-family nested CV
       - repeated nested CV
       - bias-corrected flat CV
6. Chosen project protocol and rationale
7. Fair search budgets and candidate registry
8. Candidate comparison results
9. Stability and hyperparameter-selection behavior
10. Statistical uncertainty and practical-equivalence analysis
11. Calibration and threshold-policy analysis
12. Final frozen pipeline specification
13. Single held-out test evaluation
14. Final test bootstrap intervals
15. Limitations and external-validity discussion
```

The report should clearly separate:

```text
research/design alternatives
project-specific protocol
observed comparative evidence
final independent test evidence
```

---

## 26. Key rules to preserve

```text
1. The held-out test set is never used to choose between candidates.

2. Every data-learned step lives inside the appropriate training folds.

3. A model name is not a candidate procedure until its preprocessing,
   search policy, and relevant decision components are defined.

4. Flat repeated CV can directly choose both family and configuration,
   but its winning score is selected from many estimates.

5. Per-family nested CV gives stronger evidence for which tuned family
   is preferable, but needs a later full-development-data tuning run
   to choose one exact configuration.

6. Repeated CV fold scores are dependent. Do not treat them as
   independent replicates in a naive t-test.

7. Report all candidates, but do not turn all pairwise p-values into
   the primary decision rule.

8. Small observed metric differences should be interpreted through
   practical relevance, uncertainty, stability, complexity, calibration,
   and operational constraints.

9. Calibration and threshold selection are model-selection decisions.

10. The final test result evaluates one frozen system. It is not an
    opportunity to choose another model.
```

---

## 27. Annotated research sources

The following sources are particularly useful for this methodology module.

### Cross-validation, selection bias, and nested CV

1. **Varma, S. and Simon, R. (2006).** *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics, 7, 91. DOI: 10.1186/1471-2105-7-91.

   Demonstrates the optimism that can arise when cross-validation is used both for tuning and for reporting selected-model performance. Essential source for explaining why nested evaluation exists.

2. **Cawley, G. C. and Talbot, N. L. C. (2010).** *On over-fitting in model selection and subsequent selection bias in performance evaluation.* Journal of Machine Learning Research, 11, 2079-2107.

   Explains model-selection overfitting as a consequence of variance in the selection criterion. Useful for the conceptual distinction between algorithm fitting and model selection.

3. **Arlot, S. and Celisse, A. (2010).** *A survey of cross-validation procedures for model selection.* Statistics Surveys, 4, 40-79. DOI: 10.1214/09-SS054.

   Broad reference on cross-validation design, bias-variance trade-offs, and model-selection purposes.

4. **Wainer, J. and Cawley, G. C. (2021).** *Nested cross-validation when selecting classifiers is overzealous for most practical applications.* Expert Systems with Applications, 182, 115222. Preprint: arXiv:1809.09446.

   Important counterbalance. Shows that flat CV can select practically similar algorithms in many benchmark settings, particularly with relatively few hyperparameters. Supports a nuanced rather than dogmatic treatment of nested CV.

### Bias-corrected CV alternatives

5. **Tibshirani, R. J. and Tibshirani, R. (2009).** *A bias correction for the minimum error rate in cross-validation.* Annals of Applied Statistics, 3, 822-829. Preprint: arXiv:0908.2904.

   Develops a low-cost correction for optimism in the selected minimum CV error.

6. **Tsamardinos, I., Greasidou, E., Tsagris, M., and Borboudakis, G. (2018).** *Bootstrapping the out-of-sample predictions for efficient and accurate cross-validation.* Preprint: arXiv:1708.07180.

   Introduces BBC-CV, which bootstraps selection using stored out-of-sample predictions. Especially relevant for a broad candidate library when full nested CV is expensive.

### Dependence and statistical comparisons

7. **Bengio, Y. and Grandvalet, Y. (2004).** *No unbiased estimator of the variance of K-fold cross-validation.* Journal of Machine Learning Research, 5, 1089-1105.

   Fundamental warning that variance estimation for K-fold CV is difficult. Supports the caution against treating fold-level scores as independent observations.

8. **Nadeau, C. and Bengio, Y. (2003).** *Inference for the generalization error.* Machine Learning, 52, 239-281. DOI: 10.1023/A:1024068626366.

   Provides corrected resampling-based inference ideas, including a variance correction for repeated CV comparisons.

9. **Dietterich, T. G. (1998).** *Approximate statistical tests for comparing supervised classification learning algorithms.* Neural Computation, 10, 1895-1923.

   Introduces and evaluates the 5x2 CV test family. Useful methodological alternative for pairwise algorithm comparison.

10. **Benavoli, A., Corani, G., Demšar, J., and Zaffalon, M. (2017).** *Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis.* Journal of Machine Learning Research, 18, 1-36. Preprint: arXiv:1606.04316.

    Explains Bayesian correlated comparison and practical equivalence through a ROPE. Useful for decision-oriented interpretation of small score differences.

11. **Demšar, J. (2006).** *Statistical comparisons of classifiers over multiple data sets.* Journal of Machine Learning Research, 7, 1-30.

    Important reference for Friedman, Nemenyi, and multiple-dataset comparison. Also clarifies why those methods do not naturally treat repeated folds of one dataset as independent datasets.

### Metric-specific comparisons

12. **DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988).** *Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.* Biometrics, 44, 837-845.

    Standard paired ROC-AUC comparison method. Useful as a secondary ROC analysis, but not a PR-AUC test.

13. **Saito, T. and Rehmsmeier, M. (2015).** *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLOS ONE, 10, e0118432. DOI: 10.1371/journal.pone.0118432.

    Supports the emphasis on precision-recall behavior for the churn problem.

---

## 28. Relationship to existing project notes

This note extends rather than replaces:

```text
evaluation_foundations.md
    finite-sample metrics, leakage discipline, validation/test roles

cross_validation_and_model_selection.md
    core CV mechanics, tuning, repeated CV, nested CV,
    search strategies, and fair tuning

statistical_uncertainty_and_tests.md
    bootstrap intervals, paired comparisons, ROC-AUC and hard-prediction tests

final_model_comparison_plan.md
    project-specific final-stage plan
```

After review, the existing notes should be updated with concise cross-references to this note. The project-specific final plan should then select and operationalize one exact protocol without duplicating this entire theoretical discussion.
