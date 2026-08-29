# Telco Customer Churn Classification

This repository contains an applied machine learning project on binary customer churn classification using the Telco Customer Churn dataset.

Unlike a typical predictive modelling project whose main objective is to reach the best possible score with the shortest reasonable pipeline, this project is deliberately broad. The dataset is used as a common test bed for systematically applying, comparing, and documenting a wide range of machine learning methods for tabular binary classification.

## Project purpose and philosophy

The primary goal of this project is to create a **comprehensive, reusable reference project for binary classification**.

The emphasis is therefore not only on the final predictive result. It is also on understanding and preserving the full modelling process: what different methods do, how they work mathematically, when they are appropriate, how they differ from one another, what assumptions they make, how they should be evaluated, and how those ideas translate into a correct implementation.

Breadth is intentional. A model or method is not excluded simply because another approach is expected to perform better. Some techniques may be simpler, weaker, partly redundant, or less suitable for this particular dataset, but they can still be valuable to include because the project is designed to:

- apply as many relevant classification models and modelling techniques as reasonably possible;
- compare alternative approaches rather than jumping directly to a single preferred model;
- explain important methods in substantial technical and mathematical detail;
- connect mathematical definitions, loss functions, probability models, optimization ideas, and model assumptions to their practical implementation;
- document preprocessing, feature engineering, feature selection, validation, diagnostics, evaluation, thresholding, calibration, and interpretation as first-class parts of the modelling workflow;
- preserve the reasoning behind modelling decisions, including approaches that are ultimately not selected;
- create material that can be reused and referenced in future data science projects instead of re-deriving the same foundations from scratch.

For that reason, the project is intentionally more extensive than a minimal end-to-end churn solution. The final results still need to be methodologically sound and evaluated on properly held-out data, but **the project itself is also intended to function as a technical reference library and learning record for supervised classification**.

## Project status

In progress.

## Main objectives

- Formulate customer churn prediction rigorously as a supervised binary classification problem.
- Build a clean, reproducible Python workflow with strict train, validation, and test discipline.
- Explore the data carefully before modelling and investigate data quality, missing values, unusual observations, feature distributions, and relationships with the target.
- Study alternative preprocessing strategies for numerical and categorical variables, scaling, transformations, missing-data handling, and potential class imbalance.
- Investigate feature creation and multiple approaches to feature selection.
- Establish simple baselines before introducing more flexible models.
- Apply and compare a broad collection of classification methods, including linear models, k-nearest neighbours, probabilistic classifiers, decision trees, ensemble methods, support vector machines, and neural networks.
- Study important variants within model families, including regularization, tree depth and pruning behaviour, bagging, random forests, boosting, AdaBoost, gradient boosting, linear and nonlinear kernels, and multilayer perceptrons.
- Explain the mathematical foundations of the major models and methods in enough detail that the report can serve as a future technical reference.
- Distinguish model fitting, hyperparameter selection, threshold selection, and final unbiased evaluation.
- Evaluate models using metrics appropriate for binary classification rather than relying on accuracy alone.
- Investigate ROC and precision-recall behaviour, confusion-matrix tradeoffs, probability calibration, and decision-threshold tuning.
- Examine overfitting, underfitting, bias-variance tradeoffs, model complexity, and generalization throughout the workflow.
- Interpret model behaviour and discuss practical churn-related tradeoffs without confusing predictive association with causality.
- Produce an extensive LaTeX report documenting methodology, mathematical foundations, experiments, model comparisons, results, limitations, and conclusions.

## Repository structure

```text
telco-customer-churn-classification/
├── data/       Local data files, not committed to Git
├── docs/       Additional project notes
├── notebooks/  Exploratory and modelling notebooks
├── reports/    LaTeX report, figures, and tables
├── results/    Metrics, predictions, and model artifacts
├── src/        Reusable Python source code
└── README.md
```

## Dataset

Dataset: Telco Customer Churn  
Source: Kaggle  
URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The raw dataset is not committed to this repository. See `data/README.md` for download instructions.

## Environment

This project uses Python 3.11.

Install dependencies with:

```bash
pip install -r requirements.txt
```
