# Telco Customer Churn Classification

This repository contains an applied machine learning project on binary customer churn classification using the Telco Customer Churn dataset.

The goal of the project is to build a professional, reproducible classification workflow that connects machine learning theory with practical implementation. The project covers data understanding, preprocessing, baseline modelling, model comparison, threshold analysis, interpretation, and reporting.

## Project status

In progress.

## Main objectives

- Formulate customer churn prediction as a binary classification problem.
- Build a clean and reproducible Python workflow.
- Apply proper train, validation, and test methodology.
- Handle missing values, categorical variables, numerical variables, and potential class imbalance.
- Compare simple baseline models, classical machine learning models, tree-based models, and a neural network.
- Evaluate models using appropriate classification metrics.
- Interpret model behaviour and discuss business tradeoffs.
- Produce an extensive LaTeX report explaining the methodology, models, results, and limitations.

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