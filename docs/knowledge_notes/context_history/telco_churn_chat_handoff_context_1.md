# Telco Customer Churn Classification: Chat Handoff Context

## Project identity

This project is the Telco Customer Churn binary classification project.

GitHub repository:
https://github.com/shakaarlatief/telco-customer-churn-classification

This is part of a broader Data Projects portfolio folder. The goal is not only to get a good churn model, but to build a professional, portfolio-ready, reusable reference project for classification workflows.

The project should connect practical modelling to Machine Learning concepts while still being written as a standalone technical report. Do not write the report as “course notes” or say “in the slides we learned.” Instead, explain the concepts directly and professionally.

A central purpose of this project is to apply the knowledge obtained from the Machine Learning course in a practical, portfolio-ready setting. The project source folder contains the Machine Learning course slide PDFs, and those slides are used as a knowledge base while building the project. We are deliberately going through the whole course and applying the relevant ideas to this classification problem: supervised learning framing, preprocessing, missing values, outliers, feature transformations, feature selection, train/validation/test discipline, overfitting and underfitting, class imbalance, model families, evaluation metrics, threshold tuning, calibration, and careful model comparison. This means the project is not only a predictive modelling exercise. It is also a structured attempt to preserve, relearn, and operationalize the course material in real Python workflows and professional report writing.

The project is meant to become a reusable reference for future classification projects. Therefore, we intentionally use and document many models, even if some are not ultimately the best model. The goal is to understand and preserve:
- preprocessing choices
- train, validation, and test discipline
- missing values and outlier logic
- feature transformation and feature selection
- class imbalance
- baseline models
- linear models
- probabilistic models
- kNN
- decision trees
- bagging
- random forests
- boosting
- AdaBoost
- gradient boosting
- SVMs
- RBF kernels
- feedforward neural networks / MLPs
- evaluation metrics
- threshold tuning
- calibration
- model comparison

## Current workflow state

The project currently has completed the initial data workflow:

1. Raw data audit and deterministic correction
2. Clean modelling dataset and train-test split
3. Training-set exploratory data analysis

The current report PDF is titled:

“Telco Customer Churn Classification: Data Audit, Splitting, and Training-Set Exploratory Analysis”

The abstract states that the report audits the raw file, validates the target, checks missing and disguised missing values, checks identifier columns, applies deterministic corrections, creates a held-out train-test split, and performs EDA only on the training set. Feature-target patterns are inspected only after splitting so the held-out test set remains reserved for final evaluation.

## Important methodology decisions already made

The full raw file may be inspected only for schema and data-quality issues before splitting. This includes:
- loading the raw file
- checking rows, columns, dtypes
- checking duplicate rows
- checking standard missing values
- checking disguised missing values such as blank strings
- validating the target labels
- checking whether `customerID` is an identifier
- correcting deterministic raw representation issues

Feature-target EDA must not be done before the split. Any patterns that can influence modelling decisions must be inspected only on the training set.

The held-out test set must remain unused until final evaluation. It should not be used for:
- feature-target EDA
- preprocessing design
- feature engineering decisions
- feature selection
- model selection
- hyperparameter tuning
- threshold tuning
- calibration decisions

The raw audit found that `TotalCharges` contained 11 blank strings. These were exactly the rows with `tenure = 0`. The correction was deterministic:
- convert `TotalCharges` to numeric
- set blank `TotalCharges` to `0.0` only where `tenure = 0`

This was not mean imputation, median imputation, model-based imputation, or target-informed correction. It was a representation correction based on the meaning of accumulated charges.

The identifier `customerID` is excluded from modelling because it is unique per row and does not represent a generalizable customer characteristic.

The binary target is:

```text
Churn_binary = 1 if Churn = Yes
Churn_binary = 0 if Churn = No
```

The positive class is churn, because churn is the event of practical interest.

The clean modelling data has:
- 7043 rows
- 19 feature columns
- 1 binary target column
- no missing values after correction

A stratified train-test split was created with:
- test size = 0.20
- random state = 42
- stratified by `Churn_binary`

Train set:
- 5634 rows
- churn rate approximately 26.54%

Test set:
- 1409 rows
- churn rate approximately 26.54%

Saved processed outputs:
- `data/processed/train.csv`
- `data/processed/test.csv`

The next stages should load only `train.csv` for modelling development.

## Existing notebook and report structure

The current notebook structure is:

```text
notebooks/01_raw_data_audit.py
notebooks/01_raw_data_audit.ipynb
notebooks/02_cleaning_and_splitting.py
notebooks/02_cleaning_and_splitting.ipynb
notebooks/03_training_set_eda.py
notebooks/03_training_set_eda.ipynb
```

The project keeps `.py` files as the source workflow files. The `.ipynb` files are also kept for readability, running cells, saved outputs, and sharing results. The notebooks can be regenerated or updated from the `.py` files.

The current LaTeX report structure is:

```text
reports/latex/main.tex
reports/latex/main.pdf
reports/latex/sections/01_raw_data_audit.tex
reports/latex/sections/02_cleaning_and_splitting.tex
reports/latex/sections/03_training_set_eda.tex
```

The report includes a table of contents and uses `\newpage` between major sections for readability.

## Current EDA findings

The EDA uses the training set only.

Target distribution:
- `Churn_binary = 0`: 73.46%
- `Churn_binary = 1`: 26.54%

This imbalance is moderate. Accuracy alone is not enough for evaluation. Later modelling sections should use metrics such as:
- confusion matrix
- precision
- recall
- specificity
- F1-score
- ROC-AUC
- PR-AUC
- threshold analysis
- calibration analysis

Numeric features:
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Numeric feature summary:
- `tenure` mean about 32.49, median 29, max 72
- `MonthlyCharges` mean about 64.93, median 70.50, max 118.75
- `TotalCharges` mean about 2299.33, median 1394.93, max 8684.80

Numeric patterns:
- `tenure` has mass near very low values and also many observations near the upper range.
- `MonthlyCharges` is not symmetric.
- `TotalCharges` is strongly right-skewed because total charges accumulate over time.
- Churners have substantially lower `tenure`.
- Churners tend to have higher `MonthlyCharges`.
- Churners tend to have lower `TotalCharges`, but this is closely related to shorter tenure.
- The scatter matrix shows churn-related patterns but no clean pairwise numeric separation.
- The clearest structural pattern is the triangular relationship between `tenure` and `TotalCharges`.
- Pearson correlation with churn:
  - `tenure`: about -0.35
  - `MonthlyCharges`: about 0.20
  - `TotalCharges`: about -0.19
- `tenure` and `TotalCharges` are strongly correlated, about 0.83.
- `MonthlyCharges` and `TotalCharges` are also positively correlated, about 0.65.

Categorical patterns:
- Important categorical churn-rate differences occur for:
  - `Contract`
  - `InternetService`
  - `PaymentMethod`
  - `OnlineSecurity`
  - `TechSupport`
  - `PaperlessBilling`
  - `SeniorCitizen`

Selected churn rates:
- Month-to-month contract: 42.75%
- One-year contract: 11.08%
- Two-year contract: 2.87%
- Fiber optic internet: 42.09%
- DSL: 18.69%
- No internet service: 7.25%
- Electronic check: 45.74%
- Paperless billing Yes: 33.80%
- Paperless billing No: 16.02%
- SeniorCitizen 1: 41.09%
- SeniorCitizen 0: 23.70%

Important interpretation:
These are associations in the training data, not causal effects. They are useful for understanding the data, designing baselines, and interpreting later models.

`No internet service` and `No phone service` are structural categories, not missing values. They should be preserved by preprocessing rather than treated as missing.

Some variables show weak marginal separation, such as `gender` and `PhoneService`. This does not automatically mean they should be removed, because they may still contribute through interactions or in combination with other features.

## Figure standards established

We agreed to use centralized figure style constants as a default baseline, but not as rigid rules. The workflow is:

1. Start with centralized report-style defaults.
2. Generate the figures.
3. Inspect each figure individually.
4. Override specific settings only where needed.
5. Keep overrides local and intentional.

The figure typography baseline includes larger, report-readable text for:
- main titles
- subplot titles
- axis labels
- tick labels
- legends
- heatmap annotations
- colorbar labels

Important principle:
The centralized figure standard prevents random inconsistent styling, but each figure must still be judged individually. Better readability is preferred unless it makes the figure visually oversized or cramped.

For categorical grid figures, we split large figures into two parts for readability:
- categorical feature distributions, Part I
- categorical feature distributions, Part II
- categorical feature churn rates, Part I
- categorical feature churn rates, Part II

Both categorical churn-rate figures are currently kept in the main report because both support interpretation. Part II is especially important because it contains strong predictors such as `Contract`, `PaymentMethod`, `OnlineSecurity`, and `TechSupport`.

The numeric correlation matrix was included smaller in LaTeX because it has only four variables and does not need full page width.

## Current generated report outputs

Current report figures include:
- `reports/figures/training_numeric_feature_distributions.png`
- `reports/figures/training_numeric_feature_boxplots_by_churn.png`
- `reports/figures/training_numeric_scatter_matrix_by_churn.png`
- `reports/figures/training_numeric_correlation_matrix.png`
- `reports/figures/training_categorical_feature_frequencies_part_1.png`
- `reports/figures/training_categorical_feature_frequencies_part_2.png`
- `reports/figures/training_categorical_feature_churn_rates_part_1.png`
- `reports/figures/training_categorical_feature_churn_rates_part_2.png`

Current report tables include:
- `reports/tables/training_numeric_summary.csv`
- `reports/tables/training_numeric_correlation_matrix.csv`
- `reports/tables/training_categorical_frequency_summary.csv`
- `reports/tables/training_categorical_churn_summary.csv`
- `reports/tables/training_selected_categorical_churn_summary.csv`

## LaTeX and local compilation setup

The local LaTeX compiler is TinyTeX / TeX Live 2024, not MiKTeX.

`pdflatex` path was:

```text
C:\Users\shaka\AppData\Roaming\TinyTeX\bin\windows\pdflatex.exe
```

`tlmgr.bat` exists at:

```text
C:\Users\shaka\AppData\Roaming\TinyTeX\bin\windows\tlmgr.bat
```

Because local TeX Live is 2024 and the remote repository was newer, the repository was set to the frozen TeX Live 2024 archive:

```bash
/c/Users/shaka/AppData/Roaming/TinyTeX/bin/windows/tlmgr.bat option repository https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/tlnet-final
```

Packages installed:
- `enumitem`
- `microtype`

To install missing LaTeX packages, use:

```bash
/c/Users/shaka/AppData/Roaming/TinyTeX/bin/windows/tlmgr.bat install PACKAGE_NAME
```

Do not use MiKTeX Console for this project unless fully switching TeX distributions later. Mixing TinyTeX and MiKTeX causes confusion because the compiler and package manager may belong to different TeX installations.

## Writing style preferences

The report should be professional, portfolio-ready, and technically detailed.

Do:
- explain modelling choices clearly
- explain mathematical or statistical ideas in standalone form
- connect workflow decisions to methodology
- write so the report can be reused as a future reference
- be precise about leakage, train/test discipline, and what was or was not used for modelling decisions
- describe assumptions and limitations
- keep notes deep enough that the user can relearn the method later

Do not:
- write “in the course” or “in the slides” inside report sections
- overclaim causality from associations
- silently remove good explanatory notes when updating code
- make unnecessary layout or modelling changes just because they are possible
- use the test set for anything before final evaluation

The user prefers no em dashes in output.

## External references to include later

Kaggle notebook:
- “Resampling strategies for imbalanced datasets”

Scikit-learn feature selection documentation:
- `VarianceThreshold`
- `SelectKBest`
- `SelectPercentile`
- `chi2`
- `f_classif`
- `mutual_info_classif`
- `RFE`
- `RFECV`
- `SelectFromModel`
- L1-based feature selection

These should be incorporated later when discussing class imbalance and feature selection.

## Important modelling roadmap

The next stage should likely be:

4. Preprocessing and baseline modelling

This stage should build training-only preprocessing pipelines. It should likely cover:
- separating numeric and categorical columns
- one-hot encoding categorical variables
- scaling numeric variables where needed
- preserving structural categories such as `No internet service`
- using `ColumnTransformer`
- using `Pipeline`
- defining baseline classifiers
- choosing validation strategy
- defining evaluation metrics
- avoiding test-set usage

Potential first models:
- majority-class baseline
- simple rule-based baseline inspired by EDA
- logistic regression baseline
- maybe dummy classifier with stratified or prior strategy

After that, continue to:
- logistic regression with regularization
- kNN
- Naive Bayes
- decision trees
- decision stumps
- bagging
- random forests
- AdaBoost
- gradient boosting
- SVMs
- MLPs
- threshold tuning
- calibration
- final model comparison
- final test evaluation only once

Important: use many models for learning and documentation, not only the best model.

## Latest commit

The latest staged update was committed with a message like:

```bash
git commit -m "Refactor data preparation and add training-set EDA report"
```

This commit replaced the earlier `01_data_understanding` style with:
- `01_raw_data_audit`
- `02_cleaning_and_splitting`
- `03_training_set_eda`

and added updated figures, tables, LaTeX sections, and compiled report PDF.
