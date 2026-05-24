# %% [markdown]
# # 01 Raw Data Audit
#
# ## Purpose
#
# This notebook performs the first raw-data audit for the Telco Customer Churn
# classification project.
#
# The goal of this stage is deliberately limited. Before model training or
# feature-target exploratory analysis, we need to verify that the raw file can
# be interpreted correctly and that the target definition is valid.
#
# This file therefore focuses on:
#
# - loading the raw dataset;
# - checking the schema, data types, row count, and column count;
# - checking standard missing values;
# - checking disguised missing values such as blank strings;
# - checking whether the target labels are valid;
# - checking whether there is a unique identifier column;
# - applying only deterministic data-quality corrections that are justified by
#   raw-data evidence;
# - saving a corrected interim dataset for the next workflow stage.
#
# ## Methodological rule
#
# This file does **not** perform feature-target exploratory analysis. It does not
# make churn-rate plots, target-colored scatter plots, feature correlations with
# the target, feature engineering decisions, feature selection, scaling,
# one-hot encoding, resampling, or model fitting.
#
# Those steps can influence model design and must therefore be performed only
# after the train-test split, using the training set.

# %% [markdown]
# ## Why raw auditing is allowed before splitting
#
# The held-out test set should not be used for model selection, preprocessing
# design, feature engineering, threshold tuning, or exploratory feature-target
# analysis. However, a minimal raw-data audit is still necessary before splitting.
#
# For example, if a column is loaded with the wrong data type because numeric
# values contain blank strings, then the train-test split would preserve a data
# representation problem. Similarly, we need to verify that the target column
# exists, has valid labels, and has no missing values before we can create a
# stratified split.
#
# The distinction used in this project is:
#
# - **Allowed before splitting:** file/schema checks, missing-value checks,
#   disguised missing-value checks, target-label validation, identifier checks,
#   and deterministic corrections of raw representation problems.
# - **Postponed until after splitting:** all feature-target EDA, model-specific
#   preprocessing, feature engineering, feature selection, resampling,
#   hyperparameter tuning, and model comparison.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:,.4f}".format)

# %% [markdown]
# ## Configuration
#
# The project root is detected by searching upward from the current working
# directory. This makes the notebook more robust when it is executed from either
# the repository root or the `notebooks/` folder.

# %%
def find_project_root(start: Path | None = None) -> Path:
    """Return the project root by searching upward for project marker files.

    The notebooks/scripts in this project may be executed either from the
    repository root or from the notebooks directory. Relying on a fixed number
    of parent directories makes the workflow fragile. This helper walks upward
    from the current working directory until it finds files or directories that
    identify the repository root.

    Parameters
    ----------
    start:
        Optional starting directory. If omitted, the current working directory is
        used.

    Returns
    -------
    Path
        Repository root directory.

    Raises
    ------
    FileNotFoundError
        If no plausible project root can be found.
    """
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        has_repo_markers = (
            (candidate / "pyproject.toml").exists()
            or (candidate / "README.md").exists()
        )
        has_project_dirs = (
            (candidate / "data").exists()
            and (candidate / "notebooks").exists()
            and (candidate / "reports").exists()
        )

        if has_repo_markers and has_project_dirs:
            return candidate

    raise FileNotFoundError(
        "Could not find the project root. Run this notebook from inside the "
        "repository, or add project marker files such as pyproject.toml."
    )

# %%
PROJECT_ROOT = find_project_root()

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_PATH = INTERIM_DATA_DIR / "telco_churn_interim.csv"

TARGET_COLUMN = "Churn"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"
IDENTIFIER_COLUMN = "customerID"

# SeniorCitizen is stored as an integer in the raw data, but semantically it is
# a binary categorical feature: 0 = not senior citizen, 1 = senior citizen.
BINARY_CATEGORICAL_COLUMNS = ["SeniorCitizen"]

# %%
path_check = pd.DataFrame(
    {
        "item": [
            "current_working_directory",
            "project_root",
            "data_path",
            "data_path_exists",
            "interim_data_path",
        ],
        "value": [
            str(Path.cwd()),
            str(PROJECT_ROOT),
            str(DATA_PATH),
            DATA_PATH.exists(),
            str(INTERIM_DATA_PATH),
        ],
    }
)

path_check

# %% [markdown]
# ## Load raw dataset
#
# The raw dataframe is kept unchanged as `df_raw`. Any deterministic correction
# is applied later to an interim copy, not to the original raw dataframe.

# %%
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Raw dataset not found at: {DATA_PATH}\n"
        "Download the Telco Customer Churn CSV from Kaggle and place it in "
        "data/raw/ before running this notebook."
    )

df_raw = pd.read_csv(DATA_PATH)

# %%
df_raw.head()

# %% [markdown]
# ## Raw dataset overview
#
# This checks the basic file structure. At this stage we only ask whether the
# file has the expected shape and whether pandas detects obvious missing values
# or exact duplicate rows.

# %%
raw_overview = pd.DataFrame(
    {
        "item": [
            "number_of_rows",
            "number_of_columns",
            "duplicate_rows",
            "total_standard_missing_values",
            "columns_with_standard_missing_values",
        ],
        "value": [
            df_raw.shape[0],
            df_raw.shape[1],
            int(df_raw.duplicated().sum()),
            int(df_raw.isna().sum().sum()),
            int((df_raw.isna().sum() > 0).sum()),
        ],
    }
)

raw_overview

# %%
raw_schema = pd.DataFrame(
    {
        "column": df_raw.columns,
        "raw_dtype": df_raw.dtypes.astype(str).values,
        "count": len(df_raw),
        "standard_missing_count": df_raw.isna().sum().values,
        "standard_missing_percentage": 100 * df_raw.isna().mean().values,
        "unique_values": df_raw.nunique(dropna=False).values,
    }
)

raw_schema

# %% [markdown]
# ## Missing-data decision framework
#
# Before choosing any cleaning strategy, we distinguish two different problems:
#
# 1. **Missing feature values**: at least one input variable `X` is missing.
# 2. **Missing target labels**: the output variable `Y` is missing.
#
# These two cases should not be handled in the same way.
#
# For missing feature values, the central production question is:
#
# > Will missing feature values occur in production?
#
# If missing feature values may occur in production, then the final model
# pipeline must be able to process them. Missing values should remain present in
# validation and test data so that evaluation simulates production. Any imputer
# or preprocessing rule must be fitted on training data only, then applied
# unchanged to validation, test, and production data.
#
# If production data will be clean but the training data contains missing
# values, then the goal is different: we want to simulate clean production. In
# that case, possible strategies include removing incomplete training rows,
# imputing training values, or removing a feature, with the choice made using
# training/validation logic rather than the held-out test set.
#
# For missing target labels, the rules are different. Missing labels in the
# training set may require training only on labelled examples, label imputation,
# or semi-supervised learning. Missing labels in the test set should not simply
# be imputed and treated as known. If test labels are missing, performance
# should be reported with uncertainty, for example using best-case and
# worst-case bounds.
#
# In this Telco dataset, the audit below checks both standard missing values and
# disguised missing values.

# %% [markdown]
# ## Disguised missing values
#
# Standard missing-value checks do not detect blank strings such as `""` or
# `" "`. These are especially important in string-like columns, because they can
# make a conceptually numeric variable appear as an object column.

# %%
string_like_columns_raw = df_raw.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

blank_string_summary = []

for column in string_like_columns_raw:
    blank_count = df_raw[column].astype(str).str.strip().eq("")

    blank_string_summary.append(
        {
            "column": column,
            "raw_dtype": str(df_raw[column].dtype),
            "count": len(df_raw),
            "blank_string_count": int(blank_count.sum()),
            "blank_string_percentage": 100 * blank_count.mean(),
            "unique_values": df_raw[column].nunique(dropna=False),
        }
    )

blank_string_summary = pd.DataFrame(blank_string_summary).sort_values(
    ["blank_string_count", "unique_values"],
    ascending=[False, False],
).reset_index(drop=True)

blank_string_summary

# %% [markdown]
# ## Target-label validation
#
# The target is checked before splitting because the split is stratified by the
# target. This is a task-definition check, not feature-target exploratory
# analysis.
#
# The raw labels should contain only:
#
# - `No`, the negative class;
# - `Yes`, the positive class.

# %%
target_distribution = (
    df_raw[TARGET_COLUMN]
    .value_counts(dropna=False)
    .rename_axis("target_level")
    .reset_index(name="count")
)

target_distribution["percentage"] = (
    100 * target_distribution["count"] / target_distribution["count"].sum()
)

target_distribution

# %%
observed_target_labels = set(df_raw[TARGET_COLUMN].dropna().unique())

target_validity_check = pd.DataFrame(
    {
        "item": [
            "expected_negative_label_present",
            "expected_positive_label_present",
            "missing_target_values",
            "unexpected_target_labels",
        ],
        "value": [
            NEGATIVE_CLASS in observed_target_labels,
            POSITIVE_CLASS in observed_target_labels,
            int(df_raw[TARGET_COLUMN].isna().sum()),
            sorted(observed_target_labels - {NEGATIVE_CLASS, POSITIVE_CLASS}),
        ],
    }
)

target_validity_check

# %% [markdown]
# ## Identifier check
#
# A unique customer identifier should not be used as a predictive feature. It
# identifies rows rather than describing a generalizable customer characteristic.
# The identifier is useful for traceability, but it should be excluded from the
# model feature set in the splitting step.

# %%
identifier_summary = pd.DataFrame(
    {
        "column": df_raw.columns,
        "unique_values": [
            df_raw[column].nunique(dropna=False) for column in df_raw.columns
        ],
        "number_of_rows": len(df_raw),
    }
)

identifier_summary["unique_value_ratio"] = (
    identifier_summary["unique_values"] / identifier_summary["number_of_rows"]
)

identifier_summary["is_unique_identifier_candidate"] = (
    identifier_summary["unique_values"] == identifier_summary["number_of_rows"]
)

identifier_summary = identifier_summary.sort_values(
    ["is_unique_identifier_candidate", "unique_value_ratio"],
    ascending=[False, False],
).reset_index(drop=True)

identifier_summary

# %% [markdown]
# ## Investigate `TotalCharges`
#
# The blank-string audit shows that `TotalCharges` contains blank strings. This
# requires further inspection because `TotalCharges` is conceptually numeric.
#
# The question is whether those blanks are ordinary missing feature values, or
# whether they have a deterministic interpretation.

# %%
total_charges_blank_mask = df_raw["TotalCharges"].astype(str).str.strip().eq("")
tenure_zero_mask = df_raw["tenure"].eq(0)

total_charges_issue_summary = pd.DataFrame(
    {
        "condition": [
            "blank TotalCharges",
            "tenure == 0",
            "blank TotalCharges and tenure == 0",
            "blank TotalCharges and tenure != 0",
            "non-blank TotalCharges and tenure == 0",
        ],
        "count": [
            int(total_charges_blank_mask.sum()),
            int(tenure_zero_mask.sum()),
            int((total_charges_blank_mask & tenure_zero_mask).sum()),
            int((total_charges_blank_mask & ~tenure_zero_mask).sum()),
            int((~total_charges_blank_mask & tenure_zero_mask).sum()),
        ],
    }
)

total_charges_issue_summary

# %%
total_charges_blank_rows = df_raw.loc[
    total_charges_blank_mask,
    [
        IDENTIFIER_COLUMN,
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        TARGET_COLUMN,
    ],
]

total_charges_blank_rows

# %%
total_charges_numeric_probe = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")

total_charges_conversion_summary = pd.DataFrame(
    {
        "item": [
            "standard_missing_before_conversion",
            "missing_after_numeric_conversion",
            "new_missing_created_by_numeric_conversion",
            "blank_string_count",
        ],
        "count": [
            int(df_raw["TotalCharges"].isna().sum()),
            int(total_charges_numeric_probe.isna().sum()),
            int(
                total_charges_numeric_probe.isna().sum()
                - df_raw["TotalCharges"].isna().sum()
            ),
            int(total_charges_blank_mask.sum()),
        ],
    }
)

total_charges_conversion_summary

# %% [markdown]
# ## Interim correction
#
# The evidence above shows:
#
# - `TotalCharges` is conceptually numeric but loaded as a string-like column;
# - the only values that cannot be converted to numeric are blank strings;
# - the blank strings occur exactly when `tenure == 0`;
# - there are no cases where `tenure == 0` and `TotalCharges` is non-blank.
#
# The correction is therefore deterministic: convert `TotalCharges` to numeric
# and set blank `TotalCharges` values to `0.0` for zero-tenure customers.
#
# This is not mean imputation, median imputation, model-based imputation, or a
# target-informed correction. It is a representation correction based on the
# meaning of accumulated charges.

# %%
df_interim = df_raw.copy()

df_interim["TotalCharges"] = pd.to_numeric(df_interim["TotalCharges"], errors="coerce")

df_interim.loc[
    df_interim["tenure"].eq(0) & df_interim["TotalCharges"].isna(),
    "TotalCharges",
] = 0.0

# %%
corrected_schema = pd.DataFrame(
    {
        "column": df_interim.columns,
        "raw_dtype": df_raw.dtypes.astype(str).values,
        "interim_dtype": df_interim.dtypes.astype(str).values,
        "count": len(df_interim),
        "missing_count": df_interim.isna().sum().values,
        "missing_percentage": 100 * df_interim.isna().mean().values,
        "unique_values": df_interim.nunique(dropna=False).values,
    }
)

corrected_schema

# %%
post_correction_check = pd.DataFrame(
    {
        "item": [
            "remaining_total_missing_values",
            "remaining_missing_TotalCharges",
            "TotalCharges_dtype",
            "minimum_TotalCharges",
            "maximum_TotalCharges",
        ],
        "value": [
            int(df_interim.isna().sum().sum()),
            int(df_interim["TotalCharges"].isna().sum()),
            str(df_interim["TotalCharges"].dtype),
            df_interim["TotalCharges"].min(),
            df_interim["TotalCharges"].max(),
        ],
    }
)

post_correction_check

# %% [markdown]
# ## Basic numeric domain-validity checks
#
# Some unusual values are natural observations and should not be removed
# automatically. However, values that violate basic domain constraints are
# different: they may indicate coding errors, measurement errors, or corrupted
# records.
#
# In this dataset, the three numeric variables should be non-negative:
#
# - `tenure` is the number of months a customer has stayed with the company;
# - `MonthlyCharges` is the customer's monthly charge amount;
# - `TotalCharges` is the accumulated charge amount.
#
# A value of zero can be valid. For example, zero tenure occurs for newly
# registered customers, and the corresponding accumulated total charge can be
# zero. Therefore, this audit checks only for negative values. This is not
# statistical outlier detection; statistical outlier analysis is postponed until
# training-set exploratory analysis.

# %%
non_negative_numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

numeric_domain_validity_rows = []

for column in non_negative_numeric_features:
    negative_mask = df_interim[column] < 0

    numeric_domain_validity_rows.append(
        {
            "feature": column,
            "minimum": df_interim[column].min(),
            "maximum": df_interim[column].max(),
            "negative_value_count": int(negative_mask.sum()),
            "negative_value_percentage": 100 * negative_mask.mean(),
        }
    )

numeric_domain_validity_summary = pd.DataFrame(numeric_domain_validity_rows)

numeric_domain_validity_summary

# %% [markdown]
# ## Save interim dataset
#
# The interim dataset preserves the raw columns but corrects the representation
# of `TotalCharges`. The original target column `Churn` remains unchanged.
#
# The next script will create `Churn_binary`, remove `customerID` from the
# modelling feature set, and create the held-out train-test split.

# %%
df_interim.to_csv(INTERIM_DATA_PATH, index=False)

save_check = pd.DataFrame(
    {
        "item": [
            "interim_data_path",
            "interim_data_path_exists",
            "number_of_rows",
            "number_of_columns",
            "total_missing_values",
        ],
        "value": [
            str(INTERIM_DATA_PATH),
            INTERIM_DATA_PATH.exists(),
            df_interim.shape[0],
            df_interim.shape[1],
            int(df_interim.isna().sum().sum()),
        ],
    }
)

save_check
