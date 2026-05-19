# %%
# 02 Cleaning and Splitting
#
# Purpose:
# This script takes the corrected interim dataset from the data-understanding
# workflow and creates the clean base dataset for supervised classification.
#
# This step performs only global cleaning and splitting decisions that are valid
# before model-specific preprocessing.
#
# It does not perform:
# - one-hot encoding
# - scaling
# - PCA
# - feature selection
# - class imbalance resampling
#
# Those steps belong inside model-specific pipelines later.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:,.4f}".format)

# %%
PROJECT_ROOT = Path.cwd().parent

INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "telco_churn_interim.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Churn"
BINARY_TARGET_COLUMN = "Churn_binary"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

IDENTIFIER_COLUMN = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.20

path_check = pd.DataFrame(
    {
        "item": [
            "current_working_directory",
            "project_root",
            "interim_data_path",
            "interim_data_path_exists",
            "processed_data_dir",
        ],
        "value": [
            str(Path.cwd()),
            str(PROJECT_ROOT),
            str(INTERIM_DATA_PATH),
            INTERIM_DATA_PATH.exists(),
            str(PROCESSED_DATA_DIR),
        ],
    }
)

path_check

# %%
df_interim = pd.read_csv(INTERIM_DATA_PATH)

df_interim.head()

# %%
interim_overview = pd.DataFrame(
    {
        "item": [
            "number_of_rows",
            "number_of_columns",
            "duplicate_rows",
            "total_missing_values",
            "columns_with_missing_values",
        ],
        "value": [
            df_interim.shape[0],
            df_interim.shape[1],
            int(df_interim.duplicated().sum()),
            int(df_interim.isna().sum().sum()),
            int((df_interim.isna().sum() > 0).sum()),
        ],
    }
)

interim_overview

# %%
interim_schema = pd.DataFrame(
    {
        "column": df_interim.columns,
        "dtype": df_interim.dtypes.astype(str).values,
        "count": len(df_interim),
        "missing_count": df_interim.isna().sum().values,
        "missing_percentage": 100 * df_interim.isna().mean().values,
        "unique_values": df_interim.nunique(dropna=False).values,
    }
)

interim_schema

# %%
# Create the binary target used by the supervised classification models.
#
# Convention:
# - Churn = Yes is the positive class and is encoded as 1.
# - Churn = No is the negative class and is encoded as 0.
df_clean = df_interim.copy()

target_mapping = {
    NEGATIVE_CLASS: 0,
    POSITIVE_CLASS: 1,
}

df_clean[BINARY_TARGET_COLUMN] = df_clean[TARGET_COLUMN].map(target_mapping)

target_encoding_check = (
    df_clean[[TARGET_COLUMN, BINARY_TARGET_COLUMN]]
    .drop_duplicates()
    .sort_values(BINARY_TARGET_COLUMN)
    .reset_index(drop=True)
)

target_encoding_check

# %%
target_missing_after_encoding = df_clean[BINARY_TARGET_COLUMN].isna().sum()

target_missing_after_encoding

# %%
# customerID is kept in df_clean for traceability, but it is excluded from the
# modelling feature columns.
modelling_excluded_columns = [
    IDENTIFIER_COLUMN,
    TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
]

feature_columns = [
    column
    for column in df_clean.columns
    if column not in modelling_excluded_columns
]

feature_columns

# %%
# Define semantic feature groups.
#
# These groups describe the clean base dataset. They do not yet determine the
# final preprocessing for every model. Later model-specific pipelines can decide
# whether to one-hot encode, scale, pass through, or otherwise transform these
# features.
numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

binary_categorical_features = [
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

nominal_categorical_features = [
    column
    for column in feature_columns
    if column not in numeric_features + binary_categorical_features
]

feature_group_table = pd.DataFrame(
    {
        "feature_group": [
            "numeric_features",
            "binary_categorical_features",
            "nominal_categorical_features",
        ],
        "count": [
            len(numeric_features),
            len(binary_categorical_features),
            len(nominal_categorical_features),
        ],
        "features": [
            numeric_features,
            binary_categorical_features,
            nominal_categorical_features,
        ],
    }
)

feature_group_table

# %%
# Validate that the feature groups cover all modelling features exactly once.
all_grouped_features = (
    numeric_features
    + binary_categorical_features
    + nominal_categorical_features
)

feature_group_validation = pd.DataFrame(
    {
        "check": [
            "number_of_feature_columns",
            "number_of_grouped_features",
            "features_missing_from_groups",
            "grouped_features_not_in_feature_columns",
            "duplicate_grouped_features",
        ],
        "value": [
            len(feature_columns),
            len(all_grouped_features),
            sorted(set(feature_columns) - set(all_grouped_features)),
            sorted(set(all_grouped_features) - set(feature_columns)),
            sorted(
                [
                    feature
                    for feature in set(all_grouped_features)
                    if all_grouped_features.count(feature) > 1
                ]
            ),
        ],
    }
)

feature_group_validation

# %%
# Check whether duplicate rows appear after excluding the unique identifier.
#
# Exact duplicate rows in the full raw/interim data were absent. However, once
# customerID is removed, duplicate feature-target combinations may exist. This is
# not automatically a data-quality problem: multiple customers can genuinely
# share the same observed characteristics and churn label.
duplicates_without_id = df_clean.drop(columns=[IDENTIFIER_COLUMN]).duplicated().sum()

pd.DataFrame(
    {
        "item": [
            "duplicates_full_clean_data",
            "duplicates_after_dropping_customer_id",
        ],
        "count": [
            int(df_clean.duplicated().sum()),
            int(duplicates_without_id),
        ],
    }
)

# %%
# Final missing-value check for the clean base dataset.
missing_summary_clean = (
    df_clean[feature_columns + [BINARY_TARGET_COLUMN]]
    .isna()
    .sum()
    .rename("missing_count")
    .reset_index()
    .rename(columns={"index": "column"})
)

missing_summary_clean["missing_percentage"] = (
    100 * missing_summary_clean["missing_count"] / len(df_clean)
)

missing_summary_clean = missing_summary_clean.sort_values(
    "missing_count",
    ascending=False,
).reset_index(drop=True)

missing_summary_clean

# %%
# Target distribution before splitting.
target_distribution_clean = (
    df_clean[BINARY_TARGET_COLUMN]
    .value_counts(dropna=False)
    .rename_axis(BINARY_TARGET_COLUMN)
    .reset_index(name="count")
)

target_distribution_clean["percentage"] = (
    100
    * target_distribution_clean["count"]
    / target_distribution_clean["count"].sum()
)

target_distribution_clean

# %%
# Create the modelling dataframe.
#
# customerID is excluded from the model features, but kept in a separate
# traceability dataframe if needed for later inspection.
df_model = df_clean[feature_columns + [BINARY_TARGET_COLUMN]].copy()

id_trace = df_clean[[IDENTIFIER_COLUMN]].copy()

df_model.head()

# %%
df_model.shape

# %%
# Stratified train/test split.
#
# The test set is held out for final evaluation. During modelling, model
# selection and hyperparameter tuning should use only the training data, for
# example with a validation split or cross-validation inside the training set.
train_df, test_df = train_test_split(
    df_model,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_model[BINARY_TARGET_COLUMN],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

split_overview = pd.DataFrame(
    {
        "split": ["train", "test"],
        "rows": [len(train_df), len(test_df)],
        "percentage": [
            100 * len(train_df) / len(df_model),
            100 * len(test_df) / len(df_model),
        ],
    }
)

split_overview

# %%
def target_distribution_by_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Return target counts and percentages by split."""
    output_tables = []

    for split_name, split_data in [
        ("train", train_data),
        ("test", test_data),
    ]:
        table = (
            split_data[target_column]
            .value_counts(dropna=False)
            .rename_axis(target_column)
            .reset_index(name="count")
        )

        table.insert(0, "split", split_name)
        table["percentage"] = 100 * table["count"] / table["count"].sum()

        output_tables.append(table)

    return pd.concat(output_tables, ignore_index=True)


split_target_distribution = target_distribution_by_split(
    train_df,
    test_df,
    BINARY_TARGET_COLUMN,
)

split_target_distribution

# %%
# Save processed train/test files locally.
#
# These files are not committed to Git because data/processed/ is ignored.
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

train_df.to_csv(TRAIN_DATA_PATH, index=False)
test_df.to_csv(TEST_DATA_PATH, index=False)

save_check = pd.DataFrame(
    {
        "item": [
            "train_data_path",
            "test_data_path",
            "train_data_path_exists",
            "test_data_path_exists",
            "train_rows",
            "test_rows",
        ],
        "value": [
            str(TRAIN_DATA_PATH),
            str(TEST_DATA_PATH),
            TRAIN_DATA_PATH.exists(),
            TEST_DATA_PATH.exists(),
            len(train_df),
            len(test_df),
        ],
    }
)

save_check