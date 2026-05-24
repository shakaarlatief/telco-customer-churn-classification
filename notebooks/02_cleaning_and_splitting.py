# %% [markdown]
# # 02 Cleaning and Splitting
#
# ## Purpose
#
# This notebook takes the corrected interim dataset from `01_raw_data_audit`
# and creates the clean base dataset for supervised binary classification.
#
# The goal of this stage is deliberately narrow:
#
# - load the corrected interim dataset;
# - create the binary modelling target;
# - define the modelling feature set;
# - exclude identifiers and non-modelling target columns;
# - define semantic feature groups;
# - validate that all modelling features are assigned to exactly one group;
# - create a stratified train-test split;
# - save the training and test datasets.
#
# This file does **not** perform exploratory feature-target analysis, one-hot
# encoding, scaling, imputation, PCA, feature selection, class-imbalance
# resampling, model training, or threshold tuning.

# %% [markdown]
# ## Why the split happens before EDA and modelling
#
# The held-out test set is meant to approximate future unseen data. Therefore,
# it should not influence feature engineering, preprocessing choices, model
# selection, hyperparameter tuning, or threshold selection.
#
# After this notebook creates `train.csv` and `test.csv`, all target-based EDA
# and all modelling decisions should use only the training set. The test set is
# kept aside until final evaluation.
#
# The only steps performed before splitting are deterministic data-quality and
# dataset-construction steps. These are necessary to build a valid modelling
# table and do not use feature-target patterns or model performance.

# %%
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:,.4f}".format)

# %% [markdown]
# ## Configuration
#
# The project root is detected by searching upward from the current working
# directory. This makes the notebook executable from either the repository root
# or the `notebooks/` folder.

# %%
def find_project_root(start: Path | None = None) -> Path:
    """Return the project root by searching upward for project marker files.

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

INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "telco_churn_interim.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

TARGET_COLUMN = "Churn"
BINARY_TARGET_COLUMN = "Churn_binary"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

IDENTIFIER_COLUMN = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# %%
path_check = pd.DataFrame(
    {
        "item": [
            "current_working_directory",
            "project_root",
            "interim_data_path",
            "interim_data_path_exists",
            "processed_data_dir",
            "train_data_path",
            "test_data_path",
        ],
        "value": [
            str(Path.cwd()),
            str(PROJECT_ROOT),
            str(INTERIM_DATA_PATH),
            INTERIM_DATA_PATH.exists(),
            str(PROCESSED_DATA_DIR),
            str(TRAIN_DATA_PATH),
            str(TEST_DATA_PATH),
        ],
    }
)

path_check

# %% [markdown]
# ## Load interim dataset
#
# The interim dataset is produced by `01_raw_data_audit`. It preserves the raw
# columns but corrects the representation of `TotalCharges`.

# %%
if not INTERIM_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Interim dataset not found at: {INTERIM_DATA_PATH}\n"
        "Run 01_raw_data_audit.py or 01_raw_data_audit.ipynb first."
    )

df_interim = pd.read_csv(INTERIM_DATA_PATH)

# %%
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

# %% [markdown]
# ## Create binary target
#
# The supervised learning target is `Churn_binary`.
#
# The positive class is churn:
#
# \[
# \texttt{Churn = Yes} \rightarrow \texttt{Churn\_binary = 1}.
# \]
#
# The negative class is non-churn:
#
# \[
# \texttt{Churn = No} \rightarrow \texttt{Churn\_binary = 0}.
# \]

# %%
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
target_missing_after_encoding = int(df_clean[BINARY_TARGET_COLUMN].isna().sum())

target_encoding_summary = pd.DataFrame(
    {
        "item": [
            "missing_values_after_binary_encoding",
            "observed_binary_target_values",
        ],
        "value": [
            target_missing_after_encoding,
            sorted(df_clean[BINARY_TARGET_COLUMN].dropna().unique().tolist()),
        ],
    }
)

target_encoding_summary

# %%
if target_missing_after_encoding != 0:
    raise ValueError(
        "The binary target contains missing values after encoding. Check the "
        "raw target labels before splitting."
    )

# %% [markdown]
# ## Define modelling feature set
#
# `customerID` is excluded because it is a unique identifier. The original
# string target `Churn` is excluded because the model target is the numeric
# binary column `Churn_binary`.
#
# The resulting modelling table contains 19 feature columns and one target
# column.

# %%
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
df_model = df_clean[feature_columns + [BINARY_TARGET_COLUMN]].copy()

# A separate traceability table is kept in memory. It is not used for modelling.
id_trace = df_clean[[IDENTIFIER_COLUMN]].copy()

model_table_overview = pd.DataFrame(
    {
        "item": [
            "model_table_rows",
            "model_table_columns",
            "number_of_feature_columns",
            "target_column",
            "identifier_excluded_from_features",
            "original_string_target_excluded_from_features",
        ],
        "value": [
            df_model.shape[0],
            df_model.shape[1],
            len(feature_columns),
            BINARY_TARGET_COLUMN,
            IDENTIFIER_COLUMN not in feature_columns,
            TARGET_COLUMN not in feature_columns,
        ],
    }
)

model_table_overview

# %%
df_model.head()

# %% [markdown]
# ## Define semantic feature groups
#
# These groups describe the clean base dataset. They do not yet determine the
# final preprocessing for every model.
#
# Later model-specific pipelines can decide whether to one-hot encode, scale,
# pass through, or otherwise transform these features. For example:
#
# - linear models, kNN, SVMs, and neural networks usually require scaling after
#   encoding;
# - tree-based models need categorical encoding in scikit-learn but do not
#   require scaling;
# - Naive Bayes variants may require different representations depending on the
#   assumed feature distribution.

# %%
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
if len(feature_columns) != len(all_grouped_features):
    raise ValueError("Feature group validation failed: feature counts do not match.")

if set(feature_columns) != set(all_grouped_features):
    raise ValueError("Feature group validation failed: grouped features do not match.")

if len(all_grouped_features) != len(set(all_grouped_features)):
    raise ValueError("Feature group validation failed: duplicate grouped features found.")

# %% [markdown]
# ## Final checks before splitting
#
# The final modelling table is checked for missing values and duplicate rows.
#
# Duplicate feature-target combinations after removing `customerID` are not
# automatically a data-quality problem. Multiple customers can genuinely share
# the same observed characteristics and churn label.

# %%
missing_summary_model = (
    df_model
    .isna()
    .sum()
    .rename("missing_count")
    .reset_index()
    .rename(columns={"index": "column"})
)

missing_summary_model["missing_percentage"] = (
    100 * missing_summary_model["missing_count"] / len(df_model)
)

missing_summary_model = missing_summary_model.sort_values(
    "missing_count",
    ascending=False,
).reset_index(drop=True)

missing_summary_model

# %%
duplicate_summary = pd.DataFrame(
    {
        "item": [
            "duplicates_full_interim_data",
            "duplicates_model_table_feature_target_rows",
        ],
        "count": [
            int(df_interim.duplicated().sum()),
            int(df_model.duplicated().sum()),
        ],
    }
)

duplicate_summary

# %% [markdown]
# ## Target distribution before splitting
#
# The target distribution is checked to justify stratified splitting. Since the
# positive class is a minority class, stratification helps preserve the churn
# proportion in both train and test sets.

# %%
target_distribution_clean = (
    df_model[BINARY_TARGET_COLUMN]
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

# %% [markdown]
# ## Stratified train-test split
#
# The test set is held out for final evaluation. After this point, target-based
# EDA, feature engineering, preprocessing choices, model selection, and
# hyperparameter tuning should use only the training set.
#
# The split uses:
#
# - `test_size = 0.20`;
# - `random_state = 42`;
# - stratification by `Churn_binary`.

# %%
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

# %% [markdown]
# ## Save processed train and test datasets
#
# The processed train and test files are saved locally. They should usually not
# be committed to Git if the project keeps data files out of version control.
#
# The next workflow step should load only `train.csv` for training-set EDA.

# %%
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
            "train_missing_values",
            "test_missing_values",
        ],
        "value": [
            str(TRAIN_DATA_PATH),
            str(TEST_DATA_PATH),
            TRAIN_DATA_PATH.exists(),
            TEST_DATA_PATH.exists(),
            len(train_df),
            len(test_df),
            int(train_df.isna().sum().sum()),
            int(test_df.isna().sum().sum()),
        ],
    }
)

save_check
