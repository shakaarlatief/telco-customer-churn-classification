"""Data-loading helpers for the Telco Customer Churn project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from telco_churn.config import (
    ALL_FEATURES,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with a clear error message when it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run the earlier data-preparation workflow stages first."
        )

    return pd.read_csv(path)


def load_train_data() -> pd.DataFrame:
    """Load the processed training dataset.

    This is the default dataset for all development-stage notebooks. The
    held-out test data should remain unused until final evaluation.
    """
    return load_csv(TRAIN_DATA_PATH)


def load_test_data() -> pd.DataFrame:
    """Load the processed held-out test dataset.

    This function exists for the final evaluation stage. Ordinary development
    notebooks should not call it.
    """
    return load_csv(TEST_DATA_PATH)


def split_features_target(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = ALL_FEATURES,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a modelling dataframe into features ``X`` and target ``y``."""
    feature_columns = list(feature_columns)
    required_columns = feature_columns + [target_column]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(
            "The dataframe is missing required modelling columns: "
            f"{missing_columns}"
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return X, y
