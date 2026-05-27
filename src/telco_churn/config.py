"""Project-wide configuration for the Telco Customer Churn project.

This module centralizes constants reused across notebooks and source modules.
The constants describe the clean modelling table produced by the earlier
data-preparation stages.
"""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"

TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

TARGET_COLUMN: str = "Churn_binary"
ORIGINAL_TARGET_COLUMN: str = "Churn"

POSITIVE_CLASS: int = 1
NEGATIVE_CLASS: int = 0

POSITIVE_CLASS_NAME: str = "Churn"
NEGATIVE_CLASS_NAME: str = "No churn"

NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_CATEGORICAL_FEATURES: list[str] = [
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

NOMINAL_CATEGORICAL_FEATURES: list[str] = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

CATEGORICAL_FEATURES: list[str] = (
    BINARY_CATEGORICAL_FEATURES + NOMINAL_CATEGORICAL_FEATURES
)

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

IDENTIFIER_COLUMNS: list[str] = ["customerID"]
NON_FEATURE_COLUMNS: list[str] = [
    *IDENTIFIER_COLUMNS,
    ORIGINAL_TARGET_COLUMN,
    TARGET_COLUMN,
]

CV_N_SPLITS: int = 5
CV_SHUFFLE: bool = True

SIMPLE_BASELINE_COMPARISON_PATH: Path = (
    TABLES_DIR / "simple_baseline_model_comparison.csv"
)
SIMPLE_BASELINE_CONFUSION_MATRIX_PATH: Path = (
    TABLES_DIR / "simple_baseline_confusion_matrices.csv"
)
