"""Reusable preprocessing pipelines for churn classification models.

The project uses scikit-learn ``Pipeline`` and ``ColumnTransformer`` objects as
the default mechanism for preprocessing. This keeps fitted preprocessing
statistics inside the relevant training fold during cross-validation.

The simple-baseline stage mostly uses dummy and rule-based estimators. The
preprocessors are still introduced here because later learned models will reuse
them.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a one-hot encoder that ignores unseen categories."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_categorical_preprocessing_pipeline() -> Pipeline:
    """Create preprocessing for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )


def make_numeric_preprocessing_pipeline(*, scale_numeric: bool) -> Pipeline:
    """Create preprocessing for numeric features."""
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    """Create a full preprocessing transformer for the clean modelling table."""
    numeric_pipeline = make_numeric_preprocessing_pipeline(
        scale_numeric=scale_numeric
    )
    categorical_pipeline = make_categorical_preprocessing_pipeline()

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_scaled_preprocessor() -> ColumnTransformer:
    """Create preprocessing for models that require or benefit from scaling."""
    return make_preprocessor(scale_numeric=True)


def make_unscaled_preprocessor() -> ColumnTransformer:
    """Create preprocessing for models that do not require numeric scaling."""
    return make_preprocessor(scale_numeric=False)
