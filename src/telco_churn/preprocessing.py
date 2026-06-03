"""Reusable preprocessing pipelines for churn classification models.

The project uses scikit-learn ``Pipeline`` and ``ColumnTransformer`` objects as
the default mechanism for preprocessing. This keeps fitted preprocessing
statistics inside the relevant training fold during cross-validation.

The simple-baseline stage mostly uses dummy and rule-based estimators. The
preprocessors are still introduced here because later learned models will reuse
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


def make_dense_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder that ignores unseen categories.

    Some estimators used in the modern boosting section either require dense
    input or are easier to inspect when the transformed matrix is dense. This
    helper mirrors ``make_one_hot_encoder`` but requests dense output from
    scikit-learn versions that support the ``sparse_output`` argument, while
    keeping compatibility with older versions that use ``sparse``.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_dense_categorical_preprocessing_pipeline() -> Pipeline:
    """Create dense one-hot preprocessing for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_dense_one_hot_encoder()),
        ]
    )


def make_dense_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    """Create a dense preprocessing transformer for estimators needing arrays.

    The standard project preprocessor allows sparse one-hot output because many
    linear models and tree ensembles can consume sparse matrices efficiently. A
    few boosting implementations are simpler and more robust when they receive a
    dense numeric matrix. This factory keeps the same train-fold-only imputation
    logic as the standard preprocessor, but it uses a dense one-hot encoder for
    categorical variables.
    """
    numeric_pipeline = make_numeric_preprocessing_pipeline(
        scale_numeric=scale_numeric
    )
    categorical_pipeline = make_dense_categorical_preprocessing_pipeline()

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_dense_unscaled_preprocessor() -> ColumnTransformer:
    """Create dense unscaled preprocessing for tree-based boosting models."""
    return make_dense_preprocessor(scale_numeric=False)


class NativeCategoricalPreprocessor(BaseEstimator, TransformerMixin):
    """Impute features while preserving native categorical columns.

    XGBoost in this project is evaluated through one-hot encoded inputs, but
    LightGBM and CatBoost can work directly with categorical columns. A standard
    ``ColumnTransformer`` is not ideal for that workflow because it can convert
    the data to an anonymous NumPy array and lose the original column names. This
    transformer keeps the modelling table as a pandas ``DataFrame`` with the
    original feature names after fold-internal imputation.

    Numeric features are imputed by their training-fold medians. Categorical
    features are imputed by their training-fold modes. When ``categorical_dtype``
    is true, categorical columns are converted to pandas ``category`` dtype,
    which lets LightGBM identify native categorical predictors. When it is false,
    categorical columns are converted to strings after imputation, which is the
    safer representation for CatBoost when categorical feature names are supplied
    to ``fit``.
    """

    def __init__(self, *, categorical_dtype: bool = True):
        self.categorical_dtype = categorical_dtype

    def fit(self, X: pd.DataFrame, y=None):
        """Fit numeric and categorical imputers on the training fold."""
        X_df = self._validate_input(X)

        self.numeric_imputer_ = SimpleImputer(strategy="median")
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")

        self.numeric_imputer_.fit(X_df[NUMERIC_FEATURES])
        self.categorical_imputer_.fit(X_df[CATEGORICAL_FEATURES])
        self.feature_names_in_ = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return an imputed pandas DataFrame with original feature names."""
        X_df = self._validate_input(X)

        numeric_values = self.numeric_imputer_.transform(X_df[NUMERIC_FEATURES])
        categorical_values = self.categorical_imputer_.transform(
            X_df[CATEGORICAL_FEATURES]
        )

        numeric_df = pd.DataFrame(
            numeric_values,
            columns=NUMERIC_FEATURES,
            index=X_df.index,
        )
        categorical_df = pd.DataFrame(
            categorical_values,
            columns=CATEGORICAL_FEATURES,
            index=X_df.index,
        )

        for column in CATEGORICAL_FEATURES:
            if self.categorical_dtype:
                categorical_df[column] = categorical_df[column].astype("category")
            else:
                categorical_df[column] = categorical_df[column].astype(str)

        output_df = pd.concat([numeric_df, categorical_df], axis=1)
        return output_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        """Validate and order the modelling columns expected by the transformer."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "NativeCategoricalPreprocessor expects a pandas DataFrame so "
                "categorical feature names can be preserved."
            )

        required_columns = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        missing_columns = [column for column in required_columns if column not in X]
        if missing_columns:
            raise ValueError(
                "Input data is missing required modelling columns: "
                f"{missing_columns}"
            )

        return X[required_columns].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names in the original modelling-table order."""
        return np.asarray(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)


def make_native_categorical_preprocessor(
    *,
    categorical_dtype: bool = True,
) -> NativeCategoricalPreprocessor:
    """Create preprocessing that preserves native categorical predictors.

    Parameters
    ----------
    categorical_dtype:
        If true, categorical outputs use pandas ``category`` dtype, which is the
        preferred representation for LightGBM native categorical handling. If
        false, categorical outputs are strings, which is convenient for CatBoost
        when categorical feature names are passed to the model at fit time.
    """
    return NativeCategoricalPreprocessor(categorical_dtype=categorical_dtype)
