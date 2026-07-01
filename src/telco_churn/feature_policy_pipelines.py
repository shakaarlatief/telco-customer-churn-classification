"""Pipeline adapters that combine feature policies with model representations.

The final-comparison registry treats a feature policy as part of the candidate
procedure.  A policy must therefore run inside the fitted scikit-learn pipeline,
before its representation-specific imputation, encoding, scaling, or native
categorical conversion is learned.  This module provides the adapter layer between
``FeaturePolicyTransformer`` and the representations needed by the core candidate
families.

The adapters deliberately leave the historical preprocessing factories unchanged.
Those factories document and reproduce the earlier individual model workflows.  The
functions here are dedicated to the resumable final-comparison system, whose feature
schemas may contain engineered numeric and categorical columns in addition to the
original raw table.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from telco_churn.feature_policies import (
    FEATURE_POLICY_RAW,
    FeaturePolicyId,
    FeaturePolicyTransformer,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
    validate_feature_policy_id,
)


FeatureRepresentation = Literal[
    "sparse_scaled_one_hot",
    "sparse_unscaled_one_hot",
    "dense_scaled_one_hot",
    "dense_unscaled_one_hot",
    "native_categorical_dtype",
    "native_categorical_string",
]

REPRESENTATION_SPARSE_SCALED: Final[FeatureRepresentation] = "sparse_scaled_one_hot"
REPRESENTATION_SPARSE_UNSCALED: Final[FeatureRepresentation] = "sparse_unscaled_one_hot"
REPRESENTATION_DENSE_SCALED: Final[FeatureRepresentation] = "dense_scaled_one_hot"
REPRESENTATION_DENSE_UNSCALED: Final[FeatureRepresentation] = "dense_unscaled_one_hot"
REPRESENTATION_NATIVE_CATEGORICAL_DTYPE: Final[FeatureRepresentation] = (
    "native_categorical_dtype"
)
REPRESENTATION_NATIVE_CATEGORICAL_STRING: Final[FeatureRepresentation] = (
    "native_categorical_string"
)


class FeaturePolicyPipelineError(ValueError):
    """Raised when a feature-policy representation is invalid or inconsistent."""


def _make_one_hot_encoder(*, dense: bool) -> OneHotEncoder:
    """Create a version-compatible encoder with a requested sparsity contract."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def make_feature_policy_one_hot_preprocessor(
    policy_id: FeaturePolicyId,
    *,
    scale_numeric: bool,
    dense: bool,
) -> ColumnTransformer:
    """Create representation preprocessing for one fixed post-policy schema.

    The column names are obtained from the declared policy contract, not inferred
    from an arbitrary validation frame.  Consequently, a cloned pipeline has the
    same schema definition in every CV fold while imputation, scaling, and one-hot
    categories are still learned exclusively from the fold's training rows.
    """
    policy_id = validate_feature_policy_id(policy_id)
    numeric_features = feature_policy_numeric_features(policy_id)
    categorical_features = feature_policy_categorical_features(policy_id)

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder(dense=dense)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


class FeaturePolicyNativeCategoricalPreprocessor(BaseEstimator, TransformerMixin):
    """Impute a policy-expanded table while preserving native categorical columns.

    LightGBM and CatBoost receive DataFrames rather than anonymous encoded arrays.
    The preceding ``FeaturePolicyTransformer`` has already created any engineered
    columns.  This transformer then performs fold-local numerical median and
    categorical mode imputation on the exact feature-policy schema and returns the
    categorical block either as pandas ``category`` dtype or as strings.
    """

    def __init__(
        self,
        *,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        categorical_dtype: bool,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.categorical_dtype = categorical_dtype

    def fit(self, X: pd.DataFrame, y=None):
        """Fit fold-local native-representation imputers."""
        X_frame = self._validate_input(X)
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
        self.numeric_imputer_.fit(X_frame.loc[:, list(self.numeric_features)])
        self.categorical_imputer_.fit(X_frame.loc[:, list(self.categorical_features)])
        self.feature_names_in_ = np.asarray(
            [*self.numeric_features, *self.categorical_features], dtype=object
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return an imputed DataFrame that preserves the declared feature schema."""
        check_is_fitted(self, ("numeric_imputer_", "categorical_imputer_"))
        X_frame = self._validate_input(X)

        numeric = pd.DataFrame(
            self.numeric_imputer_.transform(X_frame.loc[:, list(self.numeric_features)]),
            columns=self.numeric_features,
            index=X_frame.index,
        )
        categorical = pd.DataFrame(
            self.categorical_imputer_.transform(
                X_frame.loc[:, list(self.categorical_features)]
            ),
            columns=self.categorical_features,
            index=X_frame.index,
        )
        for column in self.categorical_features:
            if self.categorical_dtype:
                categorical[column] = categorical[column].astype("category")
            else:
                categorical[column] = categorical[column].astype(str)

        return pd.concat([numeric, categorical], axis=1).loc[
            :, [*self.numeric_features, *self.categorical_features]
        ]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the fixed native-column output schema."""
        return np.asarray([*self.numeric_features, *self.categorical_features], dtype=object)

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate that a feature-policy transformer produced the declared columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeaturePolicyNativeCategoricalPreprocessor expects a pandas DataFrame."
            )
        expected = [*self.numeric_features, *self.categorical_features]
        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise FeaturePolicyPipelineError(
                "Feature-policy output is missing declared native categorical columns: "
                f"{missing!r}."
            )
        return X.loc[:, expected].copy()


def make_feature_policy_native_categorical_preprocessor(
    policy_id: FeaturePolicyId,
    *,
    categorical_dtype: bool,
) -> FeaturePolicyNativeCategoricalPreprocessor:
    """Create fold-safe native categorical preprocessing for one feature policy."""
    policy_id = validate_feature_policy_id(policy_id)
    return FeaturePolicyNativeCategoricalPreprocessor(
        numeric_features=tuple(feature_policy_numeric_features(policy_id)),
        categorical_features=tuple(feature_policy_categorical_features(policy_id)),
        categorical_dtype=bool(categorical_dtype),
    )


class CloneSafeFeaturePolicyCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe CatBoost wrapper with a policy-dependent categorical column list.

    The existing historical CatBoost wrapper fixes the original raw categorical
    feature list.  F1 and F2 additionally contain the predeclared
    ``f1_contract_x_payment_method`` categorical interaction, so the final-comparison
    pipeline needs a wrapper whose categorical names are immutable constructor
    parameters.  Supplying them during ``fit`` avoids CatBoost constructor parameter
    normalization problems with scikit-learn cloning.
    """

    def __init__(
        self,
        *,
        iterations: int,
        learning_rate: float,
        depth: int,
        l2_leaf_reg: float,
        categorical_features: tuple[str, ...],
        random_state: int,
        thread_count: int = 1,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """Fit CatBoost with policy-specific categorical feature names."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for the final-comparison CatBoost candidate."
            ) from exc

        self.model_ = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            iterations=int(self.iterations),
            learning_rate=float(self.learning_rate),
            depth=int(self.depth),
            l2_leaf_reg=float(self.l2_leaf_reg),
            random_seed=int(self.random_state),
            verbose=False,
            allow_writing_files=False,
            thread_count=int(self.thread_count),
        )
        self.model_.fit(X, y, cat_features=list(self.categorical_features))
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels after policy-aware native preprocessing."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities after policy-aware native preprocessing."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict_proba(X), dtype=float)


def make_feature_policy_classifier_pipeline(
    *,
    policy_id: FeaturePolicyId = FEATURE_POLICY_RAW,
    representation: FeatureRepresentation,
    classifier,
) -> Pipeline:
    """Combine a feature policy, representation adapter, and unfitted classifier.

    All three steps are in one cloneable pipeline.  This is crucial for nested CV:
    the feature-policy fallback values, representation-level imputation and scaling,
    category vocabulary, and estimator parameters are fitted only from each relevant
    training partition.
    """
    policy_id = validate_feature_policy_id(policy_id)

    if representation == REPRESENTATION_SPARSE_SCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=True, dense=False
        )
    elif representation == REPRESENTATION_SPARSE_UNSCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=False, dense=False
        )
    elif representation == REPRESENTATION_DENSE_SCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=True, dense=True
        )
    elif representation == REPRESENTATION_DENSE_UNSCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=False, dense=True
        )
    elif representation == REPRESENTATION_NATIVE_CATEGORICAL_DTYPE:
        preprocessor = make_feature_policy_native_categorical_preprocessor(
            policy_id, categorical_dtype=True
        )
    elif representation == REPRESENTATION_NATIVE_CATEGORICAL_STRING:
        preprocessor = make_feature_policy_native_categorical_preprocessor(
            policy_id, categorical_dtype=False
        )
    else:
        raise FeaturePolicyPipelineError(
            f"Unknown feature representation {representation!r}."
        )

    return Pipeline(
        steps=[
            ("feature_policy", FeaturePolicyTransformer(policy_id=policy_id)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
