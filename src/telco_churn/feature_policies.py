"""Fold-safe deterministic feature policies for Telco churn classification.

The final comparison treats the input representation as part of a candidate procedure,
not as an informal notebook-side convenience. This module defines three predeclared,
target-free feature policies that operate on the cleaned raw modelling table:

``F0_RAW``
    The original cleaned numerical and categorical feature columns.

``F1_DOMAIN_ENRICHED``
    A compact set of domain-motivated service aggregates, safe charge-to-tenure
    summaries, one selected categorical interaction, and selected interactions whose
    interpretation is clear before model fitting.

``F2_LINEAR_EXPANDED``
    A larger but still bounded systematic expansion intended only for regularized
    linear procedures. It contains the structural F1 features, quadratic numeric
    terms, pairwise numeric products, and numeric-by-nonreference-category products.
    It intentionally does not create every categorical-by-categorical cross-product.

The transformer learns only training-partition imputation values needed to construct
features when a raw input contains missing values. It never uses target values. When it
is placed as the first step of a scikit-learn pipeline, every learned fill value is
therefore estimated from the relevant inner or outer training partition only.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from telco_churn.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


FeaturePolicyId = Literal[
    "F0_RAW",
    "F1_DOMAIN_ENRICHED",
    "F2_LINEAR_EXPANDED",
]

FEATURE_POLICY_RAW: Final[FeaturePolicyId] = "F0_RAW"
FEATURE_POLICY_DOMAIN: Final[FeaturePolicyId] = "F1_DOMAIN_ENRICHED"
FEATURE_POLICY_LINEAR_EXPANDED: Final[FeaturePolicyId] = "F2_LINEAR_EXPANDED"

FEATURE_POLICY_IDS: Final[tuple[FeaturePolicyId, ...]] = (
    FEATURE_POLICY_RAW,
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_LINEAR_EXPANDED,
)


class FeaturePolicyError(ValueError):
    """Raised when a requested feature policy or raw input contract is invalid."""


# The values are part of F2's fixed representation contract. A reference level is
# deliberately omitted from every numeric-by-category interaction family: including
# all levels would make their sum exactly equal to the original numeric predictor.
CATEGORICAL_LEVELS_BY_FEATURE: Final[Mapping[str, tuple[str, ...]]] = {
    "SeniorCitizen": ("0", "1"),
    "gender": ("Female", "Male"),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("No", "Yes"),
    "PaperlessBilling": ("No", "Yes"),
    "MultipleLines": ("No", "No phone service", "Yes"),
    "InternetService": ("No", "DSL", "Fiber optic"),
    "OnlineSecurity": ("No", "No internet service", "Yes"),
    "OnlineBackup": ("No", "No internet service", "Yes"),
    "DeviceProtection": ("No", "No internet service", "Yes"),
    "TechSupport": ("No", "No internet service", "Yes"),
    "StreamingTV": ("No", "No internet service", "Yes"),
    "StreamingMovies": ("No", "No internet service", "Yes"),
    "Contract": ("One year", "Month-to-month", "Two year"),
    "PaymentMethod": (
        "Mailed check",
        "Electronic check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

for _categorical_feature in CATEGORICAL_FEATURES:
    if _categorical_feature not in CATEGORICAL_LEVELS_BY_FEATURE:
        raise RuntimeError(
            "The F2 interaction contract has no declared levels for "
            f"{_categorical_feature!r}."
        )


F1_STRUCTURAL_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "f1_total_subscribed_service_count",
    "f1_protection_support_service_count",
    "f1_streaming_service_count",
    "f1_tenure_squared",
    "f1_log1p_tenure",
    "f1_average_charges_per_tenure",
    "f1_zero_tenure",
)

F1_CURATED_INTERACTION_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "f1_monthlycharges_x_month_to_month",
    "f1_monthlycharges_x_two_year_contract",
    "f1_monthlycharges_x_dsl",
    "f1_monthlycharges_x_fiber_optic",
    "f1_monthlycharges_x_tech_support_yes",
    "f1_monthlycharges_x_online_security_yes",
    "f1_tenure_x_month_to_month",
)

F1_ENGINEERED_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    *F1_STRUCTURAL_NUMERIC_FEATURES,
    *F1_CURATED_INTERACTION_NUMERIC_FEATURES,
)

F1_ENGINEERED_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "f1_contract_x_payment_method",
)


def _slugify(value: str) -> str:
    """Return a stable identifier fragment for a declared categorical level."""
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise FeaturePolicyError(f"Cannot create a feature-name fragment from {value!r}.")
    return result


F2_QUADRATIC_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "f2_tenure_squared",
    "f2_monthlycharges_squared",
    "f2_totalcharges_squared",
)

F2_NUMERIC_PRODUCT_FEATURES: Final[tuple[str, ...]] = (
    "f2_tenure_x_monthlycharges",
    "f2_tenure_x_totalcharges",
    "f2_monthlycharges_x_totalcharges",
)

F2_NUMERIC_BY_CATEGORY_FEATURE_SPECS: Final[
    tuple[tuple[str, str, str, str], ...]
] = tuple(
    (
        numeric_feature,
        categorical_feature,
        level,
        f"f2_{numeric_feature.lower()}_x_{categorical_feature.lower()}_{_slugify(level)}",
    )
    for numeric_feature in NUMERIC_FEATURES
    for categorical_feature in CATEGORICAL_FEATURES
    # The first declared level is the reference level and is intentionally omitted.
    for level in CATEGORICAL_LEVELS_BY_FEATURE[categorical_feature][1:]
)

F2_NUMERIC_BY_CATEGORY_FEATURES: Final[tuple[str, ...]] = tuple(
    specification[3] for specification in F2_NUMERIC_BY_CATEGORY_FEATURE_SPECS
)

F2_ADDITIONAL_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    *F2_QUADRATIC_NUMERIC_FEATURES,
    *F2_NUMERIC_PRODUCT_FEATURES,
    *F2_NUMERIC_BY_CATEGORY_FEATURES,
)


SERVICE_YES_COLUMNS: Final[tuple[str, ...]] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

PROTECTION_SUPPORT_YES_COLUMNS: Final[tuple[str, ...]] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
)

STREAMING_YES_COLUMNS: Final[tuple[str, ...]] = (
    "StreamingTV",
    "StreamingMovies",
)


def validate_feature_policy_id(policy_id: str) -> FeaturePolicyId:
    """Validate and return one declared feature-policy identifier."""
    if policy_id not in FEATURE_POLICY_IDS:
        raise FeaturePolicyError(
            f"Unknown feature policy {policy_id!r}. Expected one of "
            f"{list(FEATURE_POLICY_IDS)!r}."
        )
    return policy_id  # type: ignore[return-value]


def feature_policy_numeric_features(policy_id: str) -> list[str]:
    """Return the fixed numeric columns supplied by one feature policy."""
    policy_id = validate_feature_policy_id(policy_id)
    columns = list(NUMERIC_FEATURES)
    if policy_id in {FEATURE_POLICY_DOMAIN, FEATURE_POLICY_LINEAR_EXPANDED}:
        columns.extend(F1_ENGINEERED_NUMERIC_FEATURES)
    if policy_id == FEATURE_POLICY_LINEAR_EXPANDED:
        # F2 supersedes the F1 curated interactions with its own systematic terms.
        columns = list(NUMERIC_FEATURES) + list(F1_STRUCTURAL_NUMERIC_FEATURES)
        columns.extend(F2_ADDITIONAL_NUMERIC_FEATURES)
    return columns


def feature_policy_categorical_features(policy_id: str) -> list[str]:
    """Return the fixed categorical columns supplied by one feature policy."""
    policy_id = validate_feature_policy_id(policy_id)
    columns = list(CATEGORICAL_FEATURES)
    if policy_id in {FEATURE_POLICY_DOMAIN, FEATURE_POLICY_LINEAR_EXPANDED}:
        columns.extend(F1_ENGINEERED_CATEGORICAL_FEATURES)
    return columns


def feature_policy_output_features(policy_id: str) -> list[str]:
    """Return the complete deterministic output schema for one policy."""
    return feature_policy_numeric_features(policy_id) + feature_policy_categorical_features(
        policy_id
    )


class FeaturePolicyTransformer(BaseEstimator, TransformerMixin):
    """Construct one deterministic feature policy from the raw Telco feature table.

    The transformer is intentionally fitted even though its engineered features are
    target-free. It learns only numerical median and categorical mode values used when
    a future scoring row has missing inputs. This keeps the derived features coherent
    with fold-local imputation and avoids fitting any missing-data statistic on a
    validation or test partition.

    Parameters
    ----------
    policy_id:
        ``F0_RAW`` keeps the raw cleaned schema. ``F1_DOMAIN_ENRICHED`` adds a small,
        predeclared feature set designed for broad use. ``F2_LINEAR_EXPANDED`` adds a
        larger regularized-linear representation and should not be used indiscriminately
        for tree ensembles, RBF SVMs, MLPs, kNN, or Naive Bayes.
    """

    def __init__(self, *, policy_id: FeaturePolicyId = FEATURE_POLICY_RAW):
        self.policy_id = policy_id

    def fit(self, X: pd.DataFrame, y=None):
        """Learn fold-local fallback values required for deterministic transforms."""
        policy_id = validate_feature_policy_id(self.policy_id)
        X_raw = self._validate_raw_input(X)

        numeric_fill_values: dict[str, float] = {}
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(X_raw[column], errors="coerce")
            median = float(values.median(skipna=True))
            numeric_fill_values[column] = median if np.isfinite(median) else 0.0

        categorical_fill_values: dict[str, str] = {}
        for column in CATEGORICAL_FEATURES:
            values = X_raw[column].astype("string").dropna().astype(str)
            if values.empty:
                categorical_fill_values[column] = "__MISSING__"
                continue
            modes = sorted(values.mode().tolist())
            categorical_fill_values[column] = str(modes[0])

        self.policy_id_ = policy_id
        self.numeric_fill_values_ = numeric_fill_values
        self.categorical_fill_values_ = categorical_fill_values
        self.feature_names_in_ = np.asarray(ALL_FEATURES, dtype=object)
        self.n_features_in_ = len(ALL_FEATURES)
        self.feature_names_out_ = np.asarray(
            feature_policy_output_features(policy_id),
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature policy without using targets or external state."""
        check_is_fitted(
            self,
            attributes=(
                "policy_id_",
                "numeric_fill_values_",
                "categorical_fill_values_",
                "feature_names_out_",
            ),
        )
        X_raw = self._validate_raw_input(X)
        output = X_raw.copy()

        if self.policy_id_ == FEATURE_POLICY_RAW:
            return output.loc[:, self.feature_names_out_].copy()

        numeric, categorical = self._calculation_inputs(X_raw)
        structural = self._make_f1_structural_features(numeric, categorical)
        for column, values in structural.items():
            output[column] = values

        output["f1_contract_x_payment_method"] = (
            categorical["Contract"] + "__" + categorical["PaymentMethod"]
        )

        if self.policy_id_ == FEATURE_POLICY_DOMAIN:
            curated = self._make_f1_curated_interactions(numeric, categorical)
            for column, values in curated.items():
                output[column] = values
        elif self.policy_id_ == FEATURE_POLICY_LINEAR_EXPANDED:
            expanded = self._make_f2_linear_expansion(numeric, categorical)
            for column, values in expanded.items():
                output[column] = values
        else:
            raise RuntimeError(f"Unexpected fitted feature policy {self.policy_id_!r}.")

        return output.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the fixed output schema after fitting."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    @staticmethod
    def _validate_raw_input(X: pd.DataFrame) -> pd.DataFrame:
        """Validate and order the raw modelling columns required by every policy."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeaturePolicyTransformer expects a pandas DataFrame with the raw "
                "Telco modelling columns."
            )
        missing_columns = [column for column in ALL_FEATURES if column not in X.columns]
        if missing_columns:
            raise FeaturePolicyError(
                "Input is missing required raw modelling columns: "
                f"{missing_columns!r}."
            )
        return X.loc[:, ALL_FEATURES].copy()

    def _calculation_inputs(
        self,
        X_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return fold-locally completed numeric and categorical calculation tables."""
        numeric = pd.DataFrame(index=X_raw.index)
        for column in NUMERIC_FEATURES:
            numeric[column] = pd.to_numeric(X_raw[column], errors="coerce").fillna(
                self.numeric_fill_values_[column]
            )

        categorical = pd.DataFrame(index=X_raw.index)
        for column in CATEGORICAL_FEATURES:
            categorical[column] = (
                X_raw[column]
                .astype("string")
                .fillna(self.categorical_fill_values_[column])
                .astype(str)
            )

        return numeric, categorical

    @staticmethod
    def _yes_count(categorical: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
        """Count explicit ``Yes`` service selections across a declared feature group."""
        return categorical.loc[:, columns].eq("Yes").sum(axis=1).astype(float)

    def _make_f1_structural_features(
        self,
        numeric: pd.DataFrame,
        categorical: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """Create compact domain aggregates and safe tenure-related transformations."""
        tenure = numeric["tenure"].clip(lower=0.0)
        monthly_charges = numeric["MonthlyCharges"]
        total_charges = numeric["TotalCharges"]
        zero_tenure = tenure.eq(0.0).astype(float)
        denominator = tenure.mask(tenure.eq(0.0), 1.0)

        total_subscribed_services = (
            categorical["PhoneService"].eq("Yes").astype(float)
            + categorical["MultipleLines"].eq("Yes").astype(float)
            + categorical["InternetService"].ne("No").astype(float)
            + self._yes_count(categorical, SERVICE_YES_COLUMNS)
        )

        return {
            "f1_total_subscribed_service_count": total_subscribed_services,
            "f1_protection_support_service_count": self._yes_count(
                categorical,
                PROTECTION_SUPPORT_YES_COLUMNS,
            ),
            "f1_streaming_service_count": self._yes_count(
                categorical,
                STREAMING_YES_COLUMNS,
            ),
            "f1_tenure_squared": tenure.pow(2),
            "f1_log1p_tenure": np.log1p(tenure),
            "f1_average_charges_per_tenure": total_charges / denominator,
            "f1_zero_tenure": zero_tenure,
        }

    @staticmethod
    def _make_f1_curated_interactions(
        numeric: pd.DataFrame,
        categorical: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """Create the small, predeclared interaction set for general model families."""
        monthly_charges = numeric["MonthlyCharges"]
        tenure = numeric["tenure"].clip(lower=0.0)
        return {
            "f1_monthlycharges_x_month_to_month": monthly_charges
            * categorical["Contract"].eq("Month-to-month").astype(float),
            "f1_monthlycharges_x_two_year_contract": monthly_charges
            * categorical["Contract"].eq("Two year").astype(float),
            "f1_monthlycharges_x_dsl": monthly_charges
            * categorical["InternetService"].eq("DSL").astype(float),
            "f1_monthlycharges_x_fiber_optic": monthly_charges
            * categorical["InternetService"].eq("Fiber optic").astype(float),
            "f1_monthlycharges_x_tech_support_yes": monthly_charges
            * categorical["TechSupport"].eq("Yes").astype(float),
            "f1_monthlycharges_x_online_security_yes": monthly_charges
            * categorical["OnlineSecurity"].eq("Yes").astype(float),
            "f1_tenure_x_month_to_month": tenure
            * categorical["Contract"].eq("Month-to-month").astype(float),
        }

    @staticmethod
    def _make_f2_linear_expansion(
        numeric: pd.DataFrame,
        categorical: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """Create the bounded systematic expansion for regularized linear procedures."""
        tenure = numeric["tenure"].clip(lower=0.0)
        monthly_charges = numeric["MonthlyCharges"]
        total_charges = numeric["TotalCharges"]

        features: dict[str, pd.Series] = {
            "f2_tenure_squared": tenure.pow(2),
            "f2_monthlycharges_squared": monthly_charges.pow(2),
            "f2_totalcharges_squared": total_charges.pow(2),
            "f2_tenure_x_monthlycharges": tenure * monthly_charges,
            "f2_tenure_x_totalcharges": tenure * total_charges,
            "f2_monthlycharges_x_totalcharges": monthly_charges * total_charges,
        }

        for numeric_feature, categorical_feature, level, feature_name in (
            F2_NUMERIC_BY_CATEGORY_FEATURE_SPECS
        ):
            features[feature_name] = numeric[numeric_feature] * categorical[
                categorical_feature
            ].eq(level).astype(float)

        return features


def make_feature_policy_transformer(
    *,
    policy_id: FeaturePolicyId = FEATURE_POLICY_RAW,
) -> FeaturePolicyTransformer:
    """Create a clone-safe transformer for one declared deterministic feature policy."""
    return FeaturePolicyTransformer(policy_id=policy_id)
