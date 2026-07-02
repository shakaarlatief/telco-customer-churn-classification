"""Smoke test for the deterministic final-comparison feature policies.

The test reads ``train.csv`` only. It verifies that each policy has a stable schema,
uses training-fold-only fallback values, preserves raw columns and row order, and
creates the predeclared F1/F2 interaction families without target information.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import ALL_FEATURES, RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_IDS,
    FEATURE_POLICY_LINEAR_EXPANDED,
    FEATURE_POLICY_RAW,
    F1_ENGINEERED_CATEGORICAL_FEATURES,
    F1_ENGINEERED_NUMERIC_FEATURES,
    F2_ADDITIONAL_NUMERIC_FEATURES,
    feature_policy_output_features,
    make_feature_policy_transformer,
)


SAMPLE_SIZE = 400


def _representative_sample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic stratified subset that keeps the smoke test fast."""
    positions = np.arange(len(X), dtype=np.int64)
    selected, _ = train_test_split(
        positions,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    return X.iloc[selected].copy(), y.iloc[selected].copy()


def _assert_schema_and_raw_columns(
    *,
    policy_id: str,
    X_input: pd.DataFrame,
    transformed: pd.DataFrame,
) -> None:
    """Check deterministic schema, index preservation, and untouched raw columns."""
    expected_columns = feature_policy_output_features(policy_id)
    if list(transformed.columns) != expected_columns:
        raise AssertionError(
            f"{policy_id} returned an unexpected output schema.\n"
            f"Expected: {expected_columns}\nObserved: {list(transformed.columns)}"
        )
    if not transformed.index.equals(X_input.index):
        raise AssertionError(f"{policy_id} did not preserve row indices.")
    pd.testing.assert_frame_equal(
        transformed.loc[:, ALL_FEATURES],
        X_input.loc[:, ALL_FEATURES],
        check_dtype=True,
        check_names=False,
    )


def _make_controlled_row(X_template: pd.DataFrame) -> pd.DataFrame:
    """Create one deterministic row for value-level feature assertions."""
    row = X_template.iloc[[0]].copy()
    row.loc[:, "tenure"] = 0.0
    row.loc[:, "MonthlyCharges"] = 100.0
    row.loc[:, "TotalCharges"] = 0.0
    row.loc[:, "PhoneService"] = "Yes"
    row.loc[:, "MultipleLines"] = "Yes"
    row.loc[:, "InternetService"] = "Fiber optic"
    row.loc[:, "OnlineSecurity"] = "Yes"
    row.loc[:, "OnlineBackup"] = "Yes"
    row.loc[:, "DeviceProtection"] = "Yes"
    row.loc[:, "TechSupport"] = "Yes"
    row.loc[:, "StreamingTV"] = "Yes"
    row.loc[:, "StreamingMovies"] = "Yes"
    row.loc[:, "Contract"] = "Month-to-month"
    row.loc[:, "PaymentMethod"] = "Electronic check"
    return row


def _assert_controlled_f1_values(transformed: pd.DataFrame) -> None:
    """Verify key aggregation and zero-tenure conventions on a controlled row."""
    values = transformed.iloc[0]
    expected = {
        "f1_total_subscribed_service_count": 9.0,
        "f1_protection_support_service_count": 4.0,
        "f1_streaming_service_count": 2.0,
        "f1_tenure_squared": 0.0,
        "f1_log1p_tenure": 0.0,
        "f1_average_charges_per_tenure": 0.0,
        "f1_zero_tenure": 1.0,
        "f1_monthlycharges_x_month_to_month": 100.0,
        "f1_monthlycharges_x_two_year_contract": 0.0,
        "f1_monthlycharges_x_dsl": 0.0,
        "f1_monthlycharges_x_fiber_optic": 100.0,
        "f1_monthlycharges_x_tech_support_yes": 100.0,
        "f1_monthlycharges_x_online_security_yes": 100.0,
        "f1_tenure_x_month_to_month": 0.0,
    }
    for column, expected_value in expected.items():
        if not np.isclose(float(values[column]), expected_value):
            raise AssertionError(
                f"Controlled F1 value for {column!r} should equal {expected_value}, "
                f"observed {values[column]!r}."
            )
    if values["f1_contract_x_payment_method"] != "Month-to-month__Electronic check":
        raise AssertionError("The selected categorical interaction has an unexpected value.")


def _assert_controlled_f2_values(transformed: pd.DataFrame) -> None:
    """Verify retained quadratic and numeric-by-category F2 interactions."""
    values = transformed.iloc[0]
    expected = {
        "f2_monthlycharges_squared": 10_000.0,
        "f2_tenure_x_monthlycharges": 0.0,
        "f2_monthlycharges_x_contract_month_to_month": 100.0,
        "f2_monthlycharges_x_contract_two_year": 0.0,
        "f2_monthlycharges_x_internetservice_fiber_optic": 100.0,
        "f2_monthlycharges_x_internetservice_dsl": 0.0,
        "f2_monthlycharges_x_techsupport_yes": 100.0,
        "f2_monthlycharges_x_onlinesecurity_yes": 100.0,
    }
    for column, expected_value in expected.items():
        if not np.isclose(float(values[column]), expected_value):
            raise AssertionError(
                f"Controlled F2 value for {column!r} should equal {expected_value}, "
                f"observed {values[column]!r}."
            )


def _assert_pruned_f2_contract(transformed: pd.DataFrame) -> None:
    """Verify that F2 preserves its target-free, nonredundant frozen schema."""
    if len(F2_ADDITIONAL_NUMERIC_FEATURES) != 42:
        raise AssertionError(
            "The frozen F2 contract should contain exactly 42 additional numeric features, "
            f"observed {len(F2_ADDITIONAL_NUMERIC_FEATURES)}."
        )
    if len(feature_policy_output_features(FEATURE_POLICY_LINEAR_EXPANDED)) != 69:
        raise AssertionError("The frozen F2 policy should contain exactly 69 columns.")

    forbidden_exact = {
        "f2_tenure_squared",
        "f2_totalcharges_squared",
        "f2_tenure_x_totalcharges",
        "f2_monthlycharges_x_totalcharges",
    }
    forbidden_fragments = (
        "_x_multiplelines_no_phone_service",
        "_x_onlinesecurity_no_internet_service",
        "_x_onlinebackup_no_internet_service",
        "_x_deviceprotection_no_internet_service",
        "_x_techsupport_no_internet_service",
        "_x_streamingtv_no_internet_service",
        "_x_streamingmovies_no_internet_service",
    )
    forbidden_columns = [
        column
        for column in F2_ADDITIONAL_NUMERIC_FEATURES
        if column in forbidden_exact
        or column.startswith("f2_totalcharges_x_")
        or any(fragment in column for fragment in forbidden_fragments)
    ]
    if forbidden_columns:
        raise AssertionError(
            "F2 retains excluded redundant terms: "
            f"{sorted(forbidden_columns)!r}."
        )

    numeric_frame = transformed.select_dtypes(include=[np.number]).astype(float)
    duplicated_columns = numeric_frame.columns[numeric_frame.T.duplicated()].tolist()
    if duplicated_columns:
        raise AssertionError(
            "F2 contains exact duplicate numeric output columns: "
            f"{duplicated_columns!r}."
        )


def main() -> None:
    train_df = load_train_data()
    X_all, y_all = split_features_target(train_df)
    X_sample, y_sample = _representative_sample(X_all, y_all)
    X_train, X_validation = train_test_split(
        X_sample,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_sample,
    )

    print("Checking deterministic F0/F1/F2 feature policies on training data only...", flush=True)
    transformed_by_policy: dict[str, pd.DataFrame] = {}
    for policy_id in FEATURE_POLICY_IDS:
        transformer = make_feature_policy_transformer(policy_id=policy_id)
        fitted_train = transformer.fit_transform(X_train)
        transformed_validation = transformer.transform(X_validation)

        _assert_schema_and_raw_columns(
            policy_id=policy_id,
            X_input=X_train,
            transformed=fitted_train,
        )
        _assert_schema_and_raw_columns(
            policy_id=policy_id,
            X_input=X_validation,
            transformed=transformed_validation,
        )
        if list(transformer.get_feature_names_out()) != feature_policy_output_features(policy_id):
            raise AssertionError(f"{policy_id} exposes inconsistent feature names.")

        added_columns = [column for column in transformed_validation if column not in ALL_FEATURES]
        if added_columns:
            numeric_added = transformed_validation.loc[:, added_columns].select_dtypes(
                include=[np.number]
            )
            if not np.isfinite(numeric_added.to_numpy(dtype=float)).all():
                raise AssertionError(f"{policy_id} produced non-finite engineered values.")

        transformed_by_policy[policy_id] = transformed_validation
        print(
            f"  {policy_id}: {len(feature_policy_output_features(policy_id))} columns "
            f"({len(added_columns)} engineered).",
            flush=True,
        )

    if list(transformed_by_policy[FEATURE_POLICY_RAW].columns) != list(ALL_FEATURES):
        raise AssertionError("F0 must contain only the raw modelling schema.")
    if not set(F1_ENGINEERED_NUMERIC_FEATURES).issubset(
        transformed_by_policy[FEATURE_POLICY_DOMAIN].columns
    ):
        raise AssertionError("F1 is missing declared numeric engineered features.")
    if not set(F1_ENGINEERED_CATEGORICAL_FEATURES).issubset(
        transformed_by_policy[FEATURE_POLICY_DOMAIN].columns
    ):
        raise AssertionError("F1 is missing declared categorical engineered features.")
    if not set(F2_ADDITIONAL_NUMERIC_FEATURES).issubset(
        transformed_by_policy[FEATURE_POLICY_LINEAR_EXPANDED].columns
    ):
        raise AssertionError("F2 is missing declared systematic linear-expansion features.")
    _assert_pruned_f2_contract(transformed_by_policy[FEATURE_POLICY_LINEAR_EXPANDED])

    controlled_row = _make_controlled_row(X_train)
    f1_transformer = make_feature_policy_transformer(policy_id=FEATURE_POLICY_DOMAIN).fit(X_train)
    f2_transformer = make_feature_policy_transformer(
        policy_id=FEATURE_POLICY_LINEAR_EXPANDED
    ).fit(X_train)
    _assert_controlled_f1_values(f1_transformer.transform(controlled_row))
    _assert_controlled_f2_values(f2_transformer.transform(controlled_row))

    print("Final-comparison feature-policy smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
