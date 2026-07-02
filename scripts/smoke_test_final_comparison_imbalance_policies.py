"""Training-only smoke test for final-comparison imbalance-treatment primitives.

The test validates generic sampler mechanics before any policy is admitted to the
candidate registry. It fits feature construction and mixed-type imputation only on a
training partition, then verifies I0 through I4 on F0 and F1 inputs. The held-out test
set is never loaded.
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


try:
    import imblearn  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "imbalanced-learn is not installed. Run `pip install -r requirements.txt` "
        "from the repository root, then rerun this smoke test."
    ) from exc


from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_DOMAIN,
    FEATURE_POLICY_RAW,
    FeaturePolicyTransformer,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
)
from telco_churn.imbalance_policies import (  # noqa: E402
    FeaturePolicySamplerImputer,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_NONE,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
    balanced_class_weight_mapping,
    class_count_summary,
    make_random_resampler,
    make_smotenc_resampler,
)


SAMPLE_SIZE = 600
RESAMPLING_RATIO = 0.75
SMOTENC_NEIGHBORS = 5
TESTED_FEATURE_POLICIES = (FEATURE_POLICY_RAW, FEATURE_POLICY_DOMAIN)


def make_training_partition() -> tuple[pd.DataFrame, pd.Series]:
    """Load a deterministic development-only subset and retain its training partition."""
    train_df = load_train_data()
    X, y = split_features_target(train_df)

    if len(y) > SAMPLE_SIZE:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=SAMPLE_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE + 1,
    )
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True)


def positive_to_negative_ratio(y) -> float:
    """Return the binary minority-to-majority ratio used by sampler conventions."""
    counts = class_count_summary(y)
    if counts[1] > counts[0]:
        raise AssertionError("The Telco smoke partition must retain churn as the minority class.")
    return float(counts[1]) / float(counts[0])


def assert_ratio_close(y, *, expected: float, name: str) -> None:
    """Check a requested binary sampling ratio while allowing integer rounding."""
    observed = positive_to_negative_ratio(y)
    majority_count = class_count_summary(y)[0]
    rounding_tolerance = max(0.02, 1.0 / float(majority_count) + 1e-12)
    if not np.isclose(observed, expected, rtol=0.0, atol=rounding_tolerance):
        raise AssertionError(
            f"{name} produced class ratio {observed:.5f}, expected approximately {expected:.5f}."
        )


def make_mixed_training_table(
    X_train: pd.DataFrame,
    *,
    feature_policy: str,
) -> pd.DataFrame:
    """Fit policy and sampler imputation only on the active smoke training partition."""
    engineered = FeaturePolicyTransformer(policy_id=feature_policy).fit_transform(X_train)
    mixed = FeaturePolicySamplerImputer(policy_id=feature_policy).fit_transform(engineered)

    expected_columns = [
        *feature_policy_numeric_features(feature_policy),
        *feature_policy_categorical_features(feature_policy),
    ]
    if list(mixed.columns) != expected_columns:
        raise AssertionError("Mixed sampler input does not follow the policy schema contract.")
    if mixed.isna().any().any():
        raise AssertionError("Mixed sampler input must be complete before SMOTENC.")
    numeric = mixed.loc[:, feature_policy_numeric_features(feature_policy)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Mixed sampler input contains non-finite numeric values.")
    return mixed


def assert_smotenc_schema(
    *,
    source: pd.DataFrame,
    resampled: pd.DataFrame,
    feature_policy: str,
) -> None:
    """Verify that SMOTENC returns a valid mixed DataFrame without novel categories."""
    if not isinstance(resampled, pd.DataFrame):
        raise AssertionError("SMOTENC must preserve a pandas DataFrame schema.")
    if list(resampled.columns) != list(source.columns):
        raise AssertionError("SMOTENC altered the declared mixed feature schema.")
    if resampled.isna().any().any():
        raise AssertionError("SMOTENC returned missing mixed-feature values.")

    numeric_columns = feature_policy_numeric_features(feature_policy)
    numeric = resampled.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError("SMOTENC returned non-finite numeric feature values.")

    for column in feature_policy_categorical_features(feature_policy):
        source_levels = set(source[column].astype(str))
        resampled_levels = set(resampled[column].astype(str))
        if not resampled_levels.issubset(source_levels):
            raise AssertionError(
                f"SMOTENC generated an unseen categorical level in {column!r}: "
                f"{sorted(resampled_levels.difference(source_levels))!r}."
            )


def run_imbalance_policy_smoke(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Exercise all initial policies on each mixed feature-policy route."""
    base_counts = class_count_summary(y_train)
    if base_counts[1] >= base_counts[0]:
        raise AssertionError("The deterministic smoke partition must be imbalanced.")

    # I0 has no sampler. Its invariant is exact preservation of training-fold counts.
    if class_count_summary(y_train) != base_counts:
        raise AssertionError(f"{IMBALANCE_NONE} unexpectedly changed the target counts.")

    # I1 has no sampler. It supplies weights that equalize aggregate class mass.
    weights = balanced_class_weight_mapping(y_train)
    weighted_mass_0 = weights[0] * base_counts[0]
    weighted_mass_1 = weights[1] * base_counts[1]
    if not np.isclose(weighted_mass_0, weighted_mass_1):
        raise AssertionError(
            f"{IMBALANCE_CLASS_WEIGHT_BALANCED} did not equalize class mass."
        )

    for feature_policy in TESTED_FEATURE_POLICIES:
        mixed = make_mixed_training_table(X_train, feature_policy=feature_policy)
        print(
            f"  {feature_policy}: checking I2 random oversampling, I3 random "
            "undersampling, and I4 SMOTENC...",
            flush=True,
        )

        oversampler = make_random_resampler(
            IMBALANCE_RANDOM_OVERSAMPLING,
            sampling_strategy=RESAMPLING_RATIO,
            random_state=RANDOM_STATE,
        )
        X_over, y_over = oversampler.fit_resample(mixed, y_train)
        if len(y_over) <= len(y_train):
            raise AssertionError("Random oversampling must increase the training row count.")
        assert_ratio_close(
            y_over,
            expected=RESAMPLING_RATIO,
            name=IMBALANCE_RANDOM_OVERSAMPLING,
        )

        undersampler = make_random_resampler(
            IMBALANCE_RANDOM_UNDERSAMPLING,
            sampling_strategy=RESAMPLING_RATIO,
            random_state=RANDOM_STATE,
        )
        X_under, y_under = undersampler.fit_resample(mixed, y_train)
        if len(y_under) >= len(y_train):
            raise AssertionError("Random undersampling must reduce the training row count.")
        assert_ratio_close(
            y_under,
            expected=RESAMPLING_RATIO,
            name=IMBALANCE_RANDOM_UNDERSAMPLING,
        )

        smotenc = make_smotenc_resampler(
            n_numeric_features=len(feature_policy_numeric_features(feature_policy)),
            n_categorical_features=len(feature_policy_categorical_features(feature_policy)),
            sampling_strategy=RESAMPLING_RATIO,
            k_neighbors=SMOTENC_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        X_smote, y_smote = smotenc.fit_resample(mixed, y_train)
        if len(y_smote) <= len(y_train):
            raise AssertionError("SMOTENC must increase the training row count.")
        assert_ratio_close(y_smote, expected=RESAMPLING_RATIO, name=IMBALANCE_SMOTENC)
        assert_smotenc_schema(
            source=mixed,
            resampled=X_smote,
            feature_policy=feature_policy,
        )


def main() -> None:
    X_train, y_train = make_training_partition()
    print(
        "Checking final-comparison imbalance policies on a training-only stratified "
        "partition...",
        flush=True,
    )
    run_imbalance_policy_smoke(X_train, y_train)
    print("Final-comparison imbalance-policy smoke test passed.")


if __name__ == "__main__":
    main()
