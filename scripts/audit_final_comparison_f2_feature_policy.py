"""Target-free structural audit for the F2 regularized-linear feature policy.

This script is a pre-protocol-freeze diagnostic for the Telco final-comparison
workflow. It uses development training features only and never reads or uses
``Churn_binary``. It does not fit predictive models, tune hyperparameters, compare
candidate performance, or touch the held-out test set.

The audit addresses three representation questions before F2 is frozen:

1. whether the declared F2 schema contains exact duplicate numeric columns;
2. how closely cumulative ``TotalCharges`` follows the structural product
   ``tenure * MonthlyCharges``; and
3. which remaining numeric terms have the strongest target-free linear
   associations, especially terms involving the cumulative charge variable.

The output is descriptive. It supports a design decision but must not be treated
as model-selection evidence.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from telco_churn.config import NUMERIC_FEATURES  # noqa: E402
from telco_churn.data import load_train_data, split_features_target  # noqa: E402
from telco_churn.feature_policies import (  # noqa: E402
    FEATURE_POLICY_LINEAR_EXPANDED,
    F2_ADDITIONAL_NUMERIC_FEATURES,
    feature_policy_numeric_features,
    make_feature_policy_transformer,
)


DISPLAY_DECIMALS = 6
TOP_CORRELATION_PAIRS = 20
NEAR_PERFECT_CORRELATION = 0.995


def _format_number(value: float) -> str:
    """Format a finite diagnostic scalar consistently for terminal output."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{DISPLAY_DECIMALS}f}"


def _descriptive_summary(values: pd.Series) -> dict[str, float]:
    """Return stable descriptive statistics for one finite numeric diagnostic."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    quantiles = values.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "count": float(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p01": float(quantiles.loc[0.01]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(values.max()),
    }


def _print_summary(title: str, values: pd.Series) -> None:
    """Print a compact distribution summary for one audit quantity."""
    summary = _descriptive_summary(values)
    print(title)
    print(
        "  "
        + ", ".join(
            (
                f"n={int(summary['count'])}",
                f"mean={_format_number(summary['mean'])}",
                f"sd={_format_number(summary['std'])}",
                f"min={_format_number(summary['min'])}",
                f"p01={_format_number(summary['p01'])}",
                f"p25={_format_number(summary['p25'])}",
                f"median={_format_number(summary['median'])}",
                f"p75={_format_number(summary['p75'])}",
                f"p99={_format_number(summary['p99'])}",
                f"max={_format_number(summary['max'])}",
            )
        )
    )


def _r_squared_from_fixed_prediction(
    observed: pd.Series,
    predicted: pd.Series,
) -> float:
    """Return R-squared for a fixed, non-estimated structural prediction.

    The returned quantity compares the observed variable directly with a supplied
    deterministic formula. It does not estimate a regression and therefore does
    not use the churn target or any validation outcome.
    """
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if observed_array.shape != predicted_array.shape:
        raise ValueError("Observed and predicted arrays must have the same shape.")
    total_sum_squares = float(np.square(observed_array - observed_array.mean()).sum())
    if total_sum_squares == 0.0:
        return np.nan
    residual_sum_squares = float(np.square(observed_array - predicted_array).sum())
    return 1.0 - residual_sum_squares / total_sum_squares


def _find_exact_duplicate_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return all exact-equality pairs among the supplied numeric columns.

    F2 has a bounded numeric schema, so an exhaustive pairwise equality check is
    inexpensive and safer than relying on naming conventions alone. Equality is
    evaluated after the policy transformer has constructed its fold-safe values.
    """
    duplicates: list[tuple[str, str]] = []
    values_by_column = {
        column: frame[column].to_numpy(dtype=float, copy=False)
        for column in frame.columns
    }
    for left, right in combinations(frame.columns, 2):
        if np.array_equal(values_by_column[left], values_by_column[right]):
            duplicates.append((left, right))
    return duplicates


def _top_numeric_correlation_pairs(
    frame: pd.DataFrame,
    *,
    exclude_exact_duplicates: set[frozenset[str]],
) -> pd.DataFrame:
    """Return the strongest absolute Pearson correlations among numeric policy columns."""
    correlations = frame.corr(method="pearson")
    records: list[dict[str, object]] = []

    for row_index, left in enumerate(correlations.columns):
        for right in correlations.columns[row_index + 1 :]:
            if frozenset((left, right)) in exclude_exact_duplicates:
                continue
            correlation = float(correlations.loc[left, right])
            if not np.isfinite(correlation):
                continue
            records.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["absolute_correlation", "left", "right"],
            ascending=[False, True, True],
            kind="stable",
        )
        .head(TOP_CORRELATION_PAIRS)
        .reset_index(drop=True)
    )


def _print_correlation_pairs(pairs: pd.DataFrame) -> None:
    """Print the strongest non-duplicate numeric associations in a readable table."""
    if pairs.empty:
        print("  No finite pairwise correlations were available.")
        return

    print(
        pairs.loc[:, ["left", "right", "correlation", "absolute_correlation"]]
        .to_string(
            index=False,
            formatters={
                "correlation": _format_number,
                "absolute_correlation": _format_number,
            },
        )
    )


def main() -> None:
    """Run the F2 structural audit against development training features only."""
    print(
        "Running target-free F2 feature-policy structural audit on development data only.",
        flush=True,
    )

    train_df = load_train_data()
    X, _ = split_features_target(train_df)

    transformer = make_feature_policy_transformer(
        policy_id=FEATURE_POLICY_LINEAR_EXPANDED
    )
    transformed = transformer.fit_transform(X)
    numeric_columns = feature_policy_numeric_features(FEATURE_POLICY_LINEAR_EXPANDED)
    numeric_frame = transformed.loc[:, numeric_columns].astype(float)

    raw_numeric_count = len(NUMERIC_FEATURES)
    f2_additional_count = len(F2_ADDITIONAL_NUMERIC_FEATURES)
    print()
    print("F2 schema summary")
    print(
        f"  rows={len(X)}, raw numeric columns={raw_numeric_count}, "
        f"F2 numeric columns={len(numeric_columns)}, "
        f"F2 additional numeric columns={f2_additional_count}, "
        f"total policy columns={transformed.shape[1]}",
        flush=True,
    )

    print()
    print("Exact duplicate numeric columns")
    duplicates = _find_exact_duplicate_columns(numeric_frame)
    if duplicates:
        for left, right in duplicates:
            print(f"  {left} == {right}")
    else:
        print("  None detected.")

    tenure = pd.to_numeric(X["tenure"], errors="coerce").clip(lower=0.0)
    monthly_charges = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
    total_charges = pd.to_numeric(X["TotalCharges"], errors="coerce")
    valid_rows = tenure.notna() & monthly_charges.notna() & total_charges.notna()
    positive_tenure = valid_rows & tenure.gt(0.0)

    if not positive_tenure.any():
        raise RuntimeError("The development data contains no valid positive-tenure rows.")

    expected_total_charges = tenure.loc[positive_tenure] * monthly_charges.loc[positive_tenure]
    observed_total_charges = total_charges.loc[positive_tenure]
    historical_average_charge = observed_total_charges / tenure.loc[positive_tenure]
    charge_gap = observed_total_charges - expected_total_charges
    relative_gap = charge_gap / observed_total_charges.abs().clip(lower=1.0)

    print()
    print("Cumulative-charge structural diagnostics for rows with tenure > 0")
    print(f"  valid rows={int(valid_rows.sum())}, positive-tenure rows={int(positive_tenure.sum())}")
    print(
        "  Corr(TotalCharges, tenure * MonthlyCharges)="
        f"{_format_number(float(observed_total_charges.corr(expected_total_charges)))}"
    )
    print(
        "  Fixed-form R² for TotalCharges ≈ tenure * MonthlyCharges="
        f"{_format_number(_r_squared_from_fixed_prediction(observed_total_charges, expected_total_charges))}"
    )
    print(
        "  Corr(TotalCharges / tenure, MonthlyCharges)="
        f"{_format_number(float(historical_average_charge.corr(monthly_charges.loc[positive_tenure])))}"
    )
    _print_summary("  TotalCharges - tenure * MonthlyCharges", charge_gap)
    _print_summary("  Relative cumulative-charge gap", relative_gap)

    exact_duplicate_keys = {frozenset(pair) for pair in duplicates}
    correlation_pairs = _top_numeric_correlation_pairs(
        numeric_frame,
        exclude_exact_duplicates=exact_duplicate_keys,
    )

    print()
    print(
        f"Top {len(correlation_pairs)} non-duplicate absolute correlations among F2 numeric columns"
    )
    _print_correlation_pairs(correlation_pairs)

    near_perfect = correlation_pairs.loc[
        correlation_pairs["absolute_correlation"] >= NEAR_PERFECT_CORRELATION
    ]
    print()
    print(
        "Near-perfect non-duplicate pairs "
        f"(|r| >= {NEAR_PERFECT_CORRELATION:.3f}) among the displayed pairs"
    )
    if near_perfect.empty:
        print("  None among the top displayed pairs.")
    else:
        _print_correlation_pairs(near_perfect)

    print()
    print("Audit completed. No target values, predictive models, model scores, or test data were used.")


if __name__ == "__main__":
    main()
