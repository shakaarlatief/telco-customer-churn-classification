"""Feature-name and coefficient utilities for fitted pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_feature_names_from_preprocessor(preprocessor) -> list[str]:
    """Return transformed feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError as exc:
        raise AttributeError(
            "The fitted preprocessor does not expose get_feature_names_out()."
        ) from exc


def clean_transformed_feature_name(name: str) -> str:
    """Clean scikit-learn transformed feature names for reports.

    ColumnTransformer prefixes names with the transformer name, for example
    ``num__tenure`` or ``cat__Contract_Month-to-month``. This helper removes
    the prefix while preserving the substantive feature name.
    """
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def extract_linear_model_coefficients(
    *,
    fitted_pipeline,
    top_n: int | None = None,
    sort_by_absolute: bool = True,
) -> pd.DataFrame:
    """Extract coefficients from a fitted linear-model pipeline.

    The fitted pipeline must contain:

    - a ``preprocessor`` step with ``get_feature_names_out``;
    - a ``classifier`` step exposing ``coef_``.

    For binary logistic regression and RidgeClassifier, ``coef_`` has shape
    ``(1, n_features)``. The coefficient sign indicates the direction of the
    relationship with the positive class under the model's linear score.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = [
        clean_transformed_feature_name(name)
        for name in get_feature_names_from_preprocessor(preprocessor)
    ]

    coefficients = np.ravel(classifier.coef_)

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Number of transformed feature names does not match number of coefficients."
        )

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    coefficient_df["odds_ratio"] = np.exp(coefficient_df["coefficient"])
    coefficient_df["direction"] = np.where(
        coefficient_df["coefficient"] >= 0,
        "higher churn score",
        "lower churn score",
    )

    if sort_by_absolute:
        coefficient_df = coefficient_df.sort_values(
            by="absolute_coefficient",
            ascending=False,
        )

    if top_n is not None:
        coefficient_df = coefficient_df.head(top_n)

    return coefficient_df.reset_index(drop=True)
