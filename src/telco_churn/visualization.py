"""Visualization utilities for model evaluation and interpretation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_regularization_metric_plot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    x_column: str = "C",
    metric_columns: list[str] | None = None,
) -> None:
    """Save a line plot of metrics over a regularization grid."""
    if metric_columns is None:
        metric_columns = ["pr_auc", "roc_auc", "balanced_accuracy", "f1"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for metric in metric_columns:
        ax.plot(results_df[x_column], results_df[metric], marker="o", label=metric)

    ax.set_xscale("log")
    ax.set_xlabel(x_column)
    ax.set_ylabel("Cross-validated metric")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_coefficient_plot(
    *,
    coefficient_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar plot of the largest absolute coefficients."""
    plot_df = (
        coefficient_df.sort_values("absolute_coefficient", ascending=False)
        .head(top_n)
        .sort_values("coefficient", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(plot_df["feature"], plot_df["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_tradeoff_plot(
    *,
    threshold_df: pd.DataFrame,
    output_path: Path,
    title: str,
    metric_columns: list[str] | None = None,
    x_label: str = "Classification threshold",
    reference_threshold: float | None = None,
    reference_label: str | None = None,
) -> None:
    """Save threshold tradeoff curves for arbitrary classifier-score thresholds.

    Probability thresholds are one common use case. The same helper can also
    display signed decision-function thresholds from margin-based classifiers,
    such as SVMs. An optional reference threshold marks a natural operating
    point, for example score zero for an uncalibrated binary SVM.
    """
    if metric_columns is None:
        metric_columns = ["precision", "recall", "specificity", "f1"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for metric in metric_columns:
        ax.plot(threshold_df["threshold"], threshold_df[metric], label=metric)

    if reference_threshold is not None:
        ax.axvline(
            reference_threshold,
            linestyle="--",
            linewidth=1,
            label=(
                reference_label
                or f"Reference threshold = {float(reference_threshold):g}"
            ),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

from sklearn.metrics import auc


def save_roc_curve_plot(
    *,
    roc_curve_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Save a ROC curve plot.

    The ROC curve plots the true positive rate against the false positive rate
    over all possible classification thresholds. The diagonal reference line is
    the performance of a non-informative random ranking.
    """
    roc_auc = auc(
        roc_curve_df["false_positive_rate"],
        roc_curve_df["true_positive_rate"],
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    ax.plot(
        roc_curve_df["false_positive_rate"],
        roc_curve_df["true_positive_rate"],
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random ranking")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / recall")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve_plot(
    *,
    precision_recall_curve_df: pd.DataFrame,
    output_path: Path,
    title: str,
    positive_rate: float | None = None,
) -> None:
    """Save a precision-recall curve plot.

    The precision-recall curve plots the reliability of positive predictions
    against the fraction of actual positives recovered. When supplied, the
    positive-rate reference line gives the expected precision of a
    non-informative random ranking.
    """
    pr_auc = auc(
        precision_recall_curve_df["recall"],
        precision_recall_curve_df["precision"],
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    ax.plot(
        precision_recall_curve_df["recall"],
        precision_recall_curve_df["precision"],
        label=f"PR curve (AUC = {pr_auc:.3f})",
    )

    if positive_rate is not None:
        ax.axhline(
            positive_rate,
            linestyle="--",
            linewidth=1,
            label=f"Positive-rate baseline = {positive_rate:.3f}",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
