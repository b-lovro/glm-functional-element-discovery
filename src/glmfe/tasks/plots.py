"""Create optional visual summaries from completed reconstruction tables.

This module plots existing results and does not perform model inference.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_region_counts(
    per_region: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = per_region["feature_type"].value_counts()
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(counts.index, counts.to_numpy(), color="#4C78A8")
    axis.set_xlabel("Feature type")
    axis.set_ylabel("Number of regions")
    axis.set_title("Annotated intervals by feature type")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / "region_count_by_feature_type.png", dpi=300)


def _plot_metric_distribution(
    per_region: pd.DataFrame,
    output_dir: Path,
    feature_order: list[str],
    value_column: str,
    y_label: str,
    title: str,
    filename_stem: str,
    y_limits: tuple[float, float] | None,
    reference_line: float | None,
    rng: np.random.Generator,
) -> None:
    counts = per_region["feature_type"].value_counts()
    values = [
        per_region.loc[
            per_region["feature_type"] == feature_type,
            value_column,
        ].dropna().to_numpy()
        for feature_type in feature_order
    ]
    labels = [
        f"{feature_type}\n(n={int(counts.loc[feature_type])})"
        for feature_type in feature_order
    ]

    figure, axis = plt.subplots(figsize=(12, 7))
    for position, feature_values in enumerate(values, start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(feature_values))
        axis.scatter(
            position + jitter,
            feature_values,
            s=10,
            alpha=0.60,
            color="#4C78A8",
            edgecolors="white",
            linewidths=0.25,
            zorder=1,
        )

    axis.boxplot(
        values,
        positions=range(1, len(values) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        boxprops={
            "facecolor": "#A6C8E0",
            "edgecolor": "#1F4E79",
            "linewidth": 1.8,
            "alpha": 0.85,
        },
        whiskerprops={"color": "#1F4E79", "linewidth": 1.5},
        capprops={"color": "#1F4E79", "linewidth": 1.5},
        zorder=2,
    )
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels)

    if y_limits is not None:
        axis.set_ylim(y_limits)
    if reference_line is not None:
        axis.axhline(
            reference_line,
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            zorder=0,
        )

    axis.set_xlabel("Feature type")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / f"{filename_stem}.png", dpi=300)


def plot_reconstruction_results(
    per_region: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save per-region reconstruction summaries as PDF and PNG figures."""

    output_dir.mkdir(exist_ok=False)
    feature_order = per_region["feature_type"].value_counts().index.tolist()
    rng = np.random.default_rng(44)

    _plot_region_counts(per_region, output_dir)
    _plot_metric_distribution(
        per_region,
        output_dir,
        feature_order,
        "mean_cross_entropy_bits",
        "Mean cross-entropy per region (bits)",
        "Mean cross-entropy by feature type",
        "mean_cross_entropy_bits_by_feature_type",
        None,
        2.0,
        rng,
    )
    _plot_metric_distribution(
        per_region,
        output_dir,
        feature_order,
        "accuracy",
        "Reconstruction accuracy per region",
        "Reconstruction accuracy by feature type",
        "accuracy_by_feature_type",
        (0.0, 1.0),
        None,
        rng,
    )
    _plot_metric_distribution(
        per_region,
        output_dir,
        feature_order,
        "mean_true_base_probability",
        "Mean true-base probability per region",
        "Mean true-base probability by feature type",
        "mean_true_base_probability_by_feature_type",
        (0.0, 1.0),
        0.25,
        rng,
    )
