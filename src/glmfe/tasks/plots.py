"""Create optional visual summaries from completed task result tables.

This module plots existing results and does not perform model inference.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


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

    output_dir.mkdir(exist_ok=True)
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


def _plot_block_score_track(
    spans: pd.DataFrame,
    map_row: pd.Series,
    run_dir: Path,
    output_dir: Path,
) -> None:
    map_path = run_dir / map_row["map_path"]
    if not map_path.exists():
        raise ValueError(
            f"Dependency-map file for map {map_row['map_id']} does not "
            f"exist: {map_path}"
        )

    with np.load(map_path, allow_pickle=False) as arrays:
        if "dependency_map" not in arrays.files:
            raise ValueError(
                f"Dependency-map file for map {map_row['map_id']} does not "
                "contain dependency_map"
            )
        dependency_map = arrays["dependency_map"]

    map_id = str(map_row["map_id"])
    map_start = int(map_row["start"])
    map_end = int(map_row["end"])
    block_size = int(spans["block_size"].iloc[0])
    top_spans = spans.sort_values(
        ["block_score", "local_start"],
        ascending=[False, True],
        kind="stable",
    ).head(5)

    figure, (track_axis, map_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 11),
        height_ratios=[1.0, 3.0],
        constrained_layout=True,
    )

    x_positions = (
        spans["span_start"].to_numpy(dtype=float)
        + spans["span_length"].to_numpy(dtype=float) / 2.0
    )
    track_axis.plot(
        x_positions,
        spans["block_score"].to_numpy(dtype=float),
        color="#4C78A8",
        linewidth=1.8,
    )
    track_axis.scatter(
        x_positions,
        spans["block_score"].to_numpy(dtype=float),
        s=12,
        color="#4C78A8",
        zorder=2,
    )
    if pd.notna(map_row["region_start"]) and pd.notna(map_row["region_end"]):
        track_axis.axvspan(
            int(map_row["region_start"]),
            int(map_row["region_end"]),
            ymin=0.92,
            ymax=0.98,
            color="#54A24B",
            alpha=0.35,
            linewidth=0,
            zorder=0,
        )
    for top_span in top_spans.itertuples(index=False):
        track_axis.axvspan(
            int(top_span.span_start),
            int(top_span.span_end),
            color="#F58518",
            alpha=0.25,
            linewidth=0,
            zorder=1,
        )
    track_axis.set_xlim(map_start, map_end)
    track_axis.set_xlabel("Absolute record position")
    track_axis.set_ylabel("Block score")
    track_axis.set_title(f"{map_id}: block-score track")

    image = map_axis.imshow(
        dependency_map,
        origin="upper",
        extent=(map_start, map_end, map_end, map_start),
        aspect="equal",
        cmap="viridis",
    )
    for top_span in top_spans.itertuples(index=False):
        rectangle = Rectangle(
            (int(top_span.span_start), int(top_span.span_start)),
            block_size,
            block_size,
            fill=False,
            edgecolor="#F58518",
            linewidth=1.8,
        )
        map_axis.add_patch(rectangle)
    map_axis.set_xlim(map_start, map_end)
    map_axis.set_ylim(map_end, map_start)
    map_axis.set_xlabel("Affected position")
    map_axis.set_ylabel("Changed position")
    map_axis.set_title("Dependency map with top block-score spans")
    figure.colorbar(image, ax=map_axis, label="Dependency")
    figure.savefig(output_dir / f"{map_id}_block_score_track.png", dpi=300)
    figure.savefig(output_dir / f"{map_id}_block_score_track.pdf")
    plt.close(figure)


def plot_block_score_results(
    per_span: pd.DataFrame,
    per_map: pd.DataFrame,
    map_index: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save descriptive block-score summaries and map companion plots."""

    plot_dir = output_dir / "block_scores" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if per_map.empty:
        raise ValueError("Cannot plot block scores with no per-map rows")
    if per_span.empty:
        raise ValueError("Cannot plot block scores with no per-span rows")
    if map_index["map_id"].duplicated().any():
        raise ValueError("map_index contains duplicate map_id values")

    summary_frame = per_map.copy()
    summary_frame["feature_type"] = (
        summary_frame["feature_type"].fillna("None").astype(str)
    )
    feature_order = summary_frame["feature_type"].value_counts().index.tolist()
    rng = np.random.default_rng(44)

    _plot_metric_distribution(
        summary_frame,
        plot_dir,
        feature_order,
        "max_block_score",
        "Maximum block score per map",
        "Descriptive maximum block score by feature type",
        "max_block_score_by_feature_type",
        None,
        None,
        rng,
    )
    _plot_metric_distribution(
        summary_frame,
        plot_dir,
        feature_order,
        "median_block_score",
        "Median block score per map",
        "Descriptive median block score by feature type",
        "median_block_score_by_feature_type",
        None,
        None,
        rng,
    )

    map_index_by_id = map_index.set_index(
        "map_id",
        drop=False,
        verify_integrity=True,
    )
    for map_id in per_map["map_id"]:
        if map_id not in map_index_by_id.index:
            raise ValueError(f"map_index is missing map_id {map_id!r}")
        spans = per_span.loc[per_span["map_id"] == map_id]
        if spans.empty:
            raise ValueError(f"per_span contains no rows for map_id {map_id!r}")
        _plot_block_score_track(
            spans.sort_values("local_start", kind="stable"),
            map_index_by_id.loc[map_id],
            output_dir,
            plot_dir,
        )
