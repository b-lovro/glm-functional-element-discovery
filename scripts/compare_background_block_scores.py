"""Compare annotated dependency-map block scores with matched backgrounds.

Usage:

    python scripts/compare_background_block_scores.py outputs/runs/<run_id>

This is a one-run offline analysis. It reads only:

    <run_dir>/dependency_maps/map_index.parquet
    <run_dir>/block_scores/per_map.parquet

It does not load models, compute dependency maps, compute block scores, or
read saved ``.npz`` map files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORE_COLUMNS = (
    "mean_block_score",
    "median_block_score",
    "max_block_score",
)
REQUIRED_MAP_INDEX_COLUMNS = [
    "map_id",
    "comparison_id",
    "map_role",
    "background_index",
    "record_id",
    "region_id",
    "label",
    "feature_type",
    "region_start",
    "region_end",
    "region_length",
    "start",
    "end",
    "length",
]
PER_MAP_OUTPUT_COLUMNS = [
    "comparison_id",
    "map_id",
    "map_role",
    "background_index",
    "record_id",
    "region_id",
    "label",
    "feature_type",
    "start",
    "end",
    "length",
    "score_column",
    "segment_score",
]
PER_COMPARISON_COLUMNS = [
    "comparison_id",
    "record_id",
    "region_id",
    "label",
    "feature_type",
    "region_start",
    "region_end",
    "region_length",
    "score_column",
    "positive_score",
    "mean_background_score",
    "median_background_score",
    "std_background_score",
    "n_backgrounds",
    "contrast",
    "win_rate",
]


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in columns if column not in frame]
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required column(s): "
            + ", ".join(missing_columns)
        )


def _validate_unique_map_ids(frame: pd.DataFrame, table_name: str) -> None:
    duplicated = frame.loc[frame["map_id"].duplicated(), "map_id"]
    if not duplicated.empty:
        examples = ", ".join(duplicated.astype(str).head(5).tolist())
        raise ValueError(
            f"{table_name}.map_id must be unique; duplicate example(s): "
            f"{examples}"
        )


def _format_id_examples(values: set[object]) -> str:
    return ", ".join(sorted(str(value) for value in values)[:5])


def read_and_validate_inputs(
    run_dir: Path,
    score_column: str,
) -> pd.DataFrame:
    """Read input parquet files and return a validated merged score table."""

    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    map_index_path = run_dir / "dependency_maps" / "map_index.parquet"
    per_map_path = run_dir / "block_scores" / "per_map.parquet"
    if not map_index_path.is_file():
        raise ValueError(f"Missing map index parquet: {map_index_path}")
    if not per_map_path.is_file():
        raise ValueError(f"Missing per-map block-score parquet: {per_map_path}")

    map_index = pd.read_parquet(map_index_path, engine="pyarrow")
    per_map = pd.read_parquet(per_map_path, engine="pyarrow")

    _require_columns(map_index, REQUIRED_MAP_INDEX_COLUMNS, "map_index")
    _require_columns(per_map, ["map_id", score_column], "per_map")
    _validate_unique_map_ids(map_index, "map_index")
    _validate_unique_map_ids(per_map, "per_map")

    map_ids = set(map_index["map_id"])
    score_ids = set(per_map["map_id"])
    missing_scores = map_ids - score_ids
    missing_metadata = score_ids - map_ids
    if missing_scores or missing_metadata:
        details = []
        if missing_scores:
            details.append(
                "map_id value(s) missing from per_map: "
                + _format_id_examples(missing_scores)
            )
        if missing_metadata:
            details.append(
                "map_id value(s) missing from map_index: "
                + _format_id_examples(missing_metadata)
            )
        raise ValueError(
            "Joining per_map to map_index by map_id is not one-to-one: "
            + "; ".join(details)
        )

    score_values = pd.to_numeric(per_map[score_column], errors="raise")
    if not np.isfinite(score_values.to_numpy(dtype=float)).all():
        raise ValueError(
            f"per_map.{score_column} must contain only finite numeric values"
        )

    if map_index["comparison_id"].isna().any():
        raise ValueError("map_index.comparison_id contains null values")
    if map_index["map_role"].isna().any():
        raise ValueError("map_index.map_role contains null values")
    roles = set(map_index["map_role"].astype(str))
    allowed_roles = {"positive", "background"}
    unsupported_roles = roles - allowed_roles
    if unsupported_roles:
        raise ValueError(
            "map_index.map_role contains unsupported value(s): "
            + ", ".join(sorted(unsupported_roles))
        )

    score_frame = per_map[["map_id"]].copy()
    score_frame[score_column] = score_values.astype(float)
    merged_scores = map_index[REQUIRED_MAP_INDEX_COLUMNS].merge(
        score_frame,
        on="map_id",
        how="inner",
        validate="one_to_one",
    )
    _validate_comparison_groups(merged_scores)
    return merged_scores


def _validate_comparison_groups(merged_scores: pd.DataFrame) -> None:
    if merged_scores.empty:
        raise ValueError("No maps are available for background comparison")

    role_counts = (
        merged_scores.groupby(["comparison_id", "map_role"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for required_role in ["positive", "background"]:
        if required_role not in role_counts:
            role_counts[required_role] = 0

    bad_positive = role_counts.index[role_counts["positive"] != 1].tolist()
    if bad_positive:
        examples = ", ".join(str(value) for value in bad_positive[:5])
        raise ValueError(
            "Every comparison_id must have exactly one positive map; "
            f"bad example(s): {examples}"
        )

    bad_background = role_counts.index[
        role_counts["background"] < 1
    ].tolist()
    if bad_background:
        examples = ", ".join(str(value) for value in bad_background[:5])
        raise ValueError(
            "Every comparison_id must have at least one background map; "
            f"bad example(s): {examples}"
        )


def build_comparison_table(
    merged_scores: pd.DataFrame,
    score_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build per-map and per-comparison background score summaries."""

    _validate_comparison_groups(merged_scores)

    per_map_scores = merged_scores.copy()
    per_map_scores["score_column"] = score_column
    per_map_scores["segment_score"] = per_map_scores[score_column].astype(
        float
    )
    role_order = per_map_scores["map_role"].map(
        {"positive": 0, "background": 1}
    )
    per_map_scores = (
        per_map_scores.assign(_role_order=role_order)
        .sort_values(
            ["comparison_id", "_role_order", "background_index", "map_id"],
            kind="stable",
        )
        .drop(columns=["_role_order"])
    )
    per_map_scores = per_map_scores[PER_MAP_OUTPUT_COLUMNS]

    comparison_rows = []
    for comparison_id, group in merged_scores.groupby(
        "comparison_id",
        sort=True,
    ):
        positive = group.loc[group["map_role"] == "positive"].iloc[0]
        backgrounds = group.loc[group["map_role"] == "background"]
        positive_score = float(positive[score_column])
        background_scores = backgrounds[score_column].to_numpy(dtype=float)
        mean_background_score = float(np.mean(background_scores))
        contrast = positive_score - mean_background_score
        comparison_rows.append(
            {
                "comparison_id": comparison_id,
                "record_id": positive["record_id"],
                "region_id": positive["region_id"],
                "label": positive["label"],
                "feature_type": positive["feature_type"],
                "region_start": positive["region_start"],
                "region_end": positive["region_end"],
                "region_length": positive["region_length"],
                "score_column": score_column,
                "positive_score": positive_score,
                "mean_background_score": mean_background_score,
                "median_background_score": float(
                    np.median(background_scores)
                ),
                "std_background_score": float(
                    np.std(background_scores, ddof=0)
                ),
                "n_backgrounds": int(len(background_scores)),
                "contrast": float(contrast),
                "win_rate": float(
                    np.mean(background_scores < positive_score)
                ),
            }
        )

    per_comparison = pd.DataFrame(
        comparison_rows,
        columns=PER_COMPARISON_COLUMNS,
    )

    summary = {
        "score_column": score_column,
        "n_comparisons": int(len(per_comparison)),
        "mean_positive_score": float(
            per_comparison["positive_score"].mean()
        ),
        "mean_background_score": float(
            per_comparison["mean_background_score"].mean()
        ),
        "mean_contrast": float(per_comparison["contrast"].mean()),
        "median_contrast": float(per_comparison["contrast"].median()),
        "fraction_positive_contrast": float(
            (per_comparison["contrast"] > 0).mean()
        ),
        "mean_win_rate": float(per_comparison["win_rate"].mean()),
    }
    return per_map_scores, per_comparison, summary


def _save_figure(figure: plt.Figure, plots_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(plots_dir / f"{stem}.png", dpi=300)
    figure.savefig(plots_dir / f"{stem}.pdf")
    plt.close(figure)


def _truncate_label(label: object, max_length: int = 60) -> str:
    text = "NA" if pd.isna(label) else str(label)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def plot_raw_per_map_scores_by_annotation(
    per_map_scores: pd.DataFrame,
    per_comparison: pd.DataFrame,
    plots_dir: Path,
) -> None:
    sorted_frame = per_comparison.sort_values(
        "positive_score",
        kind="stable",
    )
    y_values = np.arange(len(sorted_frame))
    figure_height = min(36, max(4, len(sorted_frame) * 0.25))
    score_column = str(per_map_scores["score_column"].iloc[0])

    region_ids = sorted_frame["region_id"].astype(str)
    if region_ids.duplicated().any():
        labels = [
            _truncate_label(f"{row.region_id} | {row.record_id}")
            for row in sorted_frame.itertuples(index=False)
        ]
    else:
        labels = [_truncate_label(region_id) for region_id in region_ids]

    background_x = []
    background_y = []
    background_rows = per_map_scores.loc[
        per_map_scores["map_role"] == "background"
    ]
    positive_rows = per_map_scores.loc[
        per_map_scores["map_role"] == "positive"
    ].set_index("comparison_id")
    positive_scores = []
    for y_value, comparison_id in zip(
        y_values,
        sorted_frame["comparison_id"],
        strict=True,
    ):
        positive_scores.append(
            float(positive_rows.loc[comparison_id, "segment_score"])
        )
        backgrounds = background_rows.loc[
            background_rows["comparison_id"] == comparison_id
        ].sort_values(["background_index", "map_id"], kind="stable")
        background_scores = backgrounds["segment_score"].to_numpy(dtype=float)
        if len(background_scores) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.16, 0.16, len(background_scores))
        background_x.extend(background_scores.tolist())
        background_y.extend((y_value + jitter).tolist())

    figure, axis = plt.subplots(figsize=(10, figure_height))
    background_points = axis.scatter(
        background_x,
        background_y,
        s=14,
        label="Individual background",
        color="#d1d5db",
        edgecolors="none",
        alpha=0.8,
    )
    mean_background_points = axis.scatter(
        sorted_frame["mean_background_score"],
        y_values,
        s=46,
        marker="D",
        label="Mean matched background",
        color="#2563eb",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    annotated_points = axis.scatter(
        positive_scores,
        y_values,
        s=50,
        marker="o",
        label="Annotated region",
        color="#f97316",
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )
    all_scores = np.concatenate(
        [
            np.array(background_x, dtype=float),
            sorted_frame["mean_background_score"].to_numpy(dtype=float),
            np.array(positive_scores, dtype=float),
        ]
    )
    if np.min(all_scores) < 0 < np.max(all_scores):
        axis.axvline(0, color="black", linewidth=0.8)

    label_fontsize = 8
    if len(sorted_frame) > 80:
        label_fontsize = 4.5
    elif len(sorted_frame) > 30:
        label_fontsize = 6
    axis.set_yticks(y_values)
    axis.set_yticklabels(labels, fontsize=label_fontsize)
    axis.set_ylim(-0.75, len(sorted_frame) - 0.25)
    axis.set_xlabel(f"Block score ({score_column})")
    axis.set_ylabel("Annotated region")
    axis.set_title(
        "Raw annotated and matched-background block scores\n"
        "Gray points are individual sampled backgrounds; blue diamonds are "
        "their mean; orange circles are annotated regions."
    )
    axis.legend(
        handles=[
            background_points,
            mean_background_points,
            annotated_points,
        ]
    )
    _save_figure(
        figure,
        plots_dir,
        "raw_per_map_scores_by_annotation",
    )


def write_outputs(
    run_dir: Path,
    per_map_scores: pd.DataFrame,
    per_comparison: pd.DataFrame,
    summary: dict,
    overwrite: bool,
) -> Path:
    output_dir = run_dir / "background_block_comparison"
    if output_dir.exists():
        if not overwrite:
            raise ValueError(
                f"Output directory already exists: {output_dir}; pass "
                "--overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    per_map_scores.to_parquet(
        output_dir / "per_map_scores.parquet",
        index=False,
        engine="pyarrow",
    )
    per_map_scores.to_csv(output_dir / "per_map_scores.csv", index=False)
    per_comparison.to_parquet(
        output_dir / "per_comparison.parquet",
        index=False,
        engine="pyarrow",
    )
    per_comparison.to_csv(output_dir / "per_comparison.csv", index=False)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    plot_raw_per_map_scores_by_annotation(
        per_map_scores,
        per_comparison,
        plots_dir,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare annotated block scores against matched background "
            "dependency-map controls for one completed run."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Completed run directory, e.g. outputs/runs/<run_id>",
    )
    parser.add_argument(
        "--score-column",
        choices=SCORE_COLUMNS,
        default="mean_block_score",
        help="per-map block-score column to compare",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing background_block_comparison directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_scores = read_and_validate_inputs(
        args.run_dir,
        args.score_column,
    )
    per_map_scores, per_comparison, summary = build_comparison_table(
        merged_scores,
        args.score_column,
    )
    output_dir = write_outputs(
        args.run_dir,
        per_map_scores,
        per_comparison,
        summary,
        args.overwrite,
    )

    print(f"Comparisons: {summary['n_comparisons']}")
    print(f"Score metric: {summary['score_column']}")
    print(f"Mean positive score: {summary['mean_positive_score']:.6g}")
    print(f"Mean background score: {summary['mean_background_score']:.6g}")
    print(f"Mean contrast: {summary['mean_contrast']:.6g}")
    print(
        "Positive contrast fraction: "
        f"{summary['fraction_positive_contrast']:.6g}"
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
