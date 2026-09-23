"""Score local blocks from previously computed dependency maps.

This task is deterministic post-processing of saved dependency-map arrays and
does not run model inference.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


_REQUIRED_MAP_INDEX_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
    "map_id",
    "record_id",
    "region_id",
    "label",
    "feature_type",
    "region_start",
    "region_end",
    "region_length",
    "tile_index",
    "tile_start",
    "tile_end",
    "tile_length",
    "start",
    "end",
    "length",
    "map_path",
]
_PER_SPAN_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
    "map_id",
    "record_id",
    "region_id",
    "label",
    "feature_type",
    "region_start",
    "region_end",
    "region_length",
    "tile_index",
    "tile_start",
    "tile_end",
    "tile_length",
    "map_start",
    "map_end",
    "map_length",
    "local_start",
    "local_end",
    "span_start",
    "span_end",
    "span_length",
    "block_size",
    "quantile",
    "block_score",
]
_PER_MAP_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
    "map_id",
    "record_id",
    "region_id",
    "label",
    "feature_type",
    "region_start",
    "region_end",
    "region_length",
    "tile_index",
    "tile_start",
    "tile_end",
    "tile_length",
    "map_start",
    "map_end",
    "map_length",
    "block_size",
    "quantile",
    "n_spans",
    "mean_block_score",
    "median_block_score",
    "max_block_score",
]


def run_block_scores(
    map_index: pd.DataFrame,
    block_config: dict,
    output_dir: Path,
    overwrite: bool,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute block-score tables from saved dependency-map arrays."""

    block_size = int(block_config["block_size"])
    quantile = float(block_config["quantile"])
    if block_size < 2:
        raise ValueError("block_scores.block_size must be at least 2")
    if quantile < 0 or quantile > 1:
        raise ValueError("block_scores.quantile must be between 0 and 1")

    missing_columns = [
        column
        for column in _REQUIRED_MAP_INDEX_COLUMNS
        if column not in map_index.columns
    ]
    if missing_columns:
        raise ValueError(
            "map_index is missing required columns: "
            + ", ".join(missing_columns)
        )

    block_dir = output_dir / "block_scores"
    block_dir.mkdir(parents=True, exist_ok=overwrite or resume)

    per_span_path = block_dir / "per_span.parquet"
    per_map_path = block_dir / "per_map.parquet"
    if resume and per_span_path.is_file() and per_map_path.is_file():
        try:
            per_span = pd.read_parquet(per_span_path)
            per_map = pd.read_parquet(per_map_path)
            print("Block scores: already computed, loaded existing parquet files.")
            return per_span, per_map
        except Exception as e:
            print(f"Warning: Corrupted block score parquet files ({e}), recomputing.")

    off_diagonal_mask = ~np.eye(block_size, dtype=bool)

    span_rows = []
    map_rows = []
    for row in tqdm(
        map_index.itertuples(index=False),
        total=len(map_index),
        desc="Block scores",
        unit="map",
    ):
        map_path = output_dir / row.map_path
        if not map_path.exists():
            raise ValueError(
                f"Dependency-map file for map {row.map_id} does not exist: "
                f"{map_path}"
            )

        with np.load(map_path, allow_pickle=False) as arrays:
            if "dependency_map" not in arrays.files:
                raise ValueError(
                    f"Dependency-map file for map {row.map_id} does not "
                    "contain dependency_map"
                )
            dependency_map = arrays["dependency_map"]

        if (
            dependency_map.ndim != 2
            or dependency_map.shape[0] != dependency_map.shape[1]
        ):
            raise ValueError(
                f"dependency_map for map {row.map_id} must be a 2D square "
                f"matrix, found shape {dependency_map.shape}"
            )

        map_length = int(row.length)
        if dependency_map.shape[0] != map_length:
            raise ValueError(
                f"dependency_map for map {row.map_id} has side length "
                f"{dependency_map.shape[0]}, but map_index length is "
                f"{map_length}"
            )
        if map_length < block_size:
            raise ValueError(
                f"dependency_map for map {row.map_id} has length "
                f"{map_length}, which is smaller than block_size "
                f"{block_size}"
            )

        map_start = int(row.start)
        map_end = int(row.end)
        scores = []
        for local_start in range(map_length - block_size + 1):
            local_end = local_start + block_size
            dependency_block = dependency_map[
                local_start:local_end,
                local_start:local_end,
            ]
            off_diagonal_values = dependency_block[off_diagonal_mask]
            block_score = float(np.quantile(off_diagonal_values, quantile))
            span_start = map_start + local_start
            span_end = map_start + local_end
            scores.append(block_score)
            span_rows.append(
                {
                    "run_id": row.run_id,
                    "dataset_id": row.dataset_id,
                    "model_id": row.model_id,
                    "map_id": row.map_id,
                    "record_id": row.record_id,
                    "region_id": row.region_id,
                    "label": row.label,
                    "feature_type": row.feature_type,
                    "region_start": row.region_start,
                    "region_end": row.region_end,
                    "region_length": row.region_length,
                    "tile_index": row.tile_index,
                    "tile_start": row.tile_start,
                    "tile_end": row.tile_end,
                    "tile_length": row.tile_length,
                    "map_start": map_start,
                    "map_end": map_end,
                    "map_length": map_length,
                    "local_start": local_start,
                    "local_end": local_end,
                    "span_start": span_start,
                    "span_end": span_end,
                    "span_length": block_size,
                    "block_size": block_size,
                    "quantile": quantile,
                    "block_score": block_score,
                }
            )

        score_array = np.array(scores, dtype=float)
        map_rows.append(
            {
                "run_id": row.run_id,
                "dataset_id": row.dataset_id,
                "model_id": row.model_id,
                "map_id": row.map_id,
                "record_id": row.record_id,
                "region_id": row.region_id,
                "label": row.label,
                "feature_type": row.feature_type,
                "region_start": row.region_start,
                "region_end": row.region_end,
                "region_length": row.region_length,
                "tile_index": row.tile_index,
                "tile_start": row.tile_start,
                "tile_end": row.tile_end,
                "tile_length": row.tile_length,
                "map_start": map_start,
                "map_end": map_end,
                "map_length": map_length,
                "block_size": block_size,
                "quantile": quantile,
                "n_spans": len(scores),
                "mean_block_score": float(np.mean(score_array)),
                "median_block_score": float(np.median(score_array)),
                "max_block_score": float(np.max(score_array)),
            }
        )

    per_span = pd.DataFrame(span_rows, columns=_PER_SPAN_COLUMNS)
    per_map = pd.DataFrame(map_rows, columns=_PER_MAP_COLUMNS)
    per_span.to_parquet(
        block_dir / "per_span.parquet",
        index=False,
        engine="pyarrow",
    )
    per_map.to_parquet(
        block_dir / "per_map.parquet",
        index=False,
        engine="pyarrow",
    )
    return per_span, per_map
