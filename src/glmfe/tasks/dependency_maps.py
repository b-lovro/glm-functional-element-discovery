"""Compute dependency maps for configured manual or annotation regions.

Each map uses a half-open interval ``[start, end)`` within one prepared sequence
record. This task builds windows from explicit manual coordinates or matching
prepared annotations, validates and extracts each sequence, calls the external
``dependency_map`` package through the model-independent wrapper interface, and
saves compressed ``.npz`` arrays plus visualizations. Map-index coordinates
remain relative to the full source record so outputs are interpretable later.
"""

from functools import partial
from pathlib import Path

from matplotlib import cm as matplotlib_cm
from matplotlib import colormaps
import numpy as np
import pandas as pd
from dependency_map import DependencyMap, DependencyMapOptions
from tqdm import tqdm

from glmfe.seq_models.base import BaseSequenceModel


_MAP_INDEX_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
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
    "tile_index",
    "tile_start",
    "tile_end",
    "tile_length",
    "long_region_policy",
    "start",
    "end",
    "length",
    "mode",
    "dependency_by_masking",
    "with_reconstruction",
    "map_path",
    "html_plot_path",
    "pdf_plot_path",
]


def _region_value(region: object, field: str) -> object:
    if isinstance(region, pd.Series):
        return region[field]
    return getattr(region, field)


def _region_interval(region: object) -> tuple[int, int]:
    return int(_region_value(region, "start")), int(
        _region_value(region, "end")
    )


def _region_id(region: object) -> str:
    return str(_region_value(region, "region_id"))


def _infer_parent_region(
    target_region: object,
    record_regions: pd.DataFrame,
) -> pd.Series:
    target_record_id = str(_region_value(target_region, "record_id"))
    target_start, target_end = _region_interval(target_region)
    target_length = target_end - target_start
    target_region_id = _region_id(target_region)
    same_record_regions = record_regions.loc[
        record_regions["record_id"] == target_record_id
    ]

    candidate_parents = same_record_regions.loc[
        (same_record_regions["start"] <= target_start)
        & (same_record_regions["end"] >= target_end)
        & (
            (same_record_regions["end"] - same_record_regions["start"])
            > target_length
        )
    ].copy()
    if candidate_parents.empty:
        raise ValueError(
            f"Background sampling for target region {target_region_id} "
            f"on record {target_record_id} requires a strict parent "
            "annotation"
        )

    candidate_parents["_parent_length"] = (
        candidate_parents["end"] - candidate_parents["start"]
    )
    candidate_parents = candidate_parents.sort_values(
        ["_parent_length", "start", "end", "region_id"],
        kind="stable",
        ascending=False,
    )
    return candidate_parents.iloc[0]


def _intervals_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    return start_a < end_b and start_b < end_a


def sample_background_starts(
    target_region: object,
    record_regions: pd.DataFrame,
    n_per_region: int,
    min_distance_bp: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample simple same-parent background starts for one target region.

    The first sampler only uses annotation coordinates: it samples fixed-length
    windows inside the smallest strict parent while keeping a configurable
    buffer away from annotations inside that parent.
    """

    parent_region = _infer_parent_region(target_region, record_regions)
    parent_start, parent_end = _region_interval(parent_region)
    target_start, target_end = _region_interval(target_region)
    target_length = target_end - target_start
    parent_region_id = _region_id(parent_region)
    target_record_id = str(_region_value(target_region, "record_id"))
    same_record_regions = record_regions.loc[
        record_regions["record_id"] == target_record_id
    ]

    annotation_intervals = []
    for annotation in same_record_regions.itertuples(index=False):
        annotation_start, annotation_end = _region_interval(annotation)
        if _region_id(annotation) == parent_region_id:
            continue
        annotation_intervals.append((annotation_start, annotation_end))

    valid_starts = []
    for start in range(parent_start, parent_end - target_length + 1):
        buffered_start = start - min_distance_bp
        buffered_end = start + target_length + min_distance_bp
        overlaps_annotation = any(
            _intervals_overlap(
                buffered_start,
                buffered_end,
                annotation_start,
                annotation_end,
            )
            for annotation_start, annotation_end in annotation_intervals
        )
        if not overlaps_annotation:
            valid_starts.append(start)

    if len(valid_starts) < n_per_region:
        print(
            "WARNING: Background sampling for target region "
            f"{_region_id(target_region)} on record {target_record_id} found "
            f"{len(valid_starts)} valid candidate start(s), but "
            f"n_per_region={n_per_region}"
        )
        n_per_region = len(valid_starts)
    if not valid_starts:
        return []

    sampled = rng.choice(
        valid_starts,
        size=n_per_region,
        replace=False,
    )
    return [int(start) for start in sampled.tolist()]


def run_dependency_maps(
    records: pd.DataFrame,
    regions: pd.DataFrame,
    model: BaseSequenceModel,
    dependency_config: dict,
    output_dir: Path,
    run_id: str,
    dataset_id: str,
    model_id: str,
    overwrite: bool,
) -> pd.DataFrame:
    """Compute and save configured manual or region dependency maps."""

    mode = str(dependency_config["mode"])
    batch_size = int(dependency_config["batch_size"])
    dependency_by_masking = bool(dependency_config["dependency_by_masking"])
    with_reconstruction = bool(dependency_config["with_reconstruction"])
    plot_results = bool(dependency_config.get("plot_results", True))
    configured_context_length = dependency_config.get("context_length")
    context_length = (
        None
        if configured_context_length is None
        else int(configured_context_length)
    )
    if context_length is not None and context_length < 1:
        raise ValueError("dependency_maps.context_length must be at least 1")
    # Build manual or region jobs from the configuration.
    jobs = []
    if mode == "manual":
        manual_config = dependency_config["manual"]
        start = int(manual_config["start"])
        end = int(manual_config["end"])
        record_ids = list(manual_config["record_ids"])
        for configured_record_id in record_ids:
            record_id = str(configured_record_id)
            jobs.append(
                {
                    "map_id": f"{record_id}__{start}_{end}",
                    "record_id": record_id,
                    "region_id": None,
                    "label": None,
                    "feature_type": None,
                    "region_start": None,
                    "region_end": None,
                    "region_length": None,
                    "tile_index": None,
                    "tile_start": None,
                    "tile_end": None,
                    "tile_length": None,
                    "long_region_policy": None,
                    "start": start,
                    "end": end,
                    "comparison_id": None,
                    "map_role": None,
                    "background_index": None,
                }
            )
    elif mode == "region":
        region_config = dependency_config["region"]
        label = region_config["label"]
        configured_record_ids = region_config["record_ids"]
        long_region_policy = str(region_config["long_region_policy"])
        background_config = region_config.get("background", {})
        if background_config is None:
            background_config = {}
        if not isinstance(background_config, dict):
            raise ValueError(
                "dependency_maps.region.background must be a mapping"
            )
        background_enabled = bool(background_config.get("enabled", False))
        if background_enabled:
            required_keys = ["n_per_region", "min_distance_bp", "seed"]
            missing_keys = [
                key for key in required_keys if key not in background_config
            ]
            if missing_keys:
                raise ValueError(
                    "dependency_maps.region.background is enabled but "
                    "missing required key(s): "
                    + ", ".join(missing_keys)
                )

            n_backgrounds_per_region = int(
                background_config["n_per_region"]
            )
            background_min_distance_bp = int(
                background_config["min_distance_bp"]
            )
            background_seed = int(background_config["seed"])
            if n_backgrounds_per_region < 1:
                raise ValueError(
                    "dependency_maps.region.background.n_per_region must "
                    "be at least 1"
                )
            if background_min_distance_bp < 0:
                raise ValueError(
                    "dependency_maps.region.background.min_distance_bp "
                    "must be non-negative"
                )
            if background_seed < 0:
                raise ValueError(
                    "dependency_maps.region.background.seed must be "
                    "non-negative"
                )
            background_rng = np.random.default_rng(background_seed)
        else:
            background_rng = None
        if long_region_policy not in {"error", "tile"}:
            raise ValueError(
                "dependency_maps.region.long_region_policy must be "
                "'error' or 'tile'"
            )
        if long_region_policy == "tile":
            tile_length = int(region_config["tile_length"])
            tile_stride = int(region_config["tile_stride"])
            if tile_length > model.max_context_length:
                raise ValueError(
                    f"dependency_maps.region.tile_length {tile_length} "
                    f"exceeds model context length "
                    f"{model.max_context_length}"
                )
        selected_regions = regions.loc[
            (regions["label"] == label)
            | (regions["feature_type"] == label)
        ]
        if configured_record_ids == "half":
            configured_record_ids = (
                selected_regions["record_id"]
                .drop_duplicates()
                .iloc[: max(1, selected_regions["record_id"].nunique() // 2)]
                .tolist()
            )
        if configured_record_ids != "all":
            selected_regions = selected_regions.loc[
                selected_regions["record_id"].isin(configured_record_ids)
            ]
        if selected_regions.empty:
            raise ValueError(
                f"No dependency-map regions matched label or feature type "
                f"{label!r} "
                f"and record_ids {configured_record_ids!r}"
            )
        skipped_background_regions = []
        for region in selected_regions.itertuples(index=False):
            region_start = int(region.start)
            region_end = int(region.end)
            region_length = region_end - region_start
            record_id = str(region.record_id)
            region_id = str(region.region_id)
            comparison_id = f"{record_id}__{region_id}"
            if background_enabled:
                if region_length > model.max_context_length:
                    raise ValueError(
                        f"Background dependency maps for region {region_id} "
                        f"on record {record_id} require the target length "
                        f"{region_length} to be at most model context "
                        f"length {model.max_context_length}; tiled "
                        "background maps are not supported yet"
                    )

            if region_length <= model.max_context_length:
                tile_intervals = [(region_start, region_end)]
            elif long_region_policy == "error":
                raise ValueError(
                    f"Dependency-map region {region_id} for record "
                    f"{record_id} [{region_start}, {region_end}) has length "
                    f"{region_length}, which exceeds model context length "
                    f"{model.max_context_length}; use long_region_policy: "
                    "tile to split it into local dependency-map windows"
                )
            else:
                tile_intervals = []
                for tile_start in range(
                    region_start,
                    region_end - tile_length + 1,
                    tile_stride,
                ):
                    tile_intervals.append(
                        (tile_start, tile_start + tile_length)
                    )
                final_tile = (
                    region_end - tile_length,
                    region_end,
                )
                if tile_intervals[-1] != final_tile:
                    tile_intervals.append(final_tile)

            background_starts = None
            if background_enabled:
                record_regions = regions.loc[
                    regions["record_id"] == record_id
                ]
                try:
                    background_starts = sample_background_starts(
                        region,
                        record_regions,
                        n_backgrounds_per_region,
                        background_min_distance_bp,
                        background_rng,
                    )
                except ValueError as error:
                    skipped_background_regions.append(region_id)
                    print(
                        "WARNING: Skipping dependency-map region "
                        f"{region_id} on record {record_id}; "
                        "no matched backgrounds are available. "
                        f"{error}"
                    )
                    continue
                if not background_starts:
                    skipped_background_regions.append(region_id)
                    print(
                        "WARNING: Skipping dependency-map region "
                        f"{region_id} on record {record_id}; "
                        "no matched backgrounds are available."
                    )
                    continue

            for tile_index, (tile_start, tile_end) in enumerate(
                tile_intervals
            ):
                positive_map_id = (
                    f"{record_id}__{region_id}__"
                    f"tile_{tile_index:03d}__"
                    f"{tile_start}_{tile_end}"
                )
                jobs.append(
                    {
                        "map_id": positive_map_id,
                        "record_id": record_id,
                        "region_id": region_id,
                        "label": region.label,
                        "feature_type": region.feature_type,
                        "region_start": region_start,
                        "region_end": region_end,
                        "region_length": region_length,
                        "tile_index": tile_index,
                        "tile_start": tile_start,
                        "tile_end": tile_end,
                        "tile_length": tile_end - tile_start,
                        "long_region_policy": long_region_policy,
                        "start": tile_start,
                        "end": tile_end,
                        "comparison_id": comparison_id,
                        "map_role": "positive",
                        "background_index": None,
                    }
                )
                if background_enabled:
                    for background_index, background_start in enumerate(
                        background_starts
                    ):
                        background_end = background_start + region_length
                        jobs.append(
                            {
                                "map_id": (
                                    f"{positive_map_id}__background_"
                                    f"{background_index:03d}"
                                ),
                                "record_id": record_id,
                                "region_id": region_id,
                                "label": region.label,
                                "feature_type": region.feature_type,
                                "region_start": region_start,
                                "region_end": region_end,
                                "region_length": region_length,
                                "tile_index": None,
                                "tile_start": None,
                                "tile_end": None,
                                "tile_length": None,
                                "long_region_policy": long_region_policy,
                                "start": background_start,
                                "end": background_end,
                                "comparison_id": comparison_id,
                                "map_role": "background",
                                "background_index": background_index,
                            }
                        )
    elif mode == "full_sequence":
        full_sequence_config = dependency_config["full_sequence"]
        configured_record_ids = full_sequence_config["record_ids"]
        long_region_policy = str(full_sequence_config["long_region_policy"])
        tile_length = int(full_sequence_config["tile_length"])
        tile_stride = int(full_sequence_config["tile_stride"])

        if long_region_policy != "tile":
            raise ValueError(
                "dependency_maps.full_sequence.long_region_policy must be 'tile'"
            )
        if tile_length > model.max_context_length:
            raise ValueError(
                f"dependency_maps.full_sequence.tile_length {tile_length} "
                f"exceeds model context length {model.max_context_length}"
            )
        if tile_stride < 1:
            raise ValueError(
                f"dependency_maps.full_sequence.tile_stride {tile_stride} must be at least 1"
            )
        if tile_stride > tile_length:
            raise ValueError(
                f"dependency_maps.full_sequence.tile_stride {tile_stride} "
                f"cannot exceed tile_length {tile_length}"
            )

        if configured_record_ids == "half":
            selected_records = records.iloc[: max(1, len(records) // 2)]
        elif configured_record_ids == "all":
            selected_records = records
        elif isinstance(configured_record_ids, list):
            selected_records = records.loc[
                records["record_id"].isin(configured_record_ids)
            ]
        else:
            raise ValueError(
                "dependency_maps.full_sequence.record_ids must be 'all', 'half', "
                f"or a list of record IDs; got {configured_record_ids!r}"
            )

        if selected_records.empty:
            raise ValueError(
                "No records matched dependency_maps.full_sequence.record_ids: "
                f"{configured_record_ids!r}"
            )

        for record in selected_records.itertuples(index=False):
            record_id = str(record.record_id)
            sequence_length = len(record.sequence)
            region_id = f"{record_id}:full_sequence"
            comparison_id = f"{record_id}__full_sequence"

            if sequence_length <= tile_length:
                tile_intervals = [(0, sequence_length)]
            else:
                tile_intervals = []
                for tile_start in range(
                    0,
                    sequence_length - tile_length + 1,
                    tile_stride,
                ):
                    tile_intervals.append(
                        (tile_start, tile_start + tile_length)
                    )
                final_tile = (
                    sequence_length - tile_length,
                    sequence_length,
                )
                if tile_intervals[-1] != final_tile:
                    tile_intervals.append(final_tile)

            for tile_index, (tile_start, tile_end) in enumerate(tile_intervals):
                positive_map_id = (
                    f"{record_id}__full_sequence__"
                    f"tile_{tile_index:03d}__"
                    f"{tile_start}_{tile_end}"
                )
                jobs.append(
                    {
                        "map_id": positive_map_id,
                        "record_id": record_id,
                        "region_id": region_id,
                        "label": "full_sequence",
                        "feature_type": "full_sequence",
                        "region_start": 0,
                        "region_end": sequence_length,
                        "region_length": sequence_length,
                        "tile_index": tile_index,
                        "tile_start": tile_start,
                        "tile_end": tile_end,
                        "tile_length": tile_end - tile_start,
                        "long_region_policy": long_region_policy,
                        "start": tile_start,
                        "end": tile_end,
                        "comparison_id": comparison_id,
                        "map_role": "positive",
                        "background_index": None,
                    }
                )
    else:
        raise ValueError(
            f"Unsupported dependency_maps mode: {mode}; "
            "expected manual, region, or full_sequence"
        )

    # Create output directories and package callbacks.
    dependency_dir = output_dir / "dependency_maps"
    maps_dir = dependency_dir / "maps"
    dependency_dir.mkdir(exist_ok=overwrite)
    maps_dir.mkdir(exist_ok=overwrite)

    options = DependencyMapOptions(
        dependency_by_masking=dependency_by_masking,
        with_reconstruction=with_reconstruction,
        autoregressive=bool(
            getattr(model, "dependency_autoregressive", False)
        ),
    )
    print(
        "Dependency maps: "
        f"{len(jobs)} job(s), mode={mode}, batch_size={batch_size}, "
        f"autoregressive={options.autoregressive}, "
        f"dependency_by_masking={dependency_by_masking}, "
        f"with_reconstruction={with_reconstruction}, "
        f"plot_results={plot_results}"
    )

    def tokenize_func(sequence: str, mask: int | None) -> object:
        return model.dependency_tokenize(sequence, mask)

    forward_func = partial(
        model.dependency_forward,
        batch_size=batch_size,
    )

    rows = []
    # Process every job through shared validation, inference, and output logic.
    for job in tqdm(jobs, desc="Dependency maps", unit="map"):
        map_id = job["map_id"]
        record_id = job["record_id"]
        start = job["start"]
        end = job["end"]
        relative_record_maps_dir = (
            Path("dependency_maps") / "maps" / record_id
        )
        (output_dir / relative_record_maps_dir).mkdir(exist_ok=True)

        matching_records = records.loc[records["record_id"] == record_id]
        if len(matching_records) != 1:
            raise ValueError(
                f"Dependency map {map_id} requires exactly one record "
                f"{record_id}, found {len(matching_records)}"
            )

        sequence = matching_records.iloc[0]["sequence"]
        sequence_length = len(sequence)
        if not 0 <= start < end <= sequence_length:
            raise ValueError(
                f"Invalid dependency map window {map_id}: "
                f"[{start}, {end}) for sequence length {sequence_length}"
            )

        target_length = end - start
        if context_length is None:
            window_start = start
            window_end = end
            active_subset = None
        else:
            window_length = max(context_length, target_length)
            pad_total = window_length - target_length
            pad_left = pad_total // 2
            window_start = max(0, start - pad_left)
            window_end = min(sequence_length, window_start + window_length)

            actual_window_length = window_end - window_start
            if actual_window_length < window_length and window_start > 0:
                window_start = max(0, window_end - window_length)

            active_subset = (start - window_start, end - window_start)

        map_sequence = sequence[window_start:window_end]
        if len(map_sequence) > model.max_context_length:
            raise ValueError(
                f"Dependency map window {map_id} context length "
                f"{len(map_sequence)} exceeds model context length "
                f"{model.max_context_length}"
            )

        job_options = DependencyMapOptions(
            dependency_by_masking=dependency_by_masking,
            with_reconstruction=with_reconstruction,
            autoregressive=bool(
                getattr(model, "dependency_autoregressive", False)
            ),
            subset=active_subset,
        )

        sample_count = job_options.num_samples(len(map_sequence))
        tqdm.write(
            "Computing dependency map "
            f"{map_id}: record={record_id}, subset=[{start}, {end}), "
            f"context=[{window_start}, {window_end}), "
            f"context_length={len(map_sequence)}, samples={sample_count}"
        )

        # Compute and save raw map arrays.
        result = DependencyMap.compute_batched(
            map_sequence,
            tokenize_func,
            forward_func,
            batch_size=batch_size,
            options=job_options,
        )

        relative_map_path = relative_record_maps_dir / f"{map_id}.npz"
        map_path = output_dir / relative_map_path
        arrays = {
            "dependency_map": result.dependency_map,
            "sequence": np.array(list(result.sequence)),
        }
        if result.reconstruction is not None:
            arrays["reconstruction"] = result.reconstruction
        np.savez_compressed(map_path, **arrays)
        relative_html_plot_path = None
        relative_pdf_plot_path = None
        if plot_results:
            tqdm.write(f"Writing dependency map plots: {map_id}")

            # Create and save visualizations.
            relative_html_plot_path = (
                relative_record_maps_dir / f"{map_id}.html"
            )
            relative_pdf_plot_path = (
                relative_record_maps_dir / f"{map_id}.pdf"
            )
            if not hasattr(matplotlib_cm, "get_cmap"):
                matplotlib_cm.get_cmap = colormaps.get_cmap
            figure = result.plot()
            figure.update_traces(
                colorbar_title_text="Dependency",
                hovertemplate=(
                    "Affected position: %{x}<br>"
                    "Changed position: %{y}<br>"
                    "Dependency: %{z:.4f}<extra></extra>"
                ),
            )
            if mode == "region" and job["tile_index"] is not None:
                title = (
                    f"{map_id}<br>"
                    f"<sup>{record_id} region "
                    f"[{job['region_start']}, {job['region_end']}) | "
                    f"tile {job['tile_index']:03d} [{start}, {end})"
                    "</sup>"
                )
            else:
                title = (
                    f"{map_id}<br>"
                    f"<sup>{record_id} [{start}, {end})</sup>"
                )
            figure.update_layout(
                title=title,
                margin={"l": 80, "r": 80, "t": 180, "b": 80},
            )
            window_left = start / sequence_length
            window_right = end / sequence_length
            context_left = window_start / sequence_length
            context_right = window_end / sequence_length
            figure.add_shape(
                type="rect",
                x0=0,
                x1=1,
                y0=1.08,
                y1=1.12,
                xref="paper",
                yref="paper",
                fillcolor="#e5e7eb",
                line={"color": "#9ca3af", "width": 1},
            )
            figure.add_shape(
                type="rect",
                x0=context_left,
                x1=context_right,
                y0=1.08,
                y1=1.12,
                xref="paper",
                yref="paper",
                fillcolor="#93c5fd",
                line={"color": "#60a5fa", "width": 1},
            )
            figure.add_shape(
                type="rect",
                x0=window_left,
                x1=window_right,
                y0=1.08,
                y1=1.12,
                xref="paper",
                yref="paper",
                fillcolor="#2563eb",
                line={"color": "#1d4ed8", "width": 1},
            )
            figure.add_annotation(
                x=0.5,
                y=1.15,
                xref="paper",
                yref="paper",
                text=(
                    f"Full record: {sequence_length} nt | context window: "
                    f"[{window_start}, {window_end}) | active subset: "
                    f"[{start}, {end}) ({end - start} nt)"
                ),
                showarrow=False,
            )
            figure.write_html(
                output_dir / relative_html_plot_path,
                include_plotlyjs=True,
            )
            figure.write_image(
                output_dir / relative_pdf_plot_path,
                format="pdf",
            )
        tqdm.write(f"Finished dependency map: {map_id}")

        rows.append(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "map_id": map_id,
                "comparison_id": job["comparison_id"],
                "map_role": job["map_role"],
                "background_index": job["background_index"],
                "record_id": record_id,
                "region_id": job["region_id"],
                "label": job["label"],
                "feature_type": job["feature_type"],
                "region_start": job["region_start"],
                "region_end": job["region_end"],
                "region_length": job["region_length"],
                "tile_index": job["tile_index"],
                "tile_start": job["tile_start"],
                "tile_end": job["tile_end"],
                "tile_length": job["tile_length"],
                "long_region_policy": job["long_region_policy"],
                "start": start,
                "end": end,
                "length": end - start,
                "mode": mode,
                "dependency_by_masking": dependency_by_masking,
                "with_reconstruction": with_reconstruction,
                "map_path": str(relative_map_path),
                "html_plot_path": (
                    None
                    if relative_html_plot_path is None
                    else str(relative_html_plot_path)
                ),
                "pdf_plot_path": (
                    None
                    if relative_pdf_plot_path is None
                    else str(relative_pdf_plot_path)
                ),
            }
        )

    # Write the map-index table.
    map_index = pd.DataFrame(rows, columns=_MAP_INDEX_COLUMNS)
    map_index.to_parquet(
        dependency_dir / "map_index.parquet",
        index=False,
        engine="pyarrow",
    )
    return map_index
