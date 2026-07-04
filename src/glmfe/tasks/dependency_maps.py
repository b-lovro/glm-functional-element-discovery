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
                }
            )
    elif mode == "region":
        region_config = dependency_config["region"]
        label = region_config["label"]
        configured_record_ids = region_config["record_ids"]
        long_region_policy = str(region_config["long_region_policy"])
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
        for region in selected_regions.itertuples(index=False):
            region_start = int(region.start)
            region_end = int(region.end)
            region_length = region_end - region_start
            record_id = str(region.record_id)
            region_id = str(region.region_id)
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

            for tile_index, (tile_start, tile_end) in enumerate(
                tile_intervals
            ):
                jobs.append(
                    {
                        "map_id": (
                            f"{record_id}__{region_id}__"
                            f"tile_{tile_index:03d}__"
                            f"{tile_start}_{tile_end}"
                        ),
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
                    }
                )
    else:
        raise ValueError(
            f"Unsupported dependency_maps mode: {mode}; "
            "expected manual or region"
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
        f"with_reconstruction={with_reconstruction}"
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

        context_length = dependency_config.get("context_length")

        subset = (start, end)
        if context_length is not None:
            target_len = subset[1] - subset[0]
            w_size = max(context_length, target_len)
            pad_total = w_size - target_len
            pad_left = pad_total // 2
            
            window_start = max(0, subset[0] - pad_left)
            window_end = min(sequence_length, window_start + w_size)
            
            # Adjust if window_end hit the limit and we can shift left
            actual_w_size = window_end - window_start
            if actual_w_size < w_size and window_start > 0:
                window_start = max(0, window_end - w_size)
                
            active_seq = sequence[window_start:window_end]
            active_subset = (subset[0] - window_start, subset[1] - window_start)
        else:
            window_start = start
            window_end = end
            active_seq = sequence[window_start:window_end]
            active_subset = None

        if len(active_seq) > model.max_context_length:
            raise ValueError(
                f"Dependency map window {map_id} context length "
                f"{len(active_seq)} exceeds model context length "
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

        sample_count = job_options.num_samples(len(active_seq))
        tqdm.write(
            "Computing dependency map "
            f"{map_id}: record={record_id}, subset=[{start}, {end}), "
            f"context=[{window_start}, {window_end}), "
            f"context_length={len(active_seq)}, samples={sample_count}"
        )

        # Compute and save raw map arrays.
        result = DependencyMap.compute_batched(
            active_seq,
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
            "context_sequence": np.array(list(active_seq)),
        }
        if result.reconstruction is not None:
            arrays["reconstruction"] = result.reconstruction
        np.savez_compressed(map_path, **arrays)
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
        if mode == "region":
            title = (
                f"{map_id}<br>"
                f"<sup>{record_id} region "
                f"[{job['region_start']}, {job['region_end']}) | "
                f"tile {job['tile_index']:03d} [{start}, {end})</sup>"
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
                "html_plot_path": str(relative_html_plot_path),
                "pdf_plot_path": str(relative_pdf_plot_path),
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
