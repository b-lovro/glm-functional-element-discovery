"""Compute dependency maps only for explicitly configured manual windows.

A manual window is a half-open interval ``[start, end)`` within one prepared
sequence record. This task validates and extracts each requested window, calls
the external ``dependency_map`` package through the model-independent wrapper
interface, and saves compressed ``.npz`` arrays plus visualizations. Map-index
coordinates remain relative to the full source record so outputs are
interpretable later.
"""

from functools import partial
from pathlib import Path

from matplotlib import cm as matplotlib_cm
from matplotlib import colormaps
import numpy as np
import pandas as pd
from dependency_map import DependencyMap, DependencyMapOptions

from glmfe.seq_models.base import BaseSequenceModel


_MAP_INDEX_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
    "map_id",
    "record_id",
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
    model: BaseSequenceModel,
    dependency_config: dict,
    output_dir: Path,
    run_id: str,
    dataset_id: str,
    model_id: str,
    overwrite: bool,
) -> pd.DataFrame:
    """Compute and save dependency maps for configured manual windows."""

    # Validate dependency-map configuration.
    mode = dependency_config["mode"]
    if mode != "manual":
        raise ValueError(f"Unsupported dependency_maps mode: {mode}")

    batch_size = dependency_config["batch_size"]

    dependency_by_masking = bool(dependency_config["dependency_by_masking"])
    with_reconstruction = bool(dependency_config["with_reconstruction"])
    windows = list(dependency_config["windows"])

    # Create output directories and package callbacks.
    dependency_dir = output_dir / "dependency_maps"
    maps_dir = dependency_dir / "maps"
    dependency_dir.mkdir(exist_ok=overwrite)
    maps_dir.mkdir(exist_ok=overwrite)

    options = DependencyMapOptions(
        dependency_by_masking=dependency_by_masking,
        with_reconstruction=with_reconstruction,
    )

    def tokenize_func(sequence: str, mask: int | None) -> object:
        return model.dependency_tokenize(sequence, mask)

    forward_func = partial(
        model.dependency_forward,
        batch_size=batch_size,
    )

    rows = []
    for window in windows:
        # Validate and extract each configured sequence window.
        map_id = str(window["map_id"])
        record_id = str(window["record_id"])
        start = int(window["start"])
        end = int(window["end"])

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

        window_sequence = sequence[start:end]
        if len(window_sequence) > model.max_context_length:
            raise ValueError(
                f"Dependency map window {map_id} length "
                f"{len(window_sequence)} exceeds model context length "
                f"{model.max_context_length}"
            )

        # Compute and save raw map arrays.
        result = DependencyMap.compute_batched(
            window_sequence,
            tokenize_func,
            forward_func,
            batch_size=batch_size,
            options=options,
        )

        relative_map_path = Path("dependency_maps") / "maps" / f"{map_id}.npz"
        map_path = output_dir / relative_map_path
        arrays = {
            "dependency_map": result.dependency_map,
            "sequence": np.array(window_sequence),
        }
        if result.reconstruction is not None:
            arrays["reconstruction"] = result.reconstruction
        np.savez_compressed(map_path, **arrays)

        # Create and save visualizations.
        relative_html_plot_path = (
            Path("dependency_maps") / "maps" / f"{map_id}.html"
        )
        relative_pdf_plot_path = (
            Path("dependency_maps") / "maps" / f"{map_id}.pdf"
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
        figure.update_layout(
            title=(
                f"{map_id}<br>"
                f"<sup>{record_id} [{start}, {end})</sup>"
            ),
            margin={"l": 80, "r": 80, "t": 180, "b": 80},
        )
        window_left = start / sequence_length
        window_right = end / sequence_length
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
                f"Full record: {sequence_length} nt | displayed window: "
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

        rows.append(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "map_id": map_id,
                "record_id": record_id,
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
