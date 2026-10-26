#!/usr/bin/env python3
"""Plot per-pattern seqlet distance distributions relative to the TSS and 28S 3'-end (3' ETS/UTR).

This script loads a TF-MoDISco HDF5 results file (`results.h5`), candidate window metadata
(`candidate_metadata.parquet`), and rDNA genomic region annotations (`regions.parquet`),
computes each seqlet's midpoint distance from:
  1. TSS (Transcription Start Site: 3' boundary of Core or CompositePromoter)
  2. 28S 3'-end / 3' ETS start (3' boundary of 28S_rRNA, upstream of the Terminator)

For each discovered pattern (across `pos_patterns` and `neg_patterns`), it writes a
two-panel PNG into the specified output directory with:
  - X-axis: Genomic distance from the landmark (bp)
  - Y-axis: Seqlet concentration (Gaussian KDE density or local count density)
  - Points: Each individual seqlet plotted as a scatter point at (distance, concentration)

Usage:
  /mnt/Dati/miniconda/envs/group6-rinalmo/bin/python scripts/plot_seqlet_distance_distributions.py \\
    --results outputs/tfmodisco_runs/all_seq_n/results.h5 \\
    --metadata outputs/tfmodisco_inputs/all_seq/candidate_metadata.parquet \\
    --regions data/prepared/ribosome/regions.parquet \\
    --output-dir outputs/tfmodisco_runs/all_seq_n/seqlet_distance_distributions
"""

from __future__ import annotations

import os

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Dynamically allocate compute device (CUDA, MPS, or CPU)."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def pattern_sort_key(full_pattern_name: str) -> Tuple[int, int]:
    """Sort pos_patterns before neg_patterns, then numerically by pattern index."""
    parts = full_pattern_name.split(".")
    group_Order = 0 if parts[0] == "pos_patterns" else 1
    idx_str = parts[-1].split("_")[-1]
    return (group_Order, int(idx_str))


def extract_seqlet_landmarks(
    record_ids: List[str],
    regions_path: Path,
    use_terminator_end: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Extract TSS and 28S 3'-end (or Terminator) coordinates for each record_id."""
    if not regions_path.is_file():
        raise FileNotFoundError(f"Regions parquet not found at: {regions_path}")

    regions_df = pd.read_parquet(regions_path)
    required_cols = ["record_id", "feature_type", "start", "end"]
    for col in required_cols:
        if col not in regions_df.columns:
            raise KeyError(
                f"Required column '{col}' missing from regions parquet: {regions_path}"
            )

    landmarks: Dict[str, Dict[str, float]] = {}
    for rec_id in record_ids:
        sub = regions_df[regions_df["record_id"] == rec_id]
        s28 = sub[sub["feature_type"] == "28S_rRNA"]
        terms = sub[sub["feature_type"] == "Terminator"]
        cores = sub[sub["feature_type"] == "Core"]
        comps = sub[sub["feature_type"] == "CompositePromoter"]

        if use_terminator_end:
            term_pos = (
                float(terms["end"].max())
                if len(terms) > 0
                else (float(s28["end"].max()) if len(s28) > 0 else np.nan)
            )
        else:
            term_pos = float(s28["end"].max()) if len(s28) > 0 else np.nan

        tss_pos = (
            float(cores["end"].max())
            if len(cores) > 0
            else (float(comps["end"].max()) if len(comps) > 0 else np.nan)
        )

        landmarks[rec_id] = {
            "tss_pos": tss_pos,
            "term_pos": term_pos,
        }
    return landmarks


def load_pattern_seqlets(
    h5_file: h5py.File, group_name: str, pattern_name: str
) -> pd.DataFrame:
    """Extract seqlet coordinates and metadata from a single pattern group in results.h5."""
    pattern_grp = h5_file[group_name][pattern_name]
    if "seqlets" not in pattern_grp:
        raise KeyError(
            f"Missing 'seqlets' dataset in HDF5 group '{group_name}/{pattern_name}'."
        )

    seqlet_grp = pattern_grp["seqlets"]
    required_keys = ["example_idx", "start", "end", "is_revcomp"]
    for k in required_keys:
        if k not in seqlet_grp:
            raise KeyError(
                f"Missing required key '{k}' in '{group_name}/{pattern_name}/seqlets'."
            )

    example_idxs = seqlet_grp["example_idx"][:]
    starts = seqlet_grp["start"][:]
    ends = seqlet_grp["end"][:]
    is_revcomps = seqlet_grp["is_revcomp"][:]

    return pd.DataFrame(
        {
            "pattern_name": f"{group_name}.{pattern_name}",
            "group_name": group_name,
            "pattern": pattern_name,
            "example_idx": example_idxs,
            "seqlet_start": starts,
            "seqlet_end": ends,
            "is_revcomp": is_revcomps,
        }
    )


def build_seqlet_distance_table(
    results_path: Path,
    metadata_path: Path,
    regions_path: Path,
    pattern_filter: Optional[str] = None,
    use_terminator_end: bool = False,
) -> pd.DataFrame:
    """Load all seqlets from results.h5, join with window metadata, and compute landmark distances."""
    if not results_path.is_file():
        raise FileNotFoundError(f"TF-MoDISco HDF5 file not found: {results_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Candidate metadata parquet not found: {metadata_path}")

    meta_df = pd.read_parquet(metadata_path)
    required_meta_cols = ["record_id", "window_start", "window_end", "peak_B_delta"]
    for col in required_meta_cols:
        if col not in meta_df.columns:
            raise KeyError(
                f"Missing required column '{col}' in metadata parquet: {metadata_path}"
            )

    all_seqlets: List[pd.DataFrame] = []
    with h5py.File(results_path, "r") as h5:
        for group_name in ["pos_patterns", "neg_patterns"]:
            if group_name not in h5:
                continue
            for pattern_name in h5[group_name].keys():
                full_name = f"{group_name}.{pattern_name}"
                if pattern_filter is not None and pattern_filter not in (
                    full_name,
                    pattern_name,
                ):
                    continue
                seqlets_df = load_pattern_seqlets(h5, group_name, pattern_name)
                if not seqlets_df.empty:
                    all_seqlets.append(seqlets_df)

    if not all_seqlets:
        raise ValueError(
            f"No seqlets found in {results_path} matching filter={pattern_filter!r}."
        )

    combined = pd.concat(all_seqlets, ignore_index=True)
    max_idx = int(combined["example_idx"].max())
    if max_idx >= len(meta_df):
        raise IndexError(
            f"Seqlet example_idx {max_idx} exceeds candidate_metadata length {len(meta_df)}. "
            f"Verify that {metadata_path} matches {results_path}."
        )

    meta_sub = meta_df.iloc[combined["example_idx"].to_numpy()].reset_index(drop=True)
    joined = pd.concat([combined, meta_sub], axis=1)

    joined["seqlet_genomic_start"] = joined["window_start"] + joined["seqlet_start"]
    joined["seqlet_genomic_end"] = joined["window_start"] + joined["seqlet_end"]
    joined["seqlet_midpoint"] = (
        joined["seqlet_genomic_start"] + joined["seqlet_genomic_end"]
    ) / 2.0

    unique_records = sorted(joined["record_id"].astype(str).unique().tolist())
    landmarks = extract_seqlet_landmarks(
        record_ids=unique_records,
        regions_path=regions_path,
        use_terminator_end=use_terminator_end,
    )

    dist_tss_list: List[float] = []
    dist_term_list: List[float] = []
    for _, row in joined.iterrows():
        rec = str(row["record_id"])
        lm = landmarks[rec]
        mid = float(row["seqlet_midpoint"])
        tss_pos = lm["tss_pos"]
        term_pos = lm["term_pos"]
        d_tss = mid - tss_pos if not np.isnan(tss_pos) else np.nan
        d_term = mid - term_pos if not np.isnan(term_pos) else np.nan
        dist_tss_list.append(d_tss)
        dist_term_list.append(d_term)

    joined["dist_to_tss"] = dist_tss_list
    joined["dist_to_28s_end"] = dist_term_list
    return joined


def evaluate_gaussian_kde_concentration(
    distances: np.ndarray,
    grid_points: int,
    bw_factor: float,
    min_bandwidth_bp: float,
    y_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute 1D Gaussian KDE concentration curve on a grid and at exact seqlet distances.

    Args:
        distances: 1D array of valid distances (in bp), shape (N,).
        grid_points: Number of evaluation points along the x-axis grid.
        bw_factor: Multiplicative modifier on Scott's rule bandwidth (N^(-1/5)).
        min_bandwidth_bp: Minimum Gaussian kernel standard deviation in bp (prevents singularity).
        y_metric: Either 'density' (probability density in bp^-1) or
                  'count_density' (expected seqlets within +/- bandwidth window).

    Returns:
        (x_grid, y_grid, y_at_seqlets, bandwidth_bp)
    """
    n = len(distances)
    if n == 0:
        raise ValueError("Cannot compute KDE concentration on an empty distance array.")

    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    span = d_max - d_min

    std_val = float(np.std(distances, ddof=1)) if n > 1 else 0.0
    iqr_val = float(np.percentile(distances, 75) - np.percentile(distances, 25)) if n > 1 else 0.0
    scale_est = min(std_val, iqr_val / 1.34) if (iqr_val > 0.0 and std_val > 0.0) else std_val

    scott_factor = float(n ** (-0.2)) if n > 1 else 1.0
    bandwidth_bp = max(min_bandwidth_bp, scale_est * scott_factor * bw_factor)

    pad = max(bandwidth_bp * 3.5, span * 0.08, 100.0)
    x_grid = np.linspace(d_min - pad, d_max + pad, grid_points, dtype=np.float64)

    # Evaluate exact Gaussian kernel sum:
    # diff_grid shape: (grid_points, N)
    diff_grid = (x_grid[:, None] - distances[None, :]) / bandwidth_bp
    norm_const = float(n) * bandwidth_bp * np.sqrt(2.0 * np.pi)
    density_grid = np.exp(-0.5 * (diff_grid ** 2)).sum(axis=1) / norm_const

    # Evaluate at each seqlet's exact coordinate:
    # diff_pts shape: (N, N)
    diff_pts = (distances[:, None] - distances[None, :]) / bandwidth_bp
    density_pts = np.exp(-0.5 * (diff_pts ** 2)).sum(axis=1) / norm_const

    if y_metric == "density":
        y_grid = density_grid
        y_at_seqlets = density_pts
    elif y_metric == "count_density":
        # Convert probability density to local seqlet concentration (effective seqlet count in kernel)
        scale_mult = float(n) * bandwidth_bp * np.sqrt(2.0 * np.pi)
        y_grid = density_grid * scale_mult
        y_at_seqlets = density_pts * scale_mult
    else:
        raise ValueError(f"Unsupported y_metric: {y_metric!r}")

    return x_grid, y_grid, y_at_seqlets, bandwidth_bp


def plot_single_landmark_panel(
    ax: plt.Axes,
    distances: np.ndarray,
    is_revcomp: np.ndarray,
    landmark_label: str,
    x_label: str,
    y_metric: str,
    grid_points: int,
    bw_factor: float,
    min_bandwidth_bp: float,
    point_size: float,
    point_alpha: float,
    curve_color: str,
    point_color_fwd: str,
    point_color_rev: str,
    jitter_strength: float,
    rng: np.random.Generator,
) -> None:
    """Render one distance distribution panel (curve + individual seqlet points)."""
    valid_mask = ~np.isnan(distances)
    d_valid = distances[valid_mask]
    rev_valid = is_revcomp[valid_mask].astype(bool)
    n_valid = len(d_valid)

    if n_valid == 0:
        ax.text(
            0.5,
            0.5,
            f"No seqlets with valid {landmark_label} landmark",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#666666",
        )
        ax.set_xlabel(x_label, fontsize=10.5, fontweight="bold")
        return

    x_grid, y_grid, y_pts, bw_bp = evaluate_gaussian_kde_concentration(
        distances=d_valid,
        grid_points=grid_points,
        bw_factor=bw_factor,
        min_bandwidth_bp=min_bandwidth_bp,
        y_metric=y_metric,
    )

    # Plot smooth concentration curve and shaded area
    ax.plot(
        x_grid,
        y_grid,
        color=curve_color,
        linewidth=2.0,
        label=f"Seqlet concentration (std dev = {bw_bp:.0f} bp)",
        zorder=2,
    )
    ax.fill_between(x_grid, 0.0, y_grid, color=curve_color, alpha=0.18, zorder=1)

    # Apply optional small vertical micro-jitter so overlapping seqlets at identical bp offsets are visible
    y_max = float(np.max(y_grid)) if len(y_grid) > 0 else 1.0
    if jitter_strength > 0.0 and n_valid > 1:
        y_jitter = rng.uniform(-jitter_strength * y_max, jitter_strength * y_max, size=n_valid)
        y_plot = np.clip(y_pts + y_jitter, a_min=0.0, a_max=None)
    else:
        y_plot = y_pts

    fwd_mask = ~rev_valid
    rev_mask = rev_valid

    if np.any(fwd_mask):
        ax.scatter(
            d_valid[fwd_mask],
            y_plot[fwd_mask],
            s=point_size,
            color=point_color_fwd,
             alpha=point_alpha,
            edgecolors="#111111",
            linewidths=0.5,
            label=f"Seqlets (+) strand (n={int(fwd_mask.sum())})",
            zorder=4,
        )
    if np.any(rev_mask):
        ax.scatter(
            d_valid[rev_mask],
            y_plot[rev_mask],
            s=point_size,
            color=point_color_rev,
            marker="^",
            alpha=point_alpha,
            edgecolors="#111111",
            linewidths=0.5,
            label=f"Seqlets (-) strand (n={int(rev_mask.sum())})",
            zorder=4,
        )

    # Reference line at landmark (0 bp) if within or near view, plus median dashed line
    med_val = float(np.median(d_valid))
    q25 = float(np.percentile(d_valid, 25))
    q75 = float(np.percentile(d_valid, 75))

    ax.axvline(
        0.0,
        color="#d95f02",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label=f"{landmark_label} (0 bp)",
        zorder=3,
    )
    ax.axvline(
        med_val,
        color="#333333",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label=f"Median: {med_val:,.0f} bp",
        zorder=3,
    )

    y_label = (
        "Seqlet Concentration (KDE Density, $\\mathrm{bp}^{-1}$)"
        if y_metric == "density"
        else "Seqlet Concentration (Local Seqlet Count)"
    )
    ax.set_xlabel(x_label, fontsize=10.5, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=10.0, fontweight="bold")
    ax.set_ylim(bottom=0.0, top=y_max * 1.38 if y_max > 0 else 1.0)

    # Determine whether the peak is in the right half or left half of the x-range
    x_mid = 0.5 * (float(x_grid[0]) + float(x_grid[-1]))
    peak_x = float(x_grid[int(np.argmax(y_grid))])
    peak_on_right = peak_x > x_mid

    stats_text = (
        f"N = {n_valid} seqlets  |  Median = {med_val:,.0f} bp\n"
        f"IQR = [{q25:,.0f}, {q75:,.0f}] bp  |  Range = [{np.min(d_valid):,.0f}, {np.max(d_valid):,.0f}] bp"
    )
    ax.text(
        0.02 if peak_on_right else 0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left" if peak_on_right else "right",
        fontsize=8.2,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#f8f9fa", "edgecolor": "#cccccc", "alpha": 0.92},
        zorder=5,
    )

    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    legend_loc = "upper left" if peak_on_right else "upper right"
    legend_bbox = (0.01, 0.84) if peak_on_right else (0.99, 0.84)
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=8.2,
        framealpha=0.92,
    )


def generate_pattern_plots(
    df: pd.DataFrame,
    output_dir: Path,
    y_metric: str,
    grid_points: int,
    bw_factor: float,
    min_bandwidth_bp: float,
    point_size: float,
    point_alpha: float,
    jitter_strength: float,
    figsize: Tuple[float, float],
    dpi: int,
    seed: int,
    use_terminator_end: bool,
) -> List[Path]:
    """Generate and save one two-panel PNG per pattern in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    pattern_names = sorted(df["pattern_name"].unique().tolist(), key=pattern_sort_key)
    saved_paths: List[Path] = []

    second_landmark_name = "Terminator 3'-End (TTS)" if use_terminator_end else "28S rRNA 3'-End (3' ETS / UTR)"
    second_x_label = (
        "Distance from Terminator 3'-End / TTS (bp)"
        if use_terminator_end
        else "Distance from 28S rRNA 3'-End / 3' UTR (bp)"
    )

    for pat_name in pattern_names:
        sub = df[df["pattern_name"] == pat_name]
        n_seqlets = len(sub)
        n_records = int(sub["record_id"].nunique())
        mean_b_delta = float(sub["peak_B_delta"].mean())

        d_tss = sub["dist_to_tss"].to_numpy(dtype=np.float64)
        d_28s = sub["dist_to_28s_end"].to_numpy(dtype=np.float64)
        is_rev = sub["is_revcomp"].to_numpy(dtype=bool)

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        ax_tss, ax_28s = axes[0], axes[1]

        plot_single_landmark_panel(
            ax=ax_tss,
            distances=d_tss,
            is_revcomp=is_rev,
            landmark_label="TSS",
            x_label="Distance from TSS (bp)",
            y_metric=y_metric,
            grid_points=grid_points,
            bw_factor=bw_factor,
            min_bandwidth_bp=min_bandwidth_bp,
            point_size=point_size,
            point_alpha=point_alpha,
            curve_color="#1f77b4",
            point_color_fwd="#005082",
            point_color_rev="#56b4e9",
            jitter_strength=jitter_strength,
            rng=rng,
        )
        ax_tss.set_title("Distance Distribution from TSS", fontsize=11.5, fontweight="bold")

        plot_single_landmark_panel(
            ax=ax_28s,
            distances=d_28s,
            is_revcomp=is_rev,
            landmark_label=second_landmark_name,
            x_label=second_x_label,
            y_metric=y_metric,
            grid_points=grid_points,
            bw_factor=bw_factor,
            min_bandwidth_bp=min_bandwidth_bp,
            point_size=point_size,
            point_alpha=point_alpha,
            curve_color="#2ca02c",
            point_color_fwd="#1b661b",
            point_color_rev="#7fc97f",
            jitter_strength=jitter_strength,
            rng=rng,
        )
        ax_28s.set_title(
            f"Distance Distribution from {second_landmark_name}",
            fontsize=11.5,
            fontweight="bold",
        )

        plt.tight_layout()

        out_png = output_dir / f"{pat_name}.png"
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_png)

    return saved_paths


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/results.h5"),
        help="Path to TF-MoDISco HDF5 results file (default: outputs/tfmodisco_runs/all_seq_n/results.h5)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/tfmodisco_inputs/all_seq/candidate_metadata.parquet"),
        help="Path to candidate metadata parquet file (default: outputs/tfmodisco_inputs/all_seq/candidate_metadata.parquet)",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=Path("data/prepared/ribosome/regions.parquet"),
        help="Path to annotations parquet file (default: data/prepared/ribosome/regions.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/seqlet_distance_distributions"),
        help="Directory to write per-pattern PNG files (default: outputs/tfmodisco_runs/all_seq_n/seqlet_distance_distributions)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional single pattern filter (e.g., 'pos_patterns.pattern_0'). Default: plot all patterns.",
    )
    parser.add_argument(
        "--y-metric",
        choices=["density", "count_density"],
        default="count_density",
        help="Y-axis concentration metric: 'count_density' (local seqlet count within kernel) or 'density' (KDE probability density in bp^-1). Default: count_density",
    )
    parser.add_argument(
        "--bw-factor",
        type=float,
        default=1.0,
        help="Multiplicative bandwidth factor on Scott's rule for Gaussian KDE (default: 1.0)",
    )
    parser.add_argument(
        "--min-bandwidth-bp",
        type=float,
        default=25.0,
        help="Minimum KDE kernel bandwidth in bp when seqlets cluster at identical offsets (default: 25.0)",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=512,
        help="Number of evaluation points along the distance axis for the KDE curve (default: 512)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=36.0,
        help="Scatter marker size for each seqlet point (default: 36.0)",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.82,
        help="Scatter marker alpha transparency (default: 0.82)",
    )
    parser.add_argument(
        "--jitter-strength",
        type=float,
        default=0.025,
        help="Small vertical jitter fraction of y_max to separate identical-distance seqlets (default: 0.025, set 0 to disable)",
    )
    parser.add_argument(
        "--use-terminator-end",
        action="store_true",
        help="If set, measure 3' distance from the Terminator annotation 3'-end instead of 28S_rRNA 3'-end.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=15.0,
        help="Figure width in inches (default: 15.0)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.5,
        help="Figure height in inches (default: 5.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output PNG resolution in DPI (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    try:
        args = parse_args(argv)
        set_seed(seed=args.seed, deterministic=True)
        device = get_device()
        print(f"[info] Initialized plot_seqlet_distance_distributions on device={device}")
        print(f"[info] Loading seqlets from: {args.results}")
        print(f"[info] Loading metadata from: {args.metadata}")
        print(f"[info] Loading regions from: {args.regions}")

        df = build_seqlet_distance_table(
            results_path=args.results,
            metadata_path=args.metadata,
            regions_path=args.regions,
            pattern_filter=args.pattern,
            use_terminator_end=args.use_terminator_end,
        )

        print(
            f"[info] Loaded {len(df)} seqlets across {df['pattern_name'].nunique()} patterns. "
            f"Generating PNGs in: {args.output_dir}"
        )

        saved_paths = generate_pattern_plots(
            df=df,
            output_dir=args.output_dir,
            y_metric=args.y_metric,
            grid_points=args.grid_points,
            bw_factor=args.bw_factor,
            min_bandwidth_bp=args.min_bandwidth_bp,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            jitter_strength=args.jitter_strength,
            figsize=(args.fig_width, args.fig_height),
            dpi=args.dpi,
            seed=args.seed,
            use_terminator_end=args.use_terminator_end,
        )

        print(f"[info] Successfully wrote {len(saved_paths)} pattern PNG plots to {args.output_dir}")
    except Exception as exc:
        print(f"[error] Failed to generate seqlet distance plots: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
