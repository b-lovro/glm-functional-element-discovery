#!/usr/bin/env python3
"""
Generate RiNALMo (+LoRA) Dependency Maps for Exemplar TF-MoDISco Seqlets
========================================================================
Loads the RiNALMo-mega model merged with the adapted LoRA weights, ranks all
seqlets for each discovered TF-MoDISco motif by:
  1. Minimum Hamming distance (maximum exact matches to the motif consensus)
  2. High mean attribution magnitude across the seqlet
  3. High PWM log-odds score

For the top-K distinct exemplar seqlets per pattern, computes a dependency map
centered on the seqlet within a 1,022-bp context window using in-silico
mutagenesis (with reconstruction). Renders each map with top block-score bars,
nucleotide axes, genomic slider track, landmark outlines, and a centered seqlet
bounding box, and saves the output as a high-resolution PNG.
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

import argparse
from functools import partial
import logging
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure repository root, src/, and external/dependency_map are in sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root / "external/dependency_map/src") not in sys.path:
    sys.path.insert(0, str(repo_root / "external/dependency_map/src"))

from dependency_map import DependencyMap, DependencyMapOptions
from glmfe.seq_models.rinalmo import load_rinalmo_model
from scripts.tfmodisco_common import (
    DEFAULT_INPUT_DIR,
    DEFAULT_RECORDS_PARQUET,
    DEFAULT_REGIONS_PARQUET,
    DEFAULT_RUN_DIR,
    format_pattern_display_name,
    get_dynamic_device,
    pattern_sort_key,
    set_global_seed,
)

logger = logging.getLogger("plot_motif_seqlet_dependency_maps")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_base_color(ftype: str) -> Optional[str]:
    """Assigns specific colors to primary landmark feature types."""
    colors: Dict[str, str] = {
        "Core": "rgba(255, 0, 0, 1.0)",
        "CTCF": "rgba(0, 0, 255, 1.0)",
        "TTF1": "rgba(0, 255, 0, 1.0)",
        "CompositePromoter": "rgba(255, 128, 0, 1.0)",
        "Upstream": "rgba(148, 103, 189, 1.0)",
        "Terminator": "rgba(140, 86, 75, 1.0)",
        "18S_rRNA": "rgba(227, 119, 194, 1.0)",
        "5_8S_rRNA": "rgba(127, 127, 127, 1.0)",
        "28S_rRNA": "rgba(188, 189, 34, 1.0)",
        "EnhancerRepeats": "rgba(23, 190, 207, 1.0)",
    }
    for key, color in colors.items():
        if key in ftype:
            return color
    return None


def select_top_exemplar_seqlets(
    pattern_name: str,
    ppm: np.ndarray,
    example_idxs: np.ndarray,
    seqlet_starts: np.ndarray,
    seqlet_ends: np.ndarray,
    is_revcomps: np.ndarray,
    meta_df: pd.DataFrame,
    records_df: pd.DataFrame,
    one_hot_arr: np.ndarray,
    att_arr: np.ndarray,
    top_k: int = 3,
    min_genomic_separation: int = 50,
) -> List[Dict[str, Any]]:
    """Rank all seqlets of a motif and select top-K spatially distinct exemplar instances.

    Ranking criteria:
      1. Minimum Hamming distance (maximum exact sequence matches to the motif consensus)
      2. High mean attribution magnitude across the seqlet
      3. High PWM log-odds score (tie-breaker)

    Args:
        pattern_name: Full pattern name (e.g. 'pos_patterns.pattern_0').
        ppm: Position Probability Matrix of shape (L, 4).
        example_idxs: 1D array of candidate window indices, shape (N_seqlets,).
        seqlet_starts: 1D array of seqlet start offsets in candidate window, shape (N_seqlets,).
        seqlet_ends: 1D array of seqlet end offsets in candidate window, shape (N_seqlets,).
        is_revcomps: 1D boolean array of reverse-complement flags, shape (N_seqlets,).
        meta_df: Candidate metadata DataFrame matching example_idx.
        one_hot_arr: 3D array of one-hot sequences, shape (N_windows, 4, W).
        att_arr: 3D array of attribution matrices, shape (N_windows, 4, W).
        top_k: Number of exemplar seqlets to select (default: 3).
        min_genomic_separation: Minimum genomic separation in bp to prevent picking overlapping loci.

    Returns:
        List of dictionaries with selected seqlet details.
    """
    n_seqlets = len(example_idxs)
    if n_seqlets == 0:
        return []

    pattern_len = int(ppm.shape[0])
    consensus_idx = np.argmax(ppm, axis=1)  # shape: (L,)
    alphabet = ["A", "C", "G", "T"]
    consensus_str = "".join(alphabet[idx] for idx in consensus_idx)

    # Pre-clip PPM for log-odds calculation
    ppm_safe = np.clip(ppm, 1e-4, 1.0)
    log_odds_mat = np.log2(ppm_safe / 0.25)  # shape: (L, 4)

    candidates: List[Dict[str, Any]] = []

    for i in range(n_seqlets):
        ex_idx = int(example_idxs[i])
        s_start = int(seqlet_starts[i])
        s_end = int(seqlet_ends[i])
        is_rc = bool(is_revcomps[i])

        meta_row = meta_df.iloc[ex_idx]
        record_id = str(meta_row["record_id"])
        parts = record_id.split("-")
        clade = parts[1] if len(parts) > 1 else ""
        clean_species = parts[2] if len(parts) > 2 else record_id
        window_start = int(meta_row["window_start"])
        window_end = int(meta_row["window_end"])

        seqlet_genomic_start = window_start + s_start
        seqlet_genomic_end = window_start + s_end

        # Extract seqlet sequence slice (shape: 4, L)
        oh_slice = one_hot_arr[ex_idx, :, s_start:s_end]
        att_slice = att_arr[ex_idx, :, s_start:s_end]

        if is_rc:
            # Reverse along sequence length, and complement bases: A(0)<->T(3), C(1)<->G(2)
            oh_slice = oh_slice[::-1, ::-1]
            att_slice = att_slice[::-1, ::-1]

        seq_indices = np.argmax(oh_slice, axis=0)  # shape: (L,)
        seq_len = int(len(seq_indices))
        seq_str = "".join(alphabet[b] for b in seq_indices)

        hamming_dist = int(np.sum(seq_indices != consensus_idx[:seq_len]))
        exact_matches = seq_len - hamming_dist
        match_fraction = exact_matches / float(seq_len)

        pwm_score = float(np.sum(log_odds_mat[np.arange(seq_len), seq_indices]))

        per_base_attr = np.sum(np.abs(att_slice), axis=0)  # shape: (L,)
        attr_mean = float(np.mean(per_base_attr))

        candidates.append(
            {
                "pattern_name": pattern_name,
                "seqlet_index": i,
                "example_idx": ex_idx,
                "record_id": record_id,
                "species": clean_species,
                "order_clade": clade,
                "seqlet_genomic_start": seqlet_genomic_start,
                "seqlet_genomic_end": seqlet_genomic_end,
                "seqlet_length": seqlet_genomic_end - seqlet_genomic_start,
                "is_revcomp": is_rc,
                "seqlet_seq_5to3": seq_str,
                "consensus_50": consensus_str,
                "hamming_dist_50": hamming_dist,
                "exact_matches_50": exact_matches,
                "match_fraction_50": round(match_fraction, 4),
                "pwm_score_50": round(pwm_score, 2),
                "attr_mean_50": round(attr_mean, 2),
            }
        )

    # Sort candidates by:
    # 1. Ascending hamming_dist_50 (highest sequence match)
    # 2. Descending attr_mean_50 (highest attribution)
    # 3. Descending pwm_score_50 (tie-breaker)
    candidates.sort(
        key=lambda c: (
            c["hamming_dist_50"],
            -c["attr_mean_50"],
            -c["pwm_score_50"],
        )
    )

    # Select top-k spatially distinct seqlets
    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        overlap = False
        for s in selected:
            if s["record_id"] == cand["record_id"]:
                inter_start = max(s["seqlet_genomic_start"], cand["seqlet_genomic_start"])
                inter_end = min(s["seqlet_genomic_end"], cand["seqlet_genomic_end"])
                if (inter_end - inter_start) > (pattern_len - min_genomic_separation):
                    overlap = True
                    break
        if not overlap:
            cand["rank"] = len(selected) + 1
            selected.append(cand)
            if len(selected) >= top_k:
                break

    return selected


def compute_per_position_block_score(
    dep_map: np.ndarray,
    block_size: int = 6,
    quantile: float = 0.25,
) -> np.ndarray:
    """Computes per-position block scores across map_size by assigning each block to its center nucleotide.

    Args:
        dep_map: 2D dependency map matrix of shape (L, L).
        block_size: Consecutive diagonal block width in nt (default: 6).
        quantile: Quantile of off-diagonal elements in each block (default: 0.25).

    Returns:
        1D array of shape (L,) containing per-nucleotide block scores assigned to block centers.
    """
    map_size = dep_map.shape[0]
    off_diag_mask = ~np.eye(block_size, dtype=bool)
    n_spans = map_size - block_size + 1
    per_pos = np.zeros(map_size, dtype=float)
    center_offset = block_size // 2

    for i in range(n_spans):
        block = dep_map[i : i + block_size, i : i + block_size]
        per_pos[i + center_offset] = float(np.quantile(block[off_diag_mask], quantile))

    return per_pos


def move_rinalmo_adapter(seq_model: Any, target_device: torch.device) -> None:
    """Moves RiNALMoSequenceModel weights and index tensors to target_device."""
    seq_model.device = target_device
    seq_model.model = seq_model.model.to(target_device)
    seq_model.nucleotide_token_indices = seq_model.nucleotide_token_indices.to(target_device)
    seq_model.canonical_nucleotide_token_indices = seq_model.canonical_nucleotide_token_indices.to(target_device)
    seq_model.random_replace_indices = seq_model.random_replace_indices.to(target_device)


def render_and_save_dependency_map_plot(
    dep_map: np.ndarray,
    block_scores: np.ndarray,
    seqlet_info: Dict[str, Any],
    record_sequence: str,
    map_start: int,
    map_end: int,
    regions_df: pd.DataFrame,
    output_png: Path,
    map_type_title: str = "RiNALMo Adapted Nucleotide Dependency Map",
    cbar_label: str = "Adapted Nucleotide Dependency",
    block_ylabel: str = "Block Score",
    scale: int = 3,
) -> None:
    """Render dependency map plot with block scores, landmarks, and seqlet box, and save PNG.

    Includes:
      - Top 1D line/bar plot of per-position Block Score
      - Axis nucleotide characters
      - Bottom genomic overview slider / track
      - Overlapping landmark annotations on the diagonal
      - Highlighted bounding box around the centered seqlet
    """
    map_size = map_end - map_start
    seq_window = record_sequence[map_start:map_end]
    fasta_length = len(record_sequence)
    record_id = seqlet_info["record_id"]
    pattern_name = seqlet_info["pattern_name"]
    pattern_display = format_pattern_display_name(pattern_name)
    rank = seqlet_info["rank"]

    # 1. Clip dependency map to [0.0, 5.0]
    dep_map_clipped = np.clip(dep_map, 0.0, 5.0)

    # 2. Create matplotlib figure with 3 vertically stacked axes
    fig = plt.figure(figsize=(13.0, 14.0), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(
        3,
        1,
        height_ratios=[1.2, 10.0, 0.6],
        hspace=0.08,
        left=0.07,
        right=0.80,
        top=0.91,
        bottom=0.05,
    )

    ax_block = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])
    ax_track = fig.add_subplot(gs[2])

    # -------------------------------------------------------------------------
    # Panel 1: Top 1D Discrete Bar Plot of Per-Position Block Score
    # -------------------------------------------------------------------------
    x_positions = np.arange(map_size)
    ax_block.bar(
        x_positions,
        block_scores,
        width=0.82,
        color="#3B82F6",
        edgecolor="#1E3A8A",
        linewidth=0.6,
        alpha=0.85,
        zorder=3,
    )

    s_rel_start = max(0, seqlet_info["seqlet_genomic_start"] - map_start)
    s_rel_end = min(map_size, seqlet_info["seqlet_genomic_end"] - map_start)
    ax_block.axvspan(
        s_rel_start - 0.5,
        s_rel_end - 0.5,
        color="#b2182b",
        alpha=0.12,
        linestyle="--",
        linewidth=1.0,
    )

    ax_block.set_xlim(-0.5, map_size - 0.5)
    ax_block.set_ylabel(block_ylabel, fontsize=9.5, fontweight="bold")
    ax_block.set_xticks([])
    ax_block.tick_params(axis="y", labelsize=8)
    ax_block.spines["top"].set_visible(False)
    ax_block.spines["right"].set_visible(False)
    ax_block.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 2: Dependency Map Heatmap
    # -------------------------------------------------------------------------
    im = ax_map.imshow(
        dep_map_clipped,
        cmap="coolwarm",
        vmin=0.0,
        vmax=5.0,
        origin="upper",
        aspect="equal",
        extent=(-0.5, map_size - 0.5, map_size - 0.5, -0.5),
    )

    # Configure top and bottom x-axes with nucleotide characters
    font_sz = 5.0 if map_size <= 100 else 3.5
    ax_map.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_map.set_xticks(np.arange(map_size))
    ax_map.set_xticklabels(list(seq_window), fontsize=font_sz, fontfamily="monospace")

    ax_map.set_yticks(np.arange(map_size))
    ax_map.set_yticklabels(list(seq_window), fontsize=font_sz, fontfamily="monospace")

    # Overlapping landmark annotations on the diagonal
    rec_regions = regions_df[
        (regions_df["record_id"] == record_id)
        & (regions_df["start"] < map_end)
        & (regions_df["end"] > map_start)
    ]

    legend_patches: List[Any] = []
    seen_ftypes = set()

    for _, annot in rec_regions.iterrows():
        ann_start = max(0, int(annot["start"]) - map_start)
        ann_end = min(map_size, int(annot["end"]) - map_start)
        ftype = str(annot["feature_type"])
        color = get_base_color(ftype)
        if color and ann_end > ann_start:
            # Convert rgba(...) string to hex/rgba for matplotlib
            mpl_color = color.replace("rgba(", "").replace(")", "").split(",")
            r, g, b, a = [float(c.strip()) for c in mpl_color]
            c_tuple = (r / 255.0, g / 255.0, b / 255.0, a)

            rect = Rectangle(
                (ann_start - 0.5, ann_start - 0.5),
                ann_end - ann_start,
                ann_end - ann_start,
                linewidth=1.4,
                edgecolor=c_tuple,
                facecolor="none",
                linestyle=":",
                zorder=5,
            )
            ax_map.add_patch(rect)
            if ftype not in seen_ftypes:
                seen_ftypes.add(ftype)
                legend_patches.append(
                    Patch(edgecolor=c_tuple, facecolor="none", linestyle=":", linewidth=1.5, label=f"Landmark: {ftype}")
                )

    # Highlight Centered Seqlet Bounding Box
    s_len = s_rel_end - s_rel_start

    seqlet_rect = Rectangle(
        (s_rel_start - 0.5, s_rel_start - 0.5),
        s_len,
        s_len,
        linewidth=2.5,
        edgecolor="#b2182b",
        facecolor="none",
        linestyle="-",
        zorder=6,
    )
    ax_map.add_patch(seqlet_rect)

    strand_label = "(-) RevComp" if seqlet_info["is_revcomp"] else "(+) Fwd"
    legend_patches.insert(
        0,
        Patch(
            edgecolor="#b2182b",
            facecolor="none",
            linewidth=2.5,
            label=f"{pattern_display} Seqlet ({s_len} bp, {strand_label})",
        ),
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.825, 0.25, 0.02, 0.45])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # Legend
    fig.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(0.815, 0.91),
        fontsize=8.0,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    # -------------------------------------------------------------------------
    # Panel 3: Bottom Genomic Overview Slider / Track
    # -------------------------------------------------------------------------
    ax_track.axhspan(0.0, 1.0, facecolor="#e0e0e0", edgecolor="none")
    ax_track.axvspan(map_start, map_end, facecolor="#4393c3", edgecolor="#053061", linewidth=1.5, alpha=0.85)

    ax_track.set_xlim(0, fasta_length)
    ax_track.set_ylim(0.0, 1.0)
    ax_track.set_yticks([])
    ax_track.set_xticks([0, map_start, map_end, fasta_length])
    ax_track.set_xticklabels(["0", str(map_start), str(map_end), str(fasta_length)], fontsize=7.5, rotation=25)
    ax_track.set_xlabel("Full rDNA Record Coordinate (bp)", fontsize=8.5, fontweight="bold")
    ax_track.spines["top"].set_visible(False)
    ax_track.spines["left"].set_visible(False)
    ax_track.spines["right"].set_visible(False)

    # Title & Subtitle (2 clean lines without pattern_n or Record: id)
    clade_str = f" ({seqlet_info['order_clade']})" if ("order_clade" in seqlet_info and seqlet_info["order_clade"]) else ""
    fig.suptitle(
        f"Species: {seqlet_info['species']}{clade_str} | Region: [{map_start:,}, {map_end:,}) bp\n"
        f"{s_len}-bp Consensus Match: {seqlet_info['exact_matches_50']}/{s_len} bp ({seqlet_info['match_fraction_50']*100:.1f}%) | "
        f"Mean Attr: {seqlet_info['attr_mean_50']:.1f} | PWM Score: {seqlet_info['pwm_score_50']:.1f}",
        fontsize=11.0,
        fontweight="bold",
        y=0.965,
        linespacing=1.35,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dependency map to: %s", output_png)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments exposing all hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Compute and plot RiNALMo (+LoRA) dependency maps for exemplar TF-MoDISco seqlets."
    )
    parser.add_argument(
        "--modisco-h5",
        type=Path,
        default=DEFAULT_RUN_DIR / "results.h5",
        help=f"Path to TF-MoDISco results.h5 (default: {DEFAULT_RUN_DIR / 'results.h5'}).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_INPUT_DIR / "candidate_metadata.parquet",
        help=f"Path to candidate_metadata.parquet (default: {DEFAULT_INPUT_DIR / 'candidate_metadata.parquet'}).",
    )
    parser.add_argument(
        "--attributions-npz",
        type=Path,
        default=DEFAULT_INPUT_DIR / "attributions.npz",
        help=f"Path to attributions.npz (default: {DEFAULT_INPUT_DIR / 'attributions.npz'}).",
    )
    parser.add_argument(
        "--onehot-npz",
        type=Path,
        default=DEFAULT_INPUT_DIR / "one_hot.npz",
        help=f"Path to one_hot.npz (default: {DEFAULT_INPUT_DIR / 'one_hot.npz'}).",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS_PARQUET,
        help=f"Path to records.parquet containing rDNA sequences (default: {DEFAULT_RECORDS_PARQUET}).",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=DEFAULT_REGIONS_PARQUET,
        help=f"Path to regions.parquet containing landmark annotations (default: {DEFAULT_REGIONS_PARQUET}).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/rinalmo/rinalmo_mega_pretrained.pt"),
        help="Path to base RiNALMo pretrained weights (default: models/rinalmo/rinalmo_mega_pretrained.pt).",
    )
    parser.add_argument(
        "--lora-weights",
        type=Path,
        default=Path("outputs/runs/pretraining-rinalmo-bf16-lr-r-long/checkpoint_latest.pt"),
        help="Path to adapted LoRA checkpoint (default: outputs/runs/pretraining-rinalmo-bf16-lr-r-long/checkpoint_latest.pt).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="mega",
        help="RiNALMo model size (default: mega).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(get_dynamic_device()),
        help="Inference device (default: auto-detected cuda/mps/cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "dep_map_100",
        help=f"Output directory (default: {DEFAULT_RUN_DIR / 'dep_map_100'}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of exemplar seqlets to plot per motif pattern (default: 1).",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=100,
        help="Active subset size in bp for dependency map (default: 100 -> 100x100 map).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1022,
        help="Surrounding context length in bp for RiNALMo forward passes (default: 1022).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for dependency map forward passes (default: 16).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Resolution scale factor for PNG export (default: 3).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=6,
        help="Block size in bp for diagonal block score computation (default: 6).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.25,
        help="Quantile for off-diagonal values in block score (default: 0.25).",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save the raw dependency map and reconstruction as .npz alongside the PNG.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of pattern names to run (e.g. pos_patterns.pattern_0). Runs all if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip maps that have already been generated.",
    )
    parser.add_argument(
        "--from-npz",
        action="store_true",
        help="Re-render PNG plots and recompute block scores directly from existing .npz files without loading RiNALMo models.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution pipeline."""
    try:
        args = parse_args(argv)
        set_global_seed(args.seed)

        # Validate input paths
        if not args.modisco_h5.is_file():
            raise FileNotFoundError(f"TF-MoDISco HDF5 file not found: {args.modisco_h5}")
        if not args.metadata.is_file():
            raise FileNotFoundError(f"Candidate metadata parquet not found: {args.metadata}")
        if not args.attributions_npz.is_file():
            raise FileNotFoundError(f"Attributions NPZ not found: {args.attributions_npz}")
        if not args.onehot_npz.is_file():
            raise FileNotFoundError(f"One-hot NPZ not found: {args.onehot_npz}")
        if not args.records.is_file():
            raise FileNotFoundError(f"Records parquet not found: {args.records}")
        if not args.regions.is_file():
            raise FileNotFoundError(f"Regions parquet not found: {args.regions}")
        if not args.from_npz:
            if not args.weights.is_file():
                raise FileNotFoundError(f"RiNALMo base weights not found: {args.weights}")
            if not args.lora_weights.is_file():
                raise FileNotFoundError(f"LoRA weights checkpoint not found: {args.lora_weights}")

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading candidate metadata from: %s", args.metadata)
        meta_df = pd.read_parquet(args.metadata)

        logger.info("Loading records from: %s", args.records)
        records_df = pd.read_parquet(args.records).set_index("record_id")

        logger.info("Loading landmark regions from: %s", args.regions)
        regions_df = pd.read_parquet(args.regions)

        logger.info("Loading attributions from: %s", args.attributions_npz)
        att_npz = np.load(args.attributions_npz)
        att_key = list(att_npz.keys())[0]
        att_arr = att_npz[att_key]  # shape: (N_windows, 4, 400)

        logger.info("Loading one-hot sequences from: %s", args.onehot_npz)
        oh_npz = np.load(args.onehot_npz)
        oh_key = list(oh_npz.keys())[0]
        one_hot_arr = oh_npz[oh_key]  # shape: (N_windows, 4, 400)

        target_device = torch.device(args.device)
        adapted_model: Any = None
        base_model: Any = None
        if not args.from_npz:
            # Load RiNALMo-mega Adapted (+LoRA) and Base models onto CPU initially
            logger.info("Loading Adapted RiNALMo (%s) from %s with LoRA from %s...", args.model_size, args.weights, args.lora_weights)
            adapted_model = load_rinalmo_model(
                model_size=args.model_size,
                weights_path=args.weights,
                device="cpu",
                lora_weights_path=args.lora_weights,
            )
            logger.info("Loading Base RiNALMo (%s) from %s (no LoRA)...", args.model_size, args.weights)
            base_model = load_rinalmo_model(
                model_size=args.model_size,
                weights_path=args.weights,
                device="cpu",
                lora_weights_path=None,
            )
            logger.info("Both Adapted and Base RiNALMo models loaded successfully (active inference device: %s).", args.device)
        else:
            logger.info("Running in --from-npz mode: skipping RiNALMo model loading.")

        # Select patterns to process
        pattern_groups = ["pos_patterns", "neg_patterns"]
        all_pattern_names: List[str] = []
        with h5py.File(args.modisco_h5, "r") as f:
            for grp_name in pattern_groups:
                if grp_name not in f:
                    continue
                grp = f[grp_name]
                for p_name in sorted(grp.keys(), key=pattern_sort_key):
                    full_p = f"{grp_name}.{p_name}"
                    p_num = p_name.replace("pattern_", "")
                    short_alias = p_name if grp_name == "pos_patterns" else f"neg_{p_name}"
                    num_alias = p_num if grp_name == "pos_patterns" else f"neg_{p_num}"
                    if (
                        args.patterns is None
                        or full_p in args.patterns
                        or short_alias in args.patterns
                        or num_alias in args.patterns
                    ):
                        all_pattern_names.append(full_p)

        logger.info("Processing %d motif pattern(s) with top-%d exemplar seqlets each...", len(all_pattern_names), args.top_k)

        all_selected_seqlets: List[Dict[str, Any]] = []

        with h5py.File(args.modisco_h5, "r") as f:
            for full_p in all_pattern_names:
                grp_name, pat_name = full_p.split(".")
                p_grp = f[grp_name][pat_name]
                ppm = np.array(p_grp["sequence"][:], dtype=np.float64)

                s_grp = p_grp["seqlets"]
                ex_idxs = np.array(s_grp["example_idx"][:], dtype=np.int64)
                s_starts = np.array(s_grp["start"][:], dtype=np.int64)
                s_ends = np.array(s_grp["end"][:], dtype=np.int64)
                is_rcs = np.array(s_grp["is_revcomp"][:], dtype=bool)

                selected = select_top_exemplar_seqlets(
                    pattern_name=full_p,
                    ppm=ppm,
                    example_idxs=ex_idxs,
                    seqlet_starts=s_starts,
                    seqlet_ends=s_ends,
                    is_revcomps=is_rcs,
                    meta_df=meta_df,
                    records_df=records_df,
                    one_hot_arr=one_hot_arr,
                    att_arr=att_arr,
                    top_k=args.top_k,
                )
                all_selected_seqlets.extend(selected)

        # Save summary TSV of all selected exemplar seqlets
        summary_tsv = output_dir / "selected_exemplar_seqlets_summary.tsv"
        summary_df = pd.DataFrame(all_selected_seqlets)
        summary_df.to_csv(summary_tsv, sep="\t", index=False)
        logger.info("Saved selected exemplar seqlets metadata table to: %s", summary_tsv)

        # Compute dependency maps and render both Adapted and Delta plots
        tokenize_func: Any = None
        forward_adapted: Any = None
        forward_base: Any = None
        if not args.from_npz:
            def _tokenize_func(sequence: str, mask: Optional[int]) -> object:
                return adapted_model.dependency_tokenize(sequence, mask)

            tokenize_func = _tokenize_func
            forward_adapted = partial(adapted_model.dependency_forward, batch_size=args.batch_size)
            forward_base = partial(base_model.dependency_forward, batch_size=args.batch_size)
        cpu_device = torch.device("cpu")

        for seqlet in all_selected_seqlets:
            pat_name = seqlet["pattern_name"]
            pat_display = format_pattern_display_name(pat_name)
            rank = seqlet["rank"]
            rec_id = seqlet["record_id"]
            s_start = seqlet["seqlet_genomic_start"]
            s_end = seqlet["seqlet_genomic_end"]

            rec_seq = str(records_df.loc[rec_id, "sequence"]).upper()
            seq_len = len(rec_seq)

            # Center a map window around the seqlet midpoint
            seqlet_mid = (s_start + s_end) // 2
            half_map = args.map_size // 2
            map_start = max(0, min(seq_len - args.map_size, seqlet_mid - half_map))
            map_end = map_start + args.map_size

            # Slice context window
            pad_total = args.context_length - args.map_size
            pad_left = pad_total // 2
            win_start = max(0, map_start - pad_left)
            win_end = min(seq_len, win_start + args.context_length)
            if (win_end - win_start) < args.context_length and win_start > 0:
                win_start = max(0, win_end - args.context_length)

            active_subset = (map_start - win_start, map_end - win_start)
            context_seq = rec_seq[win_start:win_end]
            seq_200 = rec_seq[map_start:map_end]

            pat_out_dir = output_dir / pat_name
            pat_out_dir.mkdir(parents=True, exist_ok=True)
            safe_rec = "".join(c if c.isalnum() else "_" for c in rec_id)
            stem = f"{pat_display}_rank{rank}_{safe_rec}_{map_start}_{map_end}"
            out_png_adapted = pat_out_dir / f"{stem}_adapted.png"
            out_png_delta = pat_out_dir / f"{stem}_delta.png"
            out_npz = pat_out_dir / f"{stem}.npz"

            if args.resume and out_png_adapted.is_file() and out_png_delta.is_file():
                logger.info("Skipping existing adapted + delta maps for %s rank %d: %s", pat_display, rank, stem)
                continue

            if args.from_npz:
                if not out_npz.is_file():
                    raise FileNotFoundError(f"--from-npz requested, but NPZ file does not exist: {out_npz}")
                logger.info("Loading precomputed dependency maps from NPZ: %s", out_npz)
                loaded_npz = np.load(out_npz)
                raw_dep_adapted = np.array(loaded_npz["dependency_map_adapted"], dtype=np.float64)
                raw_dep_base = np.array(loaded_npz["dependency_map_base"], dtype=np.float64)
                recon_adapted = np.array(loaded_npz["reconstruction_adapted"])
                recon_base = np.array(loaded_npz["reconstruction_base"])
                dep_adapted = raw_dep_adapted.copy()
                dep_base = raw_dep_base.copy()
                np.fill_diagonal(dep_adapted, 0.0)
                np.fill_diagonal(dep_base, 0.0)
                dep_delta = dep_adapted - dep_base
            else:
                logger.info(
                    "Computing Adapted & Base dependency maps for %s (rank %d): record=%s, map=[%d, %d), context=[%d, %d) (len=%d)...",
                    pat_display,
                    rank,
                    rec_id,
                    map_start,
                    map_end,
                    win_start,
                    win_end,
                    len(context_seq),
                )

                job_options = DependencyMapOptions(
                    subset=active_subset,
                    with_reconstruction=True,
                    dependency_by_masking=False,
                    autoregressive=False,
                )

                # 1. Adapted RiNALMo pass
                move_rinalmo_adapter(adapted_model, target_device)
                dm_adapted = DependencyMap.compute_batched(
                    context_seq,
                    tokenize_func,
                    forward_adapted,
                    batch_size=args.batch_size,
                    enable_progress_bar=False,
                    options=job_options,
                )
                move_rinalmo_adapter(adapted_model, cpu_device)
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()

                # 2. Base RiNALMo pass
                move_rinalmo_adapter(base_model, target_device)
                dm_base = DependencyMap.compute_batched(
                    context_seq,
                    tokenize_func,
                    forward_base,
                    batch_size=args.batch_size,
                    enable_progress_bar=False,
                    options=job_options,
                )
                move_rinalmo_adapter(base_model, cpu_device)
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()

                raw_dep_adapted = dm_adapted.dependency_map
                raw_dep_base = dm_base.dependency_map
                recon_adapted = dm_adapted.reconstruction
                recon_base = dm_base.reconstruction

                # Extract zero-diagonal matrices
                dep_adapted = dm_adapted.get_depenency_map_without_diagonal()
                dep_base = dm_base.get_depenency_map_without_diagonal()
                dep_delta = dep_adapted - dep_base

            # Compute per-position Block Scores
            bs_adapted = compute_per_position_block_score(
                dep_adapted, block_size=args.block_size, quantile=args.quantile
            )
            bs_base = compute_per_position_block_score(
                dep_base, block_size=args.block_size, quantile=args.quantile
            )
            bs_delta = np.maximum(0.0, bs_adapted - bs_base)

            # Render Adapted Map
            render_and_save_dependency_map_plot(
                dep_map=dep_adapted,
                block_scores=bs_adapted,
                seqlet_info=seqlet,
                record_sequence=rec_seq,
                map_start=map_start,
                map_end=map_end,
                regions_df=regions_df,
                output_png=out_png_adapted,
                map_type_title="RiNALMo Adapted Nucleotide Dependency Map",
                cbar_label="Adapted Nucleotide Dependency",
                block_ylabel="Block Score",
                scale=args.scale,
            )

            # Render Delta (Adapted - Base) Map
            render_and_save_dependency_map_plot(
                dep_map=dep_delta,
                block_scores=bs_delta,
                seqlet_info=seqlet,
                record_sequence=rec_seq,
                map_start=map_start,
                map_end=map_end,
                regions_df=regions_df,
                output_png=out_png_delta,
                map_type_title="RiNALMo Delta (Adapted − Base) Dependency Map",
                cbar_label="Delta Dependency (Adapted − Base)",
                block_ylabel="Δ Block Score",
                scale=args.scale,
            )

            if args.save_npz:
                np.savez_compressed(
                    out_npz,
                    dependency_map_adapted=raw_dep_adapted,
                    dependency_map_base=raw_dep_base,
                    dependency_map_delta=dep_delta,
                    reconstruction_adapted=recon_adapted,
                    reconstruction_base=recon_base,
                    block_score_adapted=bs_adapted,
                    block_score_base=bs_base,
                    block_score_delta=bs_delta,
                    sequence=np.array(list(seq_200)),
                    map_start=map_start,
                    map_end=map_end,
                    seqlet_start=seqlet["seqlet_genomic_start"],
                    seqlet_end=seqlet["seqlet_genomic_end"],
                )
                logger.info("Saved raw array NPZ to: %s", out_npz)

        logger.info("Exemplar seqlet dependency map generation completed successfully.")
        return 0

    except Exception as exc:
        logger.error("Fatal error in plot_motif_seqlet_dependency_maps: %s", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(main())
