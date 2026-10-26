#!/usr/bin/env python3
"""
Generate 200x200 RiNALMo (+LoRA) Dependency Maps for Exemplar TF-MoDISco Seqlets
================================================================================
Loads the RiNALMo-mega model merged with the adapted LoRA weights, ranks all
50-bp seqlets for each discovered TF-MoDISco motif by:
  1. Minimum 50-bp Hamming distance (maximum exact matches to 50-bp motif consensus)
  2. High 50-bp mean attribution magnitude
  3. High 50-bp PWM log-odds score

For the top-K distinct exemplar seqlets per pattern, computes a 200x200 nucleotide
dependency map centered on the seqlet within a 1,022-bp context window using
in-silico mutagenesis (with reconstruction). Renders each map in the style of
scripts/plot_annotated_dependency_map.py, highlighting the centered 50-bp seqlet
and overlapping landmark annotations, and saves the output as high-resolution PNG.
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
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

# Ensure repository root and external/dependency_map are in sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root / "external/dependency_map/src") not in sys.path:
    sys.path.insert(0, str(repo_root / "external/dependency_map/src"))

from dependency_map import DependencyMap, DependencyMapOptions
from glmfe.seq_models.rinalmo import load_rinalmo_model

logger = logging.getLogger("plot_motif_seqlet_dependency_maps")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pattern_sort_key(name: str) -> Tuple[int, int]:
    """Extracts (is_neg, pattern_index) for natural sorting (pos_patterns first, then neg_patterns)."""
    is_neg = 1 if name.startswith("neg_") else 0
    parts = name.split(".")[-1].split("_")
    return (is_neg, int(parts[-1]))


def format_pattern_display_name(name: str) -> str:
    """Formats pattern identifier cleanly, preserving neg_ prefix if from neg_patterns."""
    if name.startswith("neg_patterns."):
        return f"neg_{name.split('.')[-1]}"
    return name.split(".")[-1]


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
    ppm_50: np.ndarray,
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
    """Ranks all 50-bp seqlets of a motif and selects top-K distinct exemplar instances.

    Ranking criteria:
      1. Minimum 50-bp Hamming distance (maximum exact sequence matches out of 50 to the motif consensus)
      2. High 50-bp mean attribution magnitude
      3. High 50-bp PWM log-odds score (tie-breaker)

    Args:
        pattern_name: Full pattern name (e.g. 'pos_patterns.pattern_0').
        ppm_50: Position Probability Matrix of shape (50, 4).
        example_idxs: 1D array of candidate window indices, shape (N_seqlets,).
        seqlet_starts: 1D array of seqlet start offsets in 400-bp window, shape (N_seqlets,).
        seqlet_ends: 1D array of seqlet end offsets in 400-bp window, shape (N_seqlets,).
        is_revcomps: 1D boolean array of reverse-complement flags, shape (N_seqlets,).
        meta_df: Candidate metadata DataFrame matching example_idx.
        one_hot_arr: 3D array of one-hot sequences, shape (N_windows, 4, 400).
        att_arr: 3D array of attribution matrices, shape (N_windows, 4, 400).
        top_k: Number of exemplar seqlets to select (default: 3).
        min_genomic_separation: Minimum genomic separation in bp to prevent picking overlapping loci.

    Returns:
        List of dictionaries with selected seqlet details.
    """
    n_seqlets = len(example_idxs)
    if n_seqlets == 0:
        return []

    consensus_idx = np.argmax(ppm_50, axis=1)  # shape: (50,)
    alphabet = ["A", "C", "G", "T"]
    consensus_str = "".join(alphabet[idx] for idx in consensus_idx)

    # Pre-clip PPM for log-odds calculation
    ppm_safe = np.clip(ppm_50, 1e-4, 1.0)
    log_odds_mat = np.log2(ppm_safe / 0.25)  # shape: (50, 4)

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

        # Extract 50-bp sequence slice (shape: 4, 50)
        oh_slice = one_hot_arr[ex_idx, :, s_start:s_end]
        att_slice = att_arr[ex_idx, :, s_start:s_end]

        if is_rc:
            # Reverse along sequence length, and complement bases: A(0)<->T(3), C(1)<->G(2)
            oh_slice = oh_slice[::-1, ::-1]
            att_slice = att_slice[::-1, ::-1]

        seq_indices = np.argmax(oh_slice, axis=0)  # shape: (50,)
        seq_str = "".join(alphabet[b] for b in seq_indices)

        hamming_dist = int(np.sum(seq_indices != consensus_idx))
        exact_matches = 50 - hamming_dist
        match_fraction = exact_matches / 50.0

        pwm_score = float(np.sum(log_odds_mat[np.arange(50), seq_indices]))

        per_base_attr = np.sum(np.abs(att_slice), axis=0)  # shape: (50,)
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
                if (inter_end - inter_start) > (50 - min_genomic_separation):
                    overlap = True
                    break
        if not overlap:
            cand["rank"] = len(selected) + 1
            selected.append(cand)
            if len(selected) >= top_k:
                break

    return selected


def render_and_save_dependency_map_plot(
    dm: DependencyMap,
    seqlet_info: Dict[str, Any],
    record_sequence: str,
    map_start: int,
    map_end: int,
    regions_df: pd.DataFrame,
    output_png: Path,
    scale: int = 3,
) -> None:
    """Renders 200x200 dependency map in the plot_annotated_dependency_map style and saves as PNG.

    Includes:
      - Top reconstruction sequence logo (scaled by information content in bits)
      - Axis nucleotide characters
      - Bottom genomic overview slider / track
      - Overlapping landmark annotations on the diagonal
      - Highlighted bounding box around the centered 50-bp seqlet
    """
    import logomaker
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle, Patch

    map_size = map_end - map_start
    seq_200 = record_sequence[map_start:map_end]
    fasta_length = len(record_sequence)
    record_id = seqlet_info["record_id"]
    pattern_name = seqlet_info["pattern_name"]
    pattern_display = format_pattern_display_name(pattern_name)
    rank = seqlet_info["rank"]

    # 1. Zero-out the diagonal for dependency map
    dep_map = dm.get_depenency_map_without_diagonal()
    dep_map_clipped = np.clip(dep_map, 0.0, 5.0)

    # 2. Compute Reconstruction IC Logo matrix (L, 4) in bits
    # IC_x = 2 - H(p_x)
    recon = dm.reconstruction  # shape: (200, 4)
    if recon is not None:
        p_safe = np.clip(recon, 1e-7, 1.0)
        p_safe = p_safe / np.sum(p_safe, axis=1, keepdims=True)
        entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)  # (L,)
        ic_bits = np.clip(2.0 - entropy, 0.0, 2.0)  # (L,)
        logo_matrix = p_safe * ic_bits[:, np.newaxis]  # (L, 4)
    else:
        # Fallback to one-hot sequence logo
        lut = {"A": 0, "C": 1, "G": 2, "T": 3}
        logo_matrix = np.zeros((map_size, 4), dtype=float)
        for idx, base in enumerate(seq_200):
            if base in lut:
                logo_matrix[idx, lut[base]] = 1.0

    logo_df = pd.DataFrame(logo_matrix, columns=["A", "C", "G", "T"])

    # 3. Create matplotlib figure with 3 vertically stacked axes
    fig = plt.figure(figsize=(13.0, 14.0), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(
        3,
        1,
        height_ratios=[1.1, 10.0, 0.6],
        hspace=0.08,
        left=0.07,
        right=0.80,
        top=0.88,
        bottom=0.05,
    )

    ax_logo = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])
    ax_track = fig.add_subplot(gs[2])

    # -------------------------------------------------------------------------
    # Panel 1: Top Reconstruction Sequence Logo
    # -------------------------------------------------------------------------
    color_scheme = {
        "A": "#109648",  # Green
        "C": "#255C99",  # Blue
        "G": "#F7B32B",  # Orange/Gold
        "T": "#D62839",  # Red
    }
    logomaker.Logo(logo_df, ax=ax_logo, color_scheme=color_scheme, vpad=0.02)
    ax_logo.set_xlim(-0.5, map_size - 0.5)
    ax_logo.set_ylim(0.0, 2.05)
    ax_logo.set_ylabel("Bits", fontsize=9, fontweight="bold")
    ax_logo.set_xticks([])
    ax_logo.tick_params(axis="y", labelsize=8)
    ax_logo.spines["top"].set_visible(False)
    ax_logo.spines["right"].set_visible(False)
    ax_logo.spines["bottom"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel 2: 200x200 Dependency Map Heatmap
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
    ax_map.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_map.set_xticks(np.arange(map_size))
    ax_map.set_xticklabels(list(seq_200), fontsize=3.5, fontfamily="monospace")

    ax_map.set_yticks(np.arange(map_size))
    ax_map.set_yticklabels(list(seq_200), fontsize=3.5, fontfamily="monospace")

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

    # Highlight Centered 50-bp Seqlet Bounding Box
    s_rel_start = max(0, seqlet_info["seqlet_genomic_start"] - map_start)
    s_rel_end = min(map_size, seqlet_info["seqlet_genomic_end"] - map_start)
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
            label=f"{pattern_display} Seqlet (50 bp, {strand_label})",
        ),
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.825, 0.25, 0.02, 0.45])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Nucleotide Dependency", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # Legend
    fig.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(0.815, 0.88),
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

    # Title & Subtitle (4 clean lines, well spaced without clipping)
    clade_str = f" ({seqlet_info['order_clade']})" if ("order_clade" in seqlet_info and seqlet_info["order_clade"]) else ""
    fig.suptitle(
        f"{pattern_display} (Exemplar #{rank}) — 200×200 RiNALMo Nucleotide Dependency Map\n"
        f"Species: {seqlet_info['species']}{clade_str} | Region: [{map_start:,}, {map_end:,}) bp\n"
        f"50-bp Consensus Match: {seqlet_info['exact_matches_50']}/50 bp ({seqlet_info['match_fraction_50']*100:.1f}%) | "
        f"Mean Attr: {seqlet_info['attr_mean_50']:.1f} | PWM Score: {seqlet_info['pwm_score_50']:.1f}\n"
        f"Record: {record_id}",
        fontsize=10.5,
        fontweight="bold",
        y=0.965,
        linespacing=1.3,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 200x200 dependency map to: %s", output_png)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments exposing all hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Compute and plot 200x200 RiNALMo (+LoRA) dependency maps for exemplar TF-MoDISco seqlets."
    )
    parser.add_argument(
        "--modisco-h5",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/results.h5"),
        help="Path to TF-MoDISco results.h5 (default: outputs/tfmodisco_runs/all_seq_n/results.h5).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/tfmodisco_inputs/all_seq/candidate_metadata.parquet"),
        help="Path to candidate_metadata.parquet (default: outputs/tfmodisco_inputs/all_seq/candidate_metadata.parquet).",
    )
    parser.add_argument(
        "--attributions-npz",
        type=Path,
        default=Path("outputs/tfmodisco_inputs/all_seq/attributions.npz"),
        help="Path to attributions.npz (default: outputs/tfmodisco_inputs/all_seq/attributions.npz).",
    )
    parser.add_argument(
        "--onehot-npz",
        type=Path,
        default=Path("outputs/tfmodisco_inputs/all_seq/one_hot.npz"),
        help="Path to one_hot.npz (default: outputs/tfmodisco_inputs/all_seq/one_hot.npz).",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/prepared/ribosome/records.parquet"),
        help="Path to records.parquet containing rDNA sequences (default: data/prepared/ribosome/records.parquet).",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=Path("data/prepared/ribosome/regions.parquet"),
        help="Path to regions.parquet containing landmark annotations (default: data/prepared/ribosome/regions.parquet).",
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
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device (default: cuda:0 if available, else cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/exemplar_seqlet_dependency_maps"),
        help="Output directory (default: outputs/tfmodisco_runs/all_seq_n/exemplar_seqlet_dependency_maps).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of exemplar seqlets to plot per motif pattern (default: 3).",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=200,
        help="Active subset size in bp for dependency map (default: 200 -> 200x200 map).",
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
        "--save-npz",
        action="store_true",
        help="Also save the raw 200x200 dependency map and reconstruction as .npz alongside the PNG.",
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

        # Load RiNALMo-mega + LoRA weights
        logger.info("Loading RiNALMo (%s) from %s with LoRA from %s...", args.model_size, args.weights, args.lora_weights)
        model = load_rinalmo_model(
            model_size=args.model_size,
            weights_path=args.weights,
            device=args.device,
            lora_weights_path=args.lora_weights,
        )
        logger.info("Model and LoRA weights loaded and merged successfully on %s.", args.device)

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
                    if args.patterns is None or full_p in args.patterns or p_name in args.patterns:
                        all_pattern_names.append(full_p)

        logger.info("Processing %d motif pattern(s) with top-%d exemplar seqlets each...", len(all_pattern_names), args.top_k)

        all_selected_seqlets: List[Dict[str, Any]] = []

        with h5py.File(args.modisco_h5, "r") as f:
            for full_p in all_pattern_names:
                grp_name, pat_name = full_p.split(".")
                p_grp = f[grp_name][pat_name]
                ppm_50 = np.array(p_grp["sequence"][:], dtype=np.float64)

                s_grp = p_grp["seqlets"]
                ex_idxs = np.array(s_grp["example_idx"][:], dtype=np.int64)
                s_starts = np.array(s_grp["start"][:], dtype=np.int64)
                s_ends = np.array(s_grp["end"][:], dtype=np.int64)
                is_rcs = np.array(s_grp["is_revcomp"][:], dtype=bool)

                selected = select_top_exemplar_seqlets(
                    pattern_name=full_p,
                    ppm_50=ppm_50,
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

        # Compute dependency maps and render plots
        def tokenize_func(sequence: str, mask: Optional[int]) -> object:
            return model.dependency_tokenize(sequence, mask)

        forward_func = partial(model.dependency_forward, batch_size=args.batch_size)

        for seqlet in all_selected_seqlets:
            pat_name = seqlet["pattern_name"]
            pat_display = format_pattern_display_name(pat_name)
            rank = seqlet["rank"]
            rec_id = seqlet["record_id"]
            s_start = seqlet["seqlet_genomic_start"]
            s_end = seqlet["seqlet_genomic_end"]

            rec_seq = str(records_df.loc[rec_id, "sequence"]).upper()
            seq_len = len(rec_seq)

            # Center a 200-bp map window around the 50-bp seqlet midpoint
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

            pat_out_dir = output_dir / pat_name
            pat_out_dir.mkdir(parents=True, exist_ok=True)
            safe_rec = "".join(c if c.isalnum() else "_" for c in rec_id)
            out_png = pat_out_dir / f"{pat_display}_rank{rank}_{safe_rec}_{map_start}_{map_end}.png"

            if args.resume and out_png.is_file():
                logger.info("Skipping existing map for %s rank %d: %s", pat_display, rank, out_png)
                continue

            logger.info(
                "Computing dependency map for %s (rank %d): record=%s, map=[%d, %d), context=[%d, %d) (len=%d)...",
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

            dm = DependencyMap.compute_batched(
                context_seq,
                tokenize_func,
                forward_func,
                batch_size=args.batch_size,
                enable_progress_bar=False,
                options=job_options,
            )

            render_and_save_dependency_map_plot(
                dm=dm,
                seqlet_info=seqlet,
                record_sequence=rec_seq,
                map_start=map_start,
                map_end=map_end,
                regions_df=regions_df,
                output_png=out_png,
                scale=args.scale,
            )

            if args.save_npz:
                out_npz = pat_out_dir / f"{pat_display}_rank{rank}_{safe_rec}_{map_start}_{map_end}.npz"
                np.savez_compressed(
                    out_npz,
                    dependency_map=dm.dependency_map,
                    reconstruction=dm.reconstruction,
                    sequence=np.array(list(context_seq[active_subset[0] : active_subset[1]])),
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
