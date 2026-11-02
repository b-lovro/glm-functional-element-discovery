#!/usr/bin/env python3
"""
TF-MoDISco Per-Pattern Boxplot Comparison: Motif Seqlets vs. EnhancerRepeats
=============================================================================
Generates per-pattern two-panel boxplot distributions for discovered TF-MoDISco motifs:
  - Top Panel: Per-nucleotide Delta Block Score (Adapted − Base RiNALMo, quantifying epistatic constraint)
  - Bottom Panel: Per-nucleotide gLM Reconstruction Information Content (in bits, quantifying LM sequence predictability)

For each motif, compares its 50-bp seqlet nucleotides directly against random non-overlapping 50-bp
sequences sampled strictly from annotated EnhancerRepeats regions that were NOT matched by TF-MoDISco
in seqlets (and do not overlap Core, TTF1, or CTCF landmarks).

Boxplot Specification (strictly Tukey):
  - Centerline: median
  - Box limits: first (Q1) and third (Q3) quartiles
  - Whiskers: span all data within 1.5×IQR of the lower and upper quartiles

Outputs:
  - High-resolution PNG and PDF figures for each pattern in <output_dir>/
  - Comprehensive statistical summary TSV: <output_dir>/novel_vs_enhancer_repeats_summary.tsv
"""

from __future__ import annotations

import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from glmfe.seq_models.rinalmo import load_rinalmo_model

logger = logging.getLogger("plot_tfmodisco_novel_vs_enhancer_repeats")
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


def format_pattern_display_name(name: str) -> str:
    """Formats pattern identifier cleanly, preserving neg_ prefix if from neg_patterns."""
    if name.startswith("neg_patterns."):
        return f"neg_{name.split('.')[-1]}"
    return name.split(".")[-1]


def pattern_sort_key(name: str) -> Tuple[int, int]:
    """Extracts (is_neg, pattern_index) for natural sorting."""
    is_neg = 1 if name.startswith("neg_") else 0
    parts = name.split(".")[-1].split("_")
    return (is_neg, int(parts[-1]))


def compute_information_content_bits(reconstruction: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Computes per-position Shannon Information Content in bits from reconstruction probabilities.

    Args:
        reconstruction: Probability or logit matrix of shape (L, 4).
        eps: Small constant for numerical stability.

    Returns:
        1D array of shape (L,) containing per-position Information Content in bits (0.0 to 2.0).
    """
    p_safe = np.clip(reconstruction, eps, 1.0)
    p_safe = p_safe / np.sum(p_safe, axis=1, keepdims=True)
    entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)  # shape: (L,)
    ic_bits = np.clip(2.0 - entropy, 0.0, 2.0)  # shape: (L,)
    return ic_bits


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between distributions x and y."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float((2.0 * u_stat) / (n_x * n_y) - 1.0)


def extract_motif_seqlet_metrics(
    pattern_name: str,
    dep_map_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Extracts per-nucleotide Delta Block Score and gLM Reconstruction IC for a motif from dep_map_100 NPZ.

    Args:
        pattern_name: Full pattern name (e.g. 'pos_patterns.pattern_14').
        dep_map_dir: Path to directory containing per-pattern NPZ folders.

    Returns:
        Dictionary with extracted arrays or None if NPZ not found.
    """
    pattern_dir = dep_map_dir / pattern_name
    if not pattern_dir.is_dir():
        return None

    npz_files = sorted(list(pattern_dir.glob("*.npz")))
    if not npz_files:
        return None

    # Load rank-1 exemplar NPZ
    npz_path = npz_files[0]
    data = np.load(npz_path)

    map_start = int(data["map_start"])
    map_end = int(data["map_end"])
    seqlet_start = int(data["seqlet_start"])
    seqlet_end = int(data["seqlet_end"])

    off_start = max(0, seqlet_start - map_start)
    off_end = min(map_end - map_start, seqlet_end - map_start)

    bs_delta_full = data["block_score_delta"]  # (100,)
    recon_adapted_full = data["reconstruction_adapted"]  # (100, 4)

    # Slice the exact 50-bp seqlet
    bs_delta_seqlet = bs_delta_full[off_start:off_end]  # (50,)
    recon_seqlet = recon_adapted_full[off_start:off_end]  # (50, 4)

    ic_bits_seqlet = compute_information_content_bits(recon_seqlet)  # (50,)

    return {
        "pattern_name": pattern_name,
        "npz_path": str(npz_path),
        "block_score_delta": bs_delta_seqlet,
        "recon_ic_bits": ic_bits_seqlet,
        "seqlet_length": len(bs_delta_seqlet),
    }


def get_unmatched_enhancer_repeats_metrics(
    records_path: Path,
    regions_path: Path,
    seqlet_annot_path: Path,
    single_nuc_pre_path: Path,
    single_nuc_full_path: Path,
    cache_path: Path,
    weights_path: Path,
    lora_weights_path: Path,
    device: str = "cuda:0",
    num_windows: int = 50,
    window_length: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Samples random 50-bp windows within EnhancerRepeats annotations NOT matched by TF-MoDISco seqlets.

    Criteria:
      - Window lies completely within an annotated EnhancerRepeats region (regions.parquet)
      - Window has zero overlap with any TF-MoDISco seqlet (seqlet_annotations_overlap.tsv)
      - Window has zero overlap with Core, TTF1, or CTCF annotations
      - Non-overlapping sampling across diverse species records

    Args:
        records_path: Path to records.parquet.
        regions_path: Path to regions.parquet.
        seqlet_annot_path: Path to seqlet_annotations_overlap.tsv.
        single_nuc_pre_path: Path to adapted single_nucleotide_scores.parquet.
        single_nuc_full_path: Path to base single_nucleotide_scores.parquet.
        cache_path: Path to cache file for precomputed unmatched EnhancerRepeats arrays.
        weights_path: Path to RiNALMo pretrained weights.
        lora_weights_path: Path to RiNALMo LoRA weights.
        device: Device for model inference ('cuda:0' or 'cpu').
        num_windows: Number of random 50-bp windows to sample (default: 50 -> 2,500 nt).
        window_length: Length in bp of each window (default: 50 bp).
        seed: Random seed for sampling.

    Returns:
        Dictionary containing pooled 'block_score_delta' and 'recon_ic_bits'.
    """
    if cache_path.is_file():
        logger.info("Loading Unmatched EnhancerRepeats metrics from cache: %s", cache_path)
        cached = np.load(cache_path)
        return {
            "block_score_delta": cached["block_score_delta"],
            "recon_ic_bits": cached["recon_ic_bits"],
            "num_windows": int(cached["num_windows"]),
            "num_nucleotides": len(cached["block_score_delta"]),
        }

    logger.info("Sampling random %d-bp sequences from EnhancerRepeats NOT matched by TF-MoDISco seqlets...", window_length)
    records_df = pd.read_parquet(records_path).set_index("record_id")
    regions_df = pd.read_parquet(regions_path)
    seqlets_df = pd.read_csv(seqlet_annot_path, sep="\t")

    er_regs = regions_df[regions_df["feature_type"] == "EnhancerRepeats"]
    avoid_regs = regions_df[regions_df["feature_type"].isin(["Core", "TTF1", "CTCF"])]

    # Build exclusion lookup by record_id
    exclusions: Dict[str, List[Tuple[int, int]]] = {}
    for _, r in seqlets_df.iterrows():
        rec_id = str(r["record_id"])
        exclusions.setdefault(rec_id, []).append((int(r["seqlet_genomic_start"]), int(r["seqlet_genomic_end"])))
    for _, r in avoid_regs.iterrows():
        rec_id = str(r["record_id"])
        exclusions.setdefault(rec_id, []).append((int(r["start"]), int(r["end"])))

    # Identify all valid non-overlapping 50-bp windows
    valid_candidates: List[Tuple[str, int, int]] = []
    for _, r in er_regs.iterrows():
        rec_id = str(r["record_id"])
        r_start = int(r["start"])
        r_end = int(r["end"])
        ex_list = exclusions.get(rec_id, [])

        for win_start in range(r_start, r_end - window_length + 1, window_length):
            win_end = win_start + window_length
            overlap = False
            for ex_s, ex_e in ex_list:
                if not (win_end <= ex_s or win_start >= ex_e):
                    overlap = True
                    break
            if not overlap:
                valid_candidates.append((rec_id, win_start, win_end))

    logger.info("Identified %d candidate non-overlapping unmatched %d-bp EnhancerRepeat intervals.", len(valid_candidates), window_length)
    if not valid_candidates:
        raise RuntimeError("No valid unmatched EnhancerRepeats windows found.")

    rng = random.Random(seed)
    sampled_windows = rng.sample(valid_candidates, min(num_windows, len(valid_candidates)))
    logger.info("Sampled %d random unmatched EnhancerRepeat windows.", len(sampled_windows))

    # 1. Extract Delta Block Scores from single_nucleotide_scores
    logger.info("Loading single-nucleotide scores to extract Delta Block Scores...")
    df_pre = pd.read_parquet(single_nuc_pre_path)
    df_full = pd.read_parquet(single_nuc_full_path)
    df_pre["delta_block_score"] = np.maximum(0.0, df_pre["block_score"].values - df_full["block_score"].values)
    df_pre.set_index(["record_id", "position"], inplace=True)

    delta_bs_list: List[float] = []
    windows_for_recon: List[Tuple[str, int, int, str]] = []

    for rec_id, win_start, win_end in sampled_windows:
        pos_keys = [(rec_id, p) for p in range(win_start, win_end)]
        matched_scores = df_pre.loc[df_pre.index.intersection(pos_keys), "delta_block_score"].values
        if len(matched_scores) == window_length:
            delta_bs_list.extend(matched_scores)

            rec_seq = str(records_df.loc[rec_id, "sequence"]).upper()
            seq_slice = rec_seq[win_start:win_end]
            if len(seq_slice) == window_length and set(seq_slice).issubset({"A", "C", "G", "T"}):
                windows_for_recon.append((rec_id, win_start, win_end, rec_seq))

    er_delta_bs = np.array(delta_bs_list, dtype=np.float32)

    # 2. Compute gLM Reconstruction IC using RiNALMo on GPU
    logger.info("Computing RiNALMo reconstruction Information Content on %d unmatched EnhancerRepeat windows...", len(windows_for_recon))
    inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rinalmo = load_rinalmo_model(
        model_size="mega",
        weights_path=weights_path,
        device=inference_device,
        lora_weights_path=lora_weights_path,
    )

    recon_ic_list: List[float] = []
    for rec_id, win_start, win_end, rec_seq in windows_for_recon:
        # 100 bp context window centered on 50 bp window
        half_pad = 25
        c_start = max(0, win_start - half_pad)
        c_end = min(len(rec_seq), win_end + half_pad)
        context_seq = rec_seq[c_start:c_end]
        target_offset = win_start - c_start

        target_positions = list(range(target_offset, target_offset + window_length))
        windows = [context_seq] * window_length
        probs = rinalmo.predict_masked_base_probabilities(windows, target_positions, batch_size=50)

        ic_window = compute_information_content_bits(probs)
        recon_ic_list.extend(ic_window)

    er_recon_ic = np.array(recon_ic_list, dtype=np.float32)

    del rinalmo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Unmatched EnhancerRepeats Baseline: %d Delta BS values (med=%.2f, IQR=%.2f), %d IC values (med=%.2f bits, IQR=%.2f bits)",
        len(er_delta_bs),
        float(np.median(er_delta_bs)),
        float(np.percentile(er_delta_bs, 75) - np.percentile(er_delta_bs, 25)),
        len(er_recon_ic),
        float(np.median(er_recon_ic)),
        float(np.percentile(er_recon_ic, 75) - np.percentile(er_recon_ic, 25)),
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        block_score_delta=er_delta_bs,
        recon_ic_bits=er_recon_ic,
        num_windows=len(windows_for_recon),
    )
    logger.info("Saved unmatched EnhancerRepeats metrics cache to: %s", cache_path)

    return {
        "block_score_delta": er_delta_bs,
        "recon_ic_bits": er_recon_ic,
        "num_windows": len(windows_for_recon),
        "num_nucleotides": len(er_delta_bs),
    }


def render_pattern_boxplot(
    pattern_name: str,
    pattern_display: str,
    category: str,
    best_match: str,
    motif_delta_bs: np.ndarray,
    motif_recon_ic: np.ndarray,
    er_delta_bs: np.ndarray,
    er_recon_ic: np.ndarray,
    er_label: str,
    output_png: Path,
    whisker_iqr: float = 1.5,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Renders a standalone two-panel boxplot comparing one motif against EnhancerRepeats.

    Boxplot Specification:
      - Centerline: median
      - Box limits: first and third quartiles (Q1, Q3)
      - Whiskers: span all data within 1.5×IQR of lower and upper quartiles
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(5.8, 6.8), dpi=dpi, facecolor="white"
    )
    fig.subplots_adjust(hspace=0.22, left=0.16, right=0.94, top=0.96, bottom=0.08)

    labels = ["Motif Seqlets", er_label]
    positions = [0, 1]
    palette = ["#7C3AED", "#0D9488"]  # Purple for motif, Teal for EnhancerRepeats

    # Compute statistics for summary
    m_bs_med, m_bs_q1, m_bs_q3 = float(np.median(motif_delta_bs)), float(np.percentile(motif_delta_bs, 25)), float(np.percentile(motif_delta_bs, 75))
    m_bs_iqr = m_bs_q3 - m_bs_q1
    er_bs_med, er_bs_q1, er_bs_q3 = float(np.median(er_delta_bs)), float(np.percentile(er_delta_bs, 25)), float(np.percentile(er_delta_bs, 75))
    er_bs_iqr = er_bs_q3 - er_bs_q1

    m_ic_med, m_ic_q1, m_ic_q3 = float(np.median(motif_recon_ic)), float(np.percentile(motif_recon_ic, 25)), float(np.percentile(motif_recon_ic, 75))
    m_ic_iqr = m_ic_q3 - m_ic_q1
    er_ic_med, er_ic_q1, er_ic_q3 = float(np.median(er_recon_ic)), float(np.percentile(er_recon_ic, 25)), float(np.percentile(er_recon_ic, 75))
    er_ic_iqr = er_ic_q3 - er_ic_q1

    # Statistical tests
    u_bs, p_bs = mannwhitneyu(motif_delta_bs, er_delta_bs, alternative="two-sided")
    cliff_bs = compute_cliffs_delta(motif_delta_bs, er_delta_bs)

    u_ic, p_ic = mannwhitneyu(motif_recon_ic, er_recon_ic, alternative="two-sided")
    cliff_ic = compute_cliffs_delta(motif_recon_ic, er_recon_ic)

    # -------------------------------------------------------------------------
    # Panel 1 (Top): Per-Nucleotide Delta Block Score
    # -------------------------------------------------------------------------
    bp_top = ax_top.boxplot(
        [motif_delta_bs, er_delta_bs],
        positions=positions,
        widths=0.42,
        patch_artist=True,
        whis=whisker_iqr,
        medianprops=dict(color="#111827", linewidth=2.2),
        boxprops=dict(edgecolor="#111827", linewidth=1.3),
        whiskerprops=dict(color="#111827", linewidth=1.3, linestyle="--"),
        capprops=dict(color="#111827", linewidth=1.3),
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor="#4B5563", alpha=0.35, markeredgewidth=0),
    )

    for patch, color in zip(bp_top["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    # Jitter scatter overlay for motif nucleotides (50 nt)
    np.random.seed(42)
    jitter_m = np.random.normal(0, 0.04, size=len(motif_delta_bs))
    ax_top.scatter(
        0 + jitter_m,
        motif_delta_bs,
        color="#4C1D95",
        alpha=0.45,
        s=16,
        edgecolors="none",
        zorder=3,
    )

    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(labels, fontsize=10.0, fontweight="bold")
    ax_top.set_ylabel("Per-Nucleotide Δ Block Score", fontsize=10.5, fontweight="bold")
    ax_top.grid(axis="y", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 2 (Bottom): Per-Nucleotide gLM Reconstruction Information Content
    # -------------------------------------------------------------------------
    bp_bottom = ax_bottom.boxplot(
        [motif_recon_ic, er_recon_ic],
        positions=positions,
        widths=0.42,
        patch_artist=True,
        whis=whisker_iqr,
        medianprops=dict(color="#111827", linewidth=2.2),
        boxprops=dict(edgecolor="#111827", linewidth=1.3),
        whiskerprops=dict(color="#111827", linewidth=1.3, linestyle="--"),
        capprops=dict(color="#111827", linewidth=1.3),
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor="#4B5563", alpha=0.35, markeredgewidth=0),
    )

    for patch, color in zip(bp_bottom["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    # Jitter scatter overlay for motif nucleotides
    ax_bottom.scatter(
        0 + jitter_m,
        motif_recon_ic,
        color="#4C1D95",
        alpha=0.45,
        s=16,
        edgecolors="none",
        zorder=3,
    )

    ax_bottom.set_xticks(positions)
    ax_bottom.set_xticklabels(labels, fontsize=10.0, fontweight="bold")
    ax_bottom.set_ylabel("gLM Reconstruction IC (bits)", fontsize=10.5, fontweight="bold")
    ax_bottom.set_ylim(-0.05, 2.15)
    ax_bottom.grid(axis="y", linestyle=":", alpha=0.6)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "pattern_name": pattern_name,
        "pattern_display": pattern_display,
        "category": category,
        "best_match": best_match,
        "motif_nucleotides_count": len(motif_delta_bs),
        "er_nucleotides_count": len(er_delta_bs),
        "motif_delta_bs_median": m_bs_med,
        "motif_delta_bs_iqr": m_bs_iqr,
        "motif_delta_bs_mean": float(np.mean(motif_delta_bs)),
        "motif_delta_bs_std": float(np.std(motif_delta_bs)),
        "motif_delta_bs_q1": m_bs_q1,
        "motif_delta_bs_q3": m_bs_q3,
        "er_delta_bs_median": er_bs_med,
        "er_delta_bs_iqr": er_bs_iqr,
        "er_delta_bs_mean": float(np.mean(er_delta_bs)),
        "er_delta_bs_std": float(np.std(er_delta_bs)),
        "er_delta_bs_q1": er_bs_q1,
        "er_delta_bs_q3": er_bs_q3,
        "delta_bs_mannwhitney_u": float(u_bs),
        "delta_bs_pvalue": float(p_bs),
        "delta_bs_cliffs_delta": float(cliff_bs),
        "motif_recon_ic_median": m_ic_med,
        "motif_recon_ic_iqr": m_ic_iqr,
        "motif_recon_ic_mean": float(np.mean(motif_recon_ic)),
        "motif_recon_ic_std": float(np.std(motif_recon_ic)),
        "motif_recon_ic_q1": m_ic_q1,
        "motif_recon_ic_q3": m_ic_q3,
        "er_recon_ic_median": er_ic_med,
        "er_recon_ic_iqr": er_ic_iqr,
        "er_recon_ic_mean": float(np.mean(er_recon_ic)),
        "er_recon_ic_std": float(np.std(er_recon_ic)),
        "er_recon_ic_q1": er_ic_q1,
        "er_recon_ic_q3": er_ic_q3,
        "recon_ic_mannwhitney_u": float(u_ic),
        "recon_ic_pvalue": float(p_ic),
        "recon_ic_cliffs_delta": float(cliff_ic),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot per-pattern Delta Block Score and gLM Reconstruction IC boxplots comparing motifs against EnhancerRepeats."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n"),
        help="Path to TF-MoDISco run directory (default: outputs/tfmodisco_runs/all_seq_n).",
    )
    parser.add_argument(
        "--dep-map-dir",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/dep_map_100"),
        help="Path to computed dependency map NPZ directory (default: outputs/tfmodisco_runs/all_seq_n/dep_map_100).",
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        default=Path("data/prepared/ribosome/records.parquet"),
        help="Path to records.parquet (default: data/prepared/ribosome/records.parquet).",
    )
    parser.add_argument(
        "--regions-path",
        type=Path,
        default=Path("data/prepared/ribosome/regions.parquet"),
        help="Path to regions.parquet (default: data/prepared/ribosome/regions.parquet).",
    )
    parser.add_argument(
        "--seqlet-annotations",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/seqlet_annotations_overlap.tsv"),
        help="Path to seqlet_annotations_overlap.tsv (default: outputs/tfmodisco_runs/all_seq_n/seqlet_annotations_overlap.tsv).",
    )
    parser.add_argument(
        "--pattern-classification",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/pattern_classification_summary.tsv"),
        help="Path to pattern_classification_summary.tsv (default: outputs/tfmodisco_runs/all_seq_n/pattern_classification_summary.tsv).",
    )
    parser.add_argument(
        "--single-nuc-adapted",
        type=Path,
        default=Path("outputs/runs/rinalmo_tfmodisco_pre_full/motif_evaluation/single_nucleotide_scores.parquet"),
        help="Path to adapted single_nucleotide_scores.parquet.",
    )
    parser.add_argument(
        "--single-nuc-base",
        type=Path,
        default=Path("outputs/runs/rinalmo_tfmodisco_full/motif_evaluation/single_nucleotide_scores.parquet"),
        help="Path to base single_nucleotide_scores.parquet.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/rinalmo/rinalmo_mega_pretrained.pt"),
        help="Path to base RiNALMo model weights.",
    )
    parser.add_argument(
        "--lora-weights",
        type=Path,
        default=Path("outputs/runs/pretraining-rinalmo-bf16-lr-r-long/checkpoint_latest.pt"),
        help="Path to adapted LoRA weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/novel_vs_enhancer_repeats"),
        help="Directory to save per-pattern boxplot figures and summary TSV.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Optional list of patterns to process (e.g. pattern_14 pattern_16). Default: all patterns in dep-map-dir.",
    )
    parser.add_argument(
        "--num-er-windows",
        type=int,
        default=50,
        help="Number of random non-overlapping 50-bp unmatched EnhancerRepeat windows to sample (default: 50 -> 2,500 nt).",
    )
    parser.add_argument(
        "--whisker-iqr",
        type=float,
        default=1.5,
        help="Whisker reach in units of interquartile range (default: 1.5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for exported PNGs (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution entry point."""
    args = parse_args(argv)
    set_global_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.results_dir / "unmatched_enhancer_repeats_cache.npz"

    logger.info("Initializing per-pattern boxplot generation...")
    logger.info("Results directory: %s", args.results_dir)
    logger.info("Exemplar NPZ directory: %s", args.dep_map_dir)
    logger.info("Output directory: %s", args.output_dir)

    # 1. Load classification summary
    if not args.pattern_classification.is_file():
        raise FileNotFoundError(f"Classification summary not found: {args.pattern_classification}")
    df_class = pd.read_csv(args.pattern_classification, sep="\t")
    class_map: Dict[str, Dict[str, str]] = {}
    for _, r in df_class.iterrows():
        pname = str(r["pattern"])
        class_map[pname] = {
            "category": str(r["category"]),
            "modality": str(r["best_overall_modality"]),
            "best_match": str(r["best_overall_match"]) if pd.notna(r["best_overall_match"]) else "None",
        }

    # 2. Extract or load unmatched EnhancerRepeats metrics
    er_metrics = get_unmatched_enhancer_repeats_metrics(
        records_path=args.records_path,
        regions_path=args.regions_path,
        seqlet_annot_path=args.seqlet_annotations,
        single_nuc_pre_path=args.single_nuc_adapted,
        single_nuc_full_path=args.single_nuc_base,
        cache_path=cache_path,
        weights_path=args.weights,
        lora_weights_path=args.lora_weights,
        num_windows=args.num_er_windows,
        seed=args.seed,
    )
    er_delta_bs = er_metrics["block_score_delta"]
    er_recon_ic = er_metrics["recon_ic_bits"]
    er_label = "EnhancerRepeats"

    # 3. Determine patterns to process
    available_dirs = sorted(
        [d for d in args.dep_map_dir.iterdir() if d.is_dir() and (d.name.startswith("pos_patterns.") or d.name.startswith("neg_patterns."))],
        key=lambda d: pattern_sort_key(d.name),
    )

    patterns_to_run: List[str] = []
    for d in available_dirs:
        pname = d.name
        p_num = pname.split(".")[-1].replace("pattern_", "")
        short_alias = pname.split(".")[-1]
        if (
            args.patterns is None
            or pname in args.patterns
            or short_alias in args.patterns
            or p_num in args.patterns
        ):
            patterns_to_run.append(pname)

    logger.info("Found %d patterns to process.", len(patterns_to_run))
    if not patterns_to_run:
        logger.warning("No matching pattern directories found in %s", args.dep_map_dir)
        return 0

    summary_rows: List[Dict[str, Any]] = []

    # 4. Generate per-pattern plots
    for idx, pat_name in enumerate(patterns_to_run):
        pat_display = format_pattern_display_name(pat_name)
        meta = class_map[pat_name] if pat_name in class_map else {"category": "Unknown", "modality": "Unknown", "best_match": "None"}
        category = meta["category"]
        best_match = meta["best_match"]

        logger.info(
            "[%d/%d] Processing %s (category: %s)...",
            idx + 1,
            len(patterns_to_run),
            pat_display,
            category,
        )

        motif_metrics = extract_motif_seqlet_metrics(pat_name, args.dep_map_dir)
        if motif_metrics is None:
            logger.warning("Skipping %s: no valid NPZ data found.", pat_name)
            continue

        motif_delta_bs = motif_metrics["block_score_delta"]
        motif_recon_ic = motif_metrics["recon_ic_bits"]

        out_png = args.output_dir / f"{pat_display}_vs_EnhancerRepeats.png"

        stats = render_pattern_boxplot(
            pattern_name=pat_name,
            pattern_display=pat_display,
            category=category,
            best_match=best_match,
            motif_delta_bs=motif_delta_bs,
            motif_recon_ic=motif_recon_ic,
            er_delta_bs=er_delta_bs,
            er_recon_ic=er_recon_ic,
            er_label=er_label,
            output_png=out_png,
            whisker_iqr=args.whisker_iqr,
            dpi=args.dpi,
        )
        summary_rows.append(stats)
        logger.info("Saved per-pattern plot to %s", out_png)

    # 5. Export comprehensive summary TSV
    summary_df = pd.DataFrame(summary_rows)
    tsv_path = args.output_dir / "novel_vs_enhancer_repeats_summary.tsv"
    summary_df.to_csv(tsv_path, sep="\t", index=False)
    logger.info("Saved statistical summary TSV (%d motifs) to: %s", len(summary_df), tsv_path)

    logger.info("All per-pattern boxplot comparisons successfully completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
