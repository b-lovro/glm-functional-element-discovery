#!/usr/bin/env python3
"""
Nucleotide Dependency & Base-Resolution IC Concordance Analysis
==============================================================
Implements the base-resolution conservation vs. nucleotide dependency concordance
analysis following Tomaz da Silva, Karollus, ..., & Gagneur (2025), "Nucleotide dependency
analysis of genomic language models detects functional elements", Nature Genetics.

For each discovered TF-MoDISco pattern:
1. Base-Resolution Concordance Plot:
   Plots Base Information Content IC(x) in bits vs. Mean Nucleotide Dependency S(x) across:
   - Core Motif: contiguous subsequence exceeding peak attribution threshold
   - Whole Matched Motif: full 50-bp matched motif window
   - Distal Flanking: background sequence positions (|d| >= 75 bp)

2. Core Motif CWM Attribution Magnitude vs. Sequence IC Concordance:
   Computes and displays next to the plot the correlation between per-position
   attribution magnitude |CWM_i| across the core and empirical sequence information
   content IC_i (Pearson r, Spearman rho, and p-values), alongside whole-motif
   and signal-to-noise ratio metrics.

Outputs saved to <run_dir>/dependency_footprint_validation/:
- <pattern_name>.png: Per-pattern conservation vs. nucleotide dependency plot with concordance card
- dependency_footprint_summary.tsv: Quantitative metrics table across all patterns
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import torch


logger = logging.getLogger("plot_dependency_footprint_and_ic_concordance")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


DNA_COLOR_SCHEME: Dict[str, str] = {
    "A": "#109648",
    "C": "#255C99",
    "G": "#F7B32B",
    "T": "#D62839",
}
ALPHABET: List[str] = ["A", "C", "G", "T"]


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


def get_compute_device() -> torch.device:
    """Dynamically allocates compute device (CUDA, MPS, or CPU)."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


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


def compute_information_content(ppm: np.ndarray, eps: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-position Information Content (in bits) and IC-scaled letter heights.

    Args:
        ppm: Position Probability Matrix of shape (L, 4).
        eps: Small constant for numerical stability in log2.

    Returns:
        ic_per_pos: 1D array of shape (L,) in [0, 2] bits.
        ic_matrix: 2D array of shape (L, 4) where ic_matrix[i, b] = ppm[i, b] * ic_per_pos[i].
    """
    p_safe = np.clip(ppm, eps, 1.0)
    p_safe = p_safe / np.sum(p_safe, axis=1, keepdims=True)
    entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)  # shape: (L,)
    ic_per_pos = np.clip(2.0 - entropy, 0.0, 2.0)  # shape: (L,)
    ic_matrix = p_safe * ic_per_pos[:, np.newaxis]  # shape: (L, 4)
    return ic_per_pos, ic_matrix


def find_core_subsequence(
    contrib_scores: np.ndarray,
    peak_ratio: float = 0.65,
    min_core_len: int = 6,
    smoothing_window: int = 3,
) -> Tuple[int, int, np.ndarray, np.ndarray, float]:
    """Identifies the contiguous core subsequence exceeding peak_ratio of peak attribution.

    Args:
        contrib_scores: Contribution Weight Matrix (CWM) of shape (L, 4).
        peak_ratio: Fraction of peak attribution threshold (default: 0.65).
        min_core_len: Minimum core length in bp (default: 6).
        smoothing_window: Smoothing window width in bp (default: 3).

    Returns:
        core_start: Inclusive start index (0-based).
        core_end: Exclusive end index (0-based).
        raw_attr: 1D array of shape (L,) with per-position max attribution magnitude.
        smooth_attr: 1D array of shape (L,) with smoothed attribution magnitude.
        threshold_val: Threshold value (peak_ratio * max(raw_attr)).
    """
    L = contrib_scores.shape[0]
    raw_attr = np.max(np.abs(contrib_scores), axis=1)  # shape: (L,)
    peak_attr = float(np.max(raw_attr))
    threshold_val = peak_ratio * peak_attr

    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=np.float64) / float(smoothing_window)
        smooth_attr = np.convolve(raw_attr, kernel, mode="same")
    else:
        smooth_attr = raw_attr.copy()

    mask = smooth_attr >= threshold_val

    runs: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            in_run = True
            run_start = i
        elif not val and in_run:
            in_run = False
            runs.append((run_start, i))
    if in_run:
        runs.append((run_start, L))

    if runs:
        core_start, core_end = max(runs, key=lambda r: (r[1] - r[0], float(np.sum(raw_attr[r[0]:r[1]]))))
    else:
        peak_idx = int(np.argmax(raw_attr))
        half = min_core_len // 2
        core_start = max(0, peak_idx - half)
        core_end = min(L, core_start + min_core_len)

    if (core_end - core_start) < min_core_len:
        center = (core_start + core_end) // 2
        half = min_core_len // 2
        core_start = max(0, center - half)
        core_end = min(L, core_start + min_core_len)
        if (core_end - core_start) < min_core_len and core_start > 0:
            core_start = max(0, core_end - min_core_len)

    return core_start, core_end, raw_attr, smooth_attr, threshold_val


def analyze_pattern_footprint_and_concordance(
    pattern_name: str,
    ppm_50: np.ndarray,
    cwm_50: np.ndarray,
    example_idxs: np.ndarray,
    seqlet_starts: np.ndarray,
    seqlet_ends: np.ndarray,
    is_revcomps: np.ndarray,
    att_1d: np.ndarray,
    one_hot_all: np.ndarray,
    half_window: int = 150,
    peak_ratio: float = 0.65,
    min_core_len: int = 6,
    smoothing_window: int = 3,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Computes aligned dependency footprint, empirical IC, and base-resolution concordance metrics.

    Args:
        pattern_name: Full pattern name (e.g., 'pos_patterns.pattern_0').
        ppm_50: 50-bp Position Probability Matrix from MoDISco HDF5, shape (50, 4).
        cwm_50: 50-bp Contribution Weight Matrix from MoDISco HDF5, shape (50, 4).
        example_idxs: 1D array of candidate window indices, shape (N_seqlets,).
        seqlet_starts: 1D array of seqlet start offsets in 400-bp window, shape (N_seqlets,).
        seqlet_ends: 1D array of seqlet end offsets in 400-bp window, shape (N_seqlets,).
        is_revcomps: 1D boolean array of reverse-complement flags, shape (N_seqlets,).
        att_1d: 2D array of per-position nucleotide dependency centrality, shape (N_windows, 400).
        one_hot_all: 3D array of one-hot encoded sequences, shape (N_windows, 4, 400).
        half_window: Half-width of aligned footprint window in bp (default: 150 -> 301 bp total).
        peak_ratio: Core attribution peak fraction threshold (default: 0.65).
        min_core_len: Minimum core motif length in bp (default: 6).
        smoothing_window: Smoothing filter width in bp (default: 3).

    Returns:
        metrics: Dictionary of summary statistics for this pattern.
        dep_mean: 1D array of shape (2*half_window + 1,) with mean nucleotide dependency profile.
        ic_aligned: 1D array of shape (2*half_window + 1,) with aligned sequence IC in bits.
    """
    core_start, core_end, _, _, _ = find_core_subsequence(
        cwm_50,
        peak_ratio=peak_ratio,
        min_core_len=min_core_len,
        smoothing_window=smoothing_window,
    )
    core_len = core_end - core_start
    core_mid_offset = (core_start + core_end) // 2

    ic_50, _ = compute_information_content(ppm_50)

    win_len = att_1d.shape[1]
    span_len = 2 * half_window + 1

    # Collect aligned [-half_window, +half_window] slices with NaN padding if near window edges
    n_seqlets = len(example_idxs)
    dep_mat = np.full((n_seqlets, span_len), np.nan, dtype=np.float64)
    oh_sum = np.zeros((span_len, 4), dtype=np.float64)
    oh_cnt = np.zeros(span_len, dtype=np.float64)

    for idx_i in range(n_seqlets):
        ex_i = int(example_idxs[idx_i])
        ss = int(seqlet_starts[idx_i])
        se = int(seqlet_ends[idx_i])
        rc = bool(is_revcomps[idx_i])

        w_dep = att_1d[ex_i]  # shape: (400,)
        w_oh = one_hot_all[ex_i]  # shape: (4, 400)

        if not rc:
            center_pos = ss + core_mid_offset
            for rel_x in range(-half_window, half_window + 1):
                src_pos = center_pos + rel_x
                if 0 <= src_pos < win_len:
                    out_idx = rel_x + half_window
                    dep_mat[idx_i, out_idx] = float(w_dep[src_pos])
                    oh_sum[out_idx, :] += w_oh[:, src_pos]
                    oh_cnt[out_idx] += 1.0
        else:
            center_pos = se - 1 - core_mid_offset
            for rel_x in range(-half_window, half_window + 1):
                src_pos = center_pos - rel_x
                if 0 <= src_pos < win_len:
                    out_idx = rel_x + half_window
                    dep_mat[idx_i, out_idx] = float(w_dep[src_pos])
                    # Reverse complement channels: A(0)<->T(3), C(1)<->G(2) -> [::-1]
                    oh_sum[out_idx, :] += w_oh[::-1, src_pos]
                    oh_cnt[out_idx] += 1.0

    dep_mean = np.nanmean(dep_mat, axis=0)
    valid_cnt = np.maximum(oh_cnt, 1.0)
    ppm_aligned = oh_sum / valid_cnt[:, np.newaxis]  # shape: (span_len, 4)
    ic_aligned, _ = compute_information_content(ppm_aligned)

    # Define Core, Whole Matched Motif, and Distal Flanking masks on [-half_window, +half_window]
    rel_coords = np.arange(-half_window, half_window + 1)
    core_rel_start = core_start - core_mid_offset
    core_rel_end = core_end - core_mid_offset  # exclusive
    motif_rel_start = 0 - core_mid_offset
    motif_rel_end = ppm_50.shape[0] - core_mid_offset  # exclusive

    core_mask = (rel_coords >= core_rel_start) & (rel_coords < core_rel_end)
    motif_mask = (rel_coords >= motif_rel_start) & (rel_coords < motif_rel_end)
    flank_mask = np.abs(rel_coords) >= 75

    core_dep_mean = float(np.mean(dep_mean[core_mask]))
    motif_dep_mean = float(np.mean(dep_mean[motif_mask]))
    flank_dep_mean = float(np.mean(dep_mean[flank_mask]))
    dep_snr = core_dep_mean / max(1e-6, flank_dep_mean)

    core_ic_mean = float(np.mean(ic_aligned[core_mask]))
    motif_ic_mean = float(np.mean(ic_aligned[motif_mask]))
    flank_ic_mean = float(np.mean(ic_aligned[flank_mask]))
    ic_snr = core_ic_mean / max(1e-6, flank_ic_mean)

    # Mann-Whitney U test: core dependency vs distal flanking dependency
    _, mwu_pval = mannwhitneyu(dep_mean[core_mask], dep_mean[flank_mask], alternative="greater")

    # Full-footprint correlation between IC(x) and Dependency(x) across [-half_window, +half_window]
    r_footprint, p_footprint = pearsonr(ic_aligned, dep_mean)
    rho_footprint, rho_p_footprint = spearmanr(ic_aligned, dep_mean)

    # Correlation of per-position attribution magnitude |CWM_i| across the core with empirical IC_i
    cwm_mag_50 = np.sum(np.abs(cwm_50), axis=1)  # shape: (50,)
    cwm_mag_core = cwm_mag_50[core_start:core_end]
    ic_core = ic_50[core_start:core_end]

    if len(cwm_mag_core) >= 3 and float(np.std(cwm_mag_core)) > 1e-9 and float(np.std(ic_core)) > 1e-9:
        r_core_val, p_core_val = pearsonr(cwm_mag_core, ic_core)
        rho_core_val, rho_p_core_val = spearmanr(cwm_mag_core, ic_core)
        r_core, p_core = float(r_core_val), float(p_core_val)
        rho_core, rho_p_core = float(rho_core_val), float(rho_p_core_val)
    else:
        r_core, p_core = 0.0, 1.0
        rho_core, rho_p_core = 0.0, 1.0

    # Correlation across the whole matched motif (50 bp)
    if len(cwm_mag_50) >= 3 and float(np.std(cwm_mag_50)) > 1e-9 and float(np.std(ic_50)) > 1e-9:
        r_motif_val, p_motif_val = pearsonr(cwm_mag_50, ic_50)
        rho_motif_val, rho_p_motif_val = spearmanr(cwm_mag_50, ic_50)
        r_motif, p_motif = float(r_motif_val), float(p_motif_val)
        rho_motif, rho_p_motif = float(rho_motif_val), float(rho_p_motif_val)
    else:
        r_motif, p_motif = 0.0, 1.0
        rho_motif, rho_p_motif = 0.0, 1.0

    # Footprint Full-Width at Half-Maximum (FWHM) above flanking baseline
    peak_dep = float(np.max(dep_mean[core_mask]))
    half_max = flank_dep_mean + 0.5 * max(0.0, peak_dep - flank_dep_mean)
    above_half = np.where(dep_mean >= half_max)[0]
    fwhm_bp = int(len(above_half))

    full_consensus = "".join(ALPHABET[i] for i in np.argmax(ppm_50, axis=1))
    core_consensus = full_consensus[core_start:core_end]

    metrics = {
        "pattern_name": pattern_name,
        "num_seqlets": n_seqlets,
        "core_start": core_start,
        "core_end": core_end,
        "core_length": core_len,
        "core_consensus": core_consensus,
        "core_dep_mean": round(core_dep_mean, 2),
        "motif_dep_mean": round(motif_dep_mean, 2),
        "flank_dep_mean": round(flank_dep_mean, 2),
        "dep_core_flank_snr": round(dep_snr, 3),
        "core_ic_bits": round(core_ic_mean, 3),
        "motif_ic_bits": round(motif_ic_mean, 3),
        "flank_ic_bits": round(flank_ic_mean, 3),
        "ic_core_flank_snr": round(ic_snr, 3),
        "core_cwm_ic_pearson_r": round(r_core, 3),
        "core_cwm_ic_pearson_p": p_core,
        "core_cwm_ic_spearman_rho": round(rho_core, 3),
        "core_cwm_ic_spearman_p": rho_p_core,
        "motif_cwm_ic_pearson_r": round(r_motif, 3),
        "motif_cwm_ic_pearson_p": p_motif,
        "motif_cwm_ic_spearman_rho": round(rho_motif, 3),
        "motif_cwm_ic_spearman_p": rho_p_motif,
        "footprint_ic_dep_pearson_r": round(float(r_footprint), 3),
        "footprint_ic_dep_pearson_p": float(p_footprint),
        "footprint_ic_dep_spearman_rho": round(float(rho_footprint), 3),
        "footprint_ic_dep_spearman_p": float(rho_p_footprint),
        "core_vs_flank_mwu_pval": float(mwu_pval),
        "footprint_fwhm_bp": fwhm_bp,
    }
    return metrics, dep_mean, ic_aligned


def plot_single_pattern_dependency_validation(
    pattern_name: str,
    ppm_50: np.ndarray,
    cwm_50: np.ndarray,
    metrics: Dict[str, Any],
    dep_mean: np.ndarray,
    ic_aligned: np.ndarray,
    output_png: Path,
    half_window: int = 150,
    dpi: int = 300,
) -> None:
    """Renders the Base Conservation vs. Nucleotide Dependency plot and concordance statistics."""
    core_start = int(metrics["core_start"])
    core_end = int(metrics["core_end"])
    core_len = int(metrics["core_length"])
    core_mid_offset = (core_start + core_end) // 2
    n_seqlets = int(metrics["num_seqlets"])
    disp_name = format_pattern_display_name(pattern_name)

    rel_coords = np.arange(-half_window, half_window + 1)
    core_rel_start = core_start - core_mid_offset
    core_rel_end = core_end - core_mid_offset
    motif_rel_start = 0 - core_mid_offset
    motif_rel_end = ppm_50.shape[0] - core_mid_offset

    core_mask = (rel_coords >= core_rel_start) & (rel_coords < core_rel_end)
    motif_mask = (rel_coords >= motif_rel_start) & (rel_coords < motif_rel_end)
    flank_mask = np.abs(rel_coords) >= 75

    fig = plt.figure(figsize=(13.5, 7.2), dpi=dpi, facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.3, 1.25], wspace=0.22)

    ax_scat = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    # 1. Distal Flank points
    ax_scat.scatter(
        ic_aligned[flank_mask],
        dep_mean[flank_mask],
        color="#8c8c8c",
        alpha=0.48,
        s=26,
        label=f"Distal Flank (|d| >= 75 bp, n={int(flank_mask.sum())})",
        zorder=2,
    )
    # 2. Whole Matched Motif points
    ax_scat.scatter(
        ic_aligned[motif_mask],
        dep_mean[motif_mask],
        color="#1f78b4",
        edgecolor="#08519c",
        linewidth=0.6,
        alpha=0.75,
        s=44,
        label=f"Whole Matched Motif ({ppm_50.shape[0]} bp, n={int(motif_mask.sum())})",
        zorder=3,
    )
    # 3. Core Motif points
    ax_scat.scatter(
        ic_aligned[core_mask],
        dep_mean[core_mask],
        color="#e31a1c",
        edgecolor="#111111",
        linewidth=0.8,
        alpha=0.95,
        s=72,
        label=f"Core Motif [{core_start}:{core_end}] ({core_len} bp, n={int(core_mask.sum())})",
        zorder=5,
    )

    ax_scat.set_xlabel("Base Information Content $\\mathrm{IC}(x)$ (bits)", fontsize=11.0, fontweight="bold")
    ax_scat.set_ylabel("Mean Nucleotide Dependency $S(x)$", fontsize=11.0, fontweight="bold")
    ax_scat.set_xlim(-0.05, 2.05)
    max_y = max(float(np.max(dep_mean[motif_mask])) * 1.25, float(np.max(dep_mean)) * 1.15)
    ax_scat.set_ylim(0.0, max_y)
    ax_scat.set_title(
        f"{disp_name} (N={n_seqlets} seqlets)\nBase Conservation vs. Nucleotide Dependency",
        fontsize=12.0,
        fontweight="bold",
        pad=10,
    )
    ax_scat.grid(True, linestyle=":", alpha=0.5)
    ax_scat.legend(loc="upper left", fontsize=9.2, framealpha=0.94)

    # Stats display next to the plot
    ax_stats.axis("off")

    core_r = float(metrics["core_cwm_ic_pearson_r"])
    core_p = float(metrics["core_cwm_ic_pearson_p"])
    core_rho = float(metrics["core_cwm_ic_spearman_rho"])
    core_rho_p = float(metrics["core_cwm_ic_spearman_p"])

    motif_r = float(metrics["motif_cwm_ic_pearson_r"])
    motif_p = float(metrics["motif_cwm_ic_pearson_p"])
    motif_rho = float(metrics["motif_cwm_ic_spearman_rho"])
    motif_rho_p = float(metrics["motif_cwm_ic_spearman_p"])

    sig_core = "***" if core_p < 1e-4 else ("**" if core_p < 1e-2 else ("*" if core_p < 0.05 else " (ns)"))
    sig_motif = "***" if motif_p < 1e-4 else ("**" if motif_p < 1e-2 else ("*" if motif_p < 0.05 else " (ns)"))

    card_text = (
        "=========================================\n"
        " CORE MOTIF CONCORDANCE (|CWM| vs IC)\n"
        "=========================================\n"
        f"Core Range:     [{core_start}:{core_end}] ({core_len} bp)\n"
        f"Core Consensus: {metrics['core_consensus']}\n"
        "\n"
        f"Pearson r:      {core_r:+.3f} {sig_core}\n"
        f"  p-value:      {core_p:.2e}\n"
        f"Spearman rho:   {core_rho:+.3f}\n"
        f"  p-value:      {core_rho_p:.2e}\n"
        "\n"
        "=========================================\n"
        f" WHOLE MATCHED MOTIF ({ppm_50.shape[0]} bp)\n"
        "=========================================\n"
        f"Pearson r:      {motif_r:+.3f} {sig_motif}\n"
        f"  p-value:      {motif_p:.2e}\n"
        f"Spearman rho:   {motif_rho:+.3f}\n"
        f"  p-value:      {motif_rho_p:.2e}\n"
        "\n"
        "=========================================\n"
        " CONTRAST & SIGNAL-TO-NOISE\n"
        "=========================================\n"
        f"Core Dependency Mean:  {metrics['core_dep_mean']:.2f}\n"
        f"Motif Dependency Mean: {metrics['motif_dep_mean']:.2f}\n"
        f"Flank Dependency Mean: {metrics['flank_dep_mean']:.2f}\n"
        f"Core / Flank Dep SNR:  {metrics['dep_core_flank_snr']:.2f}x\n"
        f"Core / Flank IC SNR:   {metrics['ic_core_flank_snr']:.2f}x\n"
        f"Mann-Whitney U Test:   p = {metrics['core_vs_flank_mwu_pval']:.1e}\n"
        "========================================="
    )

    ax_stats.text(
        0.02,
        0.98,
        card_text,
        transform=ax_stats.transAxes,
        fontsize=9.8,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#b0bec5", linewidth=1.2),
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Base-Resolution Conservation vs. Nucleotide Dependency Concordance plots."
    )
    parser.add_argument(
        "--modisco-h5",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/results.h5"),
        help="Path to TF-MoDISco results.h5 (default: outputs/tfmodisco_runs/all_seq_n/results.h5).",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <modisco-h5-parent>/dependency_footprint_validation).",
    )
    parser.add_argument(
        "--half-window",
        type=int,
        default=150,
        help="Half-window width in bp around the motif core center (default: 150 -> [-150, +150] bp).",
    )
    parser.add_argument(
        "--peak-ratio",
        type=float,
        default=0.65,
        help="Attribution peak fraction threshold for core trimming (default: 0.65).",
    )
    parser.add_argument(
        "--min-core-len",
        type=int,
        default=6,
        help="Minimum core motif length in bp (default: 6).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Smoothing window width in bp for core boundary detection (default: 3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for output PNGs (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
        set_global_seed(args.seed)
        _ = get_compute_device()

        h5_path = args.modisco_h5.resolve()
        if not h5_path.is_file():
            raise FileNotFoundError(f"TF-MoDISco HDF5 file not found: {h5_path}")
        if not args.attributions_npz.is_file():
            raise FileNotFoundError(f"Attributions NPZ file not found: {args.attributions_npz}")
        if not args.onehot_npz.is_file():
            raise FileNotFoundError(f"One-hot NPZ file not found: {args.onehot_npz}")

        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else (h5_path.parent / "dependency_footprint_validation")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading attributions from: %s", args.attributions_npz)
        att_npz = np.load(args.attributions_npz)
        att_key = list(att_npz.keys())[0]
        att_arr = att_npz[att_key]  # shape: (N_windows, 4, 400)
        att_1d = np.sum(np.abs(att_arr), axis=1)  # shape: (N_windows, 400)

        logger.info("Loading one-hot sequences from: %s", args.onehot_npz)
        oh_npz = np.load(args.onehot_npz)
        oh_key = list(oh_npz.keys())[0]
        one_hot_all = oh_npz[oh_key]  # shape: (N_windows, 4, 400)

        summary_rows: List[Dict[str, Any]] = []

        with h5py.File(h5_path, "r") as f:
            for group_name in ["pos_patterns", "neg_patterns"]:
                if group_name not in f:
                    continue
                grp = f[group_name]
                sorted_pats = sorted(grp.keys(), key=pattern_sort_key)

                for pat_name in sorted_pats:
                    full_name = f"{group_name}.{pat_name}"
                    p_grp = grp[pat_name]
                    ppm_50 = np.array(p_grp["sequence"][:], dtype=np.float64)
                    cwm_50 = np.array(p_grp["contrib_scores"][:], dtype=np.float64)
                    s_grp = p_grp["seqlets"]
                    ex_idxs = np.array(s_grp["example_idx"][:], dtype=np.int64)
                    s_starts = np.array(s_grp["start"][:], dtype=np.int64)
                    s_ends = np.array(s_grp["end"][:], dtype=np.int64)
                    is_rcs = np.array(s_grp["is_revcomp"][:], dtype=bool)

                    metrics, dep_mean, ic_aligned = (
                        analyze_pattern_footprint_and_concordance(
                            pattern_name=full_name,
                            ppm_50=ppm_50,
                            cwm_50=cwm_50,
                            example_idxs=ex_idxs,
                            seqlet_starts=s_starts,
                            seqlet_ends=s_ends,
                            is_revcomps=is_rcs,
                            att_1d=att_1d,
                            one_hot_all=one_hot_all,
                            half_window=args.half_window,
                            peak_ratio=args.peak_ratio,
                            min_core_len=args.min_core_len,
                            smoothing_window=args.smoothing_window,
                        )
                    )
                    summary_rows.append(metrics)

                    out_png = output_dir / f"{full_name}.png"
                    plot_single_pattern_dependency_validation(
                        pattern_name=full_name,
                        ppm_50=ppm_50,
                        cwm_50=cwm_50,
                        metrics=metrics,
                        dep_mean=dep_mean,
                        ic_aligned=ic_aligned,
                        output_png=out_png,
                        half_window=args.half_window,
                        dpi=args.dpi,
                    )

        summary_df = pd.DataFrame(summary_rows)
        tsv_path = output_dir / "dependency_footprint_summary.tsv"
        summary_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info("Saved summary TSV for %d patterns to: %s", len(summary_df), tsv_path)

        # Remove previous summary PNG if it exists to maintain clean output directory
        old_summary_png = output_dir / "dependency_footprint_summary.png"
        if old_summary_png.exists():
            old_summary_png.unlink()
            logger.info("Removed deprecated summary plot: %s", old_summary_png)

        return 0
    except Exception as exc:
        logger.error("Fatal error in plot_dependency_footprint_and_ic_concordance: %s", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(main())
