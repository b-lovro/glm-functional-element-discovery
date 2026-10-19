#!/usr/bin/env python3
"""Diagnose attribution plateaus and sequence complexity in TF-MoDISco patterns.

This script evaluates all discovered motifs for:
1. CWM attribution plateau span (distinguishing EXTENDED_PLATEAU vs FOCAL_MOTIF).
2. Continuous sequence complexity metrics:
   - Normalized K-mer Shannon Entropy (k=1, k=2) across consensus and constituent seqlets.
   - Calibrated autocorrelation matching (evaluating excess over nucleotide background probability sum(p_b^2)).
3. Classification into:
   - EXTENDED_PLATEAU: Broad, diffuse attribution spanning >= min_contiguous_span or plateau_span_total >= plateau_min_span.
   - FOCAL_MOTIF: Sharp, localized attribution peak characteristic of compact regulatory factor binding.
4. Generates the comprehensive overview grid and summary audit tables.

Author: Antigravity Agent
Target Environment: group6-rinalmo
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Set matplotlib cache directory to writable location before importing pyplot
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUCLEOTIDES = ["A", "C", "G", "T"]
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def compute_normalized_kmer_entropy(seq: str, k: int) -> float:
    """Compute normalized Shannon entropy for k-mers in a nucleotide sequence.
    
    Formula:
        H_k = - sum_w p(w) * log2(p(w))
        Normalized H_k = H_k / (k * 2.0)
    
    Range: [0.0, 1.0], where 1.0 represents uniform k-mer usage.
    """
    seq_len = len(seq)
    num_kmers = seq_len - k + 1
    if num_kmers <= 0:
        raise ValueError(f"Sequence length {seq_len} is shorter than k={k}")

    kmers = [seq[i : i + k] for i in range(num_kmers)]
    counts = Counter(kmers)
    
    entropy = 0.0
    for count in counts.values():
        p = count / float(num_kmers)
        entropy -= p * math.log2(p)
        
    # Maximum theoretical entropy is k * log2(4) = 2.0 * k
    max_theoretical_entropy = 2.0 * float(k)
    normalized_entropy = entropy / max_theoretical_entropy
    return max(0.0, min(1.0, normalized_entropy))


def compute_autocorrelation(seq: str, max_lag: int = 6) -> Tuple[int, float, Dict[int, float]]:
    """Compute base-level autocorrelation across lags [1, max_lag].
    
    Returns:
        (top_lag, top_rate, dict_of_all_lags)
    """
    seq_len = len(seq)
    lag_dict: Dict[int, float] = {}
    for tau in range(1, max_lag + 1):
        if seq_len - tau <= 0:
            lag_dict[tau] = 0.0
            continue
        matches = sum(1 for i in range(seq_len - tau) if seq[i] == seq[i + tau])
        lag_dict[tau] = matches / float(seq_len - tau)
        
    top_lag = max(lag_dict.keys(), key=lambda t: lag_dict[t])
    top_rate = lag_dict[top_lag]
    return top_lag, top_rate, lag_dict


def compute_max_contiguous_span(mask: np.ndarray) -> int:
    """Calculate the maximum length of contiguous True elements in a boolean array."""
    max_len = 0
    curr_len = 0
    for val in mask:
        if val:
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
        else:
            curr_len = 0
    return max_len


def analyze_pattern(
    pat_group: h5py.Group,
    pattern_name: str,
    plateau_ratio: float,
    plateau_min_span: int,
    min_contiguous_span: int,
    min_relaxed_contiguous_span: int,
    entropy_thresh: float,
    raw_autocorr_thresh: float,
    min_autocorr_excess: float,
) -> Dict[str, Any]:
    """Extract sequences, compute entropies, and evaluate CWM plateau characteristics."""
    # 1. Consensus and sequence matrices
    seq_mat = pat_group["sequence"][:]  # shape: (L, 4)
    contrib_mat = pat_group["contrib_scores"][:]  # shape: (L, 4)
    L = seq_mat.shape[0]

    consensus_str = "".join(NUCLEOTIDES[int(np.argmax(seq_mat[i]))] for i in range(L))
    
    # 2. Extract seqlet sequences
    seqlets_group = pat_group["seqlets"]
    raw_seqlet_mats = seqlets_group["sequence"][:]  # shape: (N, L, 4)
    num_seqlets = raw_seqlet_mats.shape[0]

    seqlet_strings: List[str] = []
    seqlet_h1: List[float] = []
    seqlet_h2: List[float] = []

    for idx in range(num_seqlets):
        s_mat = raw_seqlet_mats[idx]
        s_int = np.argmax(s_mat, axis=1)
        s_str = "".join(NUCLEOTIDES[int(b)] for b in s_int)
        seqlet_strings.append(s_str)
        seqlet_h1.append(compute_normalized_kmer_entropy(s_str, k=1))
        seqlet_h2.append(compute_normalized_kmer_entropy(s_str, k=2))

    # 3. CWM profile & Plateau analysis
    pos_attr_magnitude = np.max(np.abs(contrib_mat), axis=1)  # shape: (L,)
    peak_attr = float(np.max(pos_attr_magnitude))
    mean_attr = float(np.mean(pos_attr_magnitude))
    
    threshold_val = plateau_ratio * peak_attr
    plateau_mask = pos_attr_magnitude >= threshold_val
    plateau_span_total = int(np.sum(plateau_mask))
    raw_contiguous_span = compute_max_contiguous_span(plateau_mask)

    # 3-bp smoothing filter to capture continuous plateau while tolerating single-base dips
    smooth_attr = np.convolve(pos_attr_magnitude, np.ones(3) / 3.0, mode="same")
    smooth_mask = smooth_attr >= threshold_val
    smooth_contiguous_span = compute_max_contiguous_span(smooth_mask)
    max_contiguous_span = max(raw_contiguous_span, smooth_contiguous_span)

    # 4. Periodicity & Calibrated Autocorrelation on consensus
    top_lag, top_rate, lag_dict = compute_autocorrelation(consensus_str, max_lag=6)
    cons_h1 = compute_normalized_kmer_entropy(consensus_str, k=1)
    cons_h2 = compute_normalized_kmer_entropy(consensus_str, k=2)

    # Calculate composition null baseline probability: sum(p_b^2)
    counts = Counter(consensus_str)
    base_probs = [counts[b] / float(L) for b in NUCLEOTIDES]
    null_match_prob = sum(p**2 for p in base_probs)
    
    # Check short lags (1, 2, 3) for tandem repeats / homopolymers
    short_lags = {tau: lag_dict[tau] for tau in [1, 2, 3] if tau in lag_dict}
    best_short_lag = max(short_lags.keys(), key=lambda t: short_lags[t]) if short_lags else 1
    best_short_rate = short_lags[best_short_lag] if short_lags else 0.0
    autocorr_excess = best_short_rate - null_match_prob

    # 5. Plateau Geometry Classification:
    # A pattern qualifies as EXTENDED_PLATEAU if:
    # 1) It has a contiguous smoothed span >= min_contiguous_span (default: 20 bp), OR
    # 2) It has a large total plateau span >= plateau_min_span (default: 30 bp) with at least min_relaxed_contiguous_span (default: 15 bp).
    flag_plateau = (smooth_contiguous_span >= min_contiguous_span) or (
        plateau_span_total >= plateau_min_span and smooth_contiguous_span >= min_relaxed_contiguous_span
    )
    verdict = "EXTENDED_PLATEAU" if flag_plateau else "FOCAL_MOTIF"

    # Descriptive sequence complexity flags
    flag_low_complexity = (cons_h2 <= entropy_thresh) or (float(np.mean(seqlet_h2)) <= entropy_thresh)
    flag_periodic = (best_short_rate >= raw_autocorr_thresh) and (autocorr_excess >= min_autocorr_excess)

    return {
        "pattern_name": pattern_name,
        "num_seqlets": num_seqlets,
        "verdict": verdict,
        "flag_plateau": flag_plateau,
        "consensus_str": consensus_str,
        "peak_attr": peak_attr,
        "mean_attr": mean_attr,
        "threshold_val": threshold_val,
        "plateau_span_total": plateau_span_total,
        "raw_contiguous_span": raw_contiguous_span,
        "smooth_contiguous_span": smooth_contiguous_span,
        "max_contiguous_span": max_contiguous_span,
        "cons_h1": cons_h1,
        "cons_h2": cons_h2,
        "mean_h1": float(np.mean(seqlet_h1)),
        "median_h1": float(np.median(seqlet_h1)),
        "std_h1": float(np.std(seqlet_h1)),
        "mean_h2": float(np.mean(seqlet_h2)),
        "median_h2": float(np.median(seqlet_h2)),
        "std_h2": float(np.std(seqlet_h2)),
        "null_match_prob": float(null_match_prob),
        "top_lag": top_lag,
        "top_rate": top_rate,
        "best_short_lag": best_short_lag,
        "best_short_rate": best_short_rate,
        "autocorr_excess": float(autocorr_excess),
        "flag_low_complexity": flag_low_complexity,
        "flag_periodic": flag_periodic,
        "lag_dict": lag_dict,
        "pos_attr_magnitude": pos_attr_magnitude,
    }


def plot_summary_overview_grid(
    records: List[Dict[str, Any]],
    output_png: Path,
    dpi: int = 150,
) -> None:
    """Generate a combined overview panel showing all patterns' CWM profiles and plateau verdicts."""
    num_patterns = len(records)
    cols = 3
    rows = math.ceil(num_patterns / float(cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), dpi=dpi)
    axes_flat = axes.flatten()

    for idx, data in enumerate(records):
        ax = axes_flat[idx]
        pos_attr = data["pos_attr_magnitude"]
        thresh = data["threshold_val"]
        positions = np.arange(len(pos_attr))

        bars = ax.bar(positions, pos_attr, color="#4a7ebb", width=0.85)
        for i, b in enumerate(bars):
            if pos_attr[i] >= thresh:
                b.set_color("#d95f02")

        ax.axhline(thresh, color="red", linestyle="--", linewidth=1.0)
        
        verdict = data["verdict"]
        if verdict == "EXTENDED_PLATEAU":
            badge_color = "#7570b3"  # Deep Purple / Slate
        else:  # FOCAL_MOTIF
            badge_color = "#1b9e77"  # Deep Teal / Green

        ax.set_title(
            f"{data['pattern_name']} (N={data['num_seqlets']}) - {verdict}",
            fontsize=10,
            fontweight="bold",
            color=badge_color,
        )
        ax.set_ylim(0, max(pos_attr) * 1.15)
        ax.set_xticks(range(0, len(pos_attr), 10))
        ax.tick_params(labelsize=8)
        
        summary_txt = f"Span={data['plateau_span_total']}bp (contig={data['max_contiguous_span']}bp) | consH2={data['cons_h2']:.2f}"
        ax.text(0.04, 0.90, summary_txt, transform=ax.transAxes, fontsize=8, family="monospace")

    # Hide unused subplots
    for idx in range(num_patterns, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "TF-MoDISco Patterns: CWM Attribution Profiles & Plateau Diagnostics",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose attribution plateaus and sequence complexity in TF-MoDISco patterns."
    )
    parser.add_argument(
        "--modisco-h5",
        type=str,
        default="outputs/tfmodisco_runs/full_tuned/motif_only_results.h5",
        help="Path to TF-MoDISco results HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tfmodisco_runs/full_tuned/plateau_diagnostics",
        help="Directory to save audit tables and overview grid.",
    )
    parser.add_argument(
        "--plateau-ratio",
        type=float,
        default=0.50,
        help="Fraction of peak attribution required to qualify as plateau position (default: 0.50).",
    )
    parser.add_argument(
        "--plateau-min-span",
        type=int,
        default=30,
        help="Minimum number of total positions >= plateau_ratio to qualify as extended plateau with relaxed continuity (default: 30 bp).",
    )
    parser.add_argument(
        "--min-contiguous-span",
        type=int,
        default=20,
        help="Minimum continuous smoothed span in bp to qualify as an extended plateau unconditionally (default: 20 bp).",
    )
    parser.add_argument(
        "--min-relaxed-contiguous-span",
        type=int,
        default=15,
        help="Minimum continuous smoothed span in bp required when plateau_min_span is satisfied (default: 15 bp).",
    )
    parser.add_argument(
        "--kmer-entropy-thresh",
        type=float,
        default=0.60,
        help="Threshold for normalized 2-mer entropy below which seqlets/consensus are considered low complexity (default: 0.60).",
    )
    parser.add_argument(
        "--raw-autocorr-thresh",
        type=float,
        default=0.50,
        help="Minimum raw match rate at short lag (1-3) to consider periodic (default: 0.50).",
    )
    parser.add_argument(
        "--min-autocorr-excess",
        type=float,
        default=0.10,
        help="Minimum excess autocorrelation above null composition sum(p_b^2) (default: 0.10).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution for saved figures (default: 150).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    h5_path = Path(args.modisco_h5)
    if not h5_path.is_file():
        raise FileNotFoundError(f"TF-MoDISco HDF5 file not found at: {h5_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old plots directory if present
    plots_dir = out_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)

    print(f"Loading TF-MoDISco results from: {h5_path}")
    records: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as f:
        if "pos_patterns" not in f:
            raise KeyError("Expected 'pos_patterns' group in HDF5 file.")
        
        pos_pats = f["pos_patterns"]
        pat_keys = sorted(pos_pats.keys(), key=lambda k: int(k.split("_")[1]) if "_" in k else 0)

        for pat_name in pat_keys:
            print(f"Analyzing {pat_name}...")
            pat_grp = pos_pats[pat_name]
            data = analyze_pattern(
                pat_grp,
                pat_name,
                plateau_ratio=args.plateau_ratio,
                plateau_min_span=args.plateau_min_span,
                min_contiguous_span=args.min_contiguous_span,
                min_relaxed_contiguous_span=args.min_relaxed_contiguous_span,
                entropy_thresh=args.kmer_entropy_thresh,
                raw_autocorr_thresh=args.raw_autocorr_thresh,
                min_autocorr_excess=args.min_autocorr_excess,
            )
            records.append(data)

    # Generate combined overview figure
    overview_png = out_dir / "overview_plateau_grid.png"
    print(f"Generating overview grid at: {overview_png}")
    plot_summary_overview_grid(records, output_png=overview_png, dpi=args.dpi)

    # Save summary TSV
    tsv_path = out_dir / "metrics_summary.tsv"
    tsv_records = [
        {
            "pattern_name": r["pattern_name"],
            "num_seqlets": r["num_seqlets"],
            "verdict": r["verdict"],
            "flag_plateau": r["flag_plateau"],
            "consensus": r["consensus_str"],
            "peak_attr": round(r["peak_attr"], 2),
            "mean_attr": round(r["mean_attr"], 2),
            "threshold_val": round(r["threshold_val"], 2),
            "plateau_span_total": r["plateau_span_total"],
            "raw_contiguous_span": r["raw_contiguous_span"],
            "smooth_contiguous_span": r["smooth_contiguous_span"],
            "max_contiguous_span": r["max_contiguous_span"],
            "cons_h1": round(r["cons_h1"], 4),
            "cons_h2": round(r["cons_h2"], 4),
            "mean_h1": round(r["mean_h1"], 4),
            "median_h1": round(r["median_h1"], 4),
            "std_h1": round(r["std_h1"], 4),
            "mean_h2": round(r["mean_h2"], 4),
            "median_h2": round(r["median_h2"], 4),
            "std_h2": round(r["std_h2"], 4),
            "null_match_prob": round(r["null_match_prob"], 4),
            "top_lag": r["top_lag"],
            "top_rate": round(r["top_rate"], 4),
            "best_short_lag": r["best_short_lag"],
            "best_short_rate": round(r["best_short_rate"], 4),
            "autocorr_excess": round(r["autocorr_excess"], 4),
            "flag_low_complexity": r["flag_low_complexity"],
            "flag_periodic": r["flag_periodic"],
        }
        for r in records
    ]
    pd.DataFrame(tsv_records).to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved metrics summary to: {tsv_path}")

    print("Done!")


if __name__ == "__main__":
    main()

