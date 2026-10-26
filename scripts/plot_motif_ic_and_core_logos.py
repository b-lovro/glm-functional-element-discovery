#!/usr/bin/env python3
"""Generate Information Content (IC) sequence logos and core-trimmed (60-70% peak attribution) logos.

For every pattern in a TF-MoDISco `results.h5` file (`pos_patterns` and `neg_patterns`),
this script computes:
  1. The Position Probability Matrix (PPM) and per-position Shannon Information Content (IC, 0-2 bits):
       IC_i = 2 + sum_{b in {A,C,G,T}} p_{i,b} * log2(p_{i,b})
       IC_matrix[i, b] = p_{i,b} * IC_i
  2. The dominant per-position attribution magnitude from the CWM (`contrib_scores`):
       a_i = max_{b} |CWM_{i,b}|
  3. The contiguous core subsequence exceeding `--core-threshold` (default: 0.65, i.e., 65% of peak
     attribution a_max, with configurable gap tolerance `--max-gap` and minimum length `--min-core-len`).

Outputs written to `--output-dir`:
  - `<pattern_name>.png`: Full-length (50 bp) Information Content (0-2 bits) logo.
  - `core_trimmed_ic_logos/<pattern_name>.ic_core.fwd.png` (and `.rev.png`): Standalone zoomed core IC logos.
  - `full_ic_logos/<pattern_name>.ic_full.fwd.png` (and `.rev.png`): Standalone full-length IC logos.
  - `core_subsequence_summary.tsv`: Tabular summary of full consensus, core coordinates, core consensus,
    IUPAC string, and IC/attribution metrics for all patterns.

Usage:
  /mnt/Dati/miniconda/envs/group6-rinalmo/bin/python scripts/plot_motif_ic_and_core_logos.py \\
    --results outputs/tfmodisco_runs/all_seq_n/results.h5 \\
    --output-dir outputs/tfmodisco_runs/all_seq_n/ic_and_core_logos
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
import logomaker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


BASES: List[str] = ["A", "C", "G", "T"]

# Classic MEME / WebLogo / TF-MoDISco nucleotide color palette
NUC_COLORS: Dict[str, str] = {
    "A": "#109648",  # Green
    "C": "#255C99",  # Blue
    "G": "#F7B32B",  # Orange / Gold
    "T": "#D62839",  # Red
}

# Two-base IUPAC degeneracy codes for positions where two nucleotides dominate
IUPAC_PAIR_MAP: Dict[Tuple[str, str], str] = {
    ("A", "G"): "R",
    ("C", "T"): "Y",
    ("C", "G"): "S",
    ("A", "T"): "W",
    ("G", "T"): "K",
    ("A", "C"): "M",
}


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
    group_order = 0 if parts[0] == "pos_patterns" else 1
    idx_str = parts[-1].split("_")[-1]
    return (group_order, int(idx_str))


def compute_ic_matrix(ppm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Shannon Information Content (IC) per position and IC-scaled matrix from PPM.

    Args:
        ppm: Position Probability Matrix of shape (L, 4) with rows summing to 1.0.

    Returns:
        (ic_per_pos, ic_matrix):
          - ic_per_pos: shape (L,), values in [0.0, 2.0] bits
          - ic_matrix: shape (L, 4), where ic_matrix[i, b] = ppm[i, b] * ic_per_pos[i]
    """
    row_sums = ppm.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums > 0.0, row_sums, 1.0)
    probs = ppm / row_sums_safe
    probs_clipped = np.clip(probs, 1e-12, 1.0)

    # Shannon entropy H_i = -sum_b p_{i,b} log2(p_{i,b}); IC_i = 2 - H_i
    ic_per_pos = 2.0 + np.sum(probs * np.log2(probs_clipped), axis=1)
    ic_per_pos = np.clip(ic_per_pos, 0.0, 2.0)
    ic_matrix = probs * ic_per_pos[:, None]
    return ic_per_pos, ic_matrix


def revcomp_matrix(mat: np.ndarray) -> np.ndarray:
    """Return the reverse-complement of an (L, 4) matrix ordered [A, C, G, T]."""
    # Reverse rows (positions) and swap columns A<->T (0<->3), C<->G (1<->2) -> [3, 2, 1, 0]
    return mat[::-1, ::-1].copy()


def ppm_to_consensus_and_iupac(ppm: np.ndarray, ic_per_pos: np.ndarray) -> Tuple[str, str]:
    """Compute strict argmax consensus string and IUPAC degeneracy consensus string from PPM.

    IUPAC rules:
      - If IC_i < 0.25 bits: 'n' (low-information position)
      - If top base probability >= 0.70: uppercase single base ('A', 'C', 'G', 'T')
      - If top base probability >= 0.50 and second base < 0.25: lowercase single base ('a', 'c', 'g', 't')
      - If top two bases sum to >= 0.75: IUPAC two-base ambiguity code ('R', 'Y', 'S', 'W', 'K', 'M')
      - Otherwise: uppercase argmax base
    """
    L = ppm.shape[0]
    strict_chars: List[str] = []
    iupac_chars: List[str] = []

    for i in range(L):
        row = ppm[i]
        order = np.argsort(row)[::-1]
        b1_idx, b2_idx = int(order[0]), int(order[1])
        p1, p2 = float(row[b1_idx]), float(row[b2_idx])
        b1, b2 = BASES[b1_idx], BASES[b2_idx]
        strict_chars.append(b1)

        if float(ic_per_pos[i]) < 0.25:
            iupac_chars.append("n")
        elif p1 >= 0.70:
            iupac_chars.append(b1)
        elif (p1 + p2) >= 0.75 and p2 >= 0.25:
            pair_tuple = (min(b1, b2), max(b1, b2))
            iupac_chars.append(IUPAC_PAIR_MAP[pair_tuple])
        else:
            iupac_chars.append(b1.lower())

    return "".join(strict_chars), "".join(iupac_chars)


def extract_core_subsequence_span(
    cwm: np.ndarray,
    core_threshold: float = 0.65,
    max_gap: int = 1,
    min_core_len: int = 8,
) -> Tuple[int, int, np.ndarray, float]:
    """Identify the contiguous core subsequence exceeding `core_threshold * peak_attribution`.

    Args:
        cwm: Contribution Weight Matrix of shape (L, 4).
        core_threshold: Fraction of peak attribution (e.g., 0.65 for 65% of peak).
        max_gap: Maximum number of consecutive sub-threshold bases tolerated inside a contiguous core run.
        min_core_len: Minimum length (in bp) of the extracted core window; if the raw contiguous run
                      is shorter than `min_core_len`, it is symmetrically padded around its peak.

    Returns:
        (core_start, core_end, pos_attr_magnitude, threshold_value)
          where [core_start, core_end) is half-open in [0, L].
    """
    L = cwm.shape[0]
    # Dominant attribution magnitude at each position: a_i = max_b |CWM_{i,b}|
    pos_attr = np.max(np.abs(cwm), axis=1)
    peak_attr = float(np.max(pos_attr))
    thresh_val = core_threshold * peak_attr

    above_idx = np.where(pos_attr >= thresh_val)[0]
    if len(above_idx) == 0:
        peak_pos = int(np.argmax(pos_attr))
        half = min_core_len // 2
        s = max(0, min(L - min_core_len, peak_pos - half))
        e = min(L, s + min_core_len)
        return s, e, pos_attr, thresh_val

    # Group above-threshold indices into contiguous runs (allowing up to `max_gap` internal dips)
    runs: List[Tuple[int, int, float]] = []
    run_start = int(above_idx[0])
    prev = int(above_idx[0])
    for idx_val in above_idx[1:]:
        curr = int(idx_val)
        if curr - prev <= max_gap + 1:
            prev = curr
        else:
            run_end = prev + 1
            run_score = float(np.sum(pos_attr[run_start:run_end] - thresh_val * 0.5))
            runs.append((run_start, run_end, run_score))
            run_start = curr
            prev = curr
    run_end = prev + 1
    run_score = float(np.sum(pos_attr[run_start:run_end] - thresh_val * 0.5))
    runs.append((run_start, run_end, run_score))

    # Select the run with the highest excess attribution mass (ties broken by length)
    runs.sort(key=lambda r: (r[2], r[1] - r[0]), reverse=True)
    best_start, best_end, _ = runs[0]

    # Enforce minimum core window length centered on the best run if needed
    curr_len = best_end - best_start
    if curr_len < min_core_len and L >= min_core_len:
        deficit = min_core_len - curr_len
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        best_start = max(0, best_start - pad_left)
        best_end = min(L, best_end + pad_right)
        if best_end - best_start < min_core_len:
            if best_start == 0:
                best_end = min(L, min_core_len)
            elif best_end == L:
                best_start = max(0, L - min_core_len)

    return int(best_start), int(best_end), pos_attr, thresh_val


def render_logo_on_axis(
    ax: plt.Axes,
    mat: np.ndarray,
    start_pos: int = 0,
    y_label: str = "Information Content (bits)",
    y_lim: Optional[Tuple[float, float]] = None,
    consensus_str: Optional[str] = None,
    iupac_str: Optional[str] = None,
    highlight_span: Optional[Tuple[int, int]] = None,
    threshold_line: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    """Render a sequence logo (IC or CWM) onto a Matplotlib Axes using logomaker."""
    L = mat.shape[0]
    positions = list(range(start_pos, start_pos + L))
    df_logo = pd.DataFrame(mat, index=positions, columns=BASES)

    if highlight_span is not None:
        h_start, h_end = highlight_span
        ax.axvspan(
            h_start - 0.48,
            h_end - 0.52,
            color="#fff3cd",
            alpha=0.65,
            zorder=-10,
        )

    logomaker.Logo(
        df_logo,
        ax=ax,
        color_scheme=NUC_COLORS,
        baseline_width=0.5,
        font_name="DejaVu Sans",
        vpad=0.02,
        width=0.88,
        zorder=2,
    )

    if highlight_span is not None:
        h_start, h_end = highlight_span
        ax.axvline(h_start - 0.48, color="#d95f02", linestyle="--", linewidth=1.5, zorder=4)
        ax.axvline(h_end - 0.52, color="#d95f02", linestyle="--", linewidth=1.5, zorder=4)

    if threshold_line is not None:
        ax.axhline(
            threshold_line,
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label=f"Core cutoff ({threshold_line:.1f})",
            zorder=4,
        )
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    ax.set_xlim(start_pos - 0.6, start_pos + L - 0.4)
    ax.set_ylabel(y_label, fontsize=9.5, fontweight="bold")

    if consensus_str is not None and L <= 30:
        # Show both coordinate index and consensus/IUPAC base on every column tick
        tick_labels: List[str] = []
        for idx_offset in range(L):
            pos_num = start_pos + idx_offset
            c_base = consensus_str[idx_offset]
            if iupac_str is not None and iupac_str[idx_offset] != c_base:
                tick_labels.append(f"{c_base}\n({iupac_str[idx_offset]})\n{pos_num}")
            else:
                tick_labels.append(f"{c_base}\n{pos_num}")
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8.0, family="monospace", fontweight="bold")
    elif consensus_str is not None:
        # For full 50 bp logos, print the consensus base on every position and numbers every 5 bp
        tick_labels = []
        for idx_offset in range(L):
            pos_num = start_pos + idx_offset
            c_base = consensus_str[idx_offset]
            if pos_num % 5 == 0:
                tick_labels.append(f"{c_base}\n{pos_num}")
            else:
                tick_labels.append(f"{c_base}\n")
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7.5, family="monospace")

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    if title is not None:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def save_standalone_logo(
    mat: np.ndarray,
    out_path: Path,
    start_pos: int,
    y_label: str,
    y_lim: Optional[Tuple[float, float]],
    consensus_str: Optional[str],
    iupac_str: Optional[str],
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """Save a clean standalone single-panel logo PNG (suitable for embedding in tables/slides)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    render_logo_on_axis(
        ax=ax,
        mat=mat,
        start_pos=start_pos,
        y_label=y_label,
        y_lim=y_lim,
        consensus_str=consensus_str,
        iupac_str=iupac_str,
        highlight_span=None,
        threshold_line=None,
        title=None,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_all_patterns(
    results_path: Path,
    output_dir: Path,
    core_threshold: float,
    max_gap: int,
    min_core_len: int,
    dpi: int,
    pattern_filter: Optional[str] = None,
    save_standalone: bool = False,
) -> pd.DataFrame:
    """Process all patterns in results.h5 and generate IC + Core Subsequence logo PNGs and summary TSV."""
    if not results_path.is_file():
        raise FileNotFoundError(f"TF-MoDISco HDF5 file not found: {results_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_standalone:
        core_ic_dir = output_dir / "core_trimmed_ic_logos"
        full_ic_dir = output_dir / "full_ic_logos"
        core_ic_dir.mkdir(parents=True, exist_ok=True)
        full_ic_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    with h5py.File(results_path, "r") as h5:
        all_keys: List[Tuple[str, str]] = []
        for grp_name in ["pos_patterns", "neg_patterns"]:
            if grp_name not in h5:
                continue
            for pat_name in h5[grp_name].keys():
                full_name = f"{grp_name}.{pat_name}"
                if pattern_filter is not None and pattern_filter not in (full_name, pat_name):
                    continue
                all_keys.append((grp_name, pat_name))

        all_keys.sort(key=lambda pair: pattern_sort_key(f"{pair[0]}.{pair[1]}"))

        if not all_keys:
            raise ValueError(f"No patterns found in {results_path} matching filter={pattern_filter!r}")

        for grp_name, pat_name in all_keys:
            full_name = f"{grp_name}.{pat_name}"
            pat_grp = h5[grp_name][pat_name]

            for req_ds in ["sequence", "contrib_scores", "seqlets"]:
                if req_ds not in pat_grp:
                    raise KeyError(f"Missing dataset '{req_ds}' in {full_name}")

            ppm_fwd = np.array(pat_grp["sequence"][:], dtype=np.float64)
            cwm_fwd = np.array(pat_grp["contrib_scores"][:], dtype=np.float64)
            num_seqlets = int(pat_grp["seqlets"]["sequence"].shape[0])
            L = ppm_fwd.shape[0]

            # Compute Information Content (0-2 bits)
            ic_pos_fwd, ic_mat_fwd = compute_ic_matrix(ppm_fwd)
            cons_fwd, iupac_fwd = ppm_to_consensus_and_iupac(ppm_fwd, ic_pos_fwd)

            # Extract core trimmed subsequence exceeding core_threshold (e.g. 65% of peak attribution)
            core_s, core_e, pos_attr, thresh_val = extract_core_subsequence_span(
                cwm=cwm_fwd,
                core_threshold=core_threshold,
                max_gap=max_gap,
                min_core_len=min_core_len,
            )
            core_len = core_e - core_s
            core_cons_fwd = cons_fwd[core_s:core_e]
            core_iupac_fwd = iupac_fwd[core_s:core_e]
            core_ic_mat_fwd = ic_mat_fwd[core_s:core_e]
            core_cwm_fwd = cwm_fwd[core_s:core_e]

            if save_standalone:
                # Reverse complement matrices for standalone rev logos
                ppm_rev = revcomp_matrix(ppm_fwd)
                ic_pos_rev, ic_mat_rev = compute_ic_matrix(ppm_rev)
                cons_rev, iupac_rev = ppm_to_consensus_and_iupac(ppm_rev, ic_pos_rev)
                core_ic_mat_rev = revcomp_matrix(core_ic_mat_fwd)
                core_cons_rev = cons_rev[L - core_e : L - core_s]
                core_iupac_rev = iupac_rev[L - core_e : L - core_s]

                # 1. Save standalone full IC logos (fwd & rev)
                save_standalone_logo(
                    mat=ic_mat_fwd,
                    out_path=full_ic_dir / f"{full_name}.ic_full.fwd.png",
                    start_pos=0,
                    y_label="Bits",
                    y_lim=(0.0, 2.05),
                    consensus_str=cons_fwd,
                    iupac_str=None,
                    fig_width=10.5,
                    fig_height=2.2,
                    dpi=dpi,
                )
                save_standalone_logo(
                    mat=ic_mat_rev,
                    out_path=full_ic_dir / f"{full_name}.ic_full.rev.png",
                    start_pos=0,
                    y_label="Bits",
                    y_lim=(0.0, 2.05),
                    consensus_str=cons_rev,
                    iupac_str=None,
                    fig_width=10.5,
                    fig_height=2.2,
                    dpi=dpi,
                )

                # 2. Save standalone core-trimmed IC logos (fwd & rev)
                save_standalone_logo(
                    mat=core_ic_mat_fwd,
                    out_path=core_ic_dir / f"{full_name}.ic_core.fwd.png",
                    start_pos=core_s,
                    y_label="Bits",
                    y_lim=(0.0, 2.05),
                    consensus_str=core_cons_fwd,
                    iupac_str=core_iupac_fwd,
                    fig_width=max(5.5, min(11.0, 0.42 * core_len + 1.5)),
                    fig_height=2.4,
                    dpi=dpi,
                )
                save_standalone_logo(
                    mat=core_ic_mat_rev,
                    out_path=core_ic_dir / f"{full_name}.ic_core.rev.png",
                    start_pos=0,
                    y_label="Bits",
                    y_lim=(0.0, 2.05),
                    consensus_str=core_cons_rev,
                    iupac_str=core_iupac_rev,
                    fig_width=max(5.5, min(11.0, 0.42 * core_len + 1.5)),
                    fig_height=2.4,
                    dpi=dpi,
                )

            # 3. Save clean full-length Information Content sequence logo PNG
            fig, ax = plt.subplots(figsize=(12.5, 2.8), dpi=dpi)
            render_logo_on_axis(
                ax=ax,
                mat=ic_mat_fwd,
                start_pos=0,
                y_label="Information Content (bits)",
                y_lim=(0.0, 2.08),
                consensus_str=cons_fwd,
                iupac_str=None,
                highlight_span=None,
                threshold_line=None,
                title=None,
            )
            plt.tight_layout()

            out_logo = output_dir / f"{full_name}.png"
            fig.savefig(out_logo, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            summary_rows.append(
                {
                    "pattern_name": full_name,
                    "num_seqlets": num_seqlets,
                    "full_length": L,
                    "core_start": core_s,
                    "core_end": core_e,
                    "core_length": core_len,
                    "core_threshold_pct": round(core_threshold * 100.0, 1),
                    "peak_attribution": round(float(np.max(pos_attr)), 3),
                    "core_cutoff_attribution": round(thresh_val, 3),
                    "mean_full_ic_bits": round(float(np.mean(ic_pos_fwd)), 3),
                    "mean_core_ic_bits": round(float(np.mean(ic_pos_fwd[core_s:core_e])), 3),
                    "max_core_ic_bits": round(float(np.max(ic_pos_fwd[core_s:core_e])), 3),
                    "core_consensus": core_cons_fwd,
                    "core_iupac": core_iupac_fwd,
                    "full_consensus": cons_fwd,
                    "full_iupac": iupac_fwd,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = output_dir / "core_subsequence_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    print(f"[info] Wrote core subsequence summary table to: {summary_tsv}")
    return summary_df


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
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs/tfmodisco_runs/all_seq_n/ic_and_core_logos"),
        help="Directory to write IC and core-trimmed logo PNGs (default: outputs/tfmodisco_runs/all_seq_n/ic_and_core_logos)",
    )
    parser.add_argument(
        "--core-threshold",
        type=float,
        default=0.65,
        help="Fraction of peak attribution required for the core subsequence (default: 0.65, i.e., 65%% of peak)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=1,
        help="Maximum internal gap (in bp) below threshold tolerated within a contiguous core subsequence (default: 1)",
    )
    parser.add_argument(
        "--min-core-len",
        type=int,
        default=8,
        help="Minimum window length (in bp) for the extracted core subsequence (default: 8)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional single pattern filter (e.g., 'pos_patterns.pattern_0'). Default: all patterns.",
    )
    parser.add_argument(
        "--save-standalone",
        action="store_true",
        help="Also save standalone single-panel forward/reverse IC logos in subdirectories",
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
        print(f"[info] Initialized plot_motif_ic_and_core_logos on device={device}")
        print(
            f"[info] Extracting Information Content and Core Subsequences "
            f"(>= {args.core_threshold * 100:.0f}% of peak attribution) from: {args.results}"
        )

        summary_df = process_all_patterns(
            results_path=args.results,
            output_dir=args.output_dir,
            core_threshold=args.core_threshold,
            max_gap=args.max_gap,
            min_core_len=args.min_core_len,
            dpi=args.dpi,
            pattern_filter=args.pattern,
            save_standalone=args.save_standalone,
        )

        print(
            f"[info] Successfully generated IC and core-trimmed logos for {len(summary_df)} patterns in: {args.output_dir}"
        )
    except Exception as exc:
        print(f"[error] Failed to generate IC and core logos: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
