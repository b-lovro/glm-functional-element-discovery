#!/usr/bin/env python3
"""
Full-Dataset Robust FIMO Scanning & Evolutionary Conservation Pipeline
======================================================================
Applies local sliding-window GC-calibrated Position Weight Matrix (PWM) scanning
across all 156 mammalian rDNA genomes for de novo TF-MoDISco patterns, featuring:

1. Core Motif Trimming:
   Trims each 50 bp TF-MoDISco pattern to its core attribution peak using the smoothed
   CWM attribution thresholding logic (smoothed attribution >= 0.65 * peak) from
   diagnose_motif_plateaus.py.
2. Local Sliding-Window Background Calibration:
   Calculates position-specific nucleotide frequencies across each genome via a sliding
   window (default: 1,000 bp) to eliminate regional composition bias across compartments.
3. Exact Dinucleotide Shuffling:
   Implements the Altschul-Erickson Eulerian path algorithm to produce a matched null
   genomic sequence dataset preserving exact 1-mer and 2-mer compositions.
4. Empirical False Discovery Rate (FDR / q-value) Calibration:
   Calculates empirical FDR curves by comparing real vs null hit score distributions,
   deriving motif-specific score thresholds for a target FDR (default: 0.05).
5. Multi-Site Biological Annotation:
   Evaluates all statistically significant hits without naive global Top-1 filtering,
   capturing true biological multi-site clusters and spatial conservation.

Outputs saved to <modisco_parent>/fimo_diagnostics/:
- motif_trimming_summary.tsv
- fimo_fdr_summary.tsv
- fimo_all_hits_annotated.parquet
- fimo_conservation_summary.tsv
- fimo_vs_modisco_comparison.tsv
- fimo_vs_modisco_attribution_comparison.png
- fimo_active_vs_silent_attribution.tsv
- fimo_active_vs_silent_attribution_distribution.png
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
import math
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pyfaidx
from scipy.stats import mannwhitneyu
import seaborn as sns


def set_global_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pattern_sort_key(name: str) -> Tuple[int, int]:
    """Extracts (is_neg, pattern_index) for numerical sorting (pos_patterns first, then neg_patterns)."""
    is_neg = 1 if name.startswith("neg_") else 0
    parts = name.split(".")[-1].split("_")
    return (is_neg, int(parts[-1]))


def format_pattern_display_name(name: str) -> str:
    """Format pattern identifier cleanly, preserving neg_ prefix if from neg_patterns."""
    if name.startswith("neg_patterns."):
        return f"neg_{name.split('.')[-1]}"
    return name.split(".")[-1]


def dinucleotide_shuffle_exact(seq: str, rng: random.Random) -> str:
    """Exact Altschul-Erickson dinucleotide shuffle algorithm.

    Preserves 100% exact counts of all 16 dinucleotides, single nucleotides,
    and sequence endpoints without changing length or composition.
    """
    if len(seq) <= 2:
        return seq

    chars = ["A", "C", "G", "T"]
    last_char = seq[-1]

    # Count edges between every pair of characters
    edge_counts: Dict[str, Dict[str, int]] = {u: {v: 0 for v in chars} for u in chars}
    for i in range(len(seq) - 1):
        u_char = seq[i]
        v_char = seq[i + 1]
        if u_char in edge_counts and v_char in edge_counts[u_char]:
            edge_counts[u_char][v_char] += 1

    active_verts = [u for u in chars if sum(edge_counts[u].values()) > 0]
    tree_verts = [u for u in active_verts if u != last_char]

    # Pick last-exit edges that form an in-tree directed to last_char
    max_attempts = 1000
    attempt = 0
    while True:
        attempt += 1
        last_edge: Dict[str, str] = {}
        for u in tree_verts:
            cand_v = [v for v in chars if edge_counts[u][v] > 0]
            if not cand_v:
                continue
            last_edge[u] = rng.choice(cand_v)

        # Verify in-tree: from every u in tree_verts, path reaches last_char without cycles
        is_tree = True
        for u in tree_verts:
            curr = u
            visited = set()
            while curr != last_char and curr in last_edge:
                if curr in visited:
                    is_tree = False
                    break
                visited.add(curr)
                curr = last_edge[curr]
            if curr != last_char:
                is_tree = False
                break

        if is_tree:
            break
        if attempt >= max_attempts:
            # Fallback to random permutation in extremely rare disconnected edge cases
            seq_list = list(seq)
            rng.shuffle(seq_list)
            return "".join(seq_list)

    # Multigraph of remaining edges
    remaining_edges: Dict[str, List[str]] = {u: [] for u in chars}
    for u in chars:
        for v in chars:
            cnt = edge_counts[u][v]
            if u in last_edge and last_edge[u] == v:
                cnt -= 1
            remaining_edges[u].extend([v] * cnt)
        rng.shuffle(remaining_edges[u])
        if u in last_edge:
            remaining_edges[u].append(last_edge[u])

    # Eulerian walk
    curr = seq[0]
    result = [curr]
    for _ in range(len(seq) - 1):
        if not remaining_edges[curr]:
            # Guard against edge exhaustion
            break
        nxt = remaining_edges[curr].pop(0)
        result.append(nxt)
        curr = nxt

    if len(result) != len(seq):
        seq_list = list(seq)
        rng.shuffle(seq_list)
        return "".join(seq_list)

    return "".join(result)


def load_and_trim_modisco_pwms(
    h5_path: Path,
    plateau_ratio: float = 0.65,
    min_core_len: int = 6,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], pd.DataFrame]:
    """Loads all discovered PWMs from HDF5 pos_patterns and neg_patterns groups and trims each to its CWM core.

    Trimming logic matches scripts/diagnose_motif_plateaus.py:
    1. Computes per-position attribution magnitude max_b(|contrib_scores[i, b]|).
    2. Applies a 3-bp smoothing filter to tolerate single-base dips.
    3. Finds contiguous positions where smooth_attr >= plateau_ratio * peak_attr.
    4. Slices PWM to the maximal contiguous core span (enforcing >= min_core_len).

    Args:
        h5_path: Path to TF-MoDISco HDF5 file.
        plateau_ratio: Fraction of peak attribution to define the core span.
        min_core_len: Minimum allowed core length in bp.

    Returns:
        motifs: Mapping from pattern name to trimmed PWM with shape (4, core_len).
        consensus_dict: Mapping from pattern name to trimmed consensus sequence.
        trimming_df: Audit dataframe recording trimming bounds and lengths.
    """
    if not h5_path.is_file():
        raise FileNotFoundError(f"MoDISco HDF5 file not found at: {h5_path}")

    motifs: Dict[str, np.ndarray] = {}
    consensus_dict: Dict[str, str] = {}
    trim_records: List[Dict[str, Any]] = []
    nucs: List[str] = ["A", "C", "G", "T"]

    with h5py.File(h5_path, "r") as f:
        if "pos_patterns" not in f and "neg_patterns" not in f:
            raise KeyError("Neither pos_patterns nor neg_patterns group found in HDF5 file.")

        for group_name in ["pos_patterns", "neg_patterns"]:
            if group_name not in f:
                continue
            grp = f[group_name]
            sorted_keys = sorted(grp.keys(), key=pattern_sort_key)

            for pat_name in sorted_keys:
                full_name = f"{group_name}.{pat_name}"
                seq_matrix = grp[pat_name]["sequence"][:]  # (L, 4)
                contrib_matrix = grp[pat_name]["contrib_scores"][:]  # (L, 4)
                L = seq_matrix.shape[0]

                pos_attr_magnitude = np.max(np.abs(contrib_matrix), axis=1)  # (L,)
                peak_attr = float(np.max(pos_attr_magnitude))
                threshold_val = plateau_ratio * peak_attr

                # 3-bp smoothing filter
                smooth_attr = np.convolve(pos_attr_magnitude, np.ones(3) / 3.0, mode="same")
                smooth_mask = smooth_attr >= threshold_val

                # Find all contiguous runs meeting the threshold
                runs: List[Tuple[int, int]] = []
                in_run = False
                start_idx = 0
                for i, val in enumerate(smooth_mask):
                    if val and not in_run:
                        in_run = True
                        start_idx = i
                    elif not val and in_run:
                        in_run = False
                        runs.append((start_idx, i))
                if in_run:
                    runs.append((start_idx, len(smooth_mask)))

                if runs:
                    longest_run = max(runs, key=lambda r: r[1] - r[0])
                else:
                    peak_pos = int(np.argmax(pos_attr_magnitude))
                    longest_run = (max(0, peak_pos - 3), min(L, peak_pos + 4))

                core_start, core_end = longest_run
                # Ensure minimum core length
                if (core_end - core_start) < min_core_len:
                    center = (core_start + core_end) // 2
                    half = min_core_len // 2
                    core_start = max(0, center - half)
                    core_end = min(L, core_start + min_core_len)
                    if (core_end - core_start) < min_core_len and core_start > 0:
                        core_start = max(0, core_end - min_core_len)

                trimmed_seq = seq_matrix[core_start:core_end]  # (core_len, 4)
                trimmed_pwm = trimmed_seq.T.astype(np.float32)  # (4, core_len)
                motifs[full_name] = trimmed_pwm

                # Consensus string
                orig_consensus = "".join(nucs[i] for i in np.argmax(seq_matrix, axis=1))
                trimmed_consensus = "".join(nucs[i] for i in np.argmax(trimmed_seq, axis=1))
                consensus_dict[full_name] = trimmed_consensus

                trim_records.append({
                    "pattern_name": full_name,
                    "orig_length": L,
                    "trimmed_start": core_start,
                    "trimmed_end": core_end,
                    "trimmed_length": core_end - core_start,
                    "peak_pos": int(np.argmax(pos_attr_magnitude)),
                    "peak_attr": round(peak_attr, 2),
                    "threshold_val": round(threshold_val, 2),
                    "orig_consensus": orig_consensus,
                    "trimmed_consensus": trimmed_consensus,
                })

    trimming_df = pd.DataFrame(trim_records)
    return motifs, consensus_dict, trimming_df


def export_clean_fasta(
    records_path: Path,
    temp_fasta_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Exports all records to a sanitized FASTA without whitespace in headers."""
    if not records_path.is_file():
        raise FileNotFoundError(f"Records parquet not found at: {records_path}")

    records_df = pd.read_parquet(records_path)
    required_cols = ["record_id", "species", "sequence", "sequence_length", "a_count", "c_count", "g_count", "t_count"]
    for col in required_cols:
        if col not in records_df.columns:
            raise KeyError(f"Required column '{col}' missing from records parquet.")

    sanitized_to_orig: Dict[str, str] = {}
    with open(temp_fasta_path, "w") as f:
        for _, row in records_df.iterrows():
            orig_id = str(row["record_id"])
            sanitized_id = orig_id.replace(" ", "_")
            if sanitized_id in sanitized_to_orig:
                raise ValueError(f"Duplicate sanitized ID encountered: {sanitized_id}")
            sanitized_to_orig[sanitized_id] = orig_id
            seq = str(row["sequence"])
            f.write(f">{sanitized_id}\n{seq}\n")

    return records_df, sanitized_to_orig


def export_shuffled_fasta(
    records_df: pd.DataFrame,
    temp_fasta_path: Path,
    seed: int,
) -> None:
    """Exports matched dinucleotide-shuffled null sequences for all records."""
    rng = random.Random(seed)
    with open(temp_fasta_path, "w") as f:
        for _, row in records_df.iterrows():
            sanitized_id = str(row["record_id"]).replace(" ", "_")
            seq = str(row["sequence"]).upper()
            shuf_seq = dinucleotide_shuffle_exact(seq, rng)
            f.write(f">{sanitized_id}\n{shuf_seq}\n")


def extract_record_landmarks(
    records_df: pd.DataFrame,
    regions_path: Path,
) -> pd.DataFrame:
    """Extracts TSS, 28S termination end, and ITS coordinates for each record."""
    if not regions_path.is_file():
        raise FileNotFoundError(f"Regions parquet not found at: {regions_path}")

    regions_df = pd.read_parquet(regions_path)
    req_cols = ["record_id", "feature_type", "start", "end"]
    for col in req_cols:
        if col not in regions_df.columns:
            raise KeyError(f"Required column '{col}' missing from regions parquet.")

    landmarks_rows: List[Dict[str, Any]] = []

    for _, row in records_df.iterrows():
        rec_id = str(row["record_id"])
        sanitized_name = rec_id.replace(" ", "_")
        seq_len = int(row["sequence_length"])
        parts = rec_id.split("-")
        tax_class = parts[0]
        tax_order = parts[1]
        species = parts[2]

        sub = regions_df[regions_df["record_id"] == rec_id]

        # 28S rRNA (100% presence)
        s28 = sub[sub["feature_type"] == "28S_rRNA"]
        if len(s28) == 0:
            raise ValueError(f"Record {rec_id} lacks a 28S_rRNA annotation.")
        s28_start = float(s28["start"].iloc[0])
        s28_end = float(s28["end"].iloc[0])

        # 5.8S rRNA immediately upstream of 28S
        s58 = sub[sub["feature_type"] == "5_8S_rRNA"]
        if len(s58) == 0:
            raise ValueError(f"Record {rec_id} lacks a 5_8S_rRNA annotation.")
        s58_upstream = s58[s58["end"] <= s28_start]
        if len(s58_upstream) == 0:
            raise ValueError(f"Record {rec_id} lacks an upstream 5_8S_rRNA before 28S.")
        s58_start = float(s58_upstream["start"].iloc[0])
        s58_end = float(s58_upstream["end"].max())

        its2_start = s58_end
        its2_end = s28_start
        its2_len = its2_end - its2_start
        if its2_len <= 0:
            raise ValueError(f"Invalid ITS2 length ({its2_len}) in record {rec_id}.")

        # 18S rRNA immediately upstream of 5.8S
        s18 = sub[sub["feature_type"] == "18S_rRNA"]
        if len(s18) == 0:
            raise ValueError(f"Record {rec_id} lacks an 18S_rRNA annotation.")
        s18_upstream = s18[s18["end"] <= s58_start]
        if len(s18_upstream) == 0:
            raise ValueError(f"Record {rec_id} lacks an upstream 18S_rRNA before 5.8S.")
        s18_end = float(s18_upstream["end"].max())

        its1_start = s18_end
        its1_end = s58_start
        its1_len = its1_end - its1_start
        if its1_len <= 0:
            raise ValueError(f"Invalid ITS1 length ({its1_len}) in record {rec_id}.")

        # Core Promoter / TSS
        cores = sub[sub["feature_type"] == "Core"]
        if len(cores) > 0:
            tss_pos = float(cores["end"].max())
        else:
            comps = sub[sub["feature_type"] == "CompositePromoter"]
            if len(comps) > 0:
                tss_pos = float(comps["end"].max())
            else:
                tss_pos = np.nan

        landmarks_rows.append({
            "record_id": rec_id,
            "sequence_name": sanitized_name,
            "tax_class": tax_class,
            "tax_order": tax_order,
            "species": species,
            "seq_len": seq_len,
            "tss_pos": tss_pos,
            "term_pos": s28_end,
            "its1_start": its1_start,
            "its1_end": its1_end,
            "its1_len": its1_len,
            "its2_start": its2_start,
            "its2_end": its2_end,
            "its2_len": its2_len,
        })

    return pd.DataFrame(landmarks_rows)


def compute_chromosome_local_bg(
    seq_bytes: np.ndarray,
    window_size: int = 1000,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Calculates position-specific nucleotide background probabilities in a sliding window.

    Args:
        seq_bytes: 1D array of int8 base indices (0=A, 1=C, 2=G, 3=T, -1=N).
        window_size: Width of local sliding window in bp.
        pseudocount: Dirichlet pseudocount for regularising low counts.

    Returns:
        log_bg_seq: 1D array of shape (seq_len,) containing the log2 background
                    probability of the observed base at each sequence position.
    """
    seq_len = len(seq_bytes)
    half_w = window_size // 2

    # One-hot counts
    one_hot = np.zeros((4, seq_len), dtype=np.float32)
    for b in range(4):
        one_hot[b, seq_bytes == b] = 1.0

    pad_cumsum = np.pad(np.cumsum(one_hot, axis=1), ((0, 0), (1, 0)), mode="constant")
    starts = np.clip(np.arange(seq_len) - half_w, 0, seq_len)
    ends = np.clip(np.arange(seq_len) + half_w, 0, seq_len)

    local_counts = pad_cumsum[:, ends] - pad_cumsum[:, starts]
    local_lens = ends - starts
    local_freqs = (local_counts + (pseudocount / 4.0)) / (local_lens + pseudocount)
    log_local_bg = np.log2(local_freqs)  # shape: (4, seq_len)

    # For each position, pick the background probability corresponding to the observed base
    valid_mask = (seq_bytes >= 0) & (seq_bytes < 4)
    log_bg_seq = np.full(seq_len, -2.0, dtype=np.float32)  # default log2(0.25) for Ns
    valid_indices = np.where(valid_mask)[0]
    log_bg_seq[valid_indices] = log_local_bg[seq_bytes[valid_indices], valid_indices]

    return log_bg_seq


@numba.njit(parallel=True, fastmath=True)
def fast_hits_local_bg(
    X_arr: np.ndarray,
    X_lengths: np.ndarray,
    bg_cumsums: np.ndarray,
    bg_offsets: np.ndarray,
    log_pwms: np.ndarray,
    pwm_offsets: np.ndarray,
    score_floor: float,
) -> List[List[Tuple[int, int, int, float]]]:
    """Scans all chromosomes across all motifs using local sliding-window background subtraction.

    Args:
        X_arr: Concatenated int8 sequence base indices (0=A, 1=C, 2=G, 3=T, -1=N).
        X_lengths: Cumulative sequence boundaries with shape (n_chroms + 1,).
        bg_cumsums: Concatenated cumulative sums of log_bg_seq per chromosome.
        bg_offsets: Cumulative offsets into bg_cumsums with shape (n_chroms + 1,).
        log_pwms: Concatenated log2 PWM matrices with shape (4, total_motif_len).
        pwm_offsets: Motif boundaries in log_pwms with shape (n_motifs + 1,).
        score_floor: Minimum score to record candidate hit.

    Returns:
        hits: Nested list of hits per motif: [(chrom_idx, start, end, score), ...]
    """
    n_motifs = len(pwm_offsets) - 1
    n_chroms = len(X_lengths) - 1

    hits = []
    for _ in range(n_motifs):
        hits.append([(np.int64(0), np.int64(0), np.int64(0), np.float32(0.0)) for _ in range(0)])

    for k in numba.prange(n_motifs):
        n = pwm_offsets[k + 1] - pwm_offsets[k]
        m_offset = pwm_offsets[k]

        for l in range(n_chroms):
            start = X_lengths[l]
            end = X_lengths[l + 1]
            chrom_len = end - start
            if chrom_len < n:
                continue

            bg_start = bg_offsets[l]

            for i in range(chrom_len - n + 1):
                raw_score = np.float32(0.0)
                valid = True
                for j in range(n):
                    base = X_arr[start + i + j]
                    if base < 0 or base > 3:
                        valid = False
                        break
                    raw_score += log_pwms[base, m_offset + j]

                if valid:
                    # O(1) sum of log background probabilities across window [i, i+n)
                    bg_sum = bg_cumsums[bg_start + i + n] - bg_cumsums[bg_start + i]
                    score = raw_score - bg_sum
                    if score >= score_floor:
                        hits[k].append((np.int64(l), np.int64(i), np.int64(i + n), score))

    return hits


def encode_fasta_and_bg(
    fasta_path: Path,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encodes FASTA sequences and precomputes cumulative local background profiles."""
    fasta = pyfaidx.Fasta(str(fasta_path))
    sequence_names = np.array(list(fasta.keys()))

    X_list: List[np.ndarray] = []
    lengths = [0]
    bg_cumsums_list: List[np.ndarray] = []
    bg_offsets = [0]

    alphabet_map = np.full(256, -1, dtype=np.int8)
    alphabet_map[ord("A")] = 0
    alphabet_map[ord("a")] = 0
    alphabet_map[ord("C")] = 1
    alphabet_map[ord("c")] = 1
    alphabet_map[ord("G")] = 2
    alphabet_map[ord("g")] = 2
    alphabet_map[ord("T")] = 3
    alphabet_map[ord("t")] = 3

    for name in sequence_names:
        seq_str = fasta[name][:].seq
        raw_bytes = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        encoded_seq = alphabet_map[raw_bytes]
        X_list.append(encoded_seq)
        lengths.append(lengths[-1] + len(encoded_seq))

        # Local background computation
        log_bg_seq = compute_chromosome_local_bg(encoded_seq, window_size=window_size)
        cumsum_bg = np.pad(np.cumsum(log_bg_seq), (1, 0), mode="constant").astype(np.float32)
        bg_cumsums_list.append(cumsum_bg)
        bg_offsets.append(bg_offsets[-1] + len(cumsum_bg))

    X_arr = np.concatenate(X_list)
    X_lengths = np.array(lengths, dtype=np.int64)
    bg_cumsums = np.concatenate(bg_cumsums_list)
    bg_offsets_arr = np.array(bg_offsets, dtype=np.int64)

    return X_arr, X_lengths, bg_cumsums, bg_offsets_arr, sequence_names


def run_local_bg_scan(
    fasta_path: Path,
    motifs: Dict[str, np.ndarray],
    window_size: int = 1000,
    score_floor: float = 3.0,
    eps: float = 0.0001,
) -> Tuple[Dict[str, pd.DataFrame], np.ndarray]:
    """Runs sliding-window local background calibrated scanning across all motifs (both strands)."""
    motifs_fwd = [(name, motifs[name]) for name in motifs]
    motifs_rev = [(name + "-rc", motifs[name][::-1, ::-1]) for name in motifs]
    all_motifs = [*motifs_fwd, *motifs_rev]
    n_motifs = len(all_motifs)

    motif_names = np.array([name for name, _ in all_motifs])
    motif_lens = np.array([pwm.shape[-1] for _, pwm in all_motifs], dtype=np.int64)
    pwm_offsets = np.concatenate([np.array([0]), np.cumsum(motif_lens)]).astype(np.int64)

    # Log2 PWM probabilities with epsilon pseudocount
    log_pwms = np.concatenate([np.log2(pwm + eps).astype(np.float32) for _, pwm in all_motifs], axis=-1)

    X_arr, X_lengths, bg_cumsums, bg_offsets, sequence_names = encode_fasta_and_bg(fasta_path, window_size)

    # Execute Numba scan
    raw_hits = fast_hits_local_bg(
        X_arr=X_arr,
        X_lengths=X_lengths,
        bg_cumsums=bg_cumsums,
        bg_offsets=bg_offsets,
        log_pwms=log_pwms,
        pwm_offsets=pwm_offsets,
        score_floor=score_floor,
    )

    names = ["chrom_idx", "start", "end", "score"]
    n_orig = n_motifs // 2
    hits_per_motif: Dict[str, pd.DataFrame] = {}

    for i in range(n_orig):
        pat_name = motif_names[i]
        strand_dfs: List[pd.DataFrame] = []
        if len(raw_hits[i]) > 0:
            fwd_df = pd.DataFrame(raw_hits[i], columns=names)
            fwd_df["strand"] = "+"
            strand_dfs.append(fwd_df)
        if len(raw_hits[i + n_orig]) > 0:
            rev_df = pd.DataFrame(raw_hits[i + n_orig], columns=names)
            rev_df["strand"] = "-"
            strand_dfs.append(rev_df)

        if strand_dfs:
            m_df = pd.concat(strand_dfs, ignore_index=True)
            m_df["chrom_idx"] = m_df["chrom_idx"].astype(np.int64)
            m_df["start"] = m_df["start"].astype(np.int64)
            m_df["end"] = m_df["end"].astype(np.int64)
            m_df["score"] = m_df["score"].astype(np.float64)
            m_df["motif_name"] = pat_name
            m_df["sequence_name"] = sequence_names[m_df["chrom_idx"].values]
        else:
            m_df = pd.DataFrame({
                "chrom_idx": pd.Series(dtype=np.int64),
                "start": pd.Series(dtype=np.int64),
                "end": pd.Series(dtype=np.int64),
                "score": pd.Series(dtype=np.float64),
                "strand": pd.Series(dtype=str),
                "motif_name": pd.Series(dtype=str),
                "sequence_name": pd.Series(dtype=str),
            })

        hits_per_motif[pat_name] = m_df

    return hits_per_motif, sequence_names


def compute_empirical_fdr_thresholds(
    real_hits_per_motif: Dict[str, pd.DataFrame],
    null_hits_per_motif: Dict[str, pd.DataFrame],
    trimming_df: pd.DataFrame,
    target_fdr: float = 0.05,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], pd.DataFrame]:
    """Computes monotonic empirical False Discovery Rate (FDR) curves and thresholds per motif.

    Args:
        real_hits_per_motif: Mapping from pattern name to raw real hits dataframe.
        null_hits_per_motif: Mapping from pattern name to raw shuffled null hits dataframe.
        trimming_df: Metadata dataframe of trimmed motif lengths.
        target_fdr: Target maximum empirical False Discovery Rate (default: 0.05).

    Returns:
        thresholds: Score threshold per motif achieving empirical FDR <= target_fdr.
        filtered_hits: Dataframes containing only significant hits (score >= threshold).
        fdr_summary_df: Summary audit table detailing empirical FDR statistics per motif.
    """
    thresholds: Dict[str, float] = {}
    filtered_hits: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    trim_len_map = trimming_df.set_index("pattern_name")["trimmed_length"].to_dict()

    for pat_name, real_df in real_hits_per_motif.items():
        null_df = null_hits_per_motif[pat_name]
        real_scores = real_df["score"].values.astype(np.float64) if len(real_df) > 0 else np.array([], dtype=np.float64)
        null_scores = null_df["score"].values.astype(np.float64) if len(null_df) > 0 else np.array([], dtype=np.float64)

        trimmed_len = int(trim_len_map[pat_name])

        if len(real_scores) == 0:
            thresholds[pat_name] = float("inf")
            filtered_hits[pat_name] = pd.DataFrame()
            summary_rows.append({
                "pattern_name": pat_name,
                "trimmed_length": trimmed_len,
                "score_threshold": float("inf"),
                "total_real_hits": 0,
                "total_null_hits": len(null_scores),
                "sig_real_hits": 0,
                "sig_null_hits": 0,
                "empirical_fdr": 0.0,
            })
            continue

        sorted_real = np.sort(real_scores)
        n_real = len(sorted_real)
        sorted_null = np.sort(null_scores)
        n_null = len(sorted_null)

        # Hits >= score for each candidate score
        real_counts_ge = n_real - np.arange(n_real)
        null_idx = np.searchsorted(sorted_null, sorted_real, side="left")
        null_counts_ge = n_null - null_idx

        raw_fdr = np.minimum(1.0, null_counts_ge.astype(np.float64) / np.maximum(1, real_counts_ge).astype(np.float64))

        # Enforce right-to-left cumulative maximum to ensure monotonic FDR
        cummax_fdr = np.maximum.accumulate(raw_fdr[::-1])[::-1]
        valid_strict = np.where(cummax_fdr <= target_fdr)[0]

        if len(valid_strict) > 0:
            chosen_idx = valid_strict[0]
            score_thresh = float(sorted_real[chosen_idx])
        else:
            valid_any = np.where(raw_fdr <= target_fdr)[0]
            if len(valid_any) > 0:
                score_thresh = float(sorted_real[valid_any[0]])
            else:
                score_thresh = float("inf")

        thresholds[pat_name] = score_thresh

        if score_thresh != float("inf"):
            sig_df = real_df[real_df["score"] >= score_thresh].copy()
            # Calculate hit-level empirical q-values
            hit_scores = sig_df["score"].values
            hit_null_cnt = n_null - np.searchsorted(sorted_null, hit_scores, side="left")
            hit_real_cnt = n_real - np.searchsorted(sorted_real, hit_scores, side="left")
            hit_q = np.minimum(1.0, hit_null_cnt.astype(np.float64) / np.maximum(1, hit_real_cnt).astype(np.float64))
            sig_df["empirical_qvalue"] = np.round(hit_q, 4)
            filtered_hits[pat_name] = sig_df

            n_sig_real = len(sig_df)
            n_sig_null = int((null_scores >= score_thresh).sum())
            emp_fdr = (n_sig_null / n_sig_real) if n_sig_real > 0 else 0.0
        else:
            filtered_hits[pat_name] = pd.DataFrame()
            n_sig_real = 0
            n_sig_null = 0
            emp_fdr = 0.0

        summary_rows.append({
            "pattern_name": pat_name,
            "trimmed_length": trimmed_len,
            "score_threshold": round(score_thresh, 2) if score_thresh != float("inf") else float("inf"),
            "total_real_hits": n_real,
            "total_null_hits": n_null,
            "sig_real_hits": n_sig_real,
            "sig_null_hits": n_sig_null,
            "empirical_fdr": round(emp_fdr, 4),
        })

    fdr_summary_df = pd.DataFrame(summary_rows)
    return thresholds, filtered_hits, fdr_summary_df


def annotate_fimo_hits(
    hits_df: pd.DataFrame,
    landmarks_df: pd.DataFrame,
    regions_path: Path,
) -> pd.DataFrame:
    """Vectorised annotation of FIMO hits with spatial landmarks and ground-truth features."""
    if len(hits_df) == 0:
        return hits_df

    merged = hits_df.merge(landmarks_df, on="sequence_name", how="left")

    midpoint = (merged["start"] + merged["end"]) / 2.0
    merged["hit_midpoint"] = midpoint
    merged["dist_to_tss"] = midpoint - merged["tss_pos"]
    merged["dist_to_28s_end"] = midpoint - merged["term_pos"]
    merged["its1_rel_pos"] = (midpoint - merged["its1_start"]) / merged["its1_len"]
    merged["is_in_its1"] = (midpoint >= merged["its1_start"]) & (midpoint <= merged["its1_end"])
    merged["its2_rel_pos"] = (midpoint - merged["its2_start"]) / merged["its2_len"]
    merged["is_in_its2"] = (midpoint >= merged["its2_start"]) & (midpoint <= merged["its2_end"])

    # High-level compartment label
    conditions = [
        (merged["dist_to_tss"] >= -2000.0) & (merged["dist_to_tss"] <= 1000.0),
        (merged["dist_to_28s_end"] >= -200.0) & (merged["dist_to_28s_end"] <= 1500.0),
        merged["is_in_its2"],
    ]
    choices = ["Promoter_TSS", "Terminator", "ITS2"]
    merged["compartment"] = np.select(conditions, choices, default="IGS_Distal")

    # Fine biological compartment for fold-enrichment analysis
    fine_conditions = [
        (merged["dist_to_tss"] >= -120.0) & (merged["dist_to_tss"] <= 100.0),
        (merged["dist_to_tss"] >= -2000.0) & (merged["dist_to_tss"] < -120.0),
        (merged["dist_to_tss"] > 100.0) & (merged["dist_to_tss"] <= 1500.0),
        merged["is_in_its1"],
        merged["is_in_its2"],
        (merged["dist_to_28s_end"] >= 0.0) & (merged["dist_to_28s_end"] <= 1500.0),
    ]
    fine_choices = [
        "Core_Promoter_TSS",
        "Upstream_Promoter",
        "5prime_ETS",
        "ITS1",
        "ITS2",
        "Terminator",
    ]
    merged["fine_compartment"] = np.select(fine_conditions, fine_choices, default="Distal_IGS")

    # Overlap with ground-truth annotations from regions.parquet
    regions_df = pd.read_parquet(regions_path)
    regions_dict: Dict[str, List[Tuple[int, int, str]]] = {}
    for _, r in regions_df.iterrows():
        rec = str(r["record_id"]).replace(" ", "_")
        if rec not in regions_dict:
            regions_dict[rec] = []
        regions_dict[rec].append((int(r["start"]), int(r["end"]), str(r["feature_type"])))

    def _get_overlaps(seq_name: str, s: int, e: int) -> str:
        if seq_name not in regions_dict:
            return "(unannotated)"
        feats = set()
        for f_s, f_e, f_type in regions_dict[seq_name]:
            if max(s, f_s) < min(e, f_e):
                feats.add(f_type)
        return ",".join(sorted(feats)) if len(feats) > 0 else "(unannotated)"

    merged["overlapping_annotations"] = [
        _get_overlaps(str(row["sequence_name"]), int(row["start"]), int(row["end"]))
        for _, row in merged.iterrows()
    ]
    merged["is_unannotated"] = merged["overlapping_annotations"] == "(unannotated)"

    return merged


def compute_regional_fold_enrichment(
    fdr_hits: pd.DataFrame,
    pattern_order: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Calculates log2 fold-enrichment across all FDR-significant hits relative to compartment lengths."""
    bg_props = {
        "Core_Promoter_TSS": 0.0059,     # ~220 bp / 37,364 bp
        "Upstream_Promoter": 0.0503,     # ~1,880 bp
        "5prime_ETS": 0.0375,            # ~1,400 bp
        "ITS1": 0.0390,                  # ~1,459 bp
        "ITS2": 0.0308,                  # ~1,151 bp
        "Terminator": 0.0401,            # ~1,500 bp
        "Distal_IGS": 0.7964,            # rest (~29.7 kb)
    }

    comp_labels = [
        "Core_Promoter_TSS",
        "Upstream_Promoter",
        "5prime_ETS",
        "ITS1",
        "ITS2",
        "Terminator",
        "Distal_IGS",
    ]
    comp_display_names = [
        "Core Prom. / TSS\n(-120..+100 bp)",
        "UCE / Upstream\n(-2k..-120 bp)",
        "5' ETS\n(+100..+1.5k bp)",
        "ITS1 Spacer\n(18S-5.8S)",
        "ITS2 Spacer\n(5.8S-28S)",
        "28S Terminator\n(0..+1.5k bp)",
        "Distal IGS\n(Intergenic)",
    ]

    fe_mat = pd.DataFrame(index=pattern_order, columns=comp_labels, dtype=float)
    for p in pattern_order:
        sub = fdr_hits[fdr_hits["motif_name"] == p]
        tot = len(sub)
        for c in comp_labels:
            obs_cnt = int((sub["fine_compartment"] == c).sum())
            obs_prop = (obs_cnt + 0.5) / (tot + 0.5 * len(comp_labels))
            exp_prop = bg_props[c]
            fe_mat.loc[p, c] = float(np.log2(obs_prop / exp_prop))

    return fe_mat, comp_labels, comp_display_names


def build_conservation_summary(
    all_fdr_hits: pd.DataFrame,
    consensus_dict: Dict[str, str],
    fe_mat: pd.DataFrame,
    trimming_df: pd.DataFrame,
    total_species: int = 156,
    total_orders: int = 17,
) -> pd.DataFrame:
    """Aggregates conservation statistics, fold-enrichment, and compartment distributions across FDR hits."""
    summary_rows: List[Dict[str, Any]] = []
    pattern_names = sorted(consensus_dict.keys(), key=pattern_sort_key)
    trim_len_map = trimming_df.set_index("pattern_name")["trimmed_length"].to_dict()

    for pat in pattern_names:
        sub = all_fdr_hits[all_fdr_hits["motif_name"] == pat]
        n_hits = len(sub)
        n_sp = sub["sequence_name"].nunique() if n_hits > 0 else 0
        sp_coverage_pct = (n_sp / total_species) * 100.0
        n_ord = sub["tax_order"].nunique() if n_hits > 0 else 0
        ord_coverage_pct = (n_ord / total_orders) * 100.0

        consensus = consensus_dict[pat]
        trimmed_len = int(trim_len_map[pat])

        primary_enriched_compartment = str(fe_mat.loc[pat].idxmax()) if pat in fe_mat.index else "None"
        primary_enriched_log2_fe = round(float(fe_mat.loc[pat].max()), 2) if pat in fe_mat.index else 0.0

        # Ground-truth annotation overlaps
        unann_cnt = int((sub["overlapping_annotations"] == "(unannotated)").sum()) if n_hits > 0 else 0
        all_feats: List[str] = []
        if n_hits > 0:
            for ann_str in sub["overlapping_annotations"]:
                if ann_str != "(unannotated)":
                    for f in ann_str.split(","):
                        all_feats.append(f)
        feat_counts = pd.Series(all_feats).value_counts() if all_feats else pd.Series(dtype=int)

        parts: List[str] = []
        if unann_cnt > 0:
            parts.append(f"(unannotated): {unann_cnt} ({unann_cnt / n_hits * 100.0:.1f}%)")
        for feat, cnt in feat_counts.items():
            parts.append(f"{feat}: {cnt} ({cnt / n_hits * 100.0:.1f}%)")
        gt_overlap_str = "; ".join(parts) if parts else "(none)"

        annotated_cnt = n_hits - unann_cnt
        unann_pct = round(unann_cnt / n_hits * 100.0, 2) if n_hits > 0 else 0.0
        annotated_pct = round(annotated_cnt / n_hits * 100.0, 2) if n_hits > 0 else 0.0

        # Compartment breakdown across all significant hits
        n_core = int((sub["fine_compartment"] == "Core_Promoter_TSS").sum()) if n_hits > 0 else 0
        n_uce = int((sub["fine_compartment"] == "Upstream_Promoter").sum()) if n_hits > 0 else 0
        n_ets = int((sub["fine_compartment"] == "5prime_ETS").sum()) if n_hits > 0 else 0
        n_its1 = int((sub["fine_compartment"] == "ITS1").sum()) if n_hits > 0 else 0
        n_its2 = int((sub["fine_compartment"] == "ITS2").sum()) if n_hits > 0 else 0
        n_term = int((sub["fine_compartment"] == "Terminator").sum()) if n_hits > 0 else 0
        n_igs = int((sub["fine_compartment"] == "Distal_IGS").sum()) if n_hits > 0 else 0

        pct_core = (n_core / n_hits * 100.0) if n_hits > 0 else 0.0
        pct_uce = (n_uce / n_hits * 100.0) if n_hits > 0 else 0.0
        pct_ets = (n_ets / n_hits * 100.0) if n_hits > 0 else 0.0
        pct_its1 = (n_its1 / n_hits * 100.0) if n_hits > 0 else 0.0
        pct_its2 = (n_its2 / n_hits * 100.0) if n_hits > 0 else 0.0
        pct_term = (n_term / n_hits * 100.0) if n_hits > 0 else 0.0
        pct_igs = (n_igs / n_hits * 100.0) if n_hits > 0 else 0.0

        # Spatial medians and IQRs
        valid_tss = sub["dist_to_tss"].dropna() if n_hits > 0 else pd.Series(dtype=float)
        med_tss = float(valid_tss.median()) if len(valid_tss) > 0 else np.nan
        iqr_tss = float(valid_tss.quantile(0.75) - valid_tss.quantile(0.25)) if len(valid_tss) > 0 else np.nan

        valid_term = sub["dist_to_28s_end"].dropna() if n_hits > 0 else pd.Series(dtype=float)
        med_term = float(valid_term.median()) if len(valid_term) > 0 else np.nan
        iqr_term = float(valid_term.quantile(0.75) - valid_term.quantile(0.25)) if len(valid_term) > 0 else np.nan

        sub_its2 = sub[sub["is_in_its2"]] if n_hits > 0 else pd.DataFrame()
        med_its2 = float(sub_its2["its2_rel_pos"].median()) if len(sub_its2) > 0 else np.nan
        iqr_its2 = float(sub_its2["its2_rel_pos"].quantile(0.75) - sub_its2["its2_rel_pos"].quantile(0.25)) if len(sub_its2) > 0 else np.nan

        summary_rows.append({
            "pattern_name": pat,
            "trimmed_length": trimmed_len,
            "primary_enriched_compartment": primary_enriched_compartment,
            "primary_enriched_log2_fe": primary_enriched_log2_fe,
            "total_fdr_hits": n_hits,
            "num_species_matched": n_sp,
            "species_coverage_pct": round(sp_coverage_pct, 2),
            "num_orders_matched": n_ord,
            "orders_coverage_pct": round(ord_coverage_pct, 2),
            "unannotated_hit_count": unann_cnt,
            "unannotated_hit_pct": unann_pct,
            "annotated_hit_count": annotated_cnt,
            "annotated_hit_pct": annotated_pct,
            "ground_truth_regulatory_overlap": gt_overlap_str,
            "consensus": consensus,
            "core_promoter_count": n_core,
            "core_promoter_pct": round(pct_core, 2),
            "upstream_uce_count": n_uce,
            "upstream_uce_pct": round(pct_uce, 2),
            "5prime_ets_count": n_ets,
            "5prime_ets_pct": round(pct_ets, 2),
            "its1_count": n_its1,
            "its1_pct": round(pct_its1, 2),
            "its2_count": n_its2,
            "its2_pct": round(pct_its2, 2),
            "terminator_count": n_term,
            "terminator_pct": round(pct_term, 2),
            "distal_igs_count": n_igs,
            "distal_igs_pct": round(pct_igs, 2),
            "median_dist_tss": round(med_tss, 1) if not np.isnan(med_tss) else np.nan,
            "iqr_dist_tss": round(iqr_tss, 1) if not np.isnan(iqr_tss) else np.nan,
            "median_dist_term": round(med_term, 1) if not np.isnan(med_term) else np.nan,
            "iqr_dist_term": round(iqr_term, 1) if not np.isnan(iqr_term) else np.nan,
            "median_its2_rel": round(med_its2, 4) if not np.isnan(med_its2) else np.nan,
            "iqr_its2_rel": round(iqr_its2, 4) if not np.isnan(iqr_its2) else np.nan,
        })

    return pd.DataFrame(summary_rows)


def extract_modisco_seqlets_and_windows(
    h5_path: Path,
    metadata_path: Optional[Path],
    landmarks_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Extracts all TF-MoDISco seqlets and maps them to genomic coordinates and biological compartments."""
    seqlet_rows: List[pd.DataFrame] = []
    with h5py.File(h5_path, "r") as f:
        for group_name in ["pos_patterns", "neg_patterns"]:
            if group_name not in f:
                continue
            grp = f[group_name]
            for pat_name in sorted(grp.keys(), key=pattern_sort_key):
                full_name = f"{group_name}.{pat_name}"
                s_grp = grp[pat_name]["seqlets"]
                seqlet_rows.append(
                    pd.DataFrame({
                        "pattern_name": full_name,
                        "example_idx": s_grp["example_idx"][:].astype(np.int64),
                        "seqlet_start": s_grp["start"][:].astype(np.int64),
                        "seqlet_end": s_grp["end"][:].astype(np.int64),
                    })
                )

    seqlets_df = pd.concat(seqlet_rows, ignore_index=True)
    max_ex_idx = int(seqlets_df["example_idx"].max())

    if metadata_path is None:
        if max_ex_idx >= 2500:
            resolved_meta = Path("outputs/tfmodisco_inputs/all_seq/candidate_metadata.parquet")
        else:
            resolved_meta = Path("outputs/tfmodisco_inputs/full_no_holdout/candidate_metadata.parquet")
    else:
        resolved_meta = metadata_path

    if not resolved_meta.is_file():
        raise FileNotFoundError(f"Candidate metadata parquet not found at: {resolved_meta}")

    meta_df = pd.read_parquet(resolved_meta)
    for req_col in ["record_id", "window_start", "window_end"]:
        if req_col not in meta_df.columns:
            raise KeyError(f"Required column '{req_col}' missing from {resolved_meta}")

    if max_ex_idx >= len(meta_df):
        raise IndexError(
            f"Max seqlet example_idx ({max_ex_idx}) exceeds candidate metadata length ({len(meta_df)}) in {resolved_meta}."
        )

    matched_meta = meta_df.iloc[seqlets_df["example_idx"].values].reset_index(drop=True)
    seqlets_df["record_id"] = matched_meta["record_id"].astype(str).values
    seqlets_df["window_start"] = matched_meta["window_start"].astype(np.int64).values
    seqlets_df["window_end"] = matched_meta["window_end"].astype(np.int64).values
    seqlets_df["seqlet_genomic_start"] = seqlets_df["window_start"] + seqlets_df["seqlet_start"]
    seqlets_df["seqlet_genomic_end"] = seqlets_df["window_start"] + seqlets_df["seqlet_end"]
    seqlets_df["seqlet_midpoint"] = (seqlets_df["seqlet_genomic_start"] + seqlets_df["seqlet_genomic_end"]) / 2.0

    seqlets_df = seqlets_df.merge(landmarks_df, on="record_id", how="left")
    midpoint = seqlets_df["seqlet_midpoint"]
    dist_to_tss = midpoint - seqlets_df["tss_pos"]
    dist_to_28s_end = midpoint - seqlets_df["term_pos"]
    is_in_its1 = (midpoint >= seqlets_df["its1_start"]) & (midpoint <= seqlets_df["its1_end"])
    is_in_its2 = (midpoint >= seqlets_df["its2_start"]) & (midpoint <= seqlets_df["its2_end"])

    fine_conditions = [
        (dist_to_tss >= -120.0) & (dist_to_tss <= 100.0),
        (dist_to_tss >= -2000.0) & (dist_to_tss < -120.0),
        (dist_to_tss > 100.0) & (dist_to_tss <= 1500.0),
        is_in_its1,
        is_in_its2,
        (dist_to_28s_end >= 0.0) & (dist_to_28s_end <= 1500.0),
    ]
    fine_choices = [
        "Core_Promoter_TSS",
        "Upstream_Promoter",
        "5prime_ETS",
        "ITS1",
        "ITS2",
        "Terminator",
    ]
    seqlets_df["fine_compartment"] = np.select(fine_conditions, fine_choices, default="Distal_IGS")
    return seqlets_df, meta_df, resolved_meta


def compare_fimo_vs_modisco_attribution(
    annotated_fdr_hits: pd.DataFrame,
    seqlets_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    pattern_order: List[str],
    comp_labels: List[str],
) -> pd.DataFrame:
    """Compares full-genome FIMO sequence hits against TF-MoDISco high-attribution seqlets and candidate windows."""
    # Pre-index seqlet intervals by (record_id, pattern_name) and by record_id
    own_seqlet_intervals: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for (rec_id, pat), grp in seqlets_df.groupby(["record_id", "pattern_name"]):
        own_seqlet_intervals[(str(rec_id), str(pat))] = (
            grp["seqlet_genomic_start"].values.astype(np.int64),
            grp["seqlet_genomic_end"].values.astype(np.int64),
        )

    all_seqlet_intervals: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for rec_id, grp in seqlets_df.groupby("record_id"):
        all_seqlet_intervals[str(rec_id)] = (
            grp["seqlet_genomic_start"].values.astype(np.int64),
            grp["seqlet_genomic_end"].values.astype(np.int64),
            grp["pattern_name"].values.astype(str),
        )

    window_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for rec_id, grp in meta_df.groupby("record_id"):
        window_intervals[str(rec_id)] = (
            grp["window_start"].values.astype(np.int64),
            grp["window_end"].values.astype(np.int64),
        )

    rows: List[Dict[str, Any]] = []
    for pat in pattern_order:
        s_own = seqlets_df[seqlets_df["pattern_name"] == pat]
        f_sub = annotated_fdr_hits[annotated_fdr_hits["motif_name"] == pat] if len(annotated_fdr_hits) > 0 else pd.DataFrame()

        n_seqlets = len(s_own)
        n_fimo = len(f_sub)

        # 1. Self-recall: how many of pat's MoDISco seqlets overlap >= 1 FIMO hit of pat?
        recalled_seqlets = 0
        if n_seqlets > 0 and n_fimo > 0:
            f_by_rec: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for rec_id, f_grp in f_sub.groupby("record_id"):
                f_by_rec[str(rec_id)] = (
                    f_grp["start"].values.astype(np.int64),
                    f_grp["end"].values.astype(np.int64),
                )
            for _, s_row in s_own.iterrows():
                rec_id = str(s_row["record_id"])
                if rec_id in f_by_rec:
                    f_starts, f_ends = f_by_rec[rec_id]
                    s_s = int(s_row["seqlet_genomic_start"])
                    s_e = int(s_row["seqlet_genomic_end"])
                    if np.any((f_starts < s_e) & (f_ends > s_s)):
                        recalled_seqlets += 1

        recall_pct = (recalled_seqlets / n_seqlets * 100.0) if n_seqlets > 0 else 0.0

        # 2. 4-way classification of all genome-wide FIMO hits of pat
        hit_own_seqlet = 0
        hit_other_seqlet = 0
        hit_in_window_silent = 0
        hit_out_of_window = 0

        if n_fimo > 0:
            for _, f_row in f_sub.iterrows():
                rec_id = str(f_row["record_id"])
                f_s = int(f_row["start"])
                f_e = int(f_row["end"])

                matched_own = False
                if (rec_id, pat) in own_seqlet_intervals:
                    os_starts, os_ends = own_seqlet_intervals[(rec_id, pat)]
                    if np.any((os_starts < f_e) & (os_ends > f_s)):
                        matched_own = True

                if matched_own:
                    hit_own_seqlet += 1
                    continue

                matched_other = False
                if rec_id in all_seqlet_intervals:
                    as_starts, as_ends, as_pats = all_seqlet_intervals[rec_id]
                    ov_mask = (as_starts < f_e) & (as_ends > f_s) & (as_pats != pat)
                    if np.any(ov_mask):
                        matched_other = True

                if matched_other:
                    hit_other_seqlet += 1
                    continue

                in_win = False
                if rec_id in window_intervals:
                    w_starts, w_ends = window_intervals[rec_id]
                    if np.any((w_starts < f_e) & (w_ends > f_s)):
                        in_win = True

                if in_win:
                    hit_in_window_silent += 1
                else:
                    hit_out_of_window += 1

        row_dict: Dict[str, Any] = {
            "pattern_name": pat,
            "total_modisco_seqlets": n_seqlets,
            "recalled_modisco_seqlets": recalled_seqlets,
            "seqlet_recall_pct": round(recall_pct, 2),
            "total_fimo_hits": n_fimo,
            "fimo_in_own_seqlet_count": hit_own_seqlet,
            "fimo_in_own_seqlet_pct": round(hit_own_seqlet / n_fimo * 100.0, 2) if n_fimo > 0 else 0.0,
            "fimo_in_other_seqlet_count": hit_other_seqlet,
            "fimo_in_other_seqlet_pct": round(hit_other_seqlet / n_fimo * 100.0, 2) if n_fimo > 0 else 0.0,
            "fimo_in_window_silent_count": hit_in_window_silent,
            "fimo_in_window_silent_pct": round(hit_in_window_silent / n_fimo * 100.0, 2) if n_fimo > 0 else 0.0,
            "fimo_out_of_window_count": hit_out_of_window,
            "fimo_out_of_window_pct": round(hit_out_of_window / n_fimo * 100.0, 2) if n_fimo > 0 else 0.0,
        }

        # 3. Compartment-level comparison: FIMO sequence hits vs MoDISco seqlets
        for c in comp_labels:
            f_cnt = int((f_sub["fine_compartment"] == c).sum()) if n_fimo > 0 else 0
            s_cnt = int((s_own["fine_compartment"] == c).sum()) if n_seqlets > 0 else 0
            row_dict[f"fimo_{c}_count"] = f_cnt
            row_dict[f"seqlet_{c}_count"] = s_cnt
            row_dict[f"log2_fimo_vs_seqlet_{c}"] = round(float(np.log2((f_cnt + 1.0) / (s_cnt + 1.0))), 2)

        rows.append(row_dict)

    return pd.DataFrame(rows)


def plot_fimo_vs_modisco_attribution_comparison(
    comp_df: pd.DataFrame,
    comp_labels: List[str],
    comp_display_names: List[str],
    output_png: Path,
) -> None:
    """Plots Self-Match Recall, 4-Way FIMO Hit Attribution Breakdown, and Compartment FIMO vs. MoDISco Comparison."""
    pattern_order = comp_df["pattern_name"].tolist()
    num_patterns = len(pattern_order)
    display_names = [format_pattern_display_name(p) for p in pattern_order]
    comp_sorted = comp_df.set_index("pattern_name").loc[pattern_order].reset_index()

    fig_height = max(9.5, 0.44 * num_patterns + 2.4)
    cbar_shrink = min(0.8, 10.0 / fig_height)

    fig = plt.figure(figsize=(21, fig_height), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.35, 2.35, 4.3], wspace=0.08)

    ax_rec = fig.add_subplot(gs[0])
    ax_brk = fig.add_subplot(gs[1])
    ax_cmp = fig.add_subplot(gs[2])

    # Panel 1 (Left): MoDISco Seqlet Self-Recall (%)
    y_pos = np.arange(num_patterns)
    rec_pct = comp_sorted["seqlet_recall_pct"].values
    rec_cnt = comp_sorted["recalled_modisco_seqlets"].values
    tot_seq = comp_sorted["total_modisco_seqlets"].values

    ax_rec.barh(y_pos, rec_pct, color="#2b5c8f", alpha=0.88, height=0.75, edgecolor="#222222", linewidth=0.6)
    ax_rec.set_xlim(0, 108)
    ax_rec.set_ylim(-0.5, num_patterns - 0.5)
    ax_rec.invert_yaxis()
    ax_rec.set_yticks(y_pos)
    ax_rec.set_yticklabels(display_names, fontsize=9.5, fontweight="bold")
    ax_rec.set_xlabel("Own Seqlet Recall (%)\n[FIMO Self-Match]", fontsize=9.5, fontweight="bold")
    ax_rec.tick_params(axis="x", top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=8.5)
    ax_rec.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (r_c, t_s, pct) in enumerate(zip(rec_cnt, tot_seq, rec_pct)):
        ax_rec.text(
            pct - 3.0 if pct > 68 else pct + 2.0,
            i,
            f"{r_c}/{t_s} ({pct:.0f}%)",
            va="center",
            ha="right" if pct > 68 else "left",
            fontsize=7.8,
            fontweight="bold",
            color="white" if pct > 68 else "#222222",
        )

    # Panel 2 (Middle): 4-Way Classification of All Genome-Wide FIMO Hits (FDR <= 0.05)
    brk_cols = [
        ("fimo_in_own_seqlet", "Own Seqlet\n(Self-Match)"),
        ("fimo_in_other_seqlet", "Other Pattern\nSeqlet"),
        ("fimo_in_window_silent", "In-Window Silent\n(No Seqlet)"),
        ("fimo_out_of_window", "Outside Windows\n(Unscanned)"),
    ]
    brk_pct_mat = np.zeros((num_patterns, len(brk_cols)), dtype=float)
    brk_annot_mat = np.empty((num_patterns, len(brk_cols)), dtype=object)

    for i, (_, row) in enumerate(comp_sorted.iterrows()):
        for j, (prefix, _) in enumerate(brk_cols):
            cnt = int(row[f"{prefix}_count"])
            pct = float(row[f"{prefix}_pct"])
            brk_pct_mat[i, j] = pct
            brk_annot_mat[i, j] = f"{cnt}\n({pct:.1f}%)" if cnt > 0 else "0\n(0%)"

    brk_headers = [label for _, label in brk_cols]
    sns.heatmap(
        brk_pct_mat,
        ax=ax_brk,
        cmap="Greens",
        vmin=0.0,
        vmax=100.0,
        annot=brk_annot_mat,
        fmt="",
        annot_kws={"fontsize": 7.8, "weight": "bold"},
        cbar=False,
        linewidths=0.6,
        linecolor="#dddddd",
        yticklabels=False,
        xticklabels=brk_headers,
    )
    ax_brk.set_xticklabels(brk_headers, rotation=25, ha="right", fontsize=9.0, fontweight="bold")
    ax_brk.set_ylabel("")

    if num_patterns > 25:
        secax_brk = ax_brk.secondary_xaxis("top")
        secax_brk.set_xticks(np.arange(len(brk_headers)) + 0.5)
        secax_brk.set_xticklabels(brk_headers, rotation=25, ha="left", fontsize=9.0, fontweight="bold")

    # Panel 3 (Right): Compartment Comparison — Full-Genome FIMO Hits vs. MoDISco Seqlets
    ratio_mat = pd.DataFrame(index=pattern_order, columns=comp_labels, dtype=float)
    cmp_annot_mat = np.empty((num_patterns, len(comp_labels)), dtype=object)

    for i, pat in enumerate(pattern_order):
        row = comp_sorted.iloc[i]
        for j, c in enumerate(comp_labels):
            f_cnt = int(row[f"fimo_{c}_count"])
            s_cnt = int(row[f"seqlet_{c}_count"])
            ratio_mat.loc[pat, c] = float(row[f"log2_fimo_vs_seqlet_{c}"])
            if f_cnt == 0 and s_cnt == 0:
                cmp_annot_mat[i, j] = "-"
            else:
                cmp_annot_mat[i, j] = f"F:{f_cnt}\nM:{s_cnt}"

    sns.heatmap(
        ratio_mat,
        ax=ax_cmp,
        cmap="vlag",
        center=0.0,
        vmin=-4.0,
        vmax=4.0,
        annot=cmp_annot_mat,
        fmt="",
        annot_kws={"fontsize": 7.6, "weight": "bold"},
        cbar_kws={
            "label": "$\\log_2((\\mathrm{FIMO\\ Hits}+1) / (\\mathrm{MoDISco\\ Seqlets}+1))$ [Red: Excess FIMO | Blue: Missed Seqlets]",
            "shrink": cbar_shrink,
            "aspect": 25,
        },
        linewidths=0.6,
        linecolor="#dddddd",
        yticklabels=False,
    )
    ax_cmp.set_xticklabels(comp_display_names, rotation=25, ha="right", fontsize=9.0, fontweight="bold")
    ax_cmp.set_ylabel("")

    if num_patterns > 25:
        secax_cmp = ax_cmp.secondary_xaxis("top")
        secax_cmp.set_xticks(np.arange(len(comp_display_names)) + 0.5)
        secax_cmp.set_xticklabels(comp_display_names, rotation=25, ha="left", fontsize=9.0, fontweight="bold")

    top_edge = ax_cmp.get_position().y1
    title_y = top_edge + (1.55 / fig_height) if num_patterns > 25 else 0.985
    fig.suptitle(
        "Full-Genome Sequence Occurrences (FIMO, FDR <= 0.05) vs. High-Attribution Seqlets (TF-MoDISco)\n"
        "[Left: Own Seqlet Self-Recall | Middle: 4-Way FIMO Hit Attribution Breakdown | Right: Compartment FIMO Hits (F) vs. MoDISco Seqlets (M)]",
        fontsize=12.5,
        fontweight="bold",
        y=title_y,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"FIMO vs. MoDISco attribution comparison plot saved to: {output_png}")


def compute_active_vs_silent_fimo_attribution(
    annotated_fdr_hits: pd.DataFrame,
    seqlets_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    att_1d: np.ndarray,
    trimming_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pattern_order: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Computes per-pattern nucleotide dependency attribution distributions for Active vs. Silent FIMO hits.

    Args:
        annotated_fdr_hits: DataFrame of all genome-wide FIMO hits passing FDR <= threshold.
        seqlets_df: DataFrame of all TF-MoDISco seqlets mapped to genomic coordinates.
        meta_df: DataFrame of candidate attribution windows (row index == example_idx in att_1d).
        att_1d: 2D array of per-position nucleotide dependency centrality, shape (N_windows, 400).
        trimming_df: DataFrame of core motif trimming boundaries (trimmed_start, trimmed_end).
        summary_df: DataFrame of conservation summary containing primary_enriched_compartment.
        pattern_order: Ordered list of pattern names.

    Returns:
        act_sil_df: DataFrame of per-pattern Active vs. Silent attribution and compartment statistics.
        global_bg_stats: Dictionary with global candidate window background median, Q25, and Q75.
    """
    win_len = att_1d.shape[1]
    global_bg_stats: Dict[str, float] = {
        "bg_median": float(np.median(att_1d)),
        "bg_q25": float(np.percentile(att_1d, 25)),
        "bg_q75": float(np.percentile(att_1d, 75)),
        "bg_mean": float(np.mean(att_1d)),
    }

    # Index candidate windows by record_id -> (window_idxs, window_starts, window_ends)
    win_by_rec: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    meta_with_idx = meta_df.copy()
    meta_with_idx["win_idx"] = np.arange(len(meta_df), dtype=np.int64)
    for rec_id, grp in meta_with_idx.groupby("record_id"):
        win_by_rec[str(rec_id)] = (
            grp["win_idx"].values.astype(np.int64),
            grp["window_start"].values.astype(np.int64),
            grp["window_end"].values.astype(np.int64),
        )

    # Index seqlet intervals by (record_id, pattern_name) and by record_id
    own_seqlet_intervals: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for (rec_id, pat), grp in seqlets_df.groupby(["record_id", "pattern_name"]):
        own_seqlet_intervals[(str(rec_id), str(pat))] = (
            grp["seqlet_genomic_start"].values.astype(np.int64),
            grp["seqlet_genomic_end"].values.astype(np.int64),
        )

    all_seqlet_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for rec_id, grp in seqlets_df.groupby("record_id"):
        all_seqlet_intervals[str(rec_id)] = (
            grp["seqlet_genomic_start"].values.astype(np.int64),
            grp["seqlet_genomic_end"].values.astype(np.int64),
        )

    trim_lookup: Dict[str, Tuple[int, int]] = {}
    for _, t_row in trimming_df.iterrows():
        trim_lookup[str(t_row["pattern_name"])] = (
            int(t_row["trimmed_start"]),
            int(t_row["trimmed_end"]),
        )

    # Determine each pattern's primary conserved compartment directly from its TF-MoDISco seqlets
    bg_comp_props: Dict[str, float] = {
        "Core_Promoter_TSS": 0.0143,
        "Upstream_Promoter": 0.0786,
        "5prime_ETS": 0.0970,
        "ITS1": 0.0987,
        "ITS2": 0.0800,
        "Terminator": 0.0474,
        "Distal_IGS": 0.5840,
    }
    primary_comp_lookup: Dict[str, str] = {}
    for pat in pattern_order:
        s_own_c = seqlets_df[seqlets_df["pattern_name"] == pat]
        n_s = len(s_own_c)
        best_c = "Distal_IGS"
        best_fe = -1e9
        for c_name, bg_p in bg_comp_props.items():
            cnt_c = int((s_own_c["fine_compartment"] == c_name).sum()) if n_s > 0 else 0
            obs_p = (cnt_c + bg_p) / (n_s + 1.0)
            fe_c = float(np.log2(obs_p / bg_p))
            if fe_c > best_fe:
                best_fe = fe_c
                best_c = c_name
        primary_comp_lookup[pat] = best_c

    rows: List[Dict[str, Any]] = []
    for pat in pattern_order:
        s_own = seqlets_df[seqlets_df["pattern_name"] == pat]
        f_sub = (
            annotated_fdr_hits[annotated_fdr_hits["motif_name"] == pat]
            if len(annotated_fdr_hits) > 0
            else pd.DataFrame()
        )
        c_start, c_end = trim_lookup[pat]
        prim_comp = primary_comp_lookup[pat]

        # Compute core attribution for all MoDISco seqlets of this pattern
        modisco_core_attrs: List[float] = []
        for _, s_row in s_own.iterrows():
            ex_i = int(s_row["example_idx"])
            ss = int(s_row["seqlet_start"])
            a_s = max(0, min(win_len - 1, ss + c_start))
            a_e = max(a_s + 1, min(win_len, ss + c_end))
            modisco_core_attrs.append(float(np.mean(att_1d[ex_i, a_s:a_e])))

        active_fimo_attrs: List[float] = []
        silent_fimo_attrs: List[float] = []
        active_pwm_scores: List[float] = []
        silent_pwm_scores: List[float] = []

        active_comp_list: List[str] = []
        silent_comp_list: List[str] = []

        if len(f_sub) > 0:
            for _, f_row in f_sub.iterrows():
                rec_id = str(f_row["record_id"])
                f_s = int(f_row["start"])
                f_e = int(f_row["end"])
                f_score = float(f_row["score"])
                f_comp = str(f_row["fine_compartment"])

                matched_own = False
                if (rec_id, pat) in own_seqlet_intervals:
                    os_starts, os_ends = own_seqlet_intervals[(rec_id, pat)]
                    if np.any((os_starts < f_e) & (os_ends > f_s)):
                        matched_own = True

                matched_any_seqlet = matched_own
                if not matched_any_seqlet and rec_id in all_seqlet_intervals:
                    as_starts, as_ends = all_seqlet_intervals[rec_id]
                    if np.any((as_starts < f_e) & (as_ends > f_s)):
                        matched_any_seqlet = True

                # Check if hit overlaps a candidate attribution window in att_1d
                hit_win_attrs: List[float] = []
                if rec_id in win_by_rec:
                    w_idxs, w_starts, w_ends = win_by_rec[rec_id]
                    ov_w = np.where((w_starts < f_e) & (w_ends > f_s))[0]
                    for wi in ov_w:
                        w_idx = int(w_idxs[wi])
                        w_s = int(w_starts[wi])
                        rel_s = max(0, f_s - w_s)
                        rel_e = min(win_len, f_e - w_s)
                        if rel_e > rel_s:
                            hit_win_attrs.append(float(np.mean(att_1d[w_idx, rel_s:rel_e])))

                if matched_own:
                    active_pwm_scores.append(f_score)
                    active_comp_list.append(f_comp)
                    if hit_win_attrs:
                        active_fimo_attrs.append(float(np.max(hit_win_attrs)))
                elif not matched_any_seqlet:
                    silent_pwm_scores.append(f_score)
                    silent_comp_list.append(f_comp)
                    if hit_win_attrs:
                        silent_fimo_attrs.append(float(np.max(hit_win_attrs)))

        # Use active_fimo_attrs when present, falling back to modisco_core_attrs if FIMO had 0 self-matches
        act_arr = np.array(active_fimo_attrs if len(active_fimo_attrs) > 0 else modisco_core_attrs, dtype=np.float64)
        sil_arr = np.array(silent_fimo_attrs, dtype=np.float64)

        act_med = float(np.median(act_arr)) if len(act_arr) > 0 else np.nan
        act_q25 = float(np.percentile(act_arr, 25)) if len(act_arr) > 0 else np.nan
        act_q75 = float(np.percentile(act_arr, 75)) if len(act_arr) > 0 else np.nan
        act_mean = float(np.mean(act_arr)) if len(act_arr) > 0 else np.nan

        if len(sil_arr) > 0:
            sil_med = float(np.median(sil_arr))
            sil_q25 = float(np.percentile(sil_arr, 25))
            sil_q75 = float(np.percentile(sil_arr, 75))
            sil_mean = float(np.mean(sil_arr))
            _, mwu_p = mannwhitneyu(act_arr, sil_arr, alternative="greater")
            mwu_pval = float(mwu_p)
            attr_ratio = act_mean / max(1e-6, sil_mean)
        else:
            sil_med = np.nan
            sil_q25 = np.nan
            sil_q75 = np.nan
            sil_mean = np.nan
            mwu_pval = np.nan
            attr_ratio = act_mean / max(1e-6, global_bg_stats["bg_mean"])

        act_pwm_med = float(np.median(active_pwm_scores)) if len(active_pwm_scores) > 0 else np.nan
        sil_pwm_med = float(np.median(silent_pwm_scores)) if len(silent_pwm_scores) > 0 else np.nan

        # Primary compartment localization (%): Active vs Silent
        if len(active_comp_list) > 0:
            act_prim_pct = sum(1 for c in active_comp_list if c == prim_comp) / len(active_comp_list) * 100.0
        elif len(s_own) > 0:
            act_prim_pct = float((s_own["fine_compartment"] == prim_comp).mean() * 100.0)
        else:
            act_prim_pct = 0.0

        if len(silent_comp_list) > 0:
            sil_prim_pct = sum(1 for c in silent_comp_list if c == prim_comp) / len(silent_comp_list) * 100.0
        else:
            sil_prim_pct = 0.0

        rows.append({
            "pattern_name": pat,
            "primary_compartment": prim_comp,
            "n_active_in_window": len(active_fimo_attrs),
            "n_modisco_seqlets": len(modisco_core_attrs),
            "n_silent_in_window": len(silent_fimo_attrs),
            "n_silent_total": len(silent_comp_list),
            "active_attr_median": round(act_med, 2) if not np.isnan(act_med) else np.nan,
            "active_attr_q25": round(act_q25, 2) if not np.isnan(act_q25) else np.nan,
            "active_attr_q75": round(act_q75, 2) if not np.isnan(act_q75) else np.nan,
            "active_attr_mean": round(act_mean, 2) if not np.isnan(act_mean) else np.nan,
            "silent_attr_median": round(sil_med, 2) if not np.isnan(sil_med) else np.nan,
            "silent_attr_q25": round(sil_q25, 2) if not np.isnan(sil_q25) else np.nan,
            "silent_attr_q75": round(sil_q75, 2) if not np.isnan(sil_q75) else np.nan,
            "silent_attr_mean": round(sil_mean, 2) if not np.isnan(sil_mean) else np.nan,
            "active_vs_silent_attr_ratio": round(attr_ratio, 2),
            "active_vs_bg_attr_ratio": round(act_mean / max(1e-6, global_bg_stats["bg_mean"]), 2),
            "active_pwm_score_median": round(act_pwm_med, 2) if not np.isnan(act_pwm_med) else np.nan,
            "silent_pwm_score_median": round(sil_pwm_med, 2) if not np.isnan(sil_pwm_med) else np.nan,
            "mwu_pval_active_gt_silent": mwu_pval,
            "active_in_primary_comp_pct": round(act_prim_pct, 1),
            "silent_in_primary_comp_pct": round(sil_prim_pct, 1),
        })

    return pd.DataFrame(rows), global_bg_stats


def plot_active_vs_silent_fimo_attribution(
    act_sil_df: pd.DataFrame,
    global_bg_stats: Dict[str, float],
    output_png: Path,
) -> None:
    """Plots the 2-panel Active vs. Silent FIMO Hit Attribution Distribution & Statistical Table figure."""
    from matplotlib.colors import ListedColormap

    pattern_order = act_sil_df["pattern_name"].tolist()
    num_patterns = len(pattern_order)
    display_names = [format_pattern_display_name(p) for p in pattern_order]
    df_sorted = act_sil_df.set_index("pattern_name").loc[pattern_order].reset_index()

    fig_height = max(9.5, 0.44 * num_patterns + 2.4)

    fig = plt.figure(figsize=(13.0, fig_height), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.4, 1.6], wspace=0.06)

    ax_dist = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    y_pos = np.arange(num_patterns)

    # -------------------------------------------------------------------------
    # Panel 1 (Left): Active vs. In-Window Silent Nucleotide Dependency Distribution
    # -------------------------------------------------------------------------
    bg_med = global_bg_stats["bg_median"]
    bg_q25 = global_bg_stats["bg_q25"]
    bg_q75 = global_bg_stats["bg_q75"]

    ax_dist.axvspan(
        bg_q25,
        bg_q75,
        color="#dddddd",
        alpha=0.65,
        zorder=1,
        label=f"Window Bg IQR ({bg_q25:.1f}–{bg_q75:.1f})",
    )
    ax_dist.axvline(
        bg_med,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
        label=f"Window Bg Median ({bg_med:.1f})",
    )

    act_med = df_sorted["active_attr_median"].values
    act_q25 = df_sorted["active_attr_q25"].values
    act_q75 = df_sorted["active_attr_q75"].values

    sil_med = df_sorted["silent_attr_median"].values
    sil_q25 = df_sorted["silent_attr_q25"].values
    sil_q75 = df_sorted["silent_attr_q75"].values

    # Plot horizontal IQR intervals and medians for Active (y - 0.15) and Silent (y + 0.15)
    for i in range(num_patterns):
        if not np.isnan(act_med[i]):
            ax_dist.plot(
                [act_q25[i], act_q75[i]],
                [i - 0.15, i - 0.15],
                color="#b2182b",
                linewidth=2.3,
                alpha=0.90,
                zorder=4,
            )
            ax_dist.scatter(
                [act_med[i]],
                [i - 0.15],
                color="#b2182b",
                edgecolor="white",
                s=46,
                linewidth=0.6,
                zorder=5,
                label="Active Hit (Own Seqlet)" if i == 0 else None,
            )
        if not np.isnan(sil_med[i]):
            ax_dist.plot(
                [sil_q25[i], sil_q75[i]],
                [i + 0.15, i + 0.15],
                color="#2166ac",
                linewidth=2.3,
                alpha=0.90,
                zorder=4,
            )
            ax_dist.scatter(
                [sil_med[i]],
                [i + 0.15],
                color="#2166ac",
                edgecolor="white",
                marker="D",
                s=38,
                linewidth=0.6,
                zorder=5,
                label="In-Window Silent FIMO Hit" if i == int(np.where(~np.isnan(sil_med))[0][0]) else None,
            )

    max_x = max(
        float(np.nanmax(act_q75)) if np.any(~np.isnan(act_q75)) else 200.0,
        float(np.nanmax(sil_q75)) if np.any(~np.isnan(sil_q75)) else 100.0,
    )
    ax_dist.set_xlim(0, max_x * 1.12)
    ax_dist.set_ylim(-0.6, num_patterns - 0.4)
    ax_dist.invert_yaxis()
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels(display_names, fontsize=9.5, fontweight="bold")
    ax_dist.set_xlabel("Per-Base Nucleotide Dependency Attribution\n[Median & 25–75% IQR]", fontsize=9.5, fontweight="bold")
    ax_dist.tick_params(axis="x", top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=8.5)
    ax_dist.grid(axis="x", linestyle="--", alpha=0.5)
    ax_dist.grid(axis="y", linestyle=":", alpha=0.35)
    leg = ax_dist.legend(
        loc="upper right",
        fontsize=9.5,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#666666",
        borderpad=0.6,
        labelspacing=0.45,
        markerscale=1.25,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Panel 2 (Right): Uncolored Quantitative Attribution Ratio Table
    # -------------------------------------------------------------------------
    tbl_headers = [
        "Active Attr\nMed (N)",
        "Silent Attr\nMed (N)",
        "Active/Silent\nAttr Ratio",
    ]
    tbl_annot_mat = np.empty((num_patterns, len(tbl_headers)), dtype=object)

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        n_act = int(row["n_active_in_window"]) if int(row["n_active_in_window"]) > 0 else int(row["n_modisco_seqlets"])
        n_sil = int(row["n_silent_in_window"])
        a_m = float(row["active_attr_median"])
        s_m = float(row["silent_attr_median"])
        ratio = float(row["active_vs_silent_attr_ratio"])

        tbl_annot_mat[i, 0] = f"{a_m:.1f} (n={n_act})" if not np.isnan(a_m) else "-"
        tbl_annot_mat[i, 1] = f"{s_m:.1f} (n={n_sil})" if n_sil > 0 and not np.isnan(s_m) else "0 silent\nin-win"
        tbl_annot_mat[i, 2] = f"{ratio:.2f}x"

    cmap_white = ListedColormap(["#ffffff"])
    sns.heatmap(
        np.zeros((num_patterns, len(tbl_headers))),
        ax=ax_tbl,
        cmap=cmap_white,
        annot=tbl_annot_mat,
        fmt="",
        annot_kws={"fontsize": 8.5, "weight": "bold", "color": "#111111"},
        cbar=False,
        linewidths=0.7,
        linecolor="#d0d0d0",
        yticklabels=False,
        xticklabels=tbl_headers,
    )
    ax_tbl.set_xticklabels(tbl_headers, rotation=25, ha="right", fontsize=9.0, fontweight="bold")
    ax_tbl.set_ylabel("")

    if num_patterns > 25:
        secax_tbl = ax_tbl.secondary_xaxis("top")
        secax_tbl.set_xticks(np.arange(len(tbl_headers)) + 0.5)
        secax_tbl.set_xticklabels(tbl_headers, rotation=25, ha="left", fontsize=9.0, fontweight="bold")

    top_edge = ax_tbl.get_position().y1
    title_y = top_edge + (1.50 / fig_height) if num_patterns > 25 else 0.985
    fig.suptitle(
        "Active vs. Silent FIMO Sequence Hits: Nucleotide Dependency Attribution Distribution\n"
        "[Left: Active vs. In-Window Silent Attribution (Median & IQR) | Right: Active vs. Silent Attribution Summary]",
        fontsize=12.5,
        fontweight="bold",
        y=title_y,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Active vs. Silent FIMO hit attribution distribution plot saved to: {output_png}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments, exposing all key parameters."""
    parser = argparse.ArgumentParser(
        description="Run local sliding-window calibrated FIMO and empirical FDR analysis across 156 mammalian genomes."
    )
    parser.add_argument(
        "--modisco-h5",
        type=Path,
        default=Path("outputs/tfmodisco_runs/full_tuned/motif_only_results.h5"),
        help="Path to TF-MoDISco motif_only_results.h5.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to candidate_metadata.parquet (auto-detected if omitted).",
    )
    parser.add_argument(
        "--attributions",
        type=Path,
        default=None,
        help="Optional path to attributions.npz (auto-detected alongside candidate_metadata.parquet if omitted).",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/prepared/ribosome/records.parquet"),
        help="Path to dataset records.parquet containing rDNA sequences.",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=Path("data/prepared/ribosome/regions.parquet"),
        help="Path to dataset regions.parquet containing landmark annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory to save all annotated parquet, summary TSV, and plots (default: <modisco-h5-parent>/fimo_diagnostics).",
    )
    parser.add_argument(
        "--plateau-ratio",
        type=float,
        default=0.65,
        help="CWM peak attribution fraction threshold for core motif trimming (default: 0.65).",
    )
    parser.add_argument(
        "--min-core-len",
        type=int,
        default=6,
        help="Minimum core motif length in bp after trimming (default: 6).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Sliding-window size in bp for local nucleotide background calibration (default: 1000).",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=3.0,
        help="Minimum log-odds score floor for recording candidate hits (default: 3.0).",
    )
    parser.add_argument(
        "--fdr-thresh",
        type=float,
        default=0.05,
        help="Target empirical False Discovery Rate (FDR) threshold (default: 0.05).",
    )
    parser.add_argument(
        "--replot-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True and fimo_all_hits_annotated.parquet already exists, skip FASTA scanning and regenerate downstream tables/plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_global_seed(args.seed)

    if args.output_dir is None:
        output_dir = args.modisco_h5.parent / "fimo_diagnostics"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("Full-Dataset Robust FIMO & Empirical FDR Conservation Pipeline")
    print("=" * 85)
    print(f"MoDISco HDF5:       {args.modisco_h5}")
    print(f"Records Parquet:    {args.records}")
    print(f"Regions Parquet:    {args.regions}")
    print(f"Output Directory:   {output_dir}")
    print(f"Plateau Ratio:      {args.plateau_ratio}")
    print(f"Min Core Length:    {args.min_core_len} bp")
    print(f"Local Window Size:  {args.window_size} bp")
    print(f"Score Floor:        {args.score_floor}")
    print(f"FDR Threshold:      {args.fdr_thresh}")
    print(f"Replot Only:        {args.replot_only}")
    print(f"Seed:               {args.seed}")
    print("=" * 85)

    # 1. Load and trim PWMs
    print("\n[Step 1/8] Loading and trimming TF-MoDISco patterns to CWM cores...")
    motifs, consensus_dict, trimming_df = load_and_trim_modisco_pwms(
        h5_path=args.modisco_h5,
        plateau_ratio=args.plateau_ratio,
        min_core_len=args.min_core_len,
    )
    trimming_tsv_path = output_dir / "motif_trimming_summary.tsv"
    trimming_df.to_csv(trimming_tsv_path, sep="\t", index=False)
    print(f"Successfully trimmed {len(motifs)} motifs. Summary saved to: {trimming_tsv_path}")

    all_parquet_path = output_dir / "fimo_all_hits_annotated.parquet"
    records_df = pd.read_parquet(args.records)
    landmarks_df = extract_record_landmarks(records_df, args.regions)

    if args.replot_only and all_parquet_path.is_file():
        print(f"\n[--replot-only] Loading existing annotated FDR hits from: {all_parquet_path}")
        annotated_fdr_hits = pd.read_parquet(all_parquet_path)
    else:
        print(trimming_df[["pattern_name", "orig_length", "trimmed_length", "trimmed_consensus"]].to_string(index=False))

        # 2. Export clean temporary real FASTA and shuffled null FASTA
        print("\n[Step 2/8] Exporting real and exact dinucleotide-shuffled null FASTAs...")
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as tmp_real, \
             tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as tmp_null:
            tmp_real_path = Path(tmp_real.name)
            tmp_null_path = Path(tmp_null.name)

        try:
            records_df, _ = export_clean_fasta(args.records, tmp_real_path)
            print(f"Exported {len(records_df)} real sequences to: {tmp_real_path}")

            t0_shuf = time.time()
            export_shuffled_fasta(records_df, tmp_null_path, seed=args.seed)
            t1_shuf = time.time()
            print(f"Generated exact dinucleotide-shuffled null FASTA in {t1_shuf - t0_shuf:.2f} seconds: {tmp_null_path}")

            # 3. Extract landmarks
            print("\n[Step 3/8] Extracting biological landmarks (TSS, 28S termination, ITS)...")
            print(f"Extracted landmarks for all {len(landmarks_df)} records.")

            # 4. Run sliding-window local background scanning on real sequences
            print(f"\n[Step 4/8] Scanning real sequences with local background (W={args.window_size} bp)...")
            t0_scan_real = time.time()
            real_hits_per_motif, _ = run_local_bg_scan(
                fasta_path=tmp_real_path,
                motifs=motifs,
                window_size=args.window_size,
                score_floor=args.score_floor,
            )
            t1_scan_real = time.time()
            total_real_candidates = sum(len(df) for df in real_hits_per_motif.values())
            print(f"Real scan completed in {t1_scan_real - t0_scan_real:.2f}s ({total_real_candidates:,} candidate hits >= {args.score_floor}).")

            # 5. Run sliding-window local background scanning on shuffled null sequences
            print(f"\n[Step 5/8] Scanning shuffled null sequences with local background (W={args.window_size} bp)...")
            t0_scan_null = time.time()
            null_hits_per_motif, _ = run_local_bg_scan(
                fasta_path=tmp_null_path,
                motifs=motifs,
                window_size=args.window_size,
                score_floor=args.score_floor,
            )
            t1_scan_null = time.time()
            total_null_candidates = sum(len(df) for df in null_hits_per_motif.values())
            print(f"Null scan completed in {t1_scan_null - t0_scan_null:.2f}s ({total_null_candidates:,} candidate hits >= {args.score_floor}).")

        finally:
            for p in [tmp_real_path, tmp_null_path]:
                if p.is_file():
                    p.unlink()
                fai = Path(str(p) + ".fai")
                if fai.is_file():
                    fai.unlink()

        # 6. Compute empirical False Discovery Rate (FDR) and filter hits
        print(f"\n[Step 6/8] Calculating empirical FDR curves and filtering for FDR <= {args.fdr_thresh}...")
        thresholds, filtered_hits_per_motif, fdr_summary_df = compute_empirical_fdr_thresholds(
            real_hits_per_motif=real_hits_per_motif,
            null_hits_per_motif=null_hits_per_motif,
            trimming_df=trimming_df,
            target_fdr=args.fdr_thresh,
        )

        fdr_tsv_path = output_dir / "fimo_fdr_summary.tsv"
        fdr_summary_df.to_csv(fdr_tsv_path, sep="\t", index=False)
        print(f"FDR summary saved to: {fdr_tsv_path}")
        print(fdr_summary_df.to_string(index=False))

        all_sig_list = [df for df in filtered_hits_per_motif.values() if len(df) > 0]
        if all_sig_list:
            combined_sig_hits = pd.concat(all_sig_list, ignore_index=True)
        else:
            combined_sig_hits = pd.DataFrame()

        print(f"Total statistically significant FDR hits retained across all motifs: {len(combined_sig_hits):,}")

        # 7. Annotate significant hits with landmarks and ground-truth features
        print("\n[Step 7/8] Annotating FDR-significant hits with coordinates and biological compartments...")
        annotated_fdr_hits = annotate_fimo_hits(combined_sig_hits, landmarks_df, args.regions)

        print(f"Saving annotated FDR hits to: {all_parquet_path}")
        annotated_fdr_hits.to_parquet(all_parquet_path, index=False)

    # 8. Downstream conservation summary and FIMO vs. MoDISco attribution comparison
    print("\n[Step 8/8] Building conservation summary and FIMO vs. MoDISco attribution comparison...")
    pattern_order = sorted(consensus_dict.keys(), key=pattern_sort_key)
    fe_mat, comp_labels, comp_display_names = compute_regional_fold_enrichment(annotated_fdr_hits, pattern_order)

    summary_df = build_conservation_summary(
        all_fdr_hits=annotated_fdr_hits,
        consensus_dict=consensus_dict,
        fe_mat=fe_mat,
        trimming_df=trimming_df,
        total_species=len(records_df),
        total_orders=landmarks_df["tax_order"].nunique(),
    )

    summary_tsv_path = output_dir / "fimo_conservation_summary.tsv"
    print(f"Saving conservation summary to: {summary_tsv_path}")
    summary_df.to_csv(summary_tsv_path, sep="\t", index=False)

    seqlets_df, meta_df, resolved_meta = extract_modisco_seqlets_and_windows(
        h5_path=args.modisco_h5,
        metadata_path=args.metadata,
        landmarks_df=landmarks_df,
    )
    comp_df = compare_fimo_vs_modisco_attribution(
        annotated_fdr_hits=annotated_fdr_hits,
        seqlets_df=seqlets_df,
        meta_df=meta_df,
        pattern_order=pattern_order,
        comp_labels=comp_labels,
    )
    comp_tsv_path = output_dir / "fimo_vs_modisco_comparison.tsv"
    print(f"Saving FIMO vs. MoDISco attribution comparison table to: {comp_tsv_path}")
    comp_df.to_csv(comp_tsv_path, sep="\t", index=False)

    comp_png = output_dir / "fimo_vs_modisco_attribution_comparison.png"
    print(f"Generating FIMO vs. MoDISco Attribution Comparison Plot to: {comp_png}")
    plot_fimo_vs_modisco_attribution_comparison(comp_df, comp_labels, comp_display_names, comp_png)

    # Active vs. Silent FIMO Hit Attribution Distribution
    resolved_attr = args.attributions if args.attributions is not None else (resolved_meta.parent / "attributions.npz")
    if not resolved_attr.is_file():
        raise FileNotFoundError(f"Attributions file not found at: {resolved_attr}")
    print(f"Loading per-position attributions from: {resolved_attr}")
    att_npz = np.load(resolved_attr)
    att_arr = att_npz["arr_0"]  # shape: (N_windows, 4, 400)
    att_1d = np.sum(np.abs(att_arr), axis=1)  # shape: (N_windows, 400)

    act_sil_df, global_bg_stats = compute_active_vs_silent_fimo_attribution(
        annotated_fdr_hits=annotated_fdr_hits,
        seqlets_df=seqlets_df,
        meta_df=meta_df,
        att_1d=att_1d,
        trimming_df=trimming_df,
        summary_df=summary_df,
        pattern_order=pattern_order,
    )
    act_sil_tsv = output_dir / "fimo_active_vs_silent_attribution.tsv"
    print(f"Saving Active vs. Silent FIMO hit attribution summary to: {act_sil_tsv}")
    act_sil_df.to_csv(act_sil_tsv, sep="\t", index=False)

    act_sil_png = output_dir / "fimo_active_vs_silent_attribution_distribution.png"
    print(f"Generating Active vs. Silent FIMO Hit Attribution Distribution plot to: {act_sil_png}")
    plot_active_vs_silent_fimo_attribution(act_sil_df, global_bg_stats, act_sil_png)

    print("\n" + "=" * 120)
    print("Conservation Summary (FDR <= 0.05 Hits across 156 Mammalian rDNA Genomes):")
    print("=" * 120)
    cols_to_print = [
        "pattern_name",
        "trimmed_length",
        "primary_enriched_compartment",
        "primary_enriched_log2_fe",
        "total_fdr_hits",
        "num_species_matched",
        "species_coverage_pct",
        "unannotated_hit_pct",
        "annotated_hit_pct",
        "ground_truth_regulatory_overlap",
    ]
    print(summary_df[cols_to_print].to_string(index=False))
    print("=" * 120)
    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
