#!/usr/bin/env python3
"""
TF-MoDISco Per-Pattern Boxplot Comparison: Motif Seqlets vs. Local Flanks vs. EnhancerRepeats
=============================================================================================
Generates per-pattern two-panel, three-box (Option B) boxplot distributions across all seqlets
of each discovered TF-MoDISco motif:
  - Top Panel: Per-nucleotide Delta Block Score (Adapted − Base RiNALMo, quantifying epistatic constraint)
    across [Motif Seqlets, Local Flanks, EnhancerRepeats]
  - Bottom Panel: Per-nucleotide gLM Reconstruction Information Content (in bits, quantifying LM sequence predictability
    within a 1,022-bp context window) across [Motif Seqlets, Local Flanks, EnhancerRepeats]

Groups compared (left to right):
  1. Motif Seqlets: All motif seqlet instances belonging to the pattern
  2. Local Flanks: Pooled precedent [s - buffer_gap - flank_length, s - buffer_gap) and successive
     [e + buffer_gap, e + buffer_gap + flank_length) flanking sequences around each seqlet (default: 24 bp each)
  3. EnhancerRepeats: Random non-overlapping 50-bp sequences sampled strictly from annotated
     EnhancerRepeats regions that were NOT matched by TF-MoDISco in seqlets (and do not overlap
     Core, TTF1, or CTCF landmarks)

Boxplot Specification (strictly Tukey):
  - Centerline: median
  - Box limits: first (Q1) and third (Q3) quartiles
  - Whiskers: span all data within 1.5×IQR of the lower and upper quartiles

Outputs:
  - High-resolution PNG figures for each pattern in <output_dir>/
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
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import torch

# Ensure repository root and src/ are in sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from glmfe.seq_models.rinalmo import load_rinalmo_model
from scripts.tfmodisco_common import (
    DEFAULT_RECORDS_PARQUET,
    DEFAULT_REGIONS_PARQUET,
    DEFAULT_RUN_DIR,
    compute_shannon_ic_bits,
    format_pattern_display_name,
    get_dynamic_device,
    pattern_sort_key,
    set_global_seed,
)

logger = logging.getLogger("plot_tfmodisco_novel_vs_enhancer_repeats")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between distributions x and y."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float((2.0 * u_stat) / (n_x * n_y) - 1.0)


def build_precomputed_tile_index(
    pre_run_dirs: List[Path],
) -> Dict[str, List[Tuple[int, int, Path]]]:
    """Indexes existing precomputed 1022-bp context tile NPZ files by record_id."""
    tiles_by_rec: Dict[str, List[Tuple[int, int, Path]]] = {}
    for run_dir in pre_run_dirs:
        idx_path = run_dir / "dependency_maps" / "map_index.parquet"
        if not idx_path.is_file():
            continue
        df = pd.read_parquet(idx_path)
        for _, r in df.iterrows():
            npz_p = run_dir / str(r["map_path"])
            if npz_p.is_file():
                rec_id = str(r["record_id"])
                t_start = int(r["tile_start"])
                t_len = int(r["tile_length"])
                if rec_id not in tiles_by_rec:
                    tiles_by_rec[rec_id] = []
                tiles_by_rec[rec_id].append((t_start, t_start + t_len, npz_p))
    return tiles_by_rec


def extract_slice_from_record_tiles(
    rec_tiles: List[Tuple[int, int, Path]],
    npz_recon_cache: Dict[Path, np.ndarray],
    start: int,
    end: int,
    failed_npz_cache: Optional[Set[Path]] = None,
) -> Optional[np.ndarray]:
    """Extracts reconstruction probability slice [start, end) from precomputed tile NPZs if covered."""
    candidates: List[Tuple[int, int, Path]] = []
    for t_start, t_end, npz_p in rec_tiles:
        if failed_npz_cache is not None and npz_p in failed_npz_cache:
            continue
        if start >= t_start and end <= t_end:
            margin = min(start - t_start, t_end - end)
            candidates.append((margin, t_start, npz_p))
    candidates.sort(key=lambda x: x[0], reverse=True)

    for margin, t_start, npz_p in candidates:
        if npz_p not in npz_recon_cache:
            try:
                with np.load(npz_p) as d:
                    if "reconstruction" not in d:
                        logger.warning("NPZ file %s missing 'reconstruction' key.", npz_p)
                        if failed_npz_cache is not None:
                            failed_npz_cache.add(npz_p)
                        continue
                    npz_recon_cache[npz_p] = d["reconstruction"].astype(np.float32)
            except Exception as e:
                logger.warning("Failed to load precomputed tile NPZ %s: %s", npz_p, e, exc_info=True)
                if failed_npz_cache is not None:
                    failed_npz_cache.add(npz_p)
                continue

        recon_full = npz_recon_cache[npz_p]
        return recon_full[start - t_start : end - t_start]

    return None


def load_single_nucleotide_delta_dict(
    single_nuc_adapted_path: Path,
    single_nuc_base_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Loads precomputed single_nucleotide_scores and indexes delta arrays by record_id."""
    logger.info("Loading adapted single-nucleotide scores: %s", single_nuc_adapted_path)
    df_adapt = pd.read_parquet(single_nuc_adapted_path)
    logger.info("Loading base single-nucleotide scores: %s", single_nuc_base_path)
    df_base = pd.read_parquet(single_nuc_base_path)

    if len(df_adapt) != len(df_base):
        raise ValueError(
            f"Row count mismatch between adapted ({len(df_adapt)}) and base ({len(df_base)}) tables!"
        )

    delta_vals = np.maximum(0.0, df_adapt["block_score"].values - df_base["block_score"].values)
    df_adapt["delta_block_score"] = delta_vals

    record_dict: Dict[str, Dict[str, Any]] = {}
    for rec_id, group in df_adapt.groupby("record_id"):
        min_pos = int(group["position"].min())
        max_pos = int(group["position"].max())
        length = max_pos - min_pos + 1

        arr_delta = np.full(length, np.nan, dtype=np.float32)
        rel_idx = group["position"].values - min_pos
        arr_delta[rel_idx] = group["delta_block_score"].values

        record_dict[str(rec_id)] = {
            "min_pos": min_pos,
            "max_pos": max_pos,
            "delta": arr_delta,
        }

    logger.info("Indexed %d records successfully from single-nucleotide tables.", len(record_dict))
    return record_dict


def get_or_compute_seqlet_and_flank_recon_ic_cache(
    seqlets_df: pd.DataFrame,
    records_path: Path,
    cache_path: Path,
    weights_path: Path,
    lora_weights_path: Path,
    pre_run_dirs: List[Path],
    window_length: Optional[int] = None,
    flank_length: int = 50,
    buffer_gap: int = 5,
    context_length: int = 1022,
    batch_size: int = 100,
) -> Dict[Tuple[str, int, int], Dict[str, np.ndarray]]:
    """Loads or computes per-nucleotide gLM Reconstruction IC (in bits) at context_length=1022
    for all unique seqlets and their local flanking regions.

    Uses precomputed 1022-bp context tile NPZ reconstruction matrices from pre_run_dirs when available
    on disk, and computes 1022-bp context masked language model probabilities on GPU for any remaining
    seqlet/flank intervals.
    """
    if window_length is None:
        window_length = int(
            (seqlets_df["seqlet_genomic_end"] - seqlets_df["seqlet_genomic_start"]).iloc[0]
        )

    unique_coords_df = seqlets_df[
        ["record_id", "seqlet_genomic_start", "seqlet_genomic_end"]
    ].drop_duplicates(subset=["record_id", "seqlet_genomic_start", "seqlet_genomic_end"])

    needed_keys = [
        (str(r["record_id"]), int(r["seqlet_genomic_start"]), int(r["seqlet_genomic_end"]))
        for _, r in unique_coords_df.iterrows()
    ]

    ic_lookup: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
    if cache_path.is_file():
        logger.info("Loading seqlet & flank reconstruction IC cache from: %s", cache_path)
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if (
                    "seqlet_ic_bits" in cached
                    and "flanks_ic_bits" in cached
                    and "context_length" in cached
                    and int(cached["context_length"]) == context_length
                ):
                    rec_ids = cached["record_ids"]
                    starts = cached["starts"]
                    ends = cached["ends"]
                    seq_ic_mat = cached["seqlet_ic_bits"]
                    flk_ic_mat = cached["flanks_ic_bits"]
                    if (
                        seq_ic_mat.ndim == 2
                        and flk_ic_mat.ndim == 2
                        and seq_ic_mat.shape[1] == window_length
                        and flk_ic_mat.shape[1] == 2 * flank_length
                    ):
                        for r_id, s, e, s_ic, f_ic in zip(rec_ids, starts, ends, seq_ic_mat, flk_ic_mat):
                            ic_lookup[(str(r_id), int(s), int(e))] = {
                                "seqlet_ic": s_ic.astype(np.float32),
                                "flanks_ic": f_ic.astype(np.float32),
                            }
                    else:
                        logger.warning(
                            "Ignoring %s due to shape mismatch (expected seq=%d bp, flanks=%d bp; got seq=%s, flanks=%s).",
                            cache_path,
                            window_length,
                            2 * flank_length,
                            seq_ic_mat.shape,
                            flk_ic_mat.shape,
                        )
        except Exception as e:
            logger.warning("Failed to load existing seqlet IC cache %s: %s", cache_path, e, exc_info=True)

    missing_keys = [k for k in needed_keys if k not in ic_lookup]
    if not missing_keys:
        logger.info("All %d unique seqlet + flank genomic intervals found in 1022-bp IC cache.", len(needed_keys))
        return ic_lookup

    # Tier 1: Extract from precomputed 1022-bp context tile NPZs on disk
    tiles_by_rec = build_precomputed_tile_index(pre_run_dirs)
    npz_recon_cache: Dict[Path, np.ndarray] = {}
    failed_npz_cache: Set[Path] = set()
    loaded_from_tiles = 0

    for rec_id, s, e in missing_keys:
        if rec_id not in tiles_by_rec:
            continue
        prec_s = s - buffer_gap - flank_length
        prec_e = s - buffer_gap
        succ_s = e + buffer_gap
        succ_e = e + buffer_gap + flank_length

        rec_tiles = tiles_by_rec[rec_id]
        recon_prec = extract_slice_from_record_tiles(
            rec_tiles, npz_recon_cache, prec_s, prec_e, failed_npz_cache
        )
        recon_seq = extract_slice_from_record_tiles(
            rec_tiles, npz_recon_cache, s, e, failed_npz_cache
        )
        recon_succ = extract_slice_from_record_tiles(
            rec_tiles, npz_recon_cache, succ_s, succ_e, failed_npz_cache
        )

        if (
            recon_prec is not None
            and recon_seq is not None
            and recon_succ is not None
            and len(recon_prec) == flank_length
            and len(recon_seq) == window_length
            and len(recon_succ) == flank_length
        ):
            seq_ic = compute_shannon_ic_bits(recon_seq, eps=1e-7).astype(np.float32)
            flk_ic = compute_shannon_ic_bits(
                np.concatenate([recon_prec, recon_succ], axis=0),
                eps=1e-7,
            ).astype(np.float32)
            ic_lookup[(rec_id, s, e)] = {
                "seqlet_ic": seq_ic,
                "flanks_ic": flk_ic,
            }
            loaded_from_tiles += 1

    npz_recon_cache.clear()
    failed_npz_cache.clear()
    logger.info(
        "Extracted %d seqlet + flank IC profiles directly from precomputed 1022-bp context tile NPZs.",
        loaded_from_tiles,
    )

    still_missing = [k for k in needed_keys if k not in ic_lookup]
    if still_missing:
        logger.info(
            "Computing RiNALMo reconstruction IC (context_length=%d bp) on GPU for %d remaining seqlet + flank intervals...",
            context_length,
            len(still_missing),
        )
        records_df = pd.read_parquet(records_path).set_index("record_id")
        valid_chars = set("ACGTNRYKMSWBDHVI-")

        # Group by context sequence length for uniform tensor batching
        tasks_by_ctx_len: Dict[int, List[Tuple[Tuple[str, int, int], str, int, int, int]]] = {}
        for rec_id, s, e in still_missing:
            if e - s != window_length:
                continue
            rec_seq = str(records_df.loc[rec_id, "sequence"]).upper().replace("U", "T")
            seq_len = len(rec_seq)
            prec_s = s - buffer_gap - flank_length
            succ_e = e + buffer_gap + flank_length
            if prec_s < 0 or succ_e > seq_len:
                continue

            mid_pos = (s + e) // 2
            w_start = max(0, min(seq_len - context_length, mid_pos - context_length // 2))
            w_end = min(seq_len, w_start + context_length)
            if w_start > prec_s:
                w_start = max(0, prec_s)
                w_end = min(seq_len, w_start + context_length)
            if w_end < succ_e:
                w_end = min(seq_len, succ_e)
                w_start = max(0, w_end - context_length)

            context_seq = rec_seq[w_start:w_end]
            if set(context_seq) - valid_chars:
                continue

            prec_off = prec_s - w_start
            seq_off = s - w_start
            succ_off = (e + buffer_gap) - w_start
            ctx_len = len(context_seq)
            if ctx_len not in tasks_by_ctx_len:
                tasks_by_ctx_len[ctx_len] = []
            tasks_by_ctx_len[ctx_len].append(((rec_id, s, e), context_seq, prec_off, seq_off, succ_off))

        inference_device = str(get_dynamic_device())
        rinalmo = load_rinalmo_model(
            model_size="mega",
            weights_path=weights_path,
            device=inference_device,
            lora_weights_path=lora_weights_path,
        )

        # Mask (prec_off + i, seq_off + i, succ_off + i) in parallel across max(window_length, flank_length) steps
        max_step_len = max(window_length, flank_length)
        for ctx_len, task_list in tasks_by_ctx_len.items():
            tokenized_seqs: List[List[int]] = []
            pos_triplets: List[Tuple[int, int, int]] = []
            for _, context_seq, prec_off, seq_off, succ_off in task_list:
                base_tok = rinalmo.alphabet.batch_tokenize([context_seq])[0]
                for i in range(max_step_len):
                    tok_copy = list(base_tok)
                    p1 = prec_off + min(i, flank_length - 1)
                    p2 = seq_off + min(i, window_length - 1)
                    p3 = succ_off + min(i, flank_length - 1)
                    tok_copy[p1 + 1] = rinalmo.alphabet.mask_idx
                    tok_copy[p2 + 1] = rinalmo.alphabet.mask_idx
                    tok_copy[p3 + 1] = rinalmo.alphabet.mask_idx
                    tokenized_seqs.append(tok_copy)
                    pos_triplets.append((p1 + 1, p2 + 1, p3 + 1))

            total_samples = len(tokenized_seqs)
            probs_prec_all = np.zeros((total_samples, 4), dtype=np.float32)
            probs_seq_all = np.zeros((total_samples, 4), dtype=np.float32)
            probs_succ_all = np.zeros((total_samples, 4), dtype=np.float32)

            for b_start in range(0, total_samples, batch_size):
                b_end = min(total_samples, b_start + batch_size)
                b_len = b_end - b_start
                tokens_tensor = torch.tensor(
                    tokenized_seqs[b_start:b_end],
                    dtype=torch.int64,
                    device=rinalmo.device,
                )
                b_triplets = pos_triplets[b_start:b_end]
                p1_idx = torch.tensor([t[0] for t in b_triplets], dtype=torch.int64, device=rinalmo.device)
                p2_idx = torch.tensor([t[1] for t in b_triplets], dtype=torch.int64, device=rinalmo.device)
                p3_idx = torch.tensor([t[2] for t in b_triplets], dtype=torch.int64, device=rinalmo.device)
                b_arange = torch.arange(b_len, dtype=torch.int64, device=rinalmo.device)

                with torch.inference_mode():
                    with torch.amp.autocast(device_type=rinalmo.device.type, dtype=torch.bfloat16):
                        logits = rinalmo.model(tokens_tensor)["logits"]

                canon_idx = rinalmo.canonical_nucleotide_token_indices
                l1 = logits[b_arange, p1_idx][:, canon_idx].float()
                l2 = logits[b_arange, p2_idx][:, canon_idx].float()
                l3 = logits[b_arange, p3_idx][:, canon_idx].float()

                probs_prec_all[b_start:b_end] = torch.softmax(l1, dim=-1).cpu().numpy()
                probs_seq_all[b_start:b_end] = torch.softmax(l2, dim=-1).cpu().numpy()
                probs_succ_all[b_start:b_end] = torch.softmax(l3, dim=-1).cpu().numpy()

                if (b_end % 2000 == 0) or (b_end == total_samples):
                    logger.info(
                        "  [ctx=%dbp] Processed %d / %d masked passes (%.1f%%)",
                        ctx_len,
                        b_end,
                        total_samples,
                        100.0 * b_end / total_samples,
                    )

            ic_prec_all = compute_shannon_ic_bits(probs_prec_all, eps=1e-7).astype(np.float32)
            ic_seq_all = compute_shannon_ic_bits(probs_seq_all, eps=1e-7).astype(np.float32)
            ic_succ_all = compute_shannon_ic_bits(probs_succ_all, eps=1e-7).astype(np.float32)

            for idx, (key, _, _, _, _) in enumerate(task_list):
                base_idx = idx * max_step_len
                ic_lookup[key] = {
                    "seqlet_ic": ic_seq_all[base_idx : base_idx + window_length],
                    "flanks_ic": np.concatenate(
                        [
                            ic_prec_all[base_idx : base_idx + flank_length],
                            ic_succ_all[base_idx : base_idx + flank_length],
                        ],
                        axis=0,
                    ),
                }

        del rinalmo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save updated cache
    all_keys = sorted(ic_lookup.keys())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        context_length=np.array(context_length, dtype=np.int32),
        record_ids=np.array([k[0] for k in all_keys], dtype=str),
        starts=np.array([k[1] for k in all_keys], dtype=np.int32),
        ends=np.array([k[2] for k in all_keys], dtype=np.int32),
        seqlet_ic_bits=np.stack([ic_lookup[k]["seqlet_ic"] for k in all_keys], axis=0),
        flanks_ic_bits=np.stack([ic_lookup[k]["flanks_ic"] for k in all_keys], axis=0),
    )
    logger.info("Saved %d seqlet + flank 1022-bp IC profiles to cache: %s", len(all_keys), cache_path)
    return ic_lookup


def extract_all_seqlets_metrics_for_pattern(
    pattern_name: str,
    pattern_df: pd.DataFrame,
    delta_record_dict: Dict[str, Dict[str, Any]],
    ic_lookup: Dict[Tuple[str, int, int], Dict[str, np.ndarray]],
    flank_length: int = 50,
    buffer_gap: int = 5,
) -> Optional[Dict[str, Any]]:
    """Extracts pooled per-nucleotide Delta Block Score and gLM Reconstruction IC across ALL seqlets
    and their local flanking sequences for a given pattern.
    """
    seqlet_bs_list: List[np.ndarray] = []
    flanks_bs_list: List[np.ndarray] = []
    seqlet_ic_list: List[np.ndarray] = []
    flanks_ic_list: List[np.ndarray] = []
    valid_seqlets = 0
    total_seqlets = len(pattern_df)

    for _, row in pattern_df.iterrows():
        rec_id = str(row["record_id"])
        s = int(row["seqlet_genomic_start"])
        e = int(row["seqlet_genomic_end"])

        if rec_id not in delta_record_dict:
            continue
        rec_data = delta_record_dict[rec_id]
        min_pos = int(rec_data["min_pos"])
        max_pos = int(rec_data["max_pos"])

        prec_start = s - buffer_gap - flank_length
        prec_end = s - buffer_gap
        succ_start = e + buffer_gap
        succ_end = e + buffer_gap + flank_length

        if prec_start < min_pos or succ_end > (max_pos + 1):
            continue

        sub_prec_delta = rec_data["delta"][prec_start - min_pos : prec_end - min_pos]
        sub_seq_delta = rec_data["delta"][s - min_pos : e - min_pos]
        sub_succ_delta = rec_data["delta"][succ_start - min_pos : succ_end - min_pos]

        if (
            len(sub_seq_delta) == 0
            or len(sub_prec_delta) != flank_length
            or len(sub_succ_delta) != flank_length
            or np.any(np.isnan(sub_prec_delta))
            or np.any(np.isnan(sub_seq_delta))
            or np.any(np.isnan(sub_succ_delta))
        ):
            continue

        key = (rec_id, s, e)
        if key not in ic_lookup:
            continue
        sub_seq_ic = ic_lookup[key]["seqlet_ic"]
        sub_flk_ic = ic_lookup[key]["flanks_ic"]
        if (
            len(sub_seq_ic) != len(sub_seq_delta)
            or len(sub_flk_ic) != 2 * flank_length
            or np.any(np.isnan(sub_seq_ic))
            or np.any(np.isnan(sub_flk_ic))
        ):
            continue

        seqlet_bs_list.append(sub_seq_delta)
        flanks_bs_list.append(np.concatenate([sub_prec_delta, sub_succ_delta]))
        seqlet_ic_list.append(sub_seq_ic)
        flanks_ic_list.append(sub_flk_ic)
        valid_seqlets += 1

    if valid_seqlets == 0:
        return None

    return {
        "pattern_name": pattern_name,
        "total_seqlets": total_seqlets,
        "valid_seqlets": valid_seqlets,
        "seqlet_delta_bs": np.concatenate(seqlet_bs_list),
        "flanks_delta_bs": np.concatenate(flanks_bs_list),
        "seqlet_recon_ic": np.concatenate(seqlet_ic_list),
        "flanks_recon_ic": np.concatenate(flanks_ic_list),
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
    pre_run_dirs: List[Path],
    num_windows: int = 50,
    window_length: int = 50,
    context_length: int = 1022,
    batch_size: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Samples random 50-bp windows within EnhancerRepeats annotations NOT matched by TF-MoDISco seqlets,
    and computes Delta Block Score and 1022-bp context gLM Reconstruction IC.
    """
    if cache_path.is_file():
        try:
            with np.load(cache_path) as cached:
                if "context_length" in cached and int(cached["context_length"]) == context_length:
                    logger.info(
                        "Loading Unmatched EnhancerRepeats metrics (context_length=%d bp) from cache: %s",
                        context_length,
                        cache_path,
                    )
                    return {
                        "block_score_delta": cached["block_score_delta"],
                        "recon_ic_bits": cached["recon_ic_bits"],
                        "num_windows": int(cached["num_windows"]),
                        "num_nucleotides": len(cached["block_score_delta"]),
                    }
        except Exception as e:
            logger.warning("Failed to load existing unmatched ER cache %s: %s", cache_path, e, exc_info=True)

    logger.info(
        "Sampling random %d-bp sequences from EnhancerRepeats NOT matched by TF-MoDISco seqlets (context_length=%d bp)...",
        window_length,
        context_length,
    )
    records_df = pd.read_parquet(records_path).set_index("record_id")
    regions_df = pd.read_parquet(regions_path)
    seqlets_df = pd.read_csv(seqlet_annot_path, sep="\t")

    er_regs = regions_df[regions_df["feature_type"] == "EnhancerRepeats"]
    avoid_regs = regions_df[regions_df["feature_type"].isin(["Core", "TTF1", "CTCF"])]

    # Build exclusion lookup by record_id
    exclusions: Dict[str, List[Tuple[int, int]]] = {}
    for _, r in seqlets_df.iterrows():
        rec_id = str(r["record_id"])
        if rec_id not in exclusions:
            exclusions[rec_id] = []
        exclusions[rec_id].append((int(r["seqlet_genomic_start"]), int(r["seqlet_genomic_end"])))
    for _, r in avoid_regs.iterrows():
        rec_id = str(r["record_id"])
        if rec_id not in exclusions:
            exclusions[rec_id] = []
        exclusions[rec_id].append((int(r["start"]), int(r["end"])))

    # Identify all valid non-overlapping 50-bp windows
    valid_candidates: List[Tuple[str, int, int]] = []
    for _, r in er_regs.iterrows():
        rec_id = str(r["record_id"])
        r_start = int(r["start"])
        r_end = int(r["end"])
        ex_list = exclusions[rec_id] if rec_id in exclusions else []

        for win_start in range(r_start, r_end - window_length + 1, window_length):
            win_end = win_start + window_length
            overlap = False
            for ex_s, ex_e in ex_list:
                if not (win_end <= ex_s or win_start >= ex_e):
                    overlap = True
                    break
            if not overlap:
                valid_candidates.append((rec_id, win_start, win_end))

    logger.info(
        "Identified %d candidate non-overlapping unmatched %d-bp EnhancerRepeat intervals.",
        len(valid_candidates),
        window_length,
    )
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
    df_pre = df_pre.set_index(["record_id", "position"])

    delta_bs_list: List[float] = []
    windows_for_recon: List[Tuple[str, int, int, str]] = []

    for rec_id, win_start, win_end in sampled_windows:
        pos_keys = [(rec_id, p) for p in range(win_start, win_end)]
        matched_scores = df_pre.loc[df_pre.index.intersection(pos_keys), "delta_block_score"].values
        if len(matched_scores) == window_length:
            rec_seq = str(records_df.loc[rec_id, "sequence"]).upper().replace("U", "T")
            seq_slice = rec_seq[win_start:win_end]
            if len(seq_slice) == window_length and set(seq_slice).issubset({"A", "C", "G", "T"}):
                delta_bs_list.extend(matched_scores)
                windows_for_recon.append((rec_id, win_start, win_end, rec_seq))

    er_delta_bs = np.array(delta_bs_list, dtype=np.float32)

    # 2. Extract from precomputed 1022-bp context tile NPZs where available, else run GPU at context_length=1022
    tiles_by_rec = build_precomputed_tile_index(pre_run_dirs)
    npz_recon_cache: Dict[Path, np.ndarray] = {}
    failed_npz_cache: Set[Path] = set()
    recon_ic_by_idx: Dict[int, np.ndarray] = {}
    gpu_queue: List[Tuple[int, str, int]] = []

    for w_idx, (rec_id, win_start, win_end, rec_seq) in enumerate(windows_for_recon):
        if rec_id in tiles_by_rec:
            recon_slice = extract_slice_from_record_tiles(
                tiles_by_rec[rec_id], npz_recon_cache, win_start, win_end, failed_npz_cache
            )
            if recon_slice is not None and len(recon_slice) == window_length:
                recon_ic_by_idx[w_idx] = compute_shannon_ic_bits(recon_slice, eps=1e-7).astype(np.float32)
                continue

        seq_len = len(rec_seq)
        mid_pos = (win_start + win_end) // 2
        c_start = max(0, min(seq_len - context_length, mid_pos - context_length // 2))
        c_end = min(seq_len, c_start + context_length)
        context_seq = rec_seq[c_start:c_end]
        target_offset = win_start - c_start
        gpu_queue.append((w_idx, context_seq, target_offset))

    npz_recon_cache.clear()
    failed_npz_cache.clear()
    logger.info(
        "Unmatched EnhancerRepeats IC: %d windows from precomputed 1022-bp tiles, %d windows queued for GPU (context_length=%d bp).",
        len(recon_ic_by_idx),
        len(gpu_queue),
        context_length,
    )

    if gpu_queue:
        inference_device = str(get_dynamic_device())
        rinalmo = load_rinalmo_model(
            model_size="mega",
            weights_path=weights_path,
            device=inference_device,
            lora_weights_path=lora_weights_path,
        )

        by_ctx_len: Dict[int, List[Tuple[int, str, int]]] = {}
        for item in gpu_queue:
            c_len = len(item[1])
            if c_len not in by_ctx_len:
                by_ctx_len[c_len] = []
            by_ctx_len[c_len].append(item)

        for c_len, items in by_ctx_len.items():
            all_windows: List[str] = []
            all_positions: List[int] = []
            for _, context_seq, target_offset in items:
                all_windows.extend([context_seq] * window_length)
                all_positions.extend(range(target_offset, target_offset + window_length))

            probs = rinalmo.predict_masked_base_probabilities(
                all_windows, all_positions, batch_size=batch_size
            )
            ic_all = compute_shannon_ic_bits(probs, eps=1e-7).astype(np.float32)
            for idx, (w_idx, _, _) in enumerate(items):
                recon_ic_by_idx[w_idx] = ic_all[idx * window_length : (idx + 1) * window_length]

        del rinalmo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    er_recon_ic = np.concatenate(
        [recon_ic_by_idx[i] for i in range(len(windows_for_recon))], axis=0
    ).astype(np.float32)

    logger.info(
        "Unmatched EnhancerRepeats Baseline (context=%d bp): %d Delta BS values (med=%.2f, IQR=%.2f), %d IC values (med=%.2f bits, IQR=%.2f bits)",
        context_length,
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
        context_length=np.array(context_length, dtype=np.int32),
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


def _render_boxplot_panel(
    ax: plt.Axes,
    panel_data: List[np.ndarray],
    labels: List[str],
    positions: List[int],
    palette: List[str],
    ylabel: str,
    whisker_iqr: float,
    rng: np.random.Generator,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Render one Tukey boxplot panel with reproducible subsampled scatter overlay."""
    bp = ax.boxplot(
        panel_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        whis=whisker_iqr,
        medianprops=dict(color="#111827", linewidth=2.2),
        boxprops=dict(edgecolor="#111827", linewidth=1.3),
        whiskerprops=dict(color="#111827", linewidth=1.3, linestyle="--"),
        capprops=dict(color="#111827", linewidth=1.3),
        flierprops=dict(marker="o", markersize=3.0, markerfacecolor="#4B5563", alpha=0.30, markeredgewidth=0),
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    for pos_idx, d_arr in enumerate(panel_data):
        n_pts = len(d_arr)
        if n_pts > 250:
            sample_idx = rng.choice(n_pts, size=250, replace=False)
            plot_pts = d_arr[sample_idx]
        else:
            plot_pts = d_arr
        jitter = rng.normal(0, 0.04, size=len(plot_pts))
        ax.scatter(
            pos_idx + jitter,
            plot_pts,
            color=palette[pos_idx],
            alpha=0.35,
            s=13,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(axis="y", linestyle=":", alpha=0.6)


def render_pattern_boxplot(
    pattern_name: str,
    pattern_display: str,
    category: str,
    best_match: str,
    total_seqlets: int,
    valid_seqlets: int,
    motif_delta_bs: np.ndarray,
    flanks_delta_bs: np.ndarray,
    er_delta_bs: np.ndarray,
    motif_recon_ic: np.ndarray,
    flanks_recon_ic: np.ndarray,
    er_recon_ic: np.ndarray,
    er_label: str,
    output_png: Path,
    whisker_iqr: float = 1.5,
    dpi: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """Renders Option B: a two-panel, three-box boxplot comparing Motif Seqlets, Local Flanks,
    and EnhancerRepeats across both Per-Nucleotide Delta Block Score and gLM Reconstruction IC.

    Boxplot Specification:
      - Centerline: median
      - Box limits: first and third quartiles (Q1, Q3)
      - Whiskers: span all data within 1.5×IQR of lower and upper quartiles
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(5.8, 6.8), dpi=dpi, facecolor="white"
    )
    fig.subplots_adjust(hspace=0.22, left=0.16, right=0.94, top=0.96, bottom=0.08)

    labels = ["Motif Seqlets", "Local Flanks", er_label]
    positions = [0, 1, 2]
    # Purple (#7C3AED) for Motif Seqlets, Blue (#3B82F6) for Local Flanks, Teal (#0D9488) for EnhancerRepeats
    palette = ["#7C3AED", "#3B82F6", "#0D9488"]

    # Descriptive statistics helper
    def _stats(arr: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        med = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return med, iqr, mean, std, q1, q3

    m_bs_med, m_bs_iqr, m_bs_mean, m_bs_std, m_bs_q1, m_bs_q3 = _stats(motif_delta_bs)
    f_bs_med, f_bs_iqr, f_bs_mean, f_bs_std, f_bs_q1, f_bs_q3 = _stats(flanks_delta_bs)
    er_bs_med, er_bs_iqr, er_bs_mean, er_bs_std, er_bs_q1, er_bs_q3 = _stats(er_delta_bs)

    m_ic_med, m_ic_iqr, m_ic_mean, m_ic_std, m_ic_q1, m_ic_q3 = _stats(motif_recon_ic)
    f_ic_med, f_ic_iqr, f_ic_mean, f_ic_std, f_ic_q1, f_ic_q3 = _stats(flanks_recon_ic)
    er_ic_med, er_ic_iqr, er_ic_mean, er_ic_std, er_ic_q1, er_ic_q3 = _stats(er_recon_ic)

    # Statistical tests (Motif vs Local Flanks, and Motif vs EnhancerRepeats)
    u_bs_flk, p_bs_flk = mannwhitneyu(motif_delta_bs, flanks_delta_bs, alternative="two-sided")
    cliff_bs_flk = compute_cliffs_delta(motif_delta_bs, flanks_delta_bs)

    u_bs_er, p_bs_er = mannwhitneyu(motif_delta_bs, er_delta_bs, alternative="two-sided")
    cliff_bs_er = compute_cliffs_delta(motif_delta_bs, er_delta_bs)

    u_ic_flk, p_ic_flk = mannwhitneyu(motif_recon_ic, flanks_recon_ic, alternative="two-sided")
    cliff_ic_flk = compute_cliffs_delta(motif_recon_ic, flanks_recon_ic)

    u_ic_er, p_ic_er = mannwhitneyu(motif_recon_ic, er_recon_ic, alternative="two-sided")
    cliff_ic_er = compute_cliffs_delta(motif_recon_ic, er_recon_ic)

    _render_boxplot_panel(
        ax=ax_top,
        panel_data=[motif_delta_bs, flanks_delta_bs, er_delta_bs],
        labels=labels,
        positions=positions,
        palette=palette,
        ylabel="Per-Nucleotide Δ Block Score",
        whisker_iqr=whisker_iqr,
        rng=np.random.default_rng(seed),
    )
    _render_boxplot_panel(
        ax=ax_bottom,
        panel_data=[motif_recon_ic, flanks_recon_ic, er_recon_ic],
        labels=labels,
        positions=positions,
        palette=palette,
        ylabel="gLM Reconstruction IC (bits)",
        whisker_iqr=whisker_iqr,
        rng=np.random.default_rng(seed),
        ylim=(-0.05, 2.15),
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "pattern_name": pattern_name,
        "pattern_display": pattern_display,
        "category": category,
        "best_match": best_match,
        "total_seqlets": total_seqlets,
        "valid_seqlets": valid_seqlets,
        "motif_nucleotides_count": len(motif_delta_bs),
        "flanks_nucleotides_count": len(flanks_delta_bs),
        "er_nucleotides_count": len(er_delta_bs),
        "motif_delta_bs_median": m_bs_med,
        "motif_delta_bs_iqr": m_bs_iqr,
        "motif_delta_bs_mean": m_bs_mean,
        "motif_delta_bs_std": m_bs_std,
        "motif_delta_bs_q1": m_bs_q1,
        "motif_delta_bs_q3": m_bs_q3,
        "flanks_delta_bs_median": f_bs_med,
        "flanks_delta_bs_iqr": f_bs_iqr,
        "flanks_delta_bs_mean": f_bs_mean,
        "flanks_delta_bs_std": f_bs_std,
        "flanks_delta_bs_q1": f_bs_q1,
        "flanks_delta_bs_q3": f_bs_q3,
        "er_delta_bs_median": er_bs_med,
        "er_delta_bs_iqr": er_bs_iqr,
        "er_delta_bs_mean": er_bs_mean,
        "er_delta_bs_std": er_bs_std,
        "er_delta_bs_q1": er_bs_q1,
        "er_delta_bs_q3": er_bs_q3,
        "delta_bs_flanks_mannwhitney_u": float(u_bs_flk),
        "delta_bs_flanks_pvalue": float(p_bs_flk),
        "delta_bs_flanks_cliffs_delta": float(cliff_bs_flk),
        "delta_bs_mannwhitney_u": float(u_bs_er),
        "delta_bs_pvalue": float(p_bs_er),
        "delta_bs_cliffs_delta": float(cliff_bs_er),
        "motif_recon_ic_median": m_ic_med,
        "motif_recon_ic_iqr": m_ic_iqr,
        "motif_recon_ic_mean": m_ic_mean,
        "motif_recon_ic_std": m_ic_std,
        "motif_recon_ic_q1": m_ic_q1,
        "motif_recon_ic_q3": m_ic_q3,
        "flanks_recon_ic_median": f_ic_med,
        "flanks_recon_ic_iqr": f_ic_iqr,
        "flanks_recon_ic_mean": f_ic_mean,
        "flanks_recon_ic_std": f_ic_std,
        "flanks_recon_ic_q1": f_ic_q1,
        "flanks_recon_ic_q3": f_ic_q3,
        "er_recon_ic_median": er_ic_med,
        "er_recon_ic_iqr": er_ic_iqr,
        "er_recon_ic_mean": er_ic_mean,
        "er_recon_ic_std": er_ic_std,
        "er_recon_ic_q1": er_ic_q1,
        "er_recon_ic_q3": er_ic_q3,
        "recon_ic_flanks_mannwhitney_u": float(u_ic_flk),
        "recon_ic_flanks_pvalue": float(p_ic_flk),
        "recon_ic_flanks_cliffs_delta": float(cliff_ic_flk),
        "recon_ic_mannwhitney_u": float(u_ic_er),
        "recon_ic_pvalue": float(p_ic_er),
        "recon_ic_cliffs_delta": float(cliff_ic_er),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot per-pattern 2-panel, 3-box (Option B) Delta Block Score and gLM Reconstruction IC boxplots comparing Motif Seqlets, Local Flanks, and EnhancerRepeats."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Path to TF-MoDISco run directory (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--dep-map-dir",
        type=Path,
        default=None,
        help="Optional/deprecated: if provided without --patterns, restricts patterns to directories inside this folder.",
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        default=DEFAULT_RECORDS_PARQUET,
        help=f"Path to records.parquet (default: {DEFAULT_RECORDS_PARQUET}).",
    )
    parser.add_argument(
        "--regions-path",
        type=Path,
        default=DEFAULT_REGIONS_PARQUET,
        help=f"Path to regions.parquet (default: {DEFAULT_REGIONS_PARQUET}).",
    )
    parser.add_argument(
        "--seqlet-annotations",
        type=Path,
        default=DEFAULT_RUN_DIR / "seqlet_annotations_overlap.tsv",
        help=f"Path to seqlet_annotations_overlap.tsv (default: {DEFAULT_RUN_DIR / 'seqlet_annotations_overlap.tsv'}).",
    )
    parser.add_argument(
        "--pattern-classification",
        type=Path,
        default=DEFAULT_RUN_DIR / "pattern_classification_summary.tsv",
        help=f"Path to pattern_classification_summary.tsv (default: {DEFAULT_RUN_DIR / 'pattern_classification_summary.tsv'}).",
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
        default=DEFAULT_RUN_DIR / "novel_vs_enhancer_repeats",
        help="Directory to save per-pattern boxplot figures and summary TSV.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Optional list of patterns to process (e.g. pattern_14 pattern_16, or 'all'). Default: all patterns in seqlet-annotations.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1022,
        help="Context window length in bp for RiNALMo reconstruction IC (default: 1022).",
    )
    parser.add_argument(
        "--flank-length",
        type=int,
        default=24,
        help="Length in bp of precedent and successive local flanks (default: 24).",
    )
    parser.add_argument(
        "--buffer-gap",
        type=int,
        default=3,
        help="Buffer gap in bp between seqlet boundary and local flanks (default: 3).",
    )
    parser.add_argument(
        "--num-er-windows",
        type=int,
        default=50,
        help="Number of random non-overlapping 50-bp unmatched EnhancerRepeat windows to sample (default: 50 -> 2,500 nt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for RiNALMo masked language model inference (default: 100).",
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
    try:
        args = parse_args(argv)
        set_global_seed(args.seed)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        er_cache_path = args.results_dir / "unmatched_enhancer_repeats_cache.npz"
        seqlet_ic_cache_path = args.results_dir / "seqlet_recon_ic_cache.npz"
        pre_run_dirs = [
            Path("outputs/runs/rinalmo_tfmodisco_pre_full"),
            Path("outputs/runs/rinalmo_tfmodisco_pre_full_test"),
        ]

        logger.info(
            "Initializing Option B (2-panel, 3-box) boxplot generation across all seqlets (context_length=%d bp)...",
            args.context_length,
        )
        logger.info("Results directory: %s", args.results_dir)
        logger.info("Output directory: %s", args.output_dir)

        # 1. Load classification summary and seqlet annotations
        if not args.pattern_classification.is_file():
            raise FileNotFoundError(f"Classification summary not found: {args.pattern_classification}")
        if not args.seqlet_annotations.is_file():
            raise FileNotFoundError(f"Seqlet annotations not found: {args.seqlet_annotations}")

        df_class = pd.read_csv(args.pattern_classification, sep="\t")
        class_map: Dict[str, Dict[str, str]] = {}
        for _, r in df_class.iterrows():
            pname = str(r["pattern"])
            class_map[pname] = {
                "category": str(r["category"]),
                "modality": str(r["best_overall_modality"]),
                "best_match": str(r["best_overall_match"]) if pd.notna(r["best_overall_match"]) else "None",
            }

        seqlets_df = pd.read_csv(args.seqlet_annotations, sep="\t")

        # 2. Determine patterns to process
        all_seqlet_patterns = sorted(seqlets_df["pattern_name"].unique(), key=pattern_sort_key)
        if args.patterns is not None and "all" in args.patterns:
            patterns_to_run = all_seqlet_patterns
        elif args.patterns is not None:
            filter_set = set()
            for p in args.patterns:
                filter_set.add(p)
                filter_set.add(f"pos_patterns.{p}")
                filter_set.add(f"neg_patterns.{p}")
                if p.isdigit():
                    filter_set.add(f"pos_patterns.pattern_{p}")
            patterns_to_run = [p for p in all_seqlet_patterns if p in filter_set or p.split(".")[-1] in filter_set]
        elif args.dep_map_dir is not None:
            if not args.dep_map_dir.is_dir():
                raise FileNotFoundError(f"Requested --dep-map-dir does not exist: {args.dep_map_dir}")
            available_dirs = sorted(
                [
                    d.name
                    for d in args.dep_map_dir.iterdir()
                    if d.is_dir() and (d.name.startswith("pos_patterns.") or d.name.startswith("neg_patterns."))
                ],
                key=pattern_sort_key,
            )
            patterns_to_run = [p for p in available_dirs if p in seqlets_df["pattern_name"].values]
        else:
            patterns_to_run = all_seqlet_patterns

        logger.info("Found %d patterns to process.", len(patterns_to_run))
        if not patterns_to_run:
            logger.warning("No matching patterns found to process.")
            return 0

        # 3. Extract or load unmatched EnhancerRepeats baseline metrics (context_length=1022)
        er_metrics = get_unmatched_enhancer_repeats_metrics(
            records_path=args.records_path,
            regions_path=args.regions_path,
            seqlet_annot_path=args.seqlet_annotations,
            single_nuc_pre_path=args.single_nuc_adapted,
            single_nuc_full_path=args.single_nuc_base,
            cache_path=er_cache_path,
            weights_path=args.weights,
            lora_weights_path=args.lora_weights,
            pre_run_dirs=pre_run_dirs,
            num_windows=args.num_er_windows,
            context_length=args.context_length,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        er_delta_bs = er_metrics["block_score_delta"]
        er_recon_ic = er_metrics["recon_ic_bits"]
        er_label = "EnhancerRepeats"

        # 4. Load per-nucleotide Delta Block Scores and compute/load seqlet + flank Reconstruction IC (context_length=1022)
        delta_record_dict = load_single_nucleotide_delta_dict(
            single_nuc_adapted_path=args.single_nuc_adapted,
            single_nuc_base_path=args.single_nuc_base,
        )
        target_seqlets_df = seqlets_df[seqlets_df["pattern_name"].isin(patterns_to_run)]
        ic_lookup = get_or_compute_seqlet_and_flank_recon_ic_cache(
            seqlets_df=target_seqlets_df,
            records_path=args.records_path,
            cache_path=seqlet_ic_cache_path,
            weights_path=args.weights,
            lora_weights_path=args.lora_weights,
            pre_run_dirs=pre_run_dirs,
            flank_length=args.flank_length,
            buffer_gap=args.buffer_gap,
            context_length=args.context_length,
            batch_size=args.batch_size,
        )

        summary_rows: List[Dict[str, Any]] = []

        # 5. Generate per-pattern Option B (2-panel, 3-box) plots across all seqlets
        for idx, pat_name in enumerate(patterns_to_run):
            pat_display = format_pattern_display_name(pat_name)
            meta = (
                class_map[pat_name]
                if pat_name in class_map
                else {"category": "Unknown", "modality": "Unknown", "best_match": "None"}
            )
            category = meta["category"]
            best_match = meta["best_match"]
            pat_df = seqlets_df[seqlets_df["pattern_name"] == pat_name]

            logger.info(
                "[%d/%d] Processing %s (N=%d seqlets, category: %s)...",
                idx + 1,
                len(patterns_to_run),
                pat_display,
                len(pat_df),
                category,
            )

            motif_metrics = extract_all_seqlets_metrics_for_pattern(
                pattern_name=pat_name,
                pattern_df=pat_df,
                delta_record_dict=delta_record_dict,
                ic_lookup=ic_lookup,
                flank_length=args.flank_length,
                buffer_gap=args.buffer_gap,
            )
            if motif_metrics is None:
                logger.warning("Skipping %s: no valid seqlet metrics found.", pat_name)
                continue

            out_png = args.output_dir / f"{pat_display}_vs_EnhancerRepeats.png"

            stats = render_pattern_boxplot(
                pattern_name=pat_name,
                pattern_display=pat_display,
                category=category,
                best_match=best_match,
                total_seqlets=motif_metrics["total_seqlets"],
                valid_seqlets=motif_metrics["valid_seqlets"],
                motif_delta_bs=motif_metrics["seqlet_delta_bs"],
                flanks_delta_bs=motif_metrics["flanks_delta_bs"],
                er_delta_bs=er_delta_bs,
                motif_recon_ic=motif_metrics["seqlet_recon_ic"],
                flanks_recon_ic=motif_metrics["flanks_recon_ic"],
                er_recon_ic=er_recon_ic,
                er_label=er_label,
                output_png=out_png,
                whisker_iqr=args.whisker_iqr,
                dpi=args.dpi,
                seed=args.seed,
            )
            summary_rows.append(stats)
            logger.info("Saved per-pattern 3-box plot to %s", out_png)

        # 6. Export comprehensive summary TSV
        summary_df = pd.DataFrame(summary_rows)
        tsv_path = args.output_dir / "novel_vs_enhancer_repeats_summary.tsv"
        summary_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info("Saved statistical summary TSV (%d motifs) to: %s", len(summary_df), tsv_path)

        logger.info("All per-pattern 2-panel, 3-box comparisons successfully completed.")
        return 0

    except Exception as exc:
        logger.error("Fatal error in plot_tfmodisco_novel_vs_enhancer_repeats: %s", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(main())
