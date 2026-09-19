#!/usr/bin/env python
"""Prepare TF-MoDISco-compatible inputs from base/adapted RiNALMo promoter runs.

This script does NOT run RiNALMo. It consumes only existing saved outputs:

    outputs/runs/base_promotor
    outputs/runs/adapted_promotor
    data/prepared/ribosome/regions.parquet

Pipeline (see feasibility_report_tfmodisco.md):

  1. Load base/adapted block_scores/per_span.parquet and pair spans on
     (record_id, region_id, tile_index, span_start, span_end).
  2. B_delta = max(0, block_score_adapted - block_score_base).
  3. Filter spans by annotation-exclusion mode (strict | motif_only).
  4. Deduplicate identical record-relative spans across overlapping tiles
     (keep the highest B_delta occurrence).
  5. Select top-X% B_delta spans, merge nearby survivors, keep one local max
     per merged cluster -> candidate peaks.
  6. Around each peak, extract a fixed-size window; load matching base/adapted
     .npz dependency maps, verify sequences, compute D_delta, centrality s_k,
     one-hot sequence, and project s_k onto the observed nucleotide channel.
  7. Save one_hot_sequences / contribution_scores (N, 4, L), per-candidate
     metadata parquet, and a summary TSV. Run sanity checks throughout.

TF-MoDISco itself is intentionally NOT run here.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PAIR_KEY = ["record_id", "region_id", "tile_index", "span_start", "span_end"]
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

# Feature types that are always kept (never a reason to exclude a span).
PROMOTER_LABEL = "CompositePromoter"

# motif_only: hard-excluded fine-grained motifs.
MOTIF_ONLY_EXCLUDE = {"CTCF", "TTF1", "Core"}
# motif_only optional extra exclusions (CLI-controlled).
MOTIF_ONLY_OPTIONAL = {"Upstream", "Terminator"}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class LossCounters:
    """Track how candidates are lost at each stage, for reporting."""

    total_paired_spans: int = 0
    excluded_by_annotation: int = 0
    removed_by_dedup: int = 0
    below_topk_threshold: int = 0
    merged_into_clusters: int = 0
    lost_tile_boundary: int = 0
    lost_non_acgt: int = 0
    lost_sequence_mismatch: int = 0
    final_candidates: int = 0
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def load_paired_spans(base_run: str, adapted_run: str) -> pd.DataFrame:
    """Load and pair per-span block scores; compute B_delta."""
    base = pd.read_parquet(os.path.join(base_run, "block_scores", "per_span.parquet"))
    adapt = pd.read_parquet(
        os.path.join(adapted_run, "block_scores", "per_span.parquet")
    )

    carry = PAIR_KEY + [
        "map_id",
        "map_start",
        "map_length",
        "local_start",
        "span_length",
    ]
    merged = base[carry + ["block_score"]].merge(
        adapt[PAIR_KEY + ["block_score"]],
        on=PAIR_KEY,
        suffixes=("_base", "_adapt"),
        how="inner",
    )
    if len(merged) != len(base):
        print(
            f"[warn] paired {len(merged)}/{len(base)} base spans "
            f"({len(adapt)} adapted spans); using intersection.",
            file=sys.stderr,
        )
    merged["B_delta"] = np.maximum(
        0.0, merged["block_score_adapt"] - merged["block_score_base"]
    )
    return merged


def load_annotations(regions_path: str) -> dict:
    """Return record_id -> list of (start, end, feature_type)."""
    reg = pd.read_parquet(regions_path)
    by_record: dict = {}
    for rid, grp in reg.groupby("record_id"):
        by_record[rid] = list(
            zip(
                grp["start"].astype(int),
                grp["end"].astype(int),
                grp["feature_type"].astype(str),
            )
        )
    return by_record


def load_map_paths(run: str) -> dict:
    """Return map_id -> (map_start, map_length, absolute_npz_path) for a run."""
    mi = pd.read_parquet(os.path.join(run, "dependency_maps", "map_index.parquet"))
    out = {}
    for _, r in mi.iterrows():
        out[r["map_id"]] = (
            int(r["start"]),
            int(r["length"]),
            os.path.join(run, r["map_path"]),
        )
    return out


def load_record_tiles(run: str) -> dict:
    """Return record_id -> list of (map_id, map_start, map_length) for tile lookup."""
    mi = pd.read_parquet(os.path.join(run, "dependency_maps", "map_index.parquet"))
    by_record: dict = {}
    for _, r in mi.iterrows():
        by_record.setdefault(r["record_id"], []).append(
            (r["map_id"], int(r["start"]), int(r["length"]))
        )
    return by_record


# --------------------------------------------------------------------------- #
# Annotation logic
# --------------------------------------------------------------------------- #


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Half-open interval overlap test [a0,a1) vs [b0,b1)."""
    return a0 < b1 and b0 < a1


def excluded_feature_types(mode: str, drop_upstream: bool, drop_terminator: bool):
    """Return the set of feature types whose overlap excludes a span."""
    if mode == "strict":
        # Everything except the promoter region itself excludes a span.
        # (Resolved lazily against whatever annotations exist per record.)
        return None  # sentinel: "any non-promoter annotation"
    if mode == "motif_only":
        excl = set(MOTIF_ONLY_EXCLUDE)
        if drop_upstream:
            excl.add("Upstream")
        if drop_terminator:
            excl.add("Terminator")
        return excl
    raise ValueError(f"unknown mode {mode!r}")


def span_is_excluded(
    record_id: str,
    start: int,
    end: int,
    annotations: dict,
    excl_set,
) -> bool:
    """True if the span overlaps a feature type that the mode excludes."""
    for a0, a1, ft in annotations.get(record_id, ()):
        if not overlaps(start, end, a0, a1):
            continue
        if ft == PROMOTER_LABEL:
            continue
        if excl_set is None:  # strict: any non-promoter annotation excludes
            return True
        if ft in excl_set:
            return True
    return False


def overlapping_annotation_types(
    record_id: str, start: int, end: int, annotations: dict
) -> list:
    """All feature types overlapping [start,end), sorted unique (incl. promoter)."""
    hits = {
        ft
        for a0, a1, ft in annotations.get(record_id, ())
        if overlaps(start, end, a0, a1)
    }
    return sorted(hits)


# --------------------------------------------------------------------------- #
# Peak selection
# --------------------------------------------------------------------------- #


def dedupe_spans(df: pd.DataFrame, losses: LossCounters) -> pd.DataFrame:
    """Keep highest-B_delta occurrence of each record-relative span."""
    before = len(df)
    df = df.sort_values("B_delta", ascending=False)
    deduped = df.drop_duplicates(subset=["record_id", "span_start", "span_end"])
    losses.removed_by_dedup = before - len(deduped)
    return deduped.reset_index(drop=True)


def select_peaks(
    df: pd.DataFrame,
    top_pct: float,
    merge_distance: int,
    losses: LossCounters,
) -> tuple[pd.DataFrame, float]:
    """Top-X% threshold, then merge nearby spans and keep one local max each."""
    if df.empty:
        return df, float("nan")

    threshold = float(np.quantile(df["B_delta"].to_numpy(), 1.0 - top_pct / 100.0))
    top = df[df["B_delta"] >= threshold].copy()
    losses.below_topk_threshold = len(df) - len(top)

    top["center"] = (top["span_start"] + top["span_end"]) // 2

    peaks = []
    for _, grp in top.groupby("record_id"):
        grp = grp.sort_values("center")
        cluster = []
        prev = None
        for _, row in grp.iterrows():
            if prev is not None and row["center"] - prev > merge_distance:
                peaks.append(_cluster_max(cluster))
                cluster = []
            cluster.append(row)
            prev = row["center"]
        if cluster:
            peaks.append(_cluster_max(cluster))

    losses.merged_into_clusters = len(top) - len(peaks)
    peaks_df = pd.DataFrame(peaks).reset_index(drop=True)
    return peaks_df, threshold


def _cluster_max(cluster: list) -> pd.Series:
    return max(cluster, key=lambda r: r["B_delta"])


def select_dense(
    df: pd.DataFrame,
    top_pct: float,
    min_sep: int,
    max_windows: int,
    losses: LossCounters,
) -> tuple[pd.DataFrame, float]:
    """Emit MANY windows (not one-per-cluster) to give TF-MoDISco enough seqlets.

    Keep top-X% B_delta spans, then greedily select non-redundant windows whose
    centers are >= ``min_sep`` bp apart (per record), highest B_delta first, up
    to ``max_windows`` total.
    """
    if df.empty:
        return df, float("nan")

    threshold = float(np.quantile(df["B_delta"].to_numpy(), 1.0 - top_pct / 100.0))
    top = df[df["B_delta"] >= threshold].copy()
    losses.below_topk_threshold = len(df) - len(top)
    top["center"] = (top["span_start"] + top["span_end"]) // 2
    top = top.sort_values("B_delta", ascending=False)

    chosen_centers: dict = {}
    picks = []
    for _, row in top.iterrows():
        if len(picks) >= max_windows:
            break
        rid = row["record_id"]
        centers = chosen_centers.setdefault(rid, [])
        if any(abs(row["center"] - c) < min_sep for c in centers):
            continue
        centers.append(row["center"])
        picks.append(row)

    losses.merged_into_clusters = len(top) - len(picks)
    return pd.DataFrame(picks).reset_index(drop=True), threshold


# --------------------------------------------------------------------------- #
# Window extraction & attribution
# --------------------------------------------------------------------------- #


class NpzCache:
    """Lazily load & cache dependency-map npz files by map_id per run."""

    def __init__(self, base_paths: dict, adapt_paths: dict):
        self.base_paths = base_paths
        self.adapt_paths = adapt_paths
        self._cache: dict = {}

    def get(self, run: str, map_id: str):
        key = (run, map_id)
        if key not in self._cache:
            paths = self.base_paths if run == "base" else self.adapt_paths
            _, _, npz_path = paths[map_id]
            d = np.load(npz_path, allow_pickle=True)
            self._cache[key] = {
                "dependency_map": d["dependency_map"],
                "sequence": np.asarray(d["sequence"]),
            }
        return self._cache[key]


def choose_tile(
    record_tiles: dict,
    record_id: str,
    preferred_map_id: str,
    win_start: int,
    win_end: int,
):
    """Pick a tile fully containing [win_start, win_end).

    Prefer the peak's own tile; otherwise any containing tile. Returns
    (map_id, map_start) or None if no tile contains the window.
    """
    candidates = record_tiles.get(record_id, [])

    def contains(mid, mstart, mlen):
        return mstart <= win_start and win_end <= mstart + mlen

    for mid, mstart, mlen in candidates:
        if mid == preferred_map_id and contains(mid, mstart, mlen):
            return mid, mstart
    # fall back to any containing tile (widest margin first for stability)
    containing = [
        (mid, mstart, mlen)
        for mid, mstart, mlen in candidates
        if contains(mid, mstart, mlen)
    ]
    if not containing:
        return None
    mid, mstart, _ = containing[0]
    return mid, mstart


def compute_centrality(d_delta_window: np.ndarray) -> np.ndarray:
    """s_k = sum_{j!=k} (D[k,j] + D[j,k]) on the cropped window."""
    dz = d_delta_window.copy()
    np.fill_diagonal(dz, 0.0)
    return dz.sum(axis=1) + dz.sum(axis=0)


def one_hot_encode(seq: np.ndarray):
    """(4, L) one-hot. Returns (one_hot, ok) where ok is False if non-ACGT present."""
    L = len(seq)
    oh = np.zeros((4, L), dtype=np.float32)
    for p, ch in enumerate(seq):
        c = str(ch).upper()
        idx = NUC_TO_IDX.get(c)
        if idx is None:
            return oh, False
        oh[idx, p] = 1.0
    return oh, True


def build_window(
    peak: pd.Series,
    L: int,
    cache: NpzCache,
    record_tiles: dict,
    base_paths: dict,
    annotations: dict,
    losses: LossCounters,
):
    """Build one candidate window; return metadata dict + arrays, or None on skip."""
    center = int((peak["span_start"] + peak["span_end"]) // 2)
    win_start = center - L // 2
    win_end = win_start + L  # half-open, length exactly L

    record_id = peak["record_id"]
    chosen = choose_tile(
        record_tiles, record_id, peak["map_id"], win_start, win_end
    )
    if chosen is None:
        losses.lost_tile_boundary += 1
        return None
    map_id, map_start = chosen

    base_npz = cache.get("base", map_id)
    adapt_npz = cache.get("adapt", map_id)

    local = win_start - map_start
    seq_b = base_npz["sequence"][local : local + L]
    seq_a = adapt_npz["sequence"][local : local + L]
    if not np.array_equal(seq_b, seq_a):
        losses.lost_sequence_mismatch += 1
        return None

    oh, ok = one_hot_encode(seq_b)
    if not ok:
        losses.lost_non_acgt += 1
        return None

    dm_b = base_npz["dependency_map"][local : local + L, local : local + L]
    dm_a = adapt_npz["dependency_map"][local : local + L, local : local + L]
    d_delta = np.maximum(0.0, dm_a - dm_b)

    s_k = compute_centrality(d_delta).astype(np.float32)
    contribution = oh * s_k[None, :]  # project onto observed nucleotide channel

    overlaps_ann = overlapping_annotation_types(record_id, win_start, win_end, annotations)
    sequence_str = "".join(str(c).upper() for c in seq_b)

    meta = {
        "record_id": record_id,
        "region_id": peak["region_id"],
        "tile_index": int(peak["tile_index"]),
        "map_id": map_id,
        "window_start": int(win_start),
        "window_end": int(win_end),
        "peak_position": center,
        "peak_B_delta": float(peak["B_delta"]),
        "sequence": sequence_str,
        "overlapping_annotations": ",".join(overlaps_ann),
    }
    return meta, oh, contribution


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #


def run_sanity_checks(one_hot: np.ndarray, contrib: np.ndarray, L: int):
    """Assert TF-MoDISco array invariants. Returns list of human-readable results."""
    results = []
    N = one_hot.shape[0]

    assert one_hot.shape == (N, 4, L), f"one_hot shape {one_hot.shape} != (N,4,{L})"
    results.append(f"one_hot_sequences shape == (N,4,L): {one_hot.shape}")

    assert contrib.shape == (N, 4, L), f"contrib shape {contrib.shape} != (N,4,{L})"
    results.append(f"contribution_scores shape == (N,4,L): {contrib.shape}")

    if N > 0:
        col_sums = one_hot.sum(axis=1)  # (N, L)
        assert np.allclose(col_sums, 1.0), "one-hot columns do not all sum to 1"
        results.append("one-hot columns all sum to 1: OK")

        assert not np.isnan(contrib).any(), "NaNs present in contribution_scores"
        results.append("no NaNs in contribution_scores: OK")

        lengths_ok = one_hot.shape[2] == L
        assert lengths_ok, "window length mismatch"
        results.append(f"all windows length == L ({L}): OK")
    else:
        results.append("N == 0: no arrays to validate (see loss report)")
    return results


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_outputs(
    out_dir: str,
    mode_tag: str,
    metas: list,
    one_hot: np.ndarray,
    contrib: np.ndarray,
    threshold: float,
    losses: LossCounters,
    sanity: list,
    params: dict,
):
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"high_score_windows_{mode_tag}.npz")
    np.savez_compressed(
        npz_path,
        one_hot_sequences=one_hot,
        contribution_scores=contrib,
    )

    meta_df = pd.DataFrame(metas)
    if not meta_df.empty:
        meta_df["exclusion_mode"] = mode_tag
    meta_path = os.path.join(out_dir, f"candidate_metadata_{mode_tag}.parquet")
    meta_df.to_parquet(meta_path, index=False)

    # summary
    n = len(metas)
    b_deltas = meta_df["peak_B_delta"].to_numpy() if n else np.array([])
    n_records = meta_df["record_id"].nunique() if n else 0
    ann_counter: Counter = Counter()
    if n:
        for s in meta_df["overlapping_annotations"]:
            for ft in filter(None, s.split(",")):
                ann_counter[ft] += 1

    summary_path = os.path.join(out_dir, f"candidate_summary_{mode_tag}.tsv")
    with open(summary_path, "w") as fh:
        fh.write("metric\tvalue\n")
        fh.write(f"exclusion_mode\t{mode_tag}\n")
        for k, v in params.items():
            fh.write(f"param.{k}\t{v}\n")
        fh.write(f"num_candidates\t{n}\n")
        fh.write(f"num_records\t{n_records}\n")
        fh.write(f"B_delta_threshold_topk\t{threshold:.6f}\n")
        fh.write(
            f"B_delta_median\t{np.median(b_deltas):.6f}\n" if n else "B_delta_median\tNA\n"
        )
        fh.write(f"B_delta_max\t{b_deltas.max():.6f}\n" if n else "B_delta_max\tNA\n")
        fh.write("--- loss accounting ---\t\n")
        for k, v in vars(losses).items():
            if k == "notes":
                continue
            fh.write(f"{k}\t{v}\n")
        fh.write("--- overlapping annotation distribution ---\t\n")
        for ft, c in ann_counter.most_common():
            fh.write(f"ann.{ft}\t{c}\n")
        fh.write("--- sanity checks ---\t\n")
        for line in sanity:
            fh.write(f"sanity\t{line}\n")

    return npz_path, meta_path, summary_path


# --------------------------------------------------------------------------- #
# Driver for one mode
# --------------------------------------------------------------------------- #


def run_mode(mode: str, args, spans: pd.DataFrame, annotations, cache, record_tiles,
             base_paths):
    print(f"\n{'='*70}\nMODE: {mode}\n{'='*70}")
    losses = LossCounters()
    losses.total_paired_spans = len(spans)

    excl_set = excluded_feature_types(mode, args.exclude_upstream, args.exclude_terminator)

    # 1) annotation exclusion (per span)
    keep_mask = ~spans.apply(
        lambda r: span_is_excluded(
            r["record_id"], r["span_start"], r["span_end"], annotations, excl_set
        ),
        axis=1,
    )
    kept = spans[keep_mask].copy()
    losses.excluded_by_annotation = len(spans) - len(kept)
    print(f"  spans after annotation exclusion: {len(kept)} / {len(spans)}")

    # 2) dedup overlapping tile spans
    kept = dedupe_spans(kept, losses)
    print(f"  spans after dedup: {len(kept)} (removed {losses.removed_by_dedup})")

    # 3) peak selection
    if args.emit == "dense":
        peaks, threshold = select_dense(
            kept, args.top_pct, args.min_sep, args.max_windows, losses
        )
        print(
            f"  dense windows after top-{args.top_pct}% + min_sep {args.min_sep}bp "
            f"(cap {args.max_windows}): {len(peaks)} (threshold B_delta={threshold:.4f})"
        )
    else:
        peaks, threshold = select_peaks(kept, args.top_pct, args.merge_distance, losses)
        print(
            f"  peaks after top-{args.top_pct}% + merge<= {args.merge_distance}bp: "
            f"{len(peaks)} (threshold B_delta={threshold:.4f})"
        )

    # 4) window extraction + attribution
    metas, oh_list, contrib_list = [], [], []
    for _, peak in peaks.iterrows():
        built = build_window(
            peak, args.window, cache, record_tiles, base_paths, annotations, losses
        )
        if built is None:
            continue
        meta, oh, contrib = built
        metas.append(meta)
        oh_list.append(oh)
        contrib_list.append(contrib)

    losses.final_candidates = len(metas)
    L = args.window
    if metas:
        one_hot = np.stack(oh_list).astype(np.float32)
        contrib = np.stack(contrib_list).astype(np.float32)
    else:
        one_hot = np.zeros((0, 4, L), dtype=np.float32)
        contrib = np.zeros((0, 4, L), dtype=np.float32)

    sanity = run_sanity_checks(one_hot, contrib, L)

    params = {
        "top_pct": args.top_pct,
        "window": args.window,
        "merge_distance": args.merge_distance,
        "exclude_upstream": args.exclude_upstream,
        "exclude_terminator": args.exclude_terminator,
    }
    npz_path, meta_path, summary_path = write_outputs(
        args.out_dir, mode, metas, one_hot, contrib, threshold, losses, sanity, params
    )

    print(f"  final candidates: {losses.final_candidates}")
    print(f"  lost to tile boundary: {losses.lost_tile_boundary}, "
          f"non-ACGT: {losses.lost_non_acgt}, seq-mismatch: {losses.lost_sequence_mismatch}")
    for line in sanity:
        print(f"  [sanity] {line}")
    print(f"  wrote: {npz_path}")
    print(f"         {meta_path}")
    print(f"         {summary_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-run", default="outputs/runs/base_promotor")
    p.add_argument("--adapted-run", default="outputs/runs/adapted_promotor")
    p.add_argument("--regions", default="data/prepared/ribosome/regions.parquet")
    p.add_argument("--out-dir", default="outputs/tfmodisco_inputs")
    p.add_argument("--mode", choices=["strict", "motif_only", "both"], default="both")
    p.add_argument("--top-pct", type=float, default=1.0,
                   help="keep top X%% of B_delta spans (default 1.0)")
    p.add_argument("--window", type=int, default=100, help="window length L (default 100)")
    p.add_argument("--merge-distance", type=int, default=12,
                   help="merge peaks within this many bp (default 12)")
    p.add_argument("--emit", choices=["peaks", "dense"], default="peaks",
                   help="peaks: one window per merged cluster; "
                        "dense: many non-redundant windows (more seqlets for TF-MoDISco)")
    p.add_argument("--min-sep", type=int, default=25,
                   help="dense mode: minimum bp between selected window centers")
    p.add_argument("--max-windows", type=int, default=5000,
                   help="dense mode: cap on total emitted windows")
    p.add_argument("--exclude-upstream", action="store_true",
                   help="motif_only: also exclude Upstream")
    p.add_argument("--exclude-terminator", action="store_true",
                   help="motif_only: also exclude Terminator")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("Loading paired per-span block scores ...")
    spans = load_paired_spans(args.base_run, args.adapted_run)
    print(f"  paired spans: {len(spans)}")

    print("Loading annotations ...")
    annotations = load_annotations(args.regions)

    print("Indexing dependency maps ...")
    base_paths = load_map_paths(args.base_run)
    adapt_paths = load_map_paths(args.adapted_run)
    record_tiles = load_record_tiles(args.base_run)
    cache = NpzCache(base_paths, adapt_paths)

    modes = ["strict", "motif_only"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(mode, args, spans, annotations, cache, record_tiles, base_paths)

    print("\nDone. TF-MoDISco was NOT run (input preparation only).")


if __name__ == "__main__":
    main()
