#!/usr/bin/env python
"""Map TF-MoDISco discovered motifs and seqlets back to genomic annotations and coordinates.

This script links the `example_idx` of seqlets in a `modisco_results.h5` file
back to `candidate_metadata_<mode>.parquet`, mapping each seqlet instance to:
  - Source record / species
  - Genomic window coordinates and exact seqlet coordinates [start, end]
  - Independent intersection with each available rDNA annotation class
  - Exact overlapping functional annotations (e.g. EnhancerRepeats, CompositePromoter, unannotated)

Usage:
  python scripts/map_motifs_to_regions.py \\
    --results outputs/tfmodisco_runs/motif_only_results.h5 \\
    --metadata outputs/tfmodisco_inputs/candidate_metadata_motif_only.parquet \\
    --regions data/prepared/ribosome/regions.parquet \\
    --output outputs/tfmodisco_runs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import pandas as pd


def load_annotations(
    regions_path: Path,
) -> Tuple[Dict[str, List[Tuple[int, int, str]]], List[str]]:
    """Load annotations from parquet file.

    Returns:
        (record_id -> list of (start, end, feature_type), sorted unique feature_types).
    """
    if not regions_path.is_file():
        raise FileNotFoundError(f"Annotations parquet file not found: {regions_path}")

    df = pd.read_parquet(regions_path)
    required = ["record_id", "start", "end", "feature_type"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r} in regions parquet: {regions_path}")

    by_rec: Dict[str, List[Tuple[int, int, str]]] = {}
    all_features = sorted(df["feature_type"].unique().tolist())
    for _, row in df.iterrows():
        rid = str(row["record_id"])
        if rid not in by_rec:
            by_rec[rid] = []
        by_rec[rid].append(
            (int(row["start"]), int(row["end"]), str(row["feature_type"]))
        )
    return by_rec, all_features


def intersect_seqlet_annotations(
    record_id: str,
    start: int,
    end: int,
    annotations: Dict[str, List[Tuple[int, int, str]]],
) -> List[str]:
    """Return sorted unique feature_types overlapping the exact seqlet interval [start, end)."""
    if record_id not in annotations:
        return []

    hits = set()
    for a_start, a_end, ftype in annotations[record_id]:
        # Half-open interval intersection test: max(s1, s2) < min(e1, e2)
        if max(start, a_start) < min(end, a_end):
            hits.add(ftype)
    return sorted(hits)


def load_pattern_seqlets(
    h5_file: h5py.File, group_name: str, pattern_name: str
) -> pd.DataFrame:
    """Extract seqlet metadata from a single pattern group in modisco_results.h5."""
    pattern_grp = h5_file[group_name][pattern_name]
    if "seqlets" not in pattern_grp:
        return pd.DataFrame()

    seqlet_grp = pattern_grp["seqlets"]
    example_idxs = seqlet_grp["example_idx"][:]
    starts = seqlet_grp["start"][:]
    ends = seqlet_grp["end"][:]
    is_revcomps = seqlet_grp["is_revcomp"][:]

    return pd.DataFrame({
        "pattern_name": f"{group_name}.{pattern_name}",
        "pattern": pattern_name,
        "example_idx": example_idxs,
        "seqlet_start": starts,
        "seqlet_end": ends,
        "is_revcomp": is_revcomps,
    })


def build_origin_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated breakdown of discovered patterns and their genomic origins.

    Groups mutually exclusive exact seqlet annotation combinations.
    """
    if final_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    ann_series = (
        final_df["overlapping_annotations"].fillna("").replace("", "(unannotated)")
    )
    work_df = final_df.copy()
    work_df["genomic_origin"] = ann_series

    for pattern_name, grp in work_df.groupby("pattern_name", sort=False):
        n_tot = len(grp)
        pattern_short = str(grp["pattern"].iloc[0])

        origin_counts = grp["genomic_origin"].value_counts()
        for origin, cnt in origin_counts.items():
            subgrp = grp[grp["genomic_origin"] == origin]
            pct = round((cnt / n_tot) * 100.0, 2)
            recs = sorted(subgrp["record_id"].unique().tolist())
            mean_b = round(float(subgrp["peak_B_delta"].mean()), 4)

            rows.append({
                "pattern_name": pattern_name,
                "pattern": pattern_short,
                "genomic_origin": str(origin),
                "seqlet_count": cnt,
                "total_pattern_seqlets": n_tot,
                "seqlet_pct": pct,
                "num_records": len(recs),
                "contributing_records": ", ".join(recs),
                "mean_peak_B_delta": mean_b,
            })

    return pd.DataFrame(rows)


def map_motifs(
    results_path: Path,
    metadata_path: Path,
    regions_path: Path,
    output_dir: Path,
    pattern_filter: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map all seqlets from TF-MoDISco results to genomic coordinates and annotations.

    Performs independent intersection of exact seqlet coordinates [start, end]
    against available rDNA annotation classes.
    """
    if not results_path.is_file():
        raise FileNotFoundError(f"TF-MoDISco results file not found: {results_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Candidate metadata parquet file not found: {metadata_path}"
        )

    print(f"Loading annotations from: {regions_path}")
    annotations, all_features = load_annotations(regions_path)
    print(
        f"  Loaded annotations for {len(annotations)} records across {len(all_features)} feature classes: {all_features}"
    )

    print(f"Loading candidate metadata from: {metadata_path}")
    meta_df = pd.read_parquet(metadata_path)

    required_meta_cols = [
        "record_id",
        "window_start",
        "window_end",
        "peak_B_delta",
        "overlapping_annotations",
    ]
    for col in required_meta_cols:
        if col not in meta_df.columns:
            raise KeyError(
                f"Missing required column {col!r} in metadata parquet: {metadata_path}"
            )

    print(f"Loading TF-MoDISco patterns from: {results_path}\n")
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

                seqlets = load_pattern_seqlets(h5, group_name, pattern_name)
                if not seqlets.empty:
                    all_seqlets.append(seqlets)

    if not all_seqlets:
        print("No seqlets found matching pattern filter.")
        empty = pd.DataFrame()
        return empty, empty

    combined_seqlets = pd.concat(all_seqlets, ignore_index=True)

    # Validate that example_idx values are within metadata bounds
    max_idx = int(combined_seqlets["example_idx"].max())
    if max_idx >= len(meta_df):
        raise IndexError(
            f"Seqlet example_idx {max_idx} exceeds metadata length {len(meta_df)}. "
            "Ensure the metadata parquet matches the input arrays used for this MoDISco run."
        )

    # Join seqlets with candidate window metadata by example_idx
    meta_sub = meta_df.iloc[
        combined_seqlets["example_idx"].to_numpy()
    ].reset_index(drop=True)

    # Merge into a comprehensive seqlet annotation frame
    joined = pd.concat([combined_seqlets, meta_sub], axis=1)

    # Compute exact genomic coordinates for each seqlet [start, end]
    joined["seqlet_genomic_start"] = (
        joined["window_start"] + joined["seqlet_start"]
    )
    joined["seqlet_genomic_end"] = joined["window_start"] + joined["seqlet_end"]

    # Preserve window-level overlapping annotations for reference
    joined["window_overlapping_annotations"] = joined["overlapping_annotations"]

    # Intersect exact seqlet coordinates against annotations
    exact_overlaps: List[str] = []
    for _, row in joined.iterrows():
        rec = str(row["record_id"])
        s = int(row["seqlet_genomic_start"])
        e = int(row["seqlet_genomic_end"])
        ov = intersect_seqlet_annotations(rec, s, e, annotations)
        exact_overlaps.append(",".join(ov) if ov else "(unannotated)")

    joined["overlapping_annotations"] = exact_overlaps

    # Compute independent binary indicator columns for each available annotation class
    indicator_cols = []
    for feat in all_features:
        col_name = f"is_{feat}"
        joined[col_name] = [
            int(feat in ann.split(","))
            for ann in joined["overlapping_annotations"]
        ]
        indicator_cols.append(col_name)

    joined["is_unannotated"] = [
        int(ann == "(unannotated)")
        for ann in joined["overlapping_annotations"]
    ]
    indicator_cols.append("is_unannotated")

    # Reorder columns for readability
    ordered_cols = [
        "pattern_name",
        "pattern",
        "example_idx",
        "record_id",
        "seqlet_genomic_start",
        "seqlet_genomic_end",
        "is_revcomp",
        "overlapping_annotations",
        "window_overlapping_annotations",
        "peak_B_delta",
        "window_start",
        "window_end",
        "seqlet_start",
        "seqlet_end",
    ]
    extra_cols = [
        c
        for c in joined.columns
        if c not in ordered_cols and c not in indicator_cols
    ]
    final_df = joined[ordered_cols + indicator_cols + extra_cols]

    # Compute granular origin breakdown summary table
    summary_df = build_origin_summary(final_df)

    # Print summary per pattern to stdout
    print("=" * 80)
    print("MOTIF PATTERN TO REGION BREAKDOWN (EXACT SEQLET INTERSECTION)")
    print("=" * 80)

    for pattern_name, grp in final_df.groupby("pattern_name", sort=False):
        n_seqlets = len(grp)
        n_records = grp["record_id"].nunique()
        print(
            f"\nPattern: {pattern_name} (Total Seqlets: {n_seqlets}, Records: {n_records})"
        )
        print("-" * 60)

        print("  [Exact Origin Combinations]")
        ann_series = (
            grp["overlapping_annotations"]
            .fillna("")
            .replace("", "(unannotated)")
        )
        counts = ann_series.value_counts()
        for ann_label, cnt in counts.items():
            pct = (cnt / n_seqlets) * 100.0
            print(f"    {ann_label:45s}: {cnt:4d} ({pct:5.1f}%)")

        print("\n  [Independent Feature Class Intersections]")
        for feat in all_features:
            cnt = int(grp[f"is_{feat}"].sum())
            if cnt > 0:
                pct = (cnt / n_seqlets) * 100.0
                print(f"    {feat:45s}: {cnt:4d} ({pct:5.1f}%)")
        unann_cnt = int(grp["is_unannotated"].sum())
        if unann_cnt > 0:
            pct = (unann_cnt / n_seqlets) * 100.0
            print(f"    {'(unannotated)':45s}: {unann_cnt:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)

    # Save output TSV files inside output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seqlet_tsv = output_dir / "seqlet_annotations_overlap.tsv"
    final_df.to_csv(seqlet_tsv, sep="\t", index=False)
    print(f"Detailed per-seqlet annotations written to: {seqlet_tsv}")

    summary_tsv = output_dir / "seqlet_annotations_overlap_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    print(f"Granular origin breakdown written to: {summary_tsv}\n")

    return final_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/tfmodisco_runs/motif_only_results.h5"),
        help="Path to TF-MoDISco HDF5 results file (default: outputs/tfmodisco_runs/motif_only_results.h5)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "outputs/tfmodisco_inputs/candidate_metadata_motif_only.parquet"
        ),
        help="Path to candidate metadata parquet file (default: outputs/tfmodisco_inputs/candidate_metadata_motif_only.parquet)",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=Path("data/prepared/ribosome/regions.parquet"),
        help="Path to annotations parquet file (default: data/prepared/ribosome/regions.parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/tfmodisco_runs"),
        help="Output directory to write seqlet_annotations_overlap.tsv and seqlet_annotations_overlap_summary.tsv (default: outputs/tfmodisco_runs)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Filter for a specific pattern name (e.g. 'pos_patterns.pattern_0'). Default: all patterns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_motifs(
        results_path=args.results,
        metadata_path=args.metadata,
        regions_path=args.regions,
        output_dir=args.output,
        pattern_filter=args.pattern,
    )


if __name__ == "__main__":
    main()
