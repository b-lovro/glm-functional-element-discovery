#!/usr/bin/env python
"""Map TF-MoDISco discovered motifs and seqlets back to genomic annotations and coordinates.

This script links the `example_idx` of seqlets in a `modisco_results.h5` file
back to `candidate_metadata_<mode>.parquet`, mapping each seqlet instance to:
  - Source record / species
  - Genomic window coordinates and exact seqlet coordinates [start, end]
  - Independent intersection with each available rDNA annotation class
  - Exact overlapping functional annotations (e.g. EnhancerRepeats, CompositePromoter, unannotated)
  - Canonical 7 biological rDNA compartments (Core Promoter, UCE, 5' ETS, ITS1, ITS2, Terminator, Distal IGS)
  - Regional fold-enrichment matrix and companion publication heatmap

Usage:
  python scripts/map_motifs_to_regions.py \\
    --results outputs/tfmodisco_runs/full_tuned/motif_only_results.h5 \\
    --metadata outputs/tfmodisco_inputs/full_no_holdout/candidate_metadata.parquet \\
    --regions data/prepared/ribosome/regions.parquet \\
    --output outputs/tfmodisco_runs/full_tuned/
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Ground-truth biological rDNA annotation ordering and clean display names
ANNOTATION_DISPLAY_NAMES: Dict[str, str] = {
    "EnhancerRepeats": "Enhancer\nRepeats",
    "Upstream": "Upstream\nUCE",
    "Core": "Core\nPromoter",
    "CompositePromoter": "Composite\nPromoter",
    "18S_rRNA": "18S\nrRNA",
    "5_8S_rRNA": "5.8S\nrRNA",
    "28S_rRNA": "28S\nrRNA",
    "Terminator": "28S\nTerminator",
    "TTF1": "TTF1\nSite",
    "CTCF": "CTCF\nSite",
    "unannotated": "Distal IGS\n(Unannotated)",
}

ANNOTATION_ORDER: List[str] = [
    "EnhancerRepeats",
    "Upstream",
    "Core",
    "CompositePromoter",
    "18S_rRNA",
    "5_8S_rRNA",
    "28S_rRNA",
    "Terminator",
    "TTF1",
    "CTCF",
    "unannotated",
]


# Candidate attribution windows background proportions (Option 2: Search space N=2,238)
CANDIDATE_BG_PROPS: Dict[str, float] = {
    "Core_Promoter_TSS": 0.0143,
    "Upstream_Promoter": 0.0786,
    "5prime_ETS": 0.0970,
    "ITS1": 0.0987,
    "ITS2": 0.0800,
    "Terminator": 0.0474,
    "Distal_IGS": 0.5840,
}

COMP_LABELS: List[str] = [
    "Core_Promoter_TSS",
    "Upstream_Promoter",
    "5prime_ETS",
    "ITS1",
    "ITS2",
    "Terminator",
    "Distal_IGS",
]

COMP_DISPLAY_NAMES: List[str] = [
    "Core Prom. / TSS\n(-120..+100 bp)",
    "UCE / Upstream\n(-2k..-120 bp)",
    "5' ETS\n(+100..+1.5k bp)",
    "ITS1 Spacer\n(18S-5.8S)",
    "ITS2 Spacer\n(5.8S-28S)",
    "28S Terminator\n(0..+1.5k bp)",
    "Distal IGS\n(Intergenic)",
]


def pattern_sort_key(name: str) -> Tuple[int, int]:
    """Extract (is_neg, pattern_index) for numerical sorting (pos_patterns first, then neg_patterns)."""
    is_neg = 1 if name.startswith("neg_") else 0
    parts = name.split(".")[-1].split("_")
    return (is_neg, int(parts[-1]))


def format_pattern_display_name(name: str) -> str:
    """Format pattern identifier cleanly, preserving neg_ prefix if from neg_patterns."""
    if name.startswith("neg_patterns."):
        return f"neg_{name.split('.')[-1]}"
    return name.split(".")[-1]


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


def extract_seqlet_landmarks(
    record_ids: List[str],
    regions_path: Path,
) -> Dict[str, Dict[str, float]]:
    """Extract TSS, 28S 3'-end, ITS1, and ITS2 boundaries for all contributing records."""
    if not regions_path.is_file():
        raise FileNotFoundError(f"Regions parquet not found at: {regions_path}")

    regions_df = pd.read_parquet(regions_path)
    required = ["record_id", "feature_type", "start", "end"]
    for col in required:
        if col not in regions_df.columns:
            raise KeyError(f"Required column '{col}' missing from regions parquet: {regions_path}")

    landmarks: Dict[str, Dict[str, float]] = {}
    for rec_id in record_ids:
        sub = regions_df[regions_df["record_id"] == rec_id]
        s28 = sub[sub["feature_type"] == "28S_rRNA"]
        s58 = sub[sub["feature_type"] == "5_8S_rRNA"]
        s18 = sub[sub["feature_type"] == "18S_rRNA"]
        cores = sub[sub["feature_type"] == "Core"]
        comps = sub[sub["feature_type"] == "CompositePromoter"]

        term_pos = float(s28["end"].max()) if len(s28) > 0 else np.nan
        s28_start = float(s28["start"].min()) if len(s28) > 0 else np.nan
        s58_upstream = s58[s58["end"] <= s28_start] if not np.isnan(s28_start) else s58
        s58_end = float(s58_upstream["end"].max()) if len(s58_upstream) > 0 else np.nan
        s58_start = float(s58_upstream["start"].min()) if len(s58_upstream) > 0 else np.nan
        s18_upstream = s18[s18["end"] <= s58_start] if not np.isnan(s58_start) else s18
        s18_end = float(s18_upstream["end"].max()) if len(s18_upstream) > 0 else np.nan

        tss_pos = float(cores["end"].max()) if len(cores) > 0 else (float(comps["end"].max()) if len(comps) > 0 else np.nan)

        landmarks[rec_id] = {
            "tss_pos": tss_pos,
            "term_pos": term_pos,
            "its1_start": s18_end,
            "its1_end": s58_start,
            "its2_start": s58_end,
            "its2_end": s28_start,
        }
    return landmarks


def assign_seqlet_compartments(
    df: pd.DataFrame,
    landmarks: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Assigns each seqlet to one of 7 canonical biological rDNA compartments."""
    midpoints = (df["seqlet_genomic_start"] + df["seqlet_genomic_end"]) / 2.0
    df["seqlet_midpoint"] = midpoints

    fine_compartments: List[str] = []
    dist_tss_list: List[float] = []
    dist_term_list: List[float] = []

    for _, row in df.iterrows():
        rec = str(row["record_id"])
        lm = landmarks[rec]
        mid = float(row["seqlet_midpoint"])
        d_tss = mid - lm["tss_pos"] if not np.isnan(lm["tss_pos"]) else np.nan
        d_term = mid - lm["term_pos"] if not np.isnan(lm["term_pos"]) else np.nan
        dist_tss_list.append(d_tss)
        dist_term_list.append(d_term)

        in_its1 = (not np.isnan(lm["its1_start"])) and (not np.isnan(lm["its1_end"])) and (mid >= lm["its1_start"]) and (mid <= lm["its1_end"])
        in_its2 = (not np.isnan(lm["its2_start"])) and (not np.isnan(lm["its2_end"])) and (mid >= lm["its2_start"]) and (mid <= lm["its2_end"])

        if not np.isnan(d_tss) and (-120.0 <= d_tss <= 100.0):
            c = "Core_Promoter_TSS"
        elif not np.isnan(d_tss) and (-2000.0 <= d_tss < -120.0):
            c = "Upstream_Promoter"
        elif not np.isnan(d_tss) and (100.0 < d_tss <= 1500.0):
            c = "5prime_ETS"
        elif in_its1:
            c = "ITS1"
        elif in_its2:
            c = "ITS2"
        elif not np.isnan(d_term) and (0.0 <= d_term <= 1500.0):
            c = "Terminator"
        else:
            c = "Distal_IGS"
        fine_compartments.append(c)

    df["dist_to_tss"] = dist_tss_list
    df["dist_to_28s_end"] = dist_term_list
    df["fine_compartment"] = fine_compartments

    for comp in COMP_LABELS:
        df[f"in_comp_{comp}"] = (df["fine_compartment"] == comp).astype(int)

    return df


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


def build_origin_summary(
    final_df: pd.DataFrame,
    feature_classes: List[str] | None = None,
    include_percentages: bool = True,
) -> pd.DataFrame:
    """Build aggregated breakdown of discovered patterns and their annotation overlaps.

    Produces one row per pattern, with count (and optional percentage) columns for each
    independent annotation feature class present in the dataset (plus unannotated).
    """
    if final_df.empty:
        return pd.DataFrame()

    if feature_classes is None:
        features = sorted([
            c[3:]
            for c in final_df.columns
            if c.startswith("is_") and c not in ("is_revcomp", "is_unannotated") and c not in [f"is_{cl}" for cl in COMP_LABELS]
        ])
    else:
        features = list(feature_classes)

    all_feats = [f for f in features if f != "unannotated"]
    all_feats.append("unannotated")

    rows: List[Dict[str, Any]] = []
    pattern_names = sorted(final_df["pattern_name"].unique(), key=pattern_sort_key)
    for pattern_name in pattern_names:
        grp = final_df[final_df["pattern_name"] == pattern_name]
        n_tot = len(grp)
        pattern_short = str(grp["pattern"].iloc[0])
        recs = sorted(grp["record_id"].unique().tolist())
        mean_b = round(float(grp["peak_B_delta"].mean()), 4)

        row: Dict[str, Any] = {
            "pattern_name": pattern_name,
            "pattern": pattern_short,
            "total_pattern_seqlets": n_tot,
            "num_records": len(recs),
            "mean_peak_B_delta": mean_b,
        }

        for feat in all_feats:
            col_name = f"is_{feat}"
            if col_name in grp.columns:
                cnt = int(grp[col_name].sum())
            else:
                cnt = 0
            row[f"{feat}_count"] = cnt
            if include_percentages:
                row[f"{feat}_pct"] = (
                    round((cnt / n_tot) * 100.0, 2) if n_tot > 0 else 0.0
                )

        row["contributing_records"] = ", ".join(recs)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_candidate_search_space_background(
    meta_df: pd.DataFrame,
    regions_path: Path,
) -> Dict[str, float]:
    """Compute empirical background compartment distribution of candidate attribution windows."""
    meta_copy = meta_df.copy()
    records = sorted(meta_copy["record_id"].unique().tolist())
    landmarks = extract_seqlet_landmarks(records, regions_path)
    meta_copy["seqlet_genomic_start"] = meta_copy["window_start"]
    meta_copy["seqlet_genomic_end"] = meta_copy["window_end"]
    meta_copy["seqlet_midpoint"] = (meta_copy["window_start"] + meta_copy["window_end"]) / 2.0
    meta_comp = assign_seqlet_compartments(meta_copy, landmarks)

    comp_props = meta_comp["fine_compartment"].value_counts(normalize=True).to_dict()
    bg_props: Dict[str, float] = {}
    for c in COMP_LABELS:
        if c in comp_props:
            bg_props[c] = float(comp_props[c])
        else:
            bg_props[c] = 1e-4

    tot_sum = sum(bg_props.values())
    for c in COMP_LABELS:
        bg_props[c] = bg_props[c] / tot_sum

    return bg_props


def build_seqlet_compartment_summary(
    df: pd.DataFrame,
    bg_props: Dict[str, float],
    comp_labels: List[str],
    prior_weight: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates seqlet compartment breakdown and log2(Obs/Exp) fold-enrichment matrix using Bayesian prior."""
    pattern_names = sorted(df["pattern_name"].unique(), key=pattern_sort_key)
    fe_mat = pd.DataFrame(index=pattern_names, columns=comp_labels, dtype=float)
    rows: List[Dict[str, Any]] = []

    for pat in pattern_names:
        sub = df[df["pattern_name"] == pat]
        n_tot = len(sub)
        pattern_short = str(sub["pattern"].iloc[0])
        recs = sorted(sub["record_id"].unique().tolist())
        n_recs = len(recs)
        mean_b = round(float(sub["peak_B_delta"].mean()), 4)

        row: Dict[str, Any] = {
            "pattern_name": pat,
            "pattern": pattern_short,
            "total_seqlets": n_tot,
            "num_records": n_recs,
            "mean_peak_B_delta": mean_b,
        }

        # Calculate fold-enrichment using Bayesian Dirichlet prior proportional to bg_props[c]
        for c in comp_labels:
            obs_cnt = int((sub["fine_compartment"] == c).sum())
            exp_prop = bg_props[c]
            obs_prop = (obs_cnt + prior_weight * exp_prop) / (n_tot + prior_weight)
            fe_val = float(np.log2(obs_prop / exp_prop))
            fe_mat.loc[pat, c] = fe_val
            row[f"{c}_count"] = obs_cnt
            row[f"{c}_pct"] = round((obs_cnt / n_tot) * 100.0, 2) if n_tot > 0 else 0.0

        primary_comp = str(fe_mat.loc[pat].idxmax())
        primary_fe = round(float(fe_mat.loc[pat].max()), 2)
        row["primary_enriched_compartment"] = primary_comp
        row["primary_enriched_log2_fe"] = primary_fe
        row["contributing_records"] = ", ".join(recs)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df, fe_mat


def plot_seqlet_fold_enrichment_heatmap(
    summary_df: pd.DataFrame,
    fe_mat: pd.DataFrame,
    comp_display_names: List[str],
    output_png: Path,
) -> None:
    """Regional Fold-Enrichment Heatmap with integrated Seqlet Counts per Pattern."""
    pattern_order = fe_mat.index.tolist()
    num_patterns = len(pattern_order)
    display_names = [format_pattern_display_name(p) for p in pattern_order]
    summary_sorted = summary_df.set_index("pattern_name").loc[pattern_order].reset_index()

    fig_height = max(9.0, 0.36 * num_patterns + 2.2)
    cbar_shrink = min(0.8, 10.0 / fig_height)

    fig = plt.figure(figsize=(15, fig_height), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 4.7], wspace=0.08)

    ax_cnt = fig.add_subplot(gs[0])
    ax_hm = fig.add_subplot(gs[1])

    y_pos = np.arange(num_patterns)
    seq_cnt = summary_sorted["total_seqlets"].values
    rec_cnt = summary_sorted["num_records"].values

    ax_cnt.barh(y_pos, seq_cnt, color="#2c7bb6", alpha=0.88, height=0.75, edgecolor="#222222", linewidth=0.6)
    max_cnt = max(seq_cnt) if len(seq_cnt) > 0 else 50
    ax_cnt.set_xlim(0, max_cnt * 1.38)
    ax_cnt.set_ylim(-0.5, num_patterns - 0.5)
    ax_cnt.invert_yaxis()
    ax_cnt.set_yticks(y_pos)
    ax_cnt.set_yticklabels(display_names, fontsize=9.5, fontweight="bold")
    ax_cnt.set_xlabel("Seqlet Count\n[Contributing Records]", fontsize=9.5, fontweight="bold")
    ax_cnt.tick_params(axis="x", top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=8.5)
    ax_cnt.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (sc, rc) in enumerate(zip(seq_cnt, rec_cnt)):
        ax_cnt.text(
            sc + (max_cnt * 0.02),
            i,
            f"{sc} [{rc} sp]",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )

    # Right: Log2 Fold-Enrichment Heatmap
    sns.heatmap(
        fe_mat,
        ax=ax_hm,
        cmap="vlag",
        center=0.0,
        vmin=-4.0,
        vmax=4.0,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8.5, "weight": "bold"},
        cbar_kws={
            "label": "$\\log_2(\\mathrm{Observed} / \\mathrm{Expected})$ Fold-Enrichment",
            "shrink": cbar_shrink,
            "aspect": 25,
        },
        linewidths=0.6,
        linecolor="#dddddd",
        yticklabels=False,
    )
    ax_hm.set_xticklabels(comp_display_names, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    ax_hm.set_ylabel("")

    if num_patterns > 25:
        secax = ax_hm.secondary_xaxis("top")
        secax.set_xticks(np.arange(len(comp_display_names)) + 0.5)
        secax.set_xticklabels(comp_display_names, rotation=30, ha="left", fontsize=9.5, fontweight="bold")

    top_edge = ax_hm.get_position().y1
    title_y = top_edge + (1.45 / fig_height) if num_patterns > 25 else 0.98
    fig.suptitle(
        "TF-MoDISco Seqlet Regional Fold-Enrichment & Compartment Distribution\n"
        "[Left: Seqlet Counts & Contributing Records | Right: Fold-Enrichment relative to Candidate Attribution Windows (Search Space)]",
        fontsize=12,
        fontweight="bold",
        y=title_y,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Seqlet regional fold-enrichment heatmap saved to: {output_png}")


def plot_seqlet_counts_percentages_table(
    summary_df: pd.DataFrame,
    comp_labels: List[str],
    comp_display_names: List[str],
    output_png: Path,
) -> None:
    """Regional Counts & Percentages Heatmap Table with integrated Total Seqlets per Pattern."""
    pattern_order = summary_df["pattern_name"].tolist()
    num_patterns = len(pattern_order)
    display_names = [format_pattern_display_name(p) for p in pattern_order]
    summary_sorted = summary_df.set_index("pattern_name").loc[pattern_order].reset_index()

    pct_mat = pd.DataFrame(index=pattern_order, columns=comp_labels, dtype=float)
    annot_mat = np.empty((num_patterns, len(comp_labels)), dtype=object)

    for i, pat in enumerate(pattern_order):
        row = summary_sorted[summary_sorted["pattern_name"] == pat].iloc[0]
        for j, c in enumerate(comp_labels):
            cnt = int(row[f"{c}_count"])
            pct = float(row[f"{c}_pct"])
            pct_mat.loc[pat, c] = pct
            annot_mat[i, j] = f"{cnt} ({pct:.1f}%)"

    fig_height = max(9.0, 0.36 * num_patterns + 2.2)
    cbar_shrink = min(0.8, 10.0 / fig_height)

    fig = plt.figure(figsize=(15, fig_height), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 4.7], wspace=0.08)

    ax_cnt = fig.add_subplot(gs[0])
    ax_hm = fig.add_subplot(gs[1])

    y_pos = np.arange(num_patterns)
    seq_cnt = summary_sorted["total_seqlets"].values
    rec_cnt = summary_sorted["num_records"].values

    ax_cnt.barh(y_pos, seq_cnt, color="#2c7bb6", alpha=0.88, height=0.75, edgecolor="#222222", linewidth=0.6)
    max_cnt = max(seq_cnt) if len(seq_cnt) > 0 else 50
    ax_cnt.set_xlim(0, max_cnt * 1.38)
    ax_cnt.set_ylim(-0.5, num_patterns - 0.5)
    ax_cnt.invert_yaxis()
    ax_cnt.set_yticks(y_pos)
    ax_cnt.set_yticklabels(display_names, fontsize=9.5, fontweight="bold")
    ax_cnt.set_xlabel("Seqlet Count\n[Contributing Records]", fontsize=9.5, fontweight="bold")
    ax_cnt.tick_params(axis="x", top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=8.5)
    ax_cnt.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (sc, rc) in enumerate(zip(seq_cnt, rec_cnt)):
        ax_cnt.text(
            sc + (max_cnt * 0.02),
            i,
            f"{sc} [{rc} sp]",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )

    # Right: Percentage Heatmap with annotated Count & Percentage
    sns.heatmap(
        pct_mat,
        ax=ax_hm,
        cmap="Blues",
        vmin=0.0,
        vmax=65.0,
        annot=annot_mat,
        fmt="",
        annot_kws={"fontsize": 8.0, "weight": "bold"},
        cbar_kws={
            "label": "Seqlet Percentage in Compartment (%)",
            "shrink": cbar_shrink,
            "aspect": 25,
        },
        linewidths=0.6,
        linecolor="#dddddd",
        yticklabels=False,
    )
    ax_hm.set_xticklabels(comp_display_names, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    ax_hm.set_ylabel("")

    if num_patterns > 25:
        secax = ax_hm.secondary_xaxis("top")
        secax.set_xticks(np.arange(len(comp_display_names)) + 0.5)
        secax.set_xticklabels(comp_display_names, rotation=30, ha="left", fontsize=9.5, fontweight="bold")

    top_edge = ax_hm.get_position().y1
    title_y = top_edge + (1.45 / fig_height) if num_patterns > 25 else 0.98
    fig.suptitle(
        "TF-MoDISco Seqlet Compartment Breakdown: Counts & Percentages\n"
        "[Left: Total Pattern Seqlets & Records | Right: Seqlet Count & Percentage (%) per Biological Compartment]",
        fontsize=12,
        fontweight="bold",
        y=title_y,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Seqlet compartment counts & percentages table saved to: {output_png}")


def plot_seqlet_annotations_overlap_table(
    summary_df: pd.DataFrame,
    output_png: Path,
    dpi: int = 300,
) -> None:
    """Render a clean, uncolored publication table of exact seqlet counts and percentages across rDNA annotations."""
    if summary_df.empty:
        return
    df = summary_df.copy()
    pattern_order = df["pattern_name"].tolist()

    # Exact annotation names from the dataset in logical biological order
    exact_annots = [a for a in ANNOTATION_ORDER if f"{a}_pct" in df.columns]

    num_patterns = len(df)
    annot_mat = np.empty((num_patterns, len(exact_annots)), dtype=object)

    for i, pat in enumerate(pattern_order):
        row = df[df["pattern_name"] == pat].iloc[0]
        for j, a in enumerate(exact_annots):
            cnt = int(row[f"{a}_count"])
            pct = float(row[f"{a}_pct"])
            if cnt == 0:
                annot_mat[i, j] = "-"
            else:
                annot_mat[i, j] = f"{cnt}\n({pct:.1f}%)"

    fig_height = max(9.5, 0.45 * num_patterns + 2.2)
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=dpi)
    cmap_white = ListedColormap(["#ffffff"])

    y_labels = [
        f"{format_pattern_display_name(str(row['pattern_name']))} (N={int(row['total_pattern_seqlets'])})"
        for _, row in df.iterrows()
    ]

    sns.heatmap(
        np.zeros((num_patterns, len(exact_annots))),
        ax=ax,
        cmap=cmap_white,
        cbar=False,
        linewidths=0.8,
        linecolor="#d0d0d0",
        annot=annot_mat,
        fmt="",
        annot_kws={"fontsize": 8.0, "weight": "bold", "color": "#111111"},
        yticklabels=y_labels,
        xticklabels=exact_annots,
    )
    ax.set_xticklabels(exact_annots, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9.5)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel("")

    if num_patterns > 25:
        secax = ax.secondary_xaxis("top")
        secax.set_xticks(np.arange(len(exact_annots)) + 0.5)
        secax.set_xticklabels(exact_annots, rotation=25, ha="left", fontsize=9.5, fontweight="bold")

    top_edge = ax.get_position().y1
    title_y = top_edge + (1.15 / fig_height) if num_patterns > 25 else 0.985
    fig.suptitle(
        "TF-MoDISco Seqlet Overlap with Ground-Truth rDNA Functional Annotations",
        fontsize=13,
        fontweight="bold",
        y=title_y,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Seqlet annotations overlap summary table saved to: {output_png}")


def map_motifs(
    results_path: Path,
    metadata_path: Path,
    regions_path: Path,
    output_dir: Path,
    pattern_filter: str | None = None,
    include_percentages: bool = True,
    plot_heatmap: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map all seqlets from TF-MoDISco results to genomic coordinates, annotations, and compartments."""
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
        return empty, empty, empty

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

    # Extract landmarks and assign canonical 7 biological compartments
    unique_records = sorted(joined["record_id"].unique().tolist())
    landmarks = extract_seqlet_landmarks(unique_records, regions_path)
    joined = assign_seqlet_compartments(joined, landmarks)

    compartment_cols = [f"in_comp_{c}" for c in COMP_LABELS]

    # Reorder columns for readability
    ordered_cols = [
        "pattern_name",
        "pattern",
        "example_idx",
        "record_id",
        "seqlet_genomic_start",
        "seqlet_genomic_end",
        "seqlet_midpoint",
        "fine_compartment",
        "dist_to_tss",
        "dist_to_28s_end",
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
        if c not in ordered_cols and c not in indicator_cols and c not in compartment_cols
    ]
    final_df = joined[ordered_cols + indicator_cols + compartment_cols + extra_cols]

    # 1. Compute origin breakdown summary table (annotation features)
    summary_df = build_origin_summary(
        final_df,
        feature_classes=all_features,
        include_percentages=include_percentages,
    )

    # 2. Compute biological compartment summary table
    bg_props = compute_candidate_search_space_background(meta_df, regions_path)
    print(f"Empirical Candidate Search Space Background Proportions (N={len(meta_df)}):")
    for c in COMP_LABELS:
        print(f"  {c}: {bg_props[c]*100.0:.2f}%")

    comp_summary_df, fe_mat = build_seqlet_compartment_summary(
        final_df,
        bg_props=bg_props,
        comp_labels=COMP_LABELS,
    )

    # Print summary per pattern to stdout
    print("=" * 85)
    print("MOTIF PATTERN TO REGIONAL COMPARTMENTS (EXACT SEQLET BREAKDOWN)")
    print("=" * 85)

    cols_to_show = [
        "pattern",
        "total_seqlets",
        "num_records",
        "primary_enriched_compartment",
        "primary_enriched_log2_fe",
        "Core_Promoter_TSS_count",
        "Upstream_Promoter_count",
        "5prime_ETS_count",
        "ITS1_count",
        "ITS2_count",
        "Terminator_count",
        "Distal_IGS_count",
    ]
    print(comp_summary_df[cols_to_show].to_string(index=False))
    print("=" * 85)

    # Save output files inside output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seqlet_tsv = output_dir / "seqlet_annotations_overlap.tsv"
    final_df.to_csv(seqlet_tsv, sep="\t", index=False)
    print(f"\nDetailed per-seqlet annotations written to: {seqlet_tsv}")

    summary_tsv = output_dir / "seqlet_annotations_overlap_summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    print(f"Annotation overlap breakdown written to: {summary_tsv}")

    comp_summary_tsv = output_dir / "seqlet_compartment_overlap_summary.tsv"
    comp_summary_df.to_csv(comp_summary_tsv, sep="\t", index=False)
    print(f"Seqlet compartment summary written to: {comp_summary_tsv}")

    if plot_heatmap:
        heatmap_png = output_dir / "seqlet_regional_fold_enrichment_heatmap.png"
        print(f"Generating Seqlet Regional Fold-Enrichment Heatmap to: {heatmap_png}...")
        plot_seqlet_fold_enrichment_heatmap(
            summary_df=comp_summary_df,
            fe_mat=fe_mat,
            comp_display_names=COMP_DISPLAY_NAMES,
            output_png=heatmap_png,
        )

        counts_png = output_dir / "seqlet_regional_counts_percentages_table.png"
        print(f"Generating Seqlet Regional Counts & Percentages Table to: {counts_png}...")
        plot_seqlet_counts_percentages_table(
            summary_df=comp_summary_df,
            comp_labels=COMP_LABELS,
            comp_display_names=COMP_DISPLAY_NAMES,
            output_png=counts_png,
        )

        annot_table_png = output_dir / "seqlet_annotations_overlap_summary_table.png"
        print(f"Generating Seqlet Annotations Overlap Summary Table to: {annot_table_png}...")
        plot_seqlet_annotations_overlap_table(
            summary_df=summary_df,
            output_png=annot_table_png,
        )

    return final_df, summary_df, comp_summary_df


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
        help="Output directory to write TSV and PNG files (default: outputs/tfmodisco_runs)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Filter for a specific pattern name (e.g. 'pos_patterns.pattern_0'). Default: all patterns.",
    )
    parser.add_argument(
        "--include-percentages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include percentage columns alongside counts in the summary table (default: True)",
    )
    parser.add_argument(
        "--plot-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate the publication fold-enrichment heatmap PNG (default: True)",
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
        include_percentages=args.include_percentages,
        plot_heatmap=args.plot_heatmap,
    )


if __name__ == "__main__":
    main()
