#!/usr/bin/env python3
"""
Classify TF-MoDISco discovered patterns by modality match (RNA vs DNA vs No close match).

This script inspects TOMTOM motif matching reports across multiple reference datasets
(e.g., JASPAR, HOCOMOCO, DFAM as DNA; Ray2013 RBP as RNA) generated for a TF-MoDISco run
and partitions all discovered patterns into 3 mutually exclusive lists:
1. RNA_match: Patterns whose best significant match is from an RNA reference dataset
   (e.g., RNA-binding protein motifs).
2. DNA_match: Patterns whose best significant match is from a DNA reference dataset
   (e.g., transcription factor motifs or repetitive DNA elements).
3. No_close_match: Patterns that lack a statistically significant match in either modality
   (e.g., p-value exceeds significance threshold in both DNA and RNA reference sets,
   such as pattern_0 in distal IGS).

Usage:
  # Standard execution using defaults on full_tuned run:
  python scripts/classify_motif_matches.py \
    --input-dir outputs/tfmodisco_runs/full_tuned

  # Filter ubiquitous CTCF hits and require strict p <= 1e-5:
  python scripts/classify_motif_matches.py \
    --input-dir outputs/tfmodisco_runs/full_tuned \
    --filter-ctcf \
    --p-threshold 1e-5
"""

from __future__ import annotations

import os
# Set matplotlib cache directory to writable location before importing pyplot
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configure module logger
logger = logging.getLogger("classify_motif_matches")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_global_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across random and numpy."""
    random.seed(seed)
    np.random.seed(seed)


def parse_motifs_html_table(report_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Parse motifs.html table produced by modisco-lite report.

    Args:
        report_dir: Path to directory containing motifs.html.
        dataset_name: Human-readable identifier for the dataset.

    Returns:
        DataFrame containing parsed columns from the HTML report.

    Raises:
        FileNotFoundError: If motifs.html is not found.
        ValueError: If table parsing fails or HTML structure is invalid.
    """
    html_path = report_dir / "motifs.html"
    if not html_path.is_file():
        raise FileNotFoundError(
            f"Report file not found for dataset '{dataset_name}': {html_path}"
        )

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No <table> found in HTML report: {html_path}")

    thead = table.find("thead")
    if thead is None:
        raise ValueError(f"No <thead> found in HTML table: {html_path}")

    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    if "pattern" not in headers:
        raise ValueError(
            f"Required column 'pattern' missing from headers in {html_path}: {headers}"
        )

    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError(f"No <tbody> found in HTML table: {html_path}")

    rows: List[List[str]] = []
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) != len(headers):
            raise ValueError(
                f"Row column count mismatch in {html_path}: expected {len(headers)}, got {len(cells)}"
            )
        rows.append(cells)

    df = pd.DataFrame(rows, columns=headers)
    df["source_dataset"] = dataset_name
    return df


def is_motif_filtered(raw_match: str, filter_motifs: List[str]) -> bool:
    """Check if a motif identifier matches any filtered TF/motif name on token boundaries.

    Tokenizes the motif string on non-alphanumeric characters (whitespace, '_', '.', ':')
    so that filtering 'SP1' matches 'MA0079.5 SP1' and 'SP1_HUMAN.H11MO.0.A' without
    falsely matching 'SPI1_HUMAN' or 'MA2540.1 SP140L', and 'CTCF' matches 'CTCF'
    without falsely matching 'CTCFL'.
    """
    if not filter_motifs:
        return False
    upper_match = raw_match.upper()
    tokens = set(re.split(r"[^A-Z0-9]+", upper_match))
    for filter_kw in filter_motifs:
        kw = filter_kw.strip().upper()
        if not kw:
            continue
        if kw in tokens or kw == upper_match:
            return True
    return False


def extract_candidate_matches(
    pattern_row: pd.Series,
    dataset_name: str,
    top_n: int,
    filter_motifs: List[str],
) -> List[Tuple[float, str, str]]:
    """Extract and validate candidate matches from a single report row.

    Args:
        pattern_row: Single row from the parsed motifs DataFrame.
        dataset_name: Name of the reference dataset.
        top_n: Number of top matches to consider (1, 2, or 3).
        filter_motifs: List of motif tokens to filter out (e.g. ['CTCF', 'SP1', 'SP2', 'SP3', 'SP4']).

    Returns:
        List of tuples: (p_value, motif_name, dataset_name), sorted ascending by p_value.
    """
    hits: List[Tuple[float, str, str]] = []

    for i in range(top_n):
        match_col = f"match{i}"
        pval_col = f"pval{i}"
        qval_col = f"qval{i}"

        # Check for pval or qval column
        score_col = None
        if pval_col in pattern_row.index:
            score_col = pval_col
        elif qval_col in pattern_row.index:
            score_col = qval_col

        if match_col not in pattern_row.index or score_col is None:
            continue

        raw_match = str(pattern_row[match_col]).strip()
        raw_score = str(pattern_row[score_col]).strip()

        # Skip empty or NA matches
        if not raw_match or raw_match.upper() == "NA" or not raw_score or raw_score.upper() == "NA":
            continue

        # Check if motif is filtered out
        if is_motif_filtered(raw_match, filter_motifs):
            continue

        try:
            pval = float(raw_score)
        except ValueError as exc:
            logger.error(
                "Failed to parse numeric p-value from '%s' in row for pattern '%s' (dataset: %s)",
                raw_score,
                pattern_row["pattern"],
                dataset_name,
                exc_info=True,
            )
            raise

        hits.append((pval, raw_match, dataset_name))

    hits.sort(key=lambda hit: hit[0])
    return hits


def classify_all_patterns(
    reports_data: Dict[str, pd.DataFrame],
    dna_datasets: List[str],
    rna_datasets: List[str],
    db_p_thresholds: Dict[str, float],
    filter_motifs: List[str],
    top_n: int,
    ranking_mode: str = "calibrated_p",
    min_percentile: float = 70.0,
) -> pd.DataFrame:
    """Classify patterns across DNA and RNA reports into 3 lists.

    Args:
        reports_data: Dictionary mapping dataset_name -> parsed report DataFrame.
        dna_datasets: List of dataset names representing DNA references.
        rna_datasets: List of dataset names representing RNA references.
        db_p_thresholds: Dictionary mapping dataset_name -> maximum p-value threshold
            (calibrated for each database's size N and motif width distribution).
        filter_motifs: List of motif names/substrings to filter out.
        top_n: Number of matches to consider per report (1 to 3).
        ranking_mode: Classification mode ('calibrated_p' or 'percentile').
        min_percentile: Minimum percentile threshold (0-100) when ranking_mode is 'percentile'.

    Returns:
        Consolidated summary DataFrame with pattern classifications and percentile ranks.
    """
    for dset in dna_datasets:
        if dset not in reports_data:
            raise KeyError(f"DNA dataset '{dset}' not found in loaded reports: {list(reports_data.keys())}")
        if dset not in db_p_thresholds:
            raise KeyError(f"DNA dataset '{dset}' missing from db_p_thresholds: {list(db_p_thresholds.keys())}")
    for dset in rna_datasets:
        if dset not in reports_data:
            raise KeyError(f"RNA dataset '{dset}' not found in loaded reports: {list(reports_data.keys())}")
        if dset not in db_p_thresholds:
            raise KeyError(f"RNA dataset '{dset}' missing from db_p_thresholds: {list(db_p_thresholds.keys())}")

    # Determine master list of pattern names from the first available report
    first_report_key = list(reports_data.keys())[0]
    pattern_names: List[str] = reports_data[first_report_key]["pattern"].tolist()

    raw_records: List[Dict[str, Any]] = []

    for pattern in pattern_names:
        # Collect candidate DNA matches
        dna_hits: List[Tuple[float, str, str]] = []
        for dname in dna_datasets:
            df = reports_data[dname]
            match_rows = df[df["pattern"] == pattern]
            if not match_rows.empty:
                dna_hits.extend(
                    extract_candidate_matches(match_rows.iloc[0], dname, top_n, filter_motifs)
                )

        # Collect candidate RNA matches
        rna_hits: List[Tuple[float, str, str]] = []
        for dname in rna_datasets:
            df = reports_data[dname]
            match_rows = df[df["pattern"] == pattern]
            if not match_rows.empty:
                rna_hits.extend(
                    extract_candidate_matches(match_rows.iloc[0], dname, top_n, filter_motifs)
                )

        # Sort hits by database-calibrated threshold ratio (equivalent to E-value ranking),
        # breaking ties by raw p-value.
        dna_hits.sort(key=lambda hit: (hit[0] / db_p_thresholds[hit[2]], hit[0]))
        rna_hits.sort(key=lambda hit: (hit[0] / db_p_thresholds[hit[2]], hit[0]))

        best_dna = dna_hits[0] if dna_hits else (float("inf"), "None", "None")
        best_rna = rna_hits[0] if rna_hits else (float("inf"), "None", "None")

        record: Dict[str, Any] = {
            "pattern": pattern,
            "best_dna_match": best_dna[1],
            "best_dna_pval": best_dna[0],
            "best_dna_dataset": best_dna[2],
            "best_rna_match": best_rna[1],
            "best_rna_pval": best_rna[0],
            "best_rna_dataset": best_rna[2],
        }

        # Add individual raw dataset fields
        for dname, report_df in reports_data.items():
            pattern_match = report_df[report_df["pattern"] == pattern]
            if not pattern_match.empty:
                p_row = pattern_match.iloc[0]
                m0 = p_row["match0"] if "match0" in p_row.index else "NA"
                pv0 = p_row["pval0"] if "pval0" in p_row.index else (
                    p_row["qval0"] if "qval0" in p_row.index else "NA"
                )
                try:
                    pv0_float = float(pv0)
                except ValueError:
                    pv0_float = np.nan
                record[f"{dname}_match0"] = m0
                record[f"{dname}_pval0"] = pv0_float
                if "num_seqlets" in p_row.index and "num_seqlets" not in record:
                    try:
                        record["num_seqlets"] = int(p_row["num_seqlets"])
                    except ValueError:
                        record["num_seqlets"] = np.nan
            else:
                record[f"{dname}_match0"] = "NA"
                record[f"{dname}_pval0"] = np.nan

        raw_records.append(record)

    df_patterns = pd.DataFrame(raw_records)

    # Compute empirical Database Percentile Ranking across discovered patterns
    # Higher percentile = lower p-value = stronger match within database distribution (100% is top)
    df_patterns["best_dna_percentile"] = (
        df_patterns["best_dna_pval"].rank(pct=True, ascending=False) * 100.0
    ).round(1)
    df_patterns["best_rna_percentile"] = (
        df_patterns["best_rna_pval"].rank(pct=True, ascending=False) * 100.0
    ).round(1)

    for dname in reports_data.keys():
        pval_col = f"{dname}_pval0"
        if pval_col in df_patterns.columns:
            df_patterns[f"{dname}_percentile"] = (
                df_patterns[pval_col].rank(pct=True, ascending=False) * 100.0
            ).round(1)

    # Perform classification based on ranking_mode
    categories: List[str] = []
    best_modalities: List[str] = []
    best_matches: List[str] = []
    best_pvals: List[float] = []
    best_datasets: List[str] = []

    for _, row in df_patterns.iterrows():
        dna_pval = row["best_dna_pval"]
        rna_pval = row["best_rna_pval"]
        dna_ds = row["best_dna_dataset"]
        rna_ds = row["best_rna_dataset"]
        dna_pct = row["best_dna_percentile"]
        rna_pct = row["best_rna_percentile"]

        if ranking_mode == "percentile":
            dna_sig = dna_pct >= min_percentile
            rna_sig = rna_pct >= min_percentile
            if not dna_sig and not rna_sig:
                cat = "No_close_match"
                mod = "None"
                mat = "None"
                pv = np.nan
                ds = "None"
            elif rna_sig and not dna_sig:
                mod = "RNA"
                mat = row["best_rna_match"]
                pv = rna_pval
                ds = rna_ds
                cat = f"RNA_match ({ds})"
            elif dna_sig and not rna_sig:
                mod = "DNA"
                mat = row["best_dna_match"]
                pv = dna_pval
                ds = dna_ds
                cat = f"DNA_match ({ds})"
            else:
                if rna_pct > dna_pct:
                    mod = "RNA"
                    mat = row["best_rna_match"]
                    pv = rna_pval
                    ds = rna_ds
                    cat = f"RNA_match ({ds})"
                else:
                    mod = "DNA"
                    mat = row["best_dna_match"]
                    pv = dna_pval
                    ds = dna_ds
                    cat = f"DNA_match ({ds})"
        else:
            # Default: calibrated_p mode using per-database p-value thresholds
            dna_thresh = db_p_thresholds[dna_ds] if dna_ds in db_p_thresholds else float("inf")
            rna_thresh = db_p_thresholds[rna_ds] if rna_ds in db_p_thresholds else float("inf")

            dna_sig = dna_pval <= dna_thresh
            rna_sig = rna_pval <= rna_thresh

            if not dna_sig and not rna_sig:
                cat = "No_close_match"
                mod = "None"
                mat = "None"
                pv = np.nan
                ds = "None"
            elif rna_sig and not dna_sig:
                mod = "RNA"
                mat = row["best_rna_match"]
                pv = rna_pval
                ds = rna_ds
                cat = f"RNA_match ({ds})"
            elif dna_sig and not rna_sig:
                mod = "DNA"
                mat = row["best_dna_match"]
                pv = dna_pval
                ds = dna_ds
                cat = f"DNA_match ({ds})"
            else:
                rna_ratio = rna_pval / rna_thresh
                dna_ratio = dna_pval / dna_thresh
                if rna_ratio < dna_ratio:
                    mod = "RNA"
                    mat = row["best_rna_match"]
                    pv = rna_pval
                    ds = rna_ds
                    cat = f"RNA_match ({ds})"
                else:
                    mod = "DNA"
                    mat = row["best_dna_match"]
                    pv = dna_pval
                    ds = dna_ds
                    cat = f"DNA_match ({ds})"

        categories.append(cat)
        best_modalities.append(mod)
        best_matches.append(mat)
        best_pvals.append(pv)
        best_datasets.append(ds)

    df_patterns["category"] = categories
    df_patterns["best_overall_modality"] = best_modalities
    df_patterns["best_overall_match"] = best_matches
    df_patterns["best_overall_pval"] = best_pvals
    df_patterns["best_overall_dataset"] = best_datasets

    # Replace infinite p-values with NaN for clean TSV formatting
    df_patterns["best_dna_pval"] = df_patterns["best_dna_pval"].replace(float("inf"), np.nan)
    df_patterns["best_rna_pval"] = df_patterns["best_rna_pval"].replace(float("inf"), np.nan)

    # Reorder columns logically: identification -> category -> overall best -> best DNA -> best RNA -> raw datasets
    ordered_cols = [
        "pattern",
        "category",
        "best_overall_modality",
        "best_overall_match",
        "best_overall_pval",
        "best_overall_dataset",
        "best_dna_match",
        "best_dna_pval",
        "best_dna_percentile",
        "best_dna_dataset",
        "best_rna_match",
        "best_rna_pval",
        "best_rna_percentile",
        "best_rna_dataset",
    ]
    # Append any remaining columns (num_seqlets, dataset-specific match0/pval0/percentiles)
    for c in df_patterns.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)

    return df_patterns[ordered_cols]


def print_classification_summary(
    classified_df: pd.DataFrame, db_p_thresholds: Dict[str, float]
) -> None:
    """Print clean formatted breakdown of the 3 pattern lists to stdout."""
    rna_count = int(classified_df["category"].str.startswith("RNA_match").sum())
    dna_count = int(classified_df["category"].str.startswith("DNA_match").sum())
    no_match_count = int((classified_df["category"] == "No_close_match").sum())
    thresh_summary = ", ".join(f"{k} <= {v:.1e}" for k, v in db_p_thresholds.items())

    print("\n" + "=" * 80)
    print(f"TF-MoDISco Pattern Modality Classification Summary ({thresh_summary})")
    print("=" * 80)
    print(
        f"Total Patterns: {len(classified_df)} | "
        f"RNA matches: {rna_count} | "
        f"DNA matches: {dna_count} | "
        f"No close match: {no_match_count}"
    )
    print("-" * 80)

    category_groups = [
        ("RNA_match", classified_df[classified_df["category"].str.startswith("RNA_match")]),
        ("DNA_match", classified_df[classified_df["category"].str.startswith("DNA_match")]),
        ("No_close_match", classified_df[classified_df["category"] == "No_close_match"]),
    ]

    for cat_label, subset in category_groups:
        print(f"\n[Category: {cat_label}] ({len(subset)} patterns):")
        if subset.empty:
            print("  (none)")
            continue
        for _, row in subset.iterrows():
            pat = row["pattern"]
            row_cat = row["category"]
            best_mod = row["best_overall_modality"]
            best_mat = row["best_overall_match"]
            best_pv = row["best_overall_pval"]
            best_ds = row["best_overall_dataset"]
            dna_pv = row["best_dna_pval"]
            rna_pv = row["best_rna_pval"]

            pv_str = f"{best_pv:.2e}" if pd.notna(best_pv) else "N/A"
            dna_str = f"{dna_pv:.2e}" if pd.notna(dna_pv) else "N/A"
            rna_str = f"{rna_pv:.2e}" if pd.notna(rna_pv) else "N/A"

            print(
                f"  - {pat:24s} [{row_cat}] -> Best: [{best_mod}] {best_mat} "
                f"(p={pv_str}, dataset={best_ds}) | Best DNA p={dna_str} | Best RNA p={rna_str}"
            )
    print("\n" + "=" * 80 + "\n")


def clean_match_name(raw: str, max_len: int = 24) -> str:
    """Format and shorten database match identifier for display in table cells."""
    if pd.isna(raw) or not raw or str(raw).lower() in ("nan", "none"):
        return "-"
    s = str(raw).strip()
    if s.startswith("RNCMPT") and " " in s:
        s = s.split(" ", 1)[1]
    elif "_HUMAN" in s:
        s = s.split("_HUMAN")[0]
    elif s.startswith("MA") and " " in s:
        s = s.split(" ", 1)[1]
    elif s.startswith("DF") and " " in s:
        s = s.split(" ", 1)[1].rsplit("_", 1)[0]
        if s.startswith("COMP-subunit_"):
            s = s.replace("COMP-subunit_", "COMP_")
    if len(s) > max_len:
        s = s[: max_len - 2] + ".."
    return s


def format_pval(p: float) -> str:
    """Format p-value in concise scientific notation."""
    if pd.isna(p) or p <= 0:
        return ""
    if p >= 0.01:
        return f"p={p:.3f}"
    exponent = int(math.floor(math.log10(p)))
    mantissa = p / (10**exponent)
    return f"p={mantissa:.1f}e{exponent}"


def format_pattern_display_name(name: str) -> str:
    """Format pattern identifier cleanly, preserving neg_ prefix if from neg_patterns."""
    if name.startswith("neg_patterns."):
        return f"neg_{name.split('.')[-1]}"
    return name.replace("pos_patterns.", "")


def plot_pattern_classification_table(
    df: pd.DataFrame,
    output_png: Path,
    dpi: int = 300,
) -> None:
    """Render a clean, uncolored publication table of pattern classifications and database hits."""
    df = df.copy()
    df["is_neg"] = df["pattern"].apply(lambda p: 1 if str(p).startswith("neg_") else 0)
    df["pat_num"] = df["pattern"].apply(lambda p: int(str(p).split("_")[-1]))
    df = df.sort_values(["is_neg", "pat_num"]).reset_index(drop=True)

    num_patterns = len(df)
    columns = [
        "Category",
        "HOCOMOCO v11\n(TF DNA)",
        "JASPAR 2024\n(TF DNA)",
        "Dfam v3.8\n(Repeats)",
        "Ray2013\n(RBP RNA)",
    ]
    databases = ["hocomoco", "jaspar", "dfam", "rbp"]

    annot_mat = np.empty((num_patterns, len(columns)), dtype=object)

    for i, row in df.iterrows():
        annot_mat[i, 0] = row["category"].replace("_", " ")
        for j, db in enumerate(databases):
            pval = row[f"{db}_pval0"]
            pct = row[f"{db}_percentile"]
            match_str = clean_match_name(row[f"{db}_match0"])
            if pd.isna(pval) or pval <= 0 or match_str == "-":
                annot_mat[i, j + 1] = "-"
            else:
                p_text = format_pval(pval)
                pct_text = f"({pct:.1f}%)" if pd.notna(pct) else ""
                annot_mat[i, j + 1] = f"{match_str}\n{p_text} {pct_text}".strip()

    fig_height = max(9.5, 0.46 * num_patterns + 2.2)
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=dpi)
    cmap_white = ListedColormap(["#ffffff"])

    y_labels = [
        f"{format_pattern_display_name(str(row['pattern']))} (N={row['num_seqlets']})"
        for _, row in df.iterrows()
    ]

    sns.heatmap(
        np.zeros((num_patterns, len(columns))),
        ax=ax,
        cmap=cmap_white,
        cbar=False,
        linewidths=0.8,
        linecolor="#d0d0d0",
        annot=annot_mat,
        fmt="",
        annot_kws={"fontsize": 8.0, "weight": "bold", "color": "#111111"},
        yticklabels=y_labels,
        xticklabels=columns,
    )
    ax.set_xticklabels(columns, rotation=0, ha="center", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9.5)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel("")

    if num_patterns > 25:
        secax = ax.secondary_xaxis("top")
        secax.set_xticks(np.arange(len(columns)) + 0.5)
        secax.set_xticklabels(columns, rotation=0, ha="center", fontsize=10, fontweight="bold")

    top_edge = ax.get_position().y1
    title_y = top_edge + (0.75 / fig_height) if num_patterns > 25 else 0.985
    fig.suptitle(
        "TF-MoDISco Pattern Classification & Reference Database Matches",
        fontsize=13,
        fontweight="bold",
        y=title_y,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pattern classification table saved to: %s", output_png)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/tfmodisco_runs/full_tuned"),
        help="Input directory containing TF-MoDISco run report folders (default: outputs/tfmodisco_runs/full_tuned)",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=None,
        help="Path for output TSV file (default: <input-dir>/pattern_classification_summary.tsv)",
    )
    parser.add_argument(
        "--jaspar-folder",
        type=str,
        default="report",
        help="Folder name for JASPAR report within input-dir (default: report)",
    )
    parser.add_argument(
        "--hocomoco-folder",
        type=str,
        default="report_hocomoco",
        help="Folder name for HOCOMOCO report within input-dir (default: report_hocomoco)",
    )
    parser.add_argument(
        "--dfam-folder",
        type=str,
        default="report_dfam",
        help="Folder name for DFAM report within input-dir (default: report_dfam)",
    )
    parser.add_argument(
        "--rbp-folder",
        type=str,
        default="report_rbp",
        help="Folder name for Ray2013 RBP report within input-dir (default: report_rbp)",
    )
    parser.add_argument(
        "--jaspar-p-threshold",
        type=float,
        default=5e-5,
        help="Maximum p-value for JASPAR matches (default: 5e-5, calibrated for N=1019 motifs at Bonferroni alpha=0.05).",
    )
    parser.add_argument(
        "--hocomoco-p-threshold",
        type=float,
        default=1e-4,
        help="Maximum p-value for HOCOMOCO matches (default: 1e-4, calibrated for N=401 motifs at Bonferroni alpha=0.05).",
    )
    parser.add_argument(
        "--dfam-p-threshold",
        type=float,
        default=1e-5,
        help="Maximum p-value for DFAM matches (default: 1e-5, calibrated for N=6345 repeat motifs at Bonferroni alpha=0.05).",
    )
    parser.add_argument(
        "--rbp-p-threshold",
        type=float,
        default=5e-4,
        help="Maximum p-value for Ray2013 RBP matches (default: 5e-4, calibrated for N=102 7-8bp motifs at Bonferroni alpha=0.05).",
    )
    parser.add_argument(
        "--dna-p-threshold",
        type=float,
        default=None,
        help="Optional p-value threshold overriding all DNA databases (--jaspar-p-threshold, --hocomoco-p-threshold, --dfam-p-threshold).",
    )
    parser.add_argument(
        "--rna-p-threshold",
        type=float,
        default=None,
        help="Optional p-value threshold overriding all RNA databases (--rbp-p-threshold).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=None,
        help="Uniform p-value threshold overriding all per-database and modality thresholds if provided.",
    )
    parser.add_argument(
        "--filter-ctcf",
        dest="filter_ctcf",
        action="store_true",
        default=True,
        help="Filter out ubiquitous CTCF matches when determining best matches (default: True).",
    )
    parser.add_argument(
        "--no-filter-ctcf",
        dest="filter_ctcf",
        action="store_false",
        help="Do not filter out CTCF matches.",
    )
    parser.add_argument(
        "--filter-sp",
        dest="filter_sp",
        action="store_true",
        default=True,
        help="Filter out promiscuous GC-box SP-family matches (SP1, SP2, SP3, SP4; default: True).",
    )
    parser.add_argument(
        "--no-filter-sp",
        dest="filter_sp",
        action="store_false",
        help="Do not filter out SP-family matches.",
    )
    parser.add_argument(
        "--exclude-motifs",
        type=str,
        default="",
        help="Comma-separated motif tokens to ignore/filter (e.g. 'CTCF,SP2').",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        default="",
        help="Comma-separated dataset names to exclude from analysis (e.g. 'jaspar').",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of top matches per pattern to examine from each report (default: 1 for match0).",
    )
    parser.add_argument(
        "--ranking-mode",
        type=str,
        default="calibrated_p",
        choices=["calibrated_p", "percentile"],
        help="Classification strategy: 'calibrated_p' uses database-specific p-value thresholds "
             "(jaspar <= 5e-5, hocomoco <= 1e-4, dfam <= 1e-5, rbp <= 5e-4); "
             "'percentile' classifies based on empirical database percentile ranking across the run (default: calibrated_p).",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=70.0,
        help="Minimum percentile (0-100) required to consider a match significant when --ranking-mode is 'percentile' (default: 70.0).",
    )
    parser.add_argument(
        "--plot-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate publication PNG table of pattern classifications (default: True).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for saved table figure (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution function."""
    args = parse_args(argv)
    set_global_seed(args.seed)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Set default output file in the same input folder if not provided
    if args.output_tsv is None:
        output_tsv = input_dir / "pattern_classification_summary.tsv"
    else:
        output_tsv = args.output_tsv.resolve()

    # Map dataset keys to directories
    dataset_dirs: Dict[str, Path] = {
        "jaspar": input_dir / args.jaspar_folder,
        "hocomoco": input_dir / args.hocomoco_folder,
        "dfam": input_dir / args.dfam_folder,
        "rbp": input_dir / args.rbp_folder,
    }

    # Handle dataset exclusions if requested
    excluded_datasets = [
        item.strip() for item in args.exclude_datasets.split(",") if item.strip()
    ]
    for ex in excluded_datasets:
        if ex in dataset_dirs:
            logger.info("Excluding dataset '%s' per user request.", ex)
            del dataset_dirs[ex]

    # Load and parse all active reports
    reports_data: Dict[str, pd.DataFrame] = {}
    for dname, rdir in dataset_dirs.items():
        logger.info("Parsing report for '%s' at: %s", dname, rdir)
        reports_data[dname] = parse_motifs_html_table(rdir, dname)

    # Determine active DNA and RNA datasets
    dna_datasets = [d for d in ["jaspar", "hocomoco", "dfam"] if d in reports_data]
    rna_datasets = [d for d in ["rbp"] if d in reports_data]

    if not dna_datasets:
        raise ValueError("No valid DNA datasets configured.")
    if not rna_datasets:
        raise ValueError("No valid RNA datasets configured.")

    # Build motif filter list
    filter_motifs: List[str] = []
    if args.filter_ctcf:
        filter_motifs.append("CTCF")
    if args.filter_sp:
        filter_motifs.extend(["SP1", "SP2", "SP3", "SP4"])
    if args.exclude_motifs:
        extra_motifs = [m.strip() for m in args.exclude_motifs.split(",") if m.strip()]
        filter_motifs.extend(extra_motifs)

    if filter_motifs:
        logger.info("Filtering out motifs matching: %s", filter_motifs)

    # Determine per-database thresholds (with optional modality or uniform overrides)
    db_p_thresholds: Dict[str, float] = {
        "jaspar": args.jaspar_p_threshold,
        "hocomoco": args.hocomoco_p_threshold,
        "dfam": args.dfam_p_threshold,
        "rbp": args.rbp_p_threshold,
    }
    if args.dna_p_threshold is not None:
        for d in ["jaspar", "hocomoco", "dfam"]:
            db_p_thresholds[d] = args.dna_p_threshold
    if args.rna_p_threshold is not None:
        db_p_thresholds["rbp"] = args.rna_p_threshold
    if args.p_threshold is not None:
        for d in db_p_thresholds:
            db_p_thresholds[d] = args.p_threshold

    active_thresholds = {
        d: db_p_thresholds[d] for d in reports_data.keys()
    }
    thresh_log_str = ", ".join(f"{k} <= {v:.1e}" for k, v in active_thresholds.items())

    # Run classification
    logger.info(
        "Classifying patterns across %d DNA datasets (%s) and %d RNA datasets (%s) "
        "[mode=%s, thresholds: %s, min_percentile=%.1f%%]...",
        len(dna_datasets),
        ", ".join(dna_datasets),
        len(rna_datasets),
        ", ".join(rna_datasets),
        args.ranking_mode,
        thresh_log_str,
        args.min_percentile,
    )
    classified_df = classify_all_patterns(
        reports_data=reports_data,
        dna_datasets=dna_datasets,
        rna_datasets=rna_datasets,
        db_p_thresholds=active_thresholds,
        filter_motifs=filter_motifs,
        top_n=args.top_n,
        ranking_mode=args.ranking_mode,
        min_percentile=args.min_percentile,
    )

    # Save TSV to input folder
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    classified_df.to_csv(output_tsv, sep="\t", index=False)
    logger.info("Classification summary saved to: %s", output_tsv)

    # Generate publication summary table PNG if requested
    if args.plot_table:
        output_png = output_tsv.parent / "pattern_classification_summary_table.png"
        logger.info("Generating publication table plot at: %s", output_png)
        plot_pattern_classification_table(classified_df, output_png, dpi=args.dpi)

    # Print summary lists to stdout
    print_classification_summary(classified_df, active_thresholds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
