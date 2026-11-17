#!/usr/bin/env python3
"""Plot 17-clade mammalian cladograms aligned with regional compartment heatmaps and landmark distances.

For every TF-MoDISco pattern in `seqlet_annotations_overlap.tsv`, generates a 4-panel PNG
where a consensus 17-order mammalian cladogram (Placentalia + Marsupialia root on the left,
leaves pointing right) plus a separated bottom `Overall` summary row are pixel-aligned (`sharey`)
with:
  1. Clade leaf labels (`Order (k/N)`) and `Overall (k_tot/N_tot)` summary label
  2. 7-column `Blues` Regional Compartments heatmap (`count`, `%`, and `species` per cell + row total `n`)
  3. Per-clade & Overall `dist_to_tss` ridgeline KDE, strand-colored seqlets, dotted medians, and missing-TSS labels
  4. Per-clade & Overall `dist_to_28s_end` ridgeline KDE, strand-colored seqlets, and dotted medians

Also writes `cladogram_clade_distribution_summary.tsv` quantifying per-pattern, per-clade (and Overall)
species coverage, dominant compartment, strand counts, missing TSS counts, and median/IQR landmark distances.

Usage:
  /mnt/Dati/miniconda/envs/group6-rinalmo/bin/python scripts/plot_cladogram_aligned_distributions.py \
    --overlap-tsv outputs/tfmodisco_runs/all_seq_n_24/seqlet_annotations_overlap.tsv \
    --records data/prepared/ribosome/records.parquet \
    --output-dir outputs/tfmodisco_runs/all_seq_n_24/cladogram_17_clades
"""

from __future__ import annotations

import os

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import argparse
import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo

# Ensure repository root is importable when invoked as `python scripts/...py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tfmodisco_common import (
    COMP_LABELS,
    DEFAULT_RECORDS_PARQUET,
    DEFAULT_RUN_DIR,
    MAMMALIAN_ORDER_PHYLOGENY,
    ORDER_TO_SUPERORDER,
    extract_tax_order,
    get_dynamic_device,
    pattern_sort_key,
    set_global_seed,
)

# Consensus mammalian phylogenetic topology over the 17 dataset orders:
# Bifurcating Placentalia (Boreoeutheria + Atlantogenata) sister to Marsupialia.
MAMMAL_17_ORDERS_NEWICK = (
    "("
    "("
    "("
    "(Primates,(Rodentia,Lagomorpha)),"
    "(Eulipotyphla,(Chiroptera,((Artiodactyla,Perissodactyla),(Carnivora,Pholidota))))"
    "),"
    "("
    "((Proboscidea,Sirenia,Hyracoidea),Macroscelidea),"
    "(Cingulata,Pilosa)"
    ")"
    "),"
    "(Didelphimorphia,Dasyuromorphia)"
    ");"
)

SUPERORDER_COLORS: Dict[str, str] = {
    "Euarchontoglires": "#1d4ed8",
    "Laurasiatheria": "#0f766e",
    "Afrotheria": "#b45309",
    "Xenarthra": "#7c3aed",
    "Marsupialia": "#be123c",
}

COMPARTMENT_HEADERS: List[str] = [
    "Core\nProm.",
    "Upstream\nUCE",
    "5' ETS",
    "ITS1",
    "ITS2",
    "28S\nTerm.",
    "Distal\nIGS",
]

OVERALL_ROW_Y: float = -1.45
DIVIDER_Y: float = -0.72
ROW_RIDGE_HEIGHT: float = 0.56
ROW_CELL_HEIGHT: float = 0.86

DEFAULT_OVERLAP_TSV = DEFAULT_RUN_DIR / "seqlet_annotations_overlap.tsv"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "cladogram_17_clades"
DEFAULT_XLIM_MODE = "full"
DEFAULT_ROW_Y_SCALE = "sqrt_count"
DEFAULT_MEDIAN_PANELS = "both"
DEFAULT_COMP_VMAX = 65.0
DEFAULT_BW_FACTOR = 1.0
DEFAULT_MIN_BANDWIDTH_BP = 40.0
DEFAULT_MAX_BANDWIDTH_BP = 1500.0
DEFAULT_GRID_POINTS = 320
DEFAULT_DPI = 300
DEFAULT_SEED = 42


def load_order_totals(records_path: Path) -> Dict[str, int]:
    """Load total species/morph count N per taxonomic order from records.parquet."""
    if not records_path.is_file():
        raise FileNotFoundError(f"Records parquet not found: {records_path}")
    rec_df = pd.read_parquet(records_path)
    if "record_id" not in rec_df.columns:
        raise KeyError(f"Required column 'record_id' missing from {records_path}")

    order_counts: Dict[str, int] = {order_name: 0 for _, order_name in MAMMALIAN_ORDER_PHYLOGENY}
    for rec_id in rec_df["record_id"].astype(str).unique().tolist():
        tax_order = extract_tax_order(rec_id)
        order_counts[tax_order] += 1

    for order_name, cnt in order_counts.items():
        if cnt <= 0:
            raise ValueError(f"Order {order_name!r} has zero species in {records_path}")
    return order_counts


def load_overlap_table(overlap_tsv: Path) -> pd.DataFrame:
    """Load and validate per-seqlet overlap table with landmark distances and compartments."""
    if not overlap_tsv.is_file():
        raise FileNotFoundError(f"Seqlet overlap TSV not found: {overlap_tsv}")

    df = pd.read_csv(overlap_tsv, sep="\t")
    required_cols = [
        "pattern_name",
        "record_id",
        "fine_compartment",
        "dist_to_tss",
        "dist_to_28s_end",
        "is_revcomp",
        "peak_B_delta",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column {col!r} missing from {overlap_tsv}")

    df = df.copy()
    df["tax_order"] = [extract_tax_order(r) for r in df["record_id"].astype(str)]
    df["superorder"] = [ORDER_TO_SUPERORDER[o] for o in df["tax_order"]]
    df["is_revcomp"] = df["is_revcomp"].astype(bool)
    return df


def compute_cladogram_coords(
    tree: Phylo.BaseTree.Tree,
) -> Tuple[List[Phylo.BaseTree.Clade], Dict[Phylo.BaseTree.Clade, float], Dict[Phylo.BaseTree.Clade, float], float]:
    """Compute horizontal cladogram (x, y) coordinates with terminals aligned at max depth."""
    terminals: List[Phylo.BaseTree.Clade] = list(tree.get_terminals())
    n_leaves = len(terminals)
    # Place first terminal at top (y = n_leaves - 1) down to y = 0
    y_coords: Dict[Phylo.BaseTree.Clade, float] = {
        leaf: float(n_leaves - 1 - idx) for idx, leaf in enumerate(terminals)
    }
    x_coords: Dict[Phylo.BaseTree.Clade, float] = {}

    def assign_xy(clade: Phylo.BaseTree.Clade, depth: float) -> float:
        x_coords[clade] = depth
        if clade.is_terminal():
            return y_coords[clade]
        child_ys = [assign_xy(child, depth + 1.0) for child in clade.clades]
        y_coords[clade] = 0.5 * (min(child_ys) + max(child_ys))
        return y_coords[clade]

    assign_xy(tree.root, 0.0)
    max_depth = max(x_coords.values())
    for leaf in terminals:
        x_coords[leaf] = max_depth

    return terminals, x_coords, y_coords, max_depth


def compute_kde_bandwidth(
    distances: np.ndarray,
    bw_factor: float,
    min_bandwidth_bp: float,
    max_bandwidth_bp: float,
) -> float:
    """Compute bounded Scott's rule Gaussian KDE bandwidth in bp."""
    n = len(distances)
    if n <= 1:
        return min_bandwidth_bp
    std_val = float(np.std(distances, ddof=1))
    iqr_val = float(np.percentile(distances, 75) - np.percentile(distances, 25))
    scale_est = min(std_val, iqr_val / 1.34) if (iqr_val > 0.0 and std_val > 0.0) else std_val
    scott_factor = float(n ** (-0.2))
    bandwidth_bp = max(min_bandwidth_bp, scale_est * scott_factor * bw_factor)
    if max_bandwidth_bp > 0.0:
        bandwidth_bp = min(bandwidth_bp, max_bandwidth_bp)
    return bandwidth_bp


def compute_axis_xlim(
    values: np.ndarray,
    xlim_mode: str,
    global_bounds: Tuple[float, float],
    row_bws: Optional[List[Tuple[float, float, float]]] = None,
) -> Tuple[float, float]:
    """Compute unclipped x-axis limits covering all valid distances and KDE tails."""
    if len(values) == 0:
        return global_bounds[0], global_bounds[1]

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if xlim_mode == "global":
        return global_bounds[0], global_bounds[1]
    if xlim_mode == "full":
        span = max(vmax - vmin, 200.0)
        pad = max(0.05 * span, 220.0)
        xmin = vmin - pad
        xmax = vmax + pad
        if row_bws:
            for d_min, d_max, bw in row_bws:
                xmin = min(xmin, d_min - 2.6 * bw)
                xmax = max(xmax, d_max + 2.6 * bw)
        # Include 0 bp landmark cleanly when data starts just downstream/upstream of 0
        if vmin > 0.0 and vmin <= 0.25 * span:
            xmin = min(xmin, -max(0.04 * span, 120.0))
        elif vmax < 0.0 and abs(vmax) <= 0.25 * span:
            xmax = max(xmax, max(0.04 * span, 120.0))
        return xmin, xmax
    raise ValueError(f"Unsupported xlim_mode: {xlim_mode!r}")


def evaluate_bounded_kde(
    distances: np.ndarray,
    xmin: float,
    xmax: float,
    grid_points: int,
    bandwidth_bp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate Gaussian KDE count-density curve locally around the clade's data range.

    Args:
        distances: 1D array of valid distances (bp), shape (N,).
        xmin: Lower x-axis bound of the panel (bp).
        xmax: Upper x-axis bound of the panel (bp).
        grid_points: Number of KDE grid points around the cluster.
        bandwidth_bp: Gaussian kernel standard deviation (bp).

    Returns:
        (x_grid, y_count_density) where x_grid stays inside [xmin, xmax].
    """
    n = len(distances)
    if n == 0:
        raise ValueError("Cannot evaluate KDE on an empty distance array.")

    d_min = float(np.min(distances))
    d_max = float(np.max(distances))

    pad_kde = max(bandwidth_bp * 3.5, 120.0)
    grid_lo = max(xmin, d_min - pad_kde)
    grid_hi = min(xmax, d_max + pad_kde)
    if grid_hi <= grid_lo:
        grid_lo, grid_hi = xmin, xmax

    x_grid_span = max(grid_hi - grid_lo, 1.0)
    min_needed_pts = int(np.ceil(x_grid_span / max(0.25 * bandwidth_bp, 1.0))) + 1
    n_grid_pts = min(4096, max(grid_points, min_needed_pts))

    x_grid = np.linspace(grid_lo, grid_hi, n_grid_pts, dtype=np.float64)
    # diff_grid shape: (n_grid_pts, N)
    diff_grid = (x_grid[:, None] - distances[None, :]) / bandwidth_bp
    y_count_density = np.exp(-0.5 * (diff_grid ** 2)).sum(axis=1)
    return x_grid, y_count_density


def build_summary_row(
    pat_name: str,
    superorder: str,
    tax_order: str,
    sub: pd.DataFrame,
    total_sp: int,
) -> Dict[str, Any]:
    """Construct a single clade or Overall summary dictionary for a pattern."""
    n_seq = int(len(sub))
    obs_sp = int(sub["record_id"].nunique()) if n_seq > 0 else 0
    obs_pct = round(100.0 * obs_sp / float(total_sp), 2)

    if n_seq > 0:
        comp_counts = sub["fine_compartment"].value_counts()
        dom_comp = str(comp_counts.index[0])
        rev_cnt = int(sub["is_revcomp"].sum())
        fwd_cnt = n_seq - rev_cnt
        fwd_pct = round(100.0 * fwd_cnt / float(n_seq), 2)
    else:
        dom_comp = "None"
        rev_cnt = 0
        fwd_cnt = 0
        fwd_pct = np.nan

    tss_valid = sub["dist_to_tss"].dropna().to_numpy(dtype=np.float64)
    e28_valid = sub["dist_to_28s_end"].dropna().to_numpy(dtype=np.float64)
    tss_valid_cnt = int(len(tss_valid))

    return {
        "pattern_name": pat_name,
        "superorder": superorder,
        "tax_order": tax_order,
        "total_order_species": total_sp,
        "observed_species_count": obs_sp,
        "observed_species_pct": obs_pct,
        "seqlet_count": n_seq,
        "dominant_compartment": dom_comp,
        "fwd_count": fwd_cnt,
        "rev_count": rev_cnt,
        "fwd_pct": fwd_pct,
        "tss_valid_count": tss_valid_cnt,
        "tss_missing_count": n_seq - tss_valid_cnt,
        "tss_median_bp": round(float(np.median(tss_valid)), 1) if tss_valid_cnt > 0 else np.nan,
        "tss_std_bp": round(float(np.std(tss_valid, ddof=1)), 1)
        if tss_valid_cnt > 1
        else (0.0 if tss_valid_cnt == 1 else np.nan),
        "tss_q25_bp": round(float(np.percentile(tss_valid, 25)), 1) if tss_valid_cnt > 0 else np.nan,
        "tss_q75_bp": round(float(np.percentile(tss_valid, 75)), 1) if tss_valid_cnt > 0 else np.nan,
        "tss_iqr_bp": round(
            float(np.percentile(tss_valid, 75) - np.percentile(tss_valid, 25)), 1
        )
        if tss_valid_cnt > 1
        else (0.0 if tss_valid_cnt == 1 else np.nan),
        "end28s_valid_count": int(len(e28_valid)),
        "end28s_median_bp": round(float(np.median(e28_valid)), 1) if len(e28_valid) > 0 else np.nan,
        "end28s_std_bp": round(float(np.std(e28_valid, ddof=1)), 1)
        if len(e28_valid) > 1
        else (0.0 if len(e28_valid) == 1 else np.nan),
        "end28s_q25_bp": round(float(np.percentile(e28_valid, 25)), 1) if len(e28_valid) > 0 else np.nan,
        "end28s_q75_bp": round(float(np.percentile(e28_valid, 75)), 1) if len(e28_valid) > 0 else np.nan,
        "end28s_iqr_bp": round(
            float(np.percentile(e28_valid, 75) - np.percentile(e28_valid, 25)), 1
        )
        if len(e28_valid) > 1
        else (0.0 if len(e28_valid) == 1 else np.nan),
    }


def compute_pattern_clade_summary(
    pat_name: str,
    df_pat: pd.DataFrame,
    leaf_orders: List[str],
    order_totals: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Compute per-clade and Overall summary rows for a single pattern."""
    rows: List[Dict[str, Any]] = []
    for order_name in leaf_orders:
        sub = df_pat[df_pat["tax_order"] == order_name]
        rows.append(
            build_summary_row(
                pat_name=pat_name,
                superorder=ORDER_TO_SUPERORDER[order_name],
                tax_order=order_name,
                sub=sub,
                total_sp=order_totals[order_name],
            )
        )

    rows.append(
        build_summary_row(
            pat_name=pat_name,
            superorder="All",
            tax_order="Overall",
            sub=df_pat,
            total_sp=sum(order_totals.values()),
        )
    )
    return rows


def draw_cladogram_panel(
    ax_tree: plt.Axes,
    df_pat: pd.DataFrame,
    tree: Phylo.BaseTree.Tree,
    terminals: List[Phylo.BaseTree.Clade],
    x_coords: Dict[Phylo.BaseTree.Clade, float],
    y_coords: Dict[Phylo.BaseTree.Clade, float],
    max_depth: float,
    order_totals: Dict[str, int],
) -> None:
    """Draw left-to-right cladogram with leaf species coverage and Overall summary label."""
    def draw_clade_edges(clade: Phylo.BaseTree.Clade) -> None:
        if clade.is_terminal():
            return
        child_ys = [y_coords[c] for c in clade.clades]
        ax_tree.plot(
            [x_coords[clade], x_coords[clade]],
            [min(child_ys), max(child_ys)],
            color="#475569",
            linewidth=1.3,
            solid_capstyle="round",
            zorder=2,
        )
        for child in clade.clades:
            edge_color = (
                SUPERORDER_COLORS[ORDER_TO_SUPERORDER[str(child.name)]]
                if child.is_terminal()
                else "#475569"
            )
            ax_tree.plot(
                [x_coords[clade], x_coords[child]],
                [y_coords[child], y_coords[child]],
                color=edge_color,
                linewidth=1.45 if child.is_terminal() else 1.3,
                solid_capstyle="round",
                zorder=2,
            )
            draw_clade_edges(child)

    draw_clade_edges(tree.root)

    for leaf in terminals:
        order_name = str(leaf.name)
        superorder = ORDER_TO_SUPERORDER[order_name]
        scolor = SUPERORDER_COLORS[superorder]
        y = y_coords[leaf]
        sub_ord = df_pat[df_pat["tax_order"] == order_name]
        k_sp = int(sub_ord["record_id"].nunique())
        n_total_sp = order_totals[order_name]

        ax_tree.scatter(
            [max_depth],
            [y],
            s=28 if k_sp > 0 else 15,
            color=scolor if k_sp > 0 else "#cbd5e1",
            edgecolors="#0f172a" if k_sp > 0 else "#94a3b8",
            linewidths=0.65,
            zorder=4,
        )
        ax_tree.text(
            max_depth + 0.22,
            y,
            f"{order_name} ({k_sp}/{n_total_sp})",
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold" if k_sp > 0 else "normal",
            color="#0f172a" if k_sp > 0 else "#94a3b8",
        )

    # Separated Overall row label aligned with leaf labels
    total_sp_obs = int(df_pat["record_id"].nunique())
    total_sp_all = sum(order_totals.values())
    ax_tree.axhline(DIVIDER_Y, color="#64748b", linestyle="-", linewidth=1.1, xmin=0.05, xmax=0.98, zorder=3)
    ax_tree.scatter(
        [max_depth],
        [OVERALL_ROW_Y],
        s=34,
        color="#1e3a8a",
        edgecolors="#0f172a",
        linewidths=0.75,
        marker="D",
        zorder=4,
    )
    ax_tree.text(
        max_depth + 0.22,
        OVERALL_ROW_Y,
        f"Overall ({total_sp_obs}/{total_sp_all})",
        va="center",
        ha="left",
        fontsize=8.7,
        fontweight="bold",
        color="#1e3a8a",
    )

    ax_tree.set_xlim(-0.35, max_depth + 4.85)
    ax_tree.set_ylim(OVERALL_ROW_Y - 0.65, float(len(terminals)) - 0.35)
    ax_tree.axis("off")


def draw_compartment_heatmap_panel(
    ax_comp: plt.Axes,
    row_specs: List[Tuple[str, float, pd.DataFrame]],
    comp_vmax: float,
) -> None:
    """Render 7-column Blues heatmap matrix aligned row-by-row with count, %, and species."""
    blues_cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=0.0, vmax=comp_vmax)

    for row_label, y, sub_row in row_specs:
        n_seq = int(len(sub_row))
        is_overall = row_label == "Overall"
        half_h = ROW_CELL_HEIGHT * 0.5

        for j, comp in enumerate(COMP_LABELS):
            if n_seq == 0:
                rect = mpatches.Rectangle(
                    (float(j), y - half_h),
                    1.0,
                    ROW_CELL_HEIGHT,
                    facecolor="#f8fafc",
                    edgecolor="#cbd5e1",
                    linewidth=0.7,
                    zorder=2,
                )
                ax_comp.add_patch(rect)
                ax_comp.text(
                    float(j) + 0.5,
                    y,
                    "-",
                    va="center",
                    ha="center",
                    fontsize=7.8,
                    color="#cbd5e1",
                    zorder=3,
                )
                continue

            sub_comp = sub_row[sub_row["fine_compartment"] == comp]
            cnt = int(len(sub_comp))
            pct = 100.0 * float(cnt) / float(n_seq)
            sp_cnt = int(sub_comp["record_id"].nunique()) if cnt > 0 else 0

            rgba = blues_cmap(norm(min(pct, comp_vmax)))
            rect = mpatches.Rectangle(
                (float(j), y - half_h),
                1.0,
                ROW_CELL_HEIGHT,
                facecolor=rgba,
                edgecolor="#94a3b8" if is_overall else "#cccccc",
                linewidth=1.05 if is_overall else 0.75,
                zorder=2,
            )
            ax_comp.add_patch(rect)

            if cnt == 0:
                ax_comp.text(
                    float(j) + 0.5,
                    y,
                    "-",
                    va="center",
                    ha="center",
                    fontsize=7.8,
                    color="#94a3b8",
                    zorder=3,
                )
            else:
                # Perceived luminance for readable contrast over dark navy cells
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if luminance < 0.52 else "#0f172a"
                ax_comp.text(
                    float(j) + 0.5,
                    y,
                    f"{cnt}\n({pct:.1f}%)\n{sp_cnt} sp",
                    va="center",
                    ha="center",
                    fontsize=6.3,
                    fontweight="bold",
                    linespacing=1.05,
                    color=txt_color,
                    zorder=3,
                )

        # Row seqlet total label just outside column 7 (including partial missing-TSS count if applicable)
        n_valid_tss = int(sub_row["dist_to_tss"].notna().sum()) if n_seq > 0 else 0
        n_miss_tss = n_seq - n_valid_tss
        if n_seq == 0:
            tot_label = "0"
        elif 0 < n_valid_tss < n_seq:
            tot_label = f"n={n_seq}\n({n_miss_tss} no TSS)"
        else:
            tot_label = f"n={n_seq}"

        ax_comp.text(
            7.10,
            y,
            tot_label,
            va="center",
            ha="left",
            fontsize=7.3 if (0 < n_valid_tss < n_seq) else (7.7 if not is_overall else 8.1),
            fontweight="bold" if n_seq > 0 else "normal",
            linespacing=1.08,
            color="#1e3a8a" if is_overall else ("#334155" if n_seq > 0 else "#94a3b8"),
            zorder=3,
        )

    ax_comp.axhline(DIVIDER_Y, color="#64748b", linestyle="-", linewidth=1.1, zorder=4)
    ax_comp.set_xlim(0.0, 8.35)
    ax_comp.xaxis.tick_top()
    ax_comp.set_xticks([float(j) + 0.5 for j in range(len(COMP_LABELS))])
    ax_comp.set_xticklabels(COMPARTMENT_HEADERS, fontsize=7.8, fontweight="bold", color="#0f172a")
    ax_comp.tick_params(axis="x", length=0, pad=4)
    ax_comp.tick_params(axis="y", left=False, labelleft=False)
    ax_comp.set_xlabel("Seqlets / (% of clade) / Species in block", fontsize=8.5, fontweight="bold", color="#334155", labelpad=6)
    for spine_name in ("top", "right", "left", "bottom"):
        ax_comp.spines[spine_name].set_visible(False)


def format_row_stats_text(vals: np.ndarray) -> str:
    """Format compact 3-line summary (median, std, IQR range) for a distance row."""
    n_valid = len(vals)
    med_val = float(np.median(vals))
    if n_valid > 1:
        std_val = float(np.std(vals, ddof=1))
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
    else:
        std_val = 0.0
        q25 = med_val
        q75 = med_val
    return f"med: {med_val:,.0f}\nstd: {std_val:,.0f}\nIQR: [{q25:,.0f}, {q75:,.0f}]"


def compute_stats_col_ratio(df_pat: pd.DataFrame, col: str) -> float:
    """Compute a snug GridSpec width ratio tailored to the widest med/std/IQR string."""
    max_line_len = 14
    for _, sub_row in list(df_pat.groupby("tax_order")) + [("Overall", df_pat)]:
        vals = sub_row[col].dropna().to_numpy(dtype=np.float64)
        if len(vals) == 0:
            continue
        txt = format_row_stats_text(vals)
        for line in txt.splitlines():
            max_line_len = max(max_line_len, len(line))
    # Empirical width scaling for fontsize=6.0-6.2 (bold Overall) so stats column stays tightly fitted
    return max(0.26, min(0.52, 0.0162 * float(max_line_len) + 0.055))


def draw_distance_panel(
    ax_dist: plt.Axes,
    ax_stats: plt.Axes,
    col: str,
    row_specs: List[Tuple[str, float, pd.DataFrame]],
    df_pat: pd.DataFrame,
    global_bounds: Tuple[float, float],
    xlim_mode: str,
    row_y_scale: str,
    show_medians: bool,
    bw_factor: float,
    min_bandwidth_bp: float,
    max_bandwidth_bp: float,
    grid_points: int,
    xlabel: str,
    fill_col: str,
    line_col: str,
    rng: np.random.Generator,
) -> None:
    """Render unclipped ridgeline KDEs, strand-colored points, dotted medians, and med/std/IQR stats."""
    all_valid = df_pat[col].dropna().to_numpy(dtype=np.float64)

    # Pre-compute KDE bandwidths so unclipped x-limits also give KDE tails smooth room
    row_vals: Dict[str, np.ndarray] = {}
    row_bws_for_xlim: List[Tuple[float, float, float]] = []
    row_bandwidths: Dict[str, float] = {}
    for row_label, _, sub_row in row_specs:
        vals = sub_row[col].dropna().to_numpy(dtype=np.float64)
        row_vals[row_label] = vals
        if len(vals) >= 3:
            bw = compute_kde_bandwidth(
                distances=vals,
                bw_factor=bw_factor,
                min_bandwidth_bp=min_bandwidth_bp,
                max_bandwidth_bp=max_bandwidth_bp,
            )
            row_bandwidths[row_label] = bw
            row_bws_for_xlim.append((float(np.min(vals)), float(np.max(vals)), bw))

    xmin, xmax = compute_axis_xlim(
        all_valid,
        xlim_mode=xlim_mode,
        global_bounds=global_bounds,
        row_bws=row_bws_for_xlim,
    )
    ax_dist.set_xlim(xmin, xmax)

    if xmin <= 0.0 <= xmax:
        ax_dist.axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)

    # Pre-evaluate KDEs for rows with >= 3 points so global scaling modes can normalize consistently
    kde_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    clade_max_peaks: List[float] = []
    overall_peak = 0.0

    for row_label, _, _ in row_specs:
        vals = row_vals[row_label]
        if len(vals) >= 3:
            xg, yg_cnt = evaluate_bounded_kde(
                distances=vals,
                xmin=xmin,
                xmax=xmax,
                grid_points=grid_points,
                bandwidth_bp=row_bandwidths[row_label],
            )
            kde_cache[row_label] = (xg, yg_cnt)
            pk = float(np.max(yg_cnt))
            if row_label == "Overall":
                overall_peak = pk
            else:
                clade_max_peaks.append(pk)

    global_clade_peak = max(clade_max_peaks) if clade_max_peaks else overall_peak
    if global_clade_peak <= 0.0:
        global_clade_peak = 1.0

    for row_label, y, sub_row in row_specs:
        is_overall = row_label == "Overall"
        base_y = y - 0.18
        ax_dist.axhline(base_y, color="#cbd5e1" if is_overall else "#e2e8f0", linewidth=0.75 if is_overall else 0.65, zorder=1)

        n_seq_row = int(len(sub_row))
        valid_mask = sub_row[col].notna().to_numpy(dtype=bool)
        vals = sub_row.loc[valid_mask, col].to_numpy(dtype=np.float64)
        is_rev = sub_row.loc[valid_mask, "is_revcomp"].to_numpy(dtype=bool)
        n_valid = len(vals)

        if n_seq_row == 0:
            continue

        if n_valid == 0:
            if col == "dist_to_tss":
                ax_dist.text(
                    0.5 * (xmin + xmax),
                    y,
                    f"No TSS annotation (n={n_seq_row})",
                    va="center",
                    ha="center",
                    fontsize=7.3,
                    style="italic",
                    color="#64748b",
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.92},
                    zorder=5,
                )
            continue

        if row_label in kde_cache:
            xg, yg_cnt = kde_cache[row_label]
            max_row_pk = float(np.max(yg_cnt))
            if max_row_pk > 0.0:
                if row_y_scale == "per_row":
                    target_height = ROW_RIDGE_HEIGHT
                elif row_y_scale == "sqrt_count":
                    denom = overall_peak if (is_overall and overall_peak > 0.0) else global_clade_peak
                    target_height = ROW_RIDGE_HEIGHT * float(np.sqrt(min(max_row_pk / denom, 1.0)))
                elif row_y_scale == "global_count":
                    denom = overall_peak if (overall_peak > 0.0) else global_clade_peak
                    target_height = ROW_RIDGE_HEIGHT * float(min(max_row_pk / denom, 1.0))
                else:
                    raise ValueError(f"Unsupported row_y_scale: {row_y_scale!r}")

                yg_norm = (yg_cnt / max_row_pk) * target_height
                ax_dist.fill_between(
                    xg,
                    base_y,
                    base_y + yg_norm,
                    color="#1d4ed8" if (is_overall and col == "dist_to_tss") else ("#15803d" if is_overall else fill_col),
                    alpha=0.28 if is_overall else 0.22,
                    zorder=2,
                )
                ax_dist.plot(
                    xg,
                    base_y + yg_norm,
                    color=line_col,
                    linewidth=1.35 if is_overall else 1.05,
                    zorder=3,
                )

        # Row-confined dotted median line
        if show_medians:
            med_val = float(np.median(vals))
            ax_dist.plot(
                [med_val, med_val],
                [base_y - 0.10, base_y + ROW_RIDGE_HEIGHT],
                color="#0f172a",
                linestyle=":",
                linewidth=1.65 if is_overall else 1.4,
                alpha=0.92,
                zorder=5,
            )

        # Individual seqlet positions (all points inside unclipped bounds)
        y_pts = base_y + rng.uniform(-0.07, 0.07, size=n_valid)
        fwd_m = ~is_rev
        if np.any(fwd_m):
            ax_dist.scatter(
                vals[fwd_m],
                y_pts[fwd_m],
                s=15 if is_overall else 13,
                color="#0369a1",
                alpha=0.72,
                edgecolors="none",
                zorder=4,
            )
        if np.any(is_rev):
            ax_dist.scatter(
                vals[is_rev],
                y_pts[is_rev],
                s=15 if is_overall else 13,
                color="#ea580c",
                alpha=0.72,
                edgecolors="none",
                zorder=4,
            )

        # Render per-clade (and Overall) med, std, and IQR range to the right of the distance plot
        ax_stats.text(
            0.03,
            y,
            format_row_stats_text(vals),
            va="center",
            ha="left",
            fontsize=6.0 if not is_overall else 6.2,
            fontweight="bold" if is_overall else "normal",
            linespacing=1.04,
            color="#1e3a8a" if is_overall else "#334155",
            zorder=3,
        )

    ax_dist.axhline(DIVIDER_Y, color="#64748b", linestyle="-", linewidth=1.1, zorder=4)
    ax_dist.set_xlabel(xlabel, fontsize=9.0, fontweight="bold", color="#334155")
    ax_dist.tick_params(axis="x", labelsize=8.0)
    ax_dist.tick_params(axis="y", left=False, labelleft=False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["left"].set_color("#cbd5e1")

    ax_stats.axhline(DIVIDER_Y, color="#64748b", linestyle="-", linewidth=1.1, zorder=4)
    ax_stats.set_xlim(0.0, 1.0)
    ax_stats.axis("off")


def plot_single_pattern_cladogram(
    pat_name: str,
    df_pat: pd.DataFrame,
    tree: Phylo.BaseTree.Tree,
    terminals: List[Phylo.BaseTree.Clade],
    x_coords: Dict[Phylo.BaseTree.Clade, float],
    y_coords: Dict[Phylo.BaseTree.Clade, float],
    max_depth: float,
    order_totals: Dict[str, int],
    global_tss_bounds: Tuple[float, float],
    global_28s_bounds: Tuple[float, float],
    out_png: Path,
    xlim_mode: str,
    row_y_scale: str,
    median_panels: str,
    comp_vmax: float,
    bw_factor: float,
    min_bandwidth_bp: float,
    max_bandwidth_bp: float,
    grid_points: int,
    dpi: int,
    rng: np.random.Generator,
) -> None:
    """Render title-free figure aligning 17-order cladogram + Overall row with heatmaps, distances & stats."""
    tss_stat_ratio = compute_stats_col_ratio(df_pat, "dist_to_tss")
    e28_stat_ratio = compute_stats_col_ratio(df_pat, "dist_to_28s_end")
    base_ratios = [1.22, 1.94, 1.46, tss_stat_ratio, 1.46, e28_stat_ratio]
    fig_width = 21.2 * (sum(base_ratios) / 6.90)

    fig = plt.figure(figsize=(fig_width, 9.3), dpi=dpi, facecolor="white")
    gs = gridspec.GridSpec(1, 6, width_ratios=base_ratios, wspace=0.04)

    ax_tree = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1], sharey=ax_tree)
    ax_tss = fig.add_subplot(gs[0, 2], sharey=ax_tree)
    ax_tss_stats = fig.add_subplot(gs[0, 3], sharey=ax_tree)
    ax_28s = fig.add_subplot(gs[0, 4], sharey=ax_tree)
    ax_28s_stats = fig.add_subplot(gs[0, 5], sharey=ax_tree)

    # Alternating row shading across Panels 2..6 + subtle Overall highlight
    shaded_axes = (ax_comp, ax_tss, ax_tss_stats, ax_28s, ax_28s_stats)
    for idx, leaf in enumerate(terminals):
        y = y_coords[leaf]
        if idx % 2 == 0:
            for ax in shaded_axes:
                ax.axhspan(y - 0.48, y + 0.48, color="#f8fafc", zorder=0)
    for ax in shaded_axes:
        ax.axhspan(OVERALL_ROW_Y - 0.50, OVERALL_ROW_Y + 0.50, color="#eff6ff", zorder=0)

    row_specs: List[Tuple[str, float, pd.DataFrame]] = [
        (str(leaf.name), y_coords[leaf], df_pat[df_pat["tax_order"] == str(leaf.name)])
        for leaf in terminals
    ]
    row_specs.append(("Overall", OVERALL_ROW_Y, df_pat))

    # 1. Cladogram + Leaf Labels + Overall label
    draw_cladogram_panel(
        ax_tree=ax_tree,
        df_pat=df_pat,
        tree=tree,
        terminals=terminals,
        x_coords=x_coords,
        y_coords=y_coords,
        max_depth=max_depth,
        order_totals=order_totals,
    )

    # 2. Regional Compartments 7-column heatmap
    draw_compartment_heatmap_panel(
        ax_comp=ax_comp,
        row_specs=row_specs,
        comp_vmax=comp_vmax,
    )

    # 3 & 4. TSS and 28S 3'-End Distance panels with tightly sized med / std / IQR columns
    show_tss_med = median_panels in ("both", "tss_only")
    show_28s_med = median_panels == "both"

    draw_distance_panel(
        ax_dist=ax_tss,
        ax_stats=ax_tss_stats,
        col="dist_to_tss",
        row_specs=row_specs,
        df_pat=df_pat,
        global_bounds=global_tss_bounds,
        xlim_mode=xlim_mode,
        row_y_scale=row_y_scale,
        show_medians=show_tss_med,
        bw_factor=bw_factor,
        min_bandwidth_bp=min_bandwidth_bp,
        max_bandwidth_bp=max_bandwidth_bp,
        grid_points=grid_points,
        xlabel="Distance from TSS (bp)",
        fill_col="#2563eb",
        line_col="#1e40af",
        rng=rng,
    )

    draw_distance_panel(
        ax_dist=ax_28s,
        ax_stats=ax_28s_stats,
        col="dist_to_28s_end",
        row_specs=row_specs,
        df_pat=df_pat,
        global_bounds=global_28s_bounds,
        xlim_mode=xlim_mode,
        row_y_scale=row_y_scale,
        show_medians=show_28s_med,
        bw_factor=bw_factor,
        min_bandwidth_bp=min_bandwidth_bp,
        max_bandwidth_bp=max_bandwidth_bp,
        grid_points=grid_points,
        xlabel="Distance from 28S 3'-End (bp)",
        fill_col="#16a34a",
        line_col="#166534",
        rng=rng,
    )

    legend_handles: List[Any] = [
        mlines.Line2D([], [], color="#0369a1", marker="o", linestyle="None", markersize=5.5, label="Fwd strand (+)"),
        mlines.Line2D([], [], color="#ea580c", marker="o", linestyle="None", markersize=5.5, label="Rev strand (-)"),
        mlines.Line2D([], [], color="#dc2626", linestyle="--", linewidth=1.3, label="Landmark (0 bp)"),
    ]
    if median_panels != "none":
        legend_handles.append(
            mlines.Line2D([], [], color="#0f172a", linestyle=":", linewidth=1.6, label="Clade median (bp)")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.008),
        handlelength=1.6,
        columnspacing=1.8,
    )

    fig.subplots_adjust(top=0.935, bottom=0.09, left=0.015, right=0.988)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--overlap-tsv",
        type=Path,
        default=DEFAULT_OVERLAP_TSV,
        help=f"Path to seqlet_annotations_overlap.tsv (default: {DEFAULT_OVERLAP_TSV})",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS_PARQUET,
        help=f"Path to records.parquet (default: {DEFAULT_RECORDS_PARQUET})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write per-pattern PNGs and summary TSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional single pattern_name filter (e.g., 'pos_patterns.pattern_10').",
    )
    parser.add_argument(
        "--pos-only",
        action="store_true",
        help="Only plot pos_patterns.* (default: plot both pos_patterns and neg_patterns).",
    )
    parser.add_argument(
        "--xlim-mode",
        choices=["full", "global"],
        default=DEFAULT_XLIM_MODE,
        help=f"Unclipped x-axis range mode for distance panels (default: {DEFAULT_XLIM_MODE}).",
    )
    parser.add_argument(
        "--row-y-scale",
        choices=["per_row", "sqrt_count", "global_count"],
        default=DEFAULT_ROW_Y_SCALE,
        help=f"KDE ridgeline height scaling across clade rows (default: {DEFAULT_ROW_Y_SCALE}).",
    )
    parser.add_argument(
        "--median-panels",
        choices=["both", "tss_only", "none"],
        default=DEFAULT_MEDIAN_PANELS,
        help=f"Which distance panels display per-clade dotted median lines (default: {DEFAULT_MEDIAN_PANELS}).",
    )
    parser.add_argument(
        "--comp-vmax",
        type=float,
        default=DEFAULT_COMP_VMAX,
        help=f"Upper percentage saturation for Blues compartment heatmap (default: {DEFAULT_COMP_VMAX}).",
    )
    parser.add_argument(
        "--bw-factor",
        type=float,
        default=DEFAULT_BW_FACTOR,
        help=f"Scott's rule Gaussian KDE bandwidth multiplier (default: {DEFAULT_BW_FACTOR}).",
    )
    parser.add_argument(
        "--min-bandwidth-bp",
        type=float,
        default=DEFAULT_MIN_BANDWIDTH_BP,
        help=f"Minimum KDE kernel bandwidth in bp (default: {DEFAULT_MIN_BANDWIDTH_BP}).",
    )
    parser.add_argument(
        "--max-bandwidth-bp",
        type=float,
        default=DEFAULT_MAX_BANDWIDTH_BP,
        help=f"Maximum KDE kernel bandwidth cap in bp; <=0 disables cap (default: {DEFAULT_MAX_BANDWIDTH_BP}).",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=DEFAULT_GRID_POINTS,
        help=f"KDE evaluation grid size per clade row (default: {DEFAULT_GRID_POINTS}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output PNG resolution in DPI (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible point jitter (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for 17-clade cladogram aligned distribution generation."""
    args = parse_args(argv)
    try:
        set_global_seed(args.seed, deterministic=True)
        _ = get_dynamic_device()
        rng = np.random.default_rng(args.seed)

        df = load_overlap_table(args.overlap_tsv)
        order_totals = load_order_totals(args.records)

        all_tss = df["dist_to_tss"].dropna().to_numpy(dtype=np.float64)
        all_28s = df["dist_to_28s_end"].dropna().to_numpy(dtype=np.float64)
        global_tss_bounds = (float(np.min(all_tss)) - 300.0, float(np.max(all_tss)) + 300.0)
        global_28s_bounds = (float(np.min(all_28s)) - 300.0, float(np.max(all_28s)) + 300.0)

        pattern_names = sorted(df["pattern_name"].astype(str).unique().tolist(), key=pattern_sort_key)
        if args.pos_only:
            pattern_names = [p for p in pattern_names if p.startswith("pos_patterns.")]
        if args.pattern is not None:
            if args.pattern not in pattern_names:
                raise KeyError(f"Requested pattern {args.pattern!r} not found in {args.overlap_tsv}")
            pattern_names = [args.pattern]

        tree = Phylo.read(io.StringIO(MAMMAL_17_ORDERS_NEWICK), "newick")
        terminals, x_coords, y_coords, max_depth = compute_cladogram_coords(tree)
        leaf_orders = [str(leaf.name) for leaf in terminals]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: List[Dict[str, Any]] = []

        print(f"[info] Plotting 17-clade cladogram distributions for {len(pattern_names)} patterns -> {args.output_dir}")
        for idx, pat_name in enumerate(pattern_names, start=1):
            df_pat = df[df["pattern_name"] == pat_name]
            out_png = args.output_dir / f"{pat_name}.png"
            plot_single_pattern_cladogram(
                pat_name=pat_name,
                df_pat=df_pat,
                tree=tree,
                terminals=terminals,
                x_coords=x_coords,
                y_coords=y_coords,
                max_depth=max_depth,
                order_totals=order_totals,
                global_tss_bounds=global_tss_bounds,
                global_28s_bounds=global_28s_bounds,
                out_png=out_png,
                xlim_mode=args.xlim_mode,
                row_y_scale=args.row_y_scale,
                median_panels=args.median_panels,
                comp_vmax=args.comp_vmax,
                bw_factor=args.bw_factor,
                min_bandwidth_bp=args.min_bandwidth_bp,
                max_bandwidth_bp=args.max_bandwidth_bp,
                grid_points=args.grid_points,
                dpi=args.dpi,
                rng=rng,
            )
            summary_rows.extend(
                compute_pattern_clade_summary(
                    pat_name=pat_name,
                    df_pat=df_pat,
                    leaf_orders=leaf_orders,
                    order_totals=order_totals,
                )
            )
            if idx % 25 == 0 or idx == len(pattern_names):
                print(f"[info] Completed {idx}/{len(pattern_names)} patterns...")

        summary_df = pd.DataFrame(summary_rows)
        summary_path = args.output_dir / "cladogram_clade_distribution_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"[info] Saved clade distribution summary TSV ({len(summary_df)} rows): {summary_path}")

    except Exception as exc:
        print(f"[error] Cladogram plotting failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
