import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_runs.py <run1_dir> <run2_dir>")
        sys.exit(1)

    run1_dir = Path(sys.argv[1])
    run2_dir = Path(sys.argv[2])
    
    if not run1_dir.is_dir() or not run2_dir.is_dir():
        print("Error: Both arguments must be valid run directories.")
        sys.exit(1)
        
    run1_id = run1_dir.name
    run2_id = run2_dir.name
    
    print(f"Comparing {run1_id} and {run2_id}...")

    def load_run(run_dir):
        per_span_path = run_dir / "block_scores" / "per_span.parquet"
        map_index_path = run_dir / "dependency_maps" / "map_index.parquet"
        if not per_span_path.exists() or not map_index_path.exists():
            print(f"Error: Missing parquet files in {run_dir}")
            sys.exit(1)
            
        per_span = pd.read_parquet(per_span_path)
        map_index = pd.read_parquet(map_index_path)
        
        if "map_role" not in map_index.columns:
            print(f"Error: map_role column missing in {run_dir}")
            sys.exit(1)
            
        merged = per_span.merge(map_index[["map_id", "map_role"]], on="map_id", how="inner")
        return merged

    df1 = load_run(run1_dir)
    df2 = load_run(run2_dir)

    plot_dir = Path("outputs/comparisons") / f"{run1_id}_vs_{run2_id}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Compute individual AUROCs
    def compute_auroc(df, run_name):
        df_filtered = df[df["map_role"].isin(["positive", "background"])]
        if df_filtered.empty:
            print(f"No positive/background roles found in {run_name}")
            return None, None, None
        
        y_true = (df_filtered["map_role"] == "positive").astype(int)
        y_score = df_filtered["block_score"]
        try:
            auroc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            print(f"{run_name} AUROC: {auroc:.4f}")
            return fpr, tpr, auroc
        except ValueError as e:
            print(f"{run_name} AUROC Error: {e}")
            return None, None, None

    fpr1, tpr1, auroc1 = compute_auroc(df1, run1_id)
    fpr2, tpr2, auroc2 = compute_auroc(df2, run2_id)
    
    if auroc1 is not None and auroc2 is not None:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        ax_roc.plot(fpr1, tpr1, color="#A6C8E0", lw=2, label=f"{run1_id} (AUC = {auroc1:.2f})")
        ax_roc.plot(fpr2, tpr2, color="#FFB77F", lw=2, label=f"{run2_id} (AUC = {auroc2:.2f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random Guesser")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves Comparison")
        ax_roc.legend(loc="lower right")
        
        roc_path = plot_dir / "runs_roc_comparison.png"
        fig_roc.savefig(roc_path, dpi=300)
        plt.close(fig_roc)
        print(f"Saved ROC comparison to {roc_path}")

    # Prepare paired dataframe for delta scores
    df1_subset = df1[["map_id", "span_start", "map_role", "block_score"]].rename(columns={"block_score": "score1"})
    df2_subset = df2[["map_id", "span_start", "map_role", "block_score"]].rename(columns={"block_score": "score2"})
    
    paired_df = df1_subset.merge(df2_subset, on=["map_id", "span_start", "map_role"], how="inner")
    paired_df["delta_score"] = paired_df["score2"] - paired_df["score1"]
    
    # Save the intermediate dataframe
    parquet_path = plot_dir / "paired_comparison.parquet"
    paired_df.to_parquet(parquet_path, index=False)
    print(f"Saved paired intermediate DataFrame to {parquet_path}")

    # Compute delta AUROC and plot ROC curve
    paired_filtered = paired_df[paired_df["map_role"].isin(["positive", "background"])]
    if not paired_filtered.empty:
        y_true = (paired_filtered["map_role"] == "positive").astype(int)
        y_score = paired_filtered["delta_score"]
        try:
            delta_auroc = roc_auc_score(y_true, y_score)
            print(f"Delta Score ({run2_id} - {run1_id}) AUROC: {delta_auroc:.4f}")
            
        except ValueError as e:
            print(f"Delta Score AUROC Error: {e}")

    # Box plot of Delta Scores
    delta_values = []
    delta_labels = []
    
    if 'paired_filtered' in locals() and not paired_filtered.empty:
        for role in ["positive", "background"]:
            scores = paired_filtered.loc[paired_filtered["map_role"] == role, "delta_score"].dropna().to_numpy()
            if len(scores) > 0:
                delta_values.append(scores)
                delta_labels.append(f"{role.capitalize()}\n(n={len(scores)})")
                
        if delta_values:
            fig_delta_box, ax_delta_box = plt.subplots(figsize=(6, 6))
            
            bplot_delta = ax_delta_box.boxplot(
                delta_values,
                positions=range(1, len(delta_values) + 1),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 2.2},
                boxprops={"linewidth": 1.8, "alpha": 0.85},
                whiskerprops={"color": "black", "linewidth": 1.5},
                capprops={"color": "black", "linewidth": 1.5},
                zorder=2,
            )
            
            for patch, role in zip(bplot_delta['boxes'], ["positive", "background"]):
                color = "#A6C8E0" if role == "positive" else "#FFB77F"
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                
            # Add a zero line reference
            ax_delta_box.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)

            ax_delta_box.set_xticks(range(1, len(delta_labels) + 1))
            ax_delta_box.set_xticklabels(delta_labels)
            ax_delta_box.set_ylabel(f"Delta Score ({run2_id} - {run1_id})")
            ax_delta_box.set_title(f"Delta Score Distributions")
            fig_delta_box.tight_layout()
            
            delta_box_path = plot_dir / "delta_score_boxplot.png"
            fig_delta_box.savefig(delta_box_path, dpi=300)
            plt.close(fig_delta_box)
            print(f"Saved delta score box plot to {delta_box_path}")

    # Grouped Box Plot
    values = []
    labels = []
    
    for df, run_name in [(df1, run1_id), (df2, run2_id)]:
        for role in ["positive", "background"]:
            scores = df.loc[df["map_role"] == role, "block_score"].dropna().to_numpy()
            if len(scores) > 0:
                values.append(scores)
                labels.append(f"{run_name}\n{role.capitalize()}\n(n={len(scores)})")
                
    if not values:
        print("No data to plot.")
        sys.exit(1)

    figure, axis = plt.subplots(figsize=(10, 6))
    
    colors = []
    for lbl in labels:
        if run1_id in lbl:
            colors.append("#A6C8E0")
        else:
            colors.append("#FFB77F")
            
    bplot = axis.boxplot(
        values,
        positions=range(1, len(values) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        boxprops={"linewidth": 1.8, "alpha": 0.85},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        zorder=2,
    )
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Block score")
    axis.set_title(f"Comparison: {run1_id} vs {run2_id}")
    figure.tight_layout()
    
    plot_path = plot_dir / "grouped_box_plots.png"
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)
    print(f"Saved grouped box plots to {plot_path}")

if __name__ == "__main__":
    main()
