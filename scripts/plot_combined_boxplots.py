import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_scores(df, model_name):
    # Background
    bg_scores = df.loc[~df['is_annotated_strict'], 'block_score'].dropna().values
    
    # Core
    core_scores = df.loc[df['feature_Core'] == True, 'block_score'].dropna().values
    
    # CTCF
    ctcf_scores = df.loc[df['feature_CTCF'] == True, 'block_score'].dropna().values
    
    # TTF1
    ttf1_scores = df.loc[df['feature_TTF1'] == True, 'block_score'].dropna().values
    
    # To save memory, we can subsample the background if it's too large, but seaborn can handle it.
    rows = []
    for s in bg_scores: rows.append({'Feature': 'Background', 'Score': s, 'Model': model_name})
    for s in core_scores: rows.append({'Feature': 'Core Motif', 'Score': s, 'Model': model_name})
    for s in ctcf_scores: rows.append({'Feature': 'CTCF Motif', 'Score': s, 'Model': model_name})
    for s in ttf1_scores: rows.append({'Feature': 'TTF1 Motif', 'Score': s, 'Model': model_name})
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Plot combined boxplots comparing two models.")
    parser.add_argument("pretrained_dir", type=str, help="Path to pretrained run directory (e.g. outputs/runs/pretraining_...)")
    parser.add_argument("scratch_dir", type=str, help="Path to from-scratch run directory (e.g. outputs/runs/rinalmo_...)")
    parser.add_argument("--out-dir", type=str, default="outputs/runs", help="Output directory for the combined plot")
    
    args = parser.parse_args()
    
    pretrained_path = Path(args.pretrained_dir) / "motif_evaluation" / "single_nucleotide_scores.parquet"
    scratch_path = Path(args.scratch_dir) / "motif_evaluation" / "single_nucleotide_scores.parquet"
    
    if not pretrained_path.exists():
        print(f"Error: {pretrained_path} not found.")
        return
    if not scratch_path.exists():
        print(f"Error: {scratch_path} not found.")
        return
        
    print("Loading Pretrained RiNALMo scores...")
    df_pre = pd.read_parquet(pretrained_path)
    print("Loading RiNALMo (From Scratch) scores...")
    df_scr = pd.read_parquet(scratch_path)
    
    print("Extracting features...")
    plot_df1 = extract_scores(df_pre, "adapted RINALMo")
    plot_df2 = extract_scores(df_scr, "base RINALMo")
    
    combined_df = pd.concat([plot_df1, plot_df2], ignore_index=True)
    
    print("Generating combined boxplot...")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Ensure Background is plotted last
    order = ["Core Motif", "CTCF Motif", "TTF1 Motif", "Background"]
    
    ax = sns.boxplot(
        data=combined_df, 
        x='Feature', 
        y='Score', 
        hue='Model',
        order=order,
        palette="Set2",
        showfliers=False # Optional: hide massive amount of outliers for cleaner box visibility, or keep True.
    )
    
    plt.title('Block Score Distributions by Feature Type (adapted RINALMo vs base RINALMo)', fontsize=14, pad=15)
    plt.ylabel('Block Score', fontsize=12)
    plt.xlabel('Genomic Feature Category', fontsize=12)
    plt.legend(title=None, fontsize=11, loc='upper right')
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_model_comparison_boxplot.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to {out_path}")

if __name__ == "__main__":
    main()
