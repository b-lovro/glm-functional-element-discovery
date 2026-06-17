import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Aggregate individual reconstruction accuracy CSVs into a global summary.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing individual species CSV files.")
    parser.add_argument("--output_file", type=str, default="global_metrics.csv", help="Filename for the aggregated summary output CSV.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        return

    csv_files = list(input_dir.glob("*_reconstruction_accuracy.csv"))
    # Filter out any existing global or master files just in case
    csv_files = [f for f in csv_files if not f.name.startswith("all_species") and not f.name.startswith("global")]

    if not csv_files:
        print(f"No valid *_reconstruction_accuracy.csv files found in {input_dir}")
        return

    print(f"Aggregating {len(csv_files)} files...")

    # Load and concatenate all individual CSVs
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Remove any trailing summary rows if they accidentally exist in legacy files
            df = df[df["Species"] != "SUMMARY"]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    if not dfs:
        print("No valid data found.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure Length is numeric for weighting
    full_df["Length"] = pd.to_numeric(full_df["Length"], errors='coerce')
    full_df = full_df.dropna(subset=["Length"])

    metrics = ["Accuracy", "Baseline Accuracy", "Cross-Entropy", "Baseline CE", "Avg Confidence"]
    
    # Helper for weighted average
    def weighted_avg(group):
        res = {"Type": group["Type"].iloc[0], "Count": len(group), "Total Length": group["Length"].sum()}
        for metric in metrics:
            if metric in group.columns:
                d = group["Length"]
                w = pd.to_numeric(group[metric], errors='coerce')
                # drop NaNs
                valid = ~w.isna()
                if valid.sum() > 0:
                    res[metric] = (d[valid] * w[valid]).sum() / d[valid].sum()
                else:
                    res[metric] = np.nan
        return pd.Series(res)

    print("\nComputing metrics by sequence Type...")
    summary_by_type = full_df.groupby("Type").apply(weighted_avg).reset_index(drop=True)
    
    # Calculate Overall Global Statistics
    print("Computing overall global metrics...")
    total_len = full_df["Length"].sum()
    total_count = len(full_df)
    
    overall_mean = {
        "Type": "OVERALL MEAN",
        "Count": total_count,
        "Total Length": total_len
    }
    overall_median = {
        "Type": "OVERALL MEDIAN",
        "Count": total_count,
        "Total Length": total_len
    }
    overall_weighted = {
        "Type": "OVERALL WEIGHTED MEAN",
        "Count": total_count,
        "Total Length": total_len
    }
    
    for metric in metrics:
        if metric in full_df.columns:
            w = pd.to_numeric(full_df[metric], errors='coerce')
            overall_mean[metric] = w.mean()
            overall_median[metric] = w.median()
            
            valid = ~w.isna()
            d = full_df["Length"]
            if valid.sum() > 0:
                overall_weighted[metric] = (d[valid] * w[valid]).sum() / d[valid].sum()
            else:
                overall_weighted[metric] = np.nan

    # Combine into a final summary dataframe
    final_rows = [overall_mean, overall_median, overall_weighted]
    final_df = pd.concat([summary_by_type, pd.DataFrame(final_rows)], ignore_index=True)

    # Output to terminal
    print("\n" + "="*80)
    print("GLOBAL EVALUATION METRICS")
    print("="*80)
    # Format for clean printing
    print_df = final_df.copy()
    for m in ["Accuracy", "Baseline Accuracy"]:
        if m in print_df.columns:
            print_df[m] = (print_df[m] * 100).round(2).astype(str) + "%"
    for m in ["Cross-Entropy", "Baseline CE", "Avg Confidence"]:
        if m in print_df.columns:
            print_df[m] = print_df[m].round(4)
            
    print(print_df.to_string(index=False))
    
    # Save to file
    output_path = input_dir / args.output_file
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved global metrics to {output_path}")

if __name__ == "__main__":
    main()
