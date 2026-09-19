#!/usr/bin/env python3
"""
Create comparison plots for RiNALMo reconstruction.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def load_parts(root: Path, prefix: str, max_part=None):

    dfs = []

    run_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )

    if max_part is not None:
        run_dirs = [
            d for d in run_dirs
            if int(d.name.split("part")[-1]) <= max_part
        ]

    print(f"Found {len(run_dirs)} candidate runs")

    for run in run_dirs:

        file = run / "reconstruction" / "per_base.csv"

        if file.exists():
            print(f"Loading {run.name}")
            dfs.append(pd.read_csv(file))
        else:
            print(f"Skipping {run.name}")

    if not dfs:
        raise RuntimeError(f"No completed runs found for {prefix}")

    return pd.concat(dfs, ignore_index=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base_prefix", required=True)

    parser.add_argument("--ft", type=Path, required=True)
    parser.add_argument("--ft_prefix", required=True)

    parser.add_argument("--max_part", type=int)

    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading base...")
    base = load_parts(
        args.base,
        args.base_prefix,
        args.max_part,
    )

    print()

    print("Loading fine-tuned...")
    ft = load_parts(
        args.ft,
        args.ft_prefix,
        args.max_part,
    )

    ##########################################################
    # Sampling for visualization
    ##########################################################

    sample_size = 500000

    base_sample = base.sample(
        min(sample_size, len(base)),
        random_state=42,
    )

    ft_sample = ft.sample(
        min(sample_size, len(ft)),
        random_state=42,
    )

    ##########################################################
    # Accuracy bar chart
    ##########################################################

    plt.figure(figsize=(8, 6))

    plt.bar(
        ["Base", "Fine-tuned"],
        [
            base["correct"].mean(),
            ft["correct"].mean(),
        ],
    )

    plt.ylabel("Reconstruction Accuracy")
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        args.output / "accuracy_barplot.png",
        dpi=300,
    )

    plt.close()

    ##########################################################
    # True-base probability histogram
    ##########################################################

    plt.figure(figsize=(10, 6))

    plt.hist(
        base_sample["true_base_probability"],
        bins=50,
        density=True,
        alpha=0.5,
        label="Base",
    )

    plt.hist(
        ft_sample["true_base_probability"],
        bins=50,
        density=True,
        alpha=0.5,
        label="Fine-tuned",
    )

    plt.xlabel("True-base probability")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        args.output / "true_probability_hist.png",
        dpi=300,
    )

    plt.close()

    ##########################################################
    # Cross entropy histogram
    ##########################################################

    plt.figure(figsize=(10, 6))

    plt.hist(
        base_sample["cross_entropy_bits"],
        bins=50,
        density=True,
        alpha=0.5,
        label="Base",
    )

    plt.hist(
        ft_sample["cross_entropy_bits"],
        bins=50,
        density=True,
        alpha=0.5,
        label="Fine-tuned",
    )

    plt.xlabel("Cross entropy (bits)")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        args.output / "cross_entropy_hist.png",
        dpi=300,
    )

    plt.close()

    ##########################################################
    # Cross entropy boxplot
    ##########################################################

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [
            base_sample["cross_entropy_bits"],
            ft_sample["cross_entropy_bits"],
        ],
        labels=["Base", "Fine-tuned"],
    )

    plt.ylabel("Cross entropy (bits)")

    plt.tight_layout()

    plt.savefig(
        args.output / "cross_entropy_boxplot.png",
        dpi=300,
    )

    plt.close()

    ##########################################################
    # True probability boxplot
    ##########################################################

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [
            base_sample["true_base_probability"],
            ft_sample["true_base_probability"],
        ],
        labels=["Base", "Fine-tuned"],
    )

    plt.ylabel("True-base probability")

    plt.tight_layout()

    plt.savefig(
        args.output / "true_probability_boxplot.png",
        dpi=300,
    )

    plt.close()

    print()
    print("=" * 60)
    print("Finished.")
    print(f"Plots written to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()