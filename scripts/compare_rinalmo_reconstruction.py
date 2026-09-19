#!/usr/bin/env python3
"""
Compare reconstruction performance between two RiNALMo runs.

Example
-------
python scripts/compare_rinalmo_reconstruction.py \
    --base outputs/runs \
    --base_prefix rinalmo_rfam_base_part \
    --ft outputs/runs \
    --ft_prefix rinalmo_rfam_ft_part \
    --max_part 7 \
    --output outputs/rfam_comparison
"""

from pathlib import Path
import argparse
import pandas as pd


def load_parts(
    root: Path,
    prefix: str,
    max_part: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate reconstruction outputs."""

    per_base = []
    per_region = []

    run_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )

    if not run_dirs:
        raise FileNotFoundError(
            f"No runs found with prefix '{prefix}'"
        )

    if max_part is not None:
        filtered = []
        for d in run_dirs:
            try:
                part = int(d.name.split("part")[-1])
                if part <= max_part:
                    filtered.append(d)
            except ValueError:
                pass
        run_dirs = filtered

    print(f"Found {len(run_dirs)} candidate runs")

    for run in run_dirs:

        base_file = run / "reconstruction" / "per_base.csv"
        region_file = run / "reconstruction" / "per_region.csv"

        if not base_file.exists():
            print(f"Skipping {run.name} (missing per_base.csv)")
            continue

        if not region_file.exists():
            print(f"Skipping {run.name} (missing per_region.csv)")
            continue

        print(f"Loading {run.name}")

        per_base.append(pd.read_csv(base_file))
        per_region.append(pd.read_csv(region_file))

    if not per_base:
        raise RuntimeError(
            f"No completed reconstruction outputs found for '{prefix}'."
        )

    return (
        pd.concat(per_base, ignore_index=True),
        pd.concat(per_region, ignore_index=True),
    )


def summarize(
    per_base: pd.DataFrame,
    per_region: pd.DataFrame,
) -> dict:

    return {
        "bases": len(per_base),
        "regions": len(per_region),
        "accuracy": per_base["correct"].mean(),
        "mean_true_probability":
            per_base["true_base_probability"].mean(),
        "mean_cross_entropy":
            per_base["cross_entropy_bits"].mean(),
        "median_cross_entropy":
            per_base["cross_entropy_bits"].median(),
        "region_accuracy":
            per_region["accuracy"].mean(),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Compare RiNALMo reconstruction runs."
    )

    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base_prefix", required=True)

    parser.add_argument("--ft", type=Path, required=True)
    parser.add_argument("--ft_prefix", required=True)

    parser.add_argument(
        "--max_part",
        type=int,
        default=None,
        help="Only compare parts up to this number.",
    )

    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading base model")
    print("=" * 80)

    base_per_base, base_per_region = load_parts(
        args.base,
        args.base_prefix,
        args.max_part,
    )

    print()

    print("=" * 80)
    print("Loading fine-tuned model")
    print("=" * 80)

    ft_per_base, ft_per_region = load_parts(
        args.ft,
        args.ft_prefix,
        args.max_part,
    )

    base = summarize(base_per_base, base_per_region)
    ft = summarize(ft_per_base, ft_per_region)

    comparison = pd.DataFrame({
        "Metric": [
            "Evaluated Bases",
            "Regions",
            "Reconstruction Accuracy",
            "Mean True-base Probability",
            "Mean Cross Entropy (bits)",
            "Median Cross Entropy (bits)",
            "Region Accuracy",
        ],
        "Base": [
            base["bases"],
            base["regions"],
            base["accuracy"],
            base["mean_true_probability"],
            base["mean_cross_entropy"],
            base["median_cross_entropy"],
            base["region_accuracy"],
        ],
        "Fine-tuned": [
            ft["bases"],
            ft["regions"],
            ft["accuracy"],
            ft["mean_true_probability"],
            ft["mean_cross_entropy"],
            ft["median_cross_entropy"],
            ft["region_accuracy"],
        ],
    })

    comparison["Difference"] = (
        comparison["Fine-tuned"] - comparison["Base"]
    )

    print()
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(comparison.to_string(index=False))
    print("=" * 80)

    output_file = args.output / "comparison.csv"

    comparison.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Saved comparison to: {output_file}")


if __name__ == "__main__":
    main()