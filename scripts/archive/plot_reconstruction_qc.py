"""Create minimal descriptive QC plots for one reconstruction run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "feature_type",
    "mean_cross_entropy_bits",
    "accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reconstruction QC metrics by feature type."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def load_per_region(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reconstruction" / "per_region.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reconstruction table: {path}")
    per_region = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(per_region.columns)
    if missing_columns:
        raise ValueError(
            f"Missing per-region columns: {sorted(missing_columns)}"
        )
    if per_region.empty:
        raise ValueError("per_region.csv is empty")
    return per_region


def save_region_counts(
    per_region: pd.DataFrame,
    output_path: Path,
) -> None:
    counts = per_region["feature_type"].value_counts()
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(counts.index, counts.to_numpy(), color="#4C78A8")
    axis.set_xlabel("Feature type")
    axis.set_ylabel("Number of regions")
    axis.set_title("Annotated intervals by feature type")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_boxplot(
    per_region: pd.DataFrame,
    feature_order: list[str],
    value_column: str,
    y_label: str,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float] | None,
    reference_line: float | None,
    rng: np.random.Generator,
) -> None:
    counts = per_region["feature_type"].value_counts()
    values = [
        per_region.loc[
            per_region["feature_type"] == feature_type,
            value_column,
        ]
        for feature_type in feature_order
    ]
    labels = [
        f"{feature_type}\n(n={int(counts.loc[feature_type])})"
        for feature_type in feature_order
    ]

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.boxplot(values, showfliers=True)
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels)

    for position, feature_values in enumerate(values, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(feature_values))
        axis.scatter(
            position + jitter,
            feature_values,
            s=12,
            alpha=0.35,
            color="#4C78A8",
            edgecolors="none",
        )

    if y_limits is not None:
        axis.set_ylim(y_limits)
    if reference_line is not None:
        axis.axhline(
            reference_line,
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
        )
    axis.set_xlabel("Feature type")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    per_region = load_per_region(args.run_dir)
    output_dir = args.run_dir / "plots" / "reconstruction_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_order = per_region["feature_type"].value_counts().index.tolist()
    rng = np.random.default_rng(44)

    save_region_counts(
        per_region,
        output_dir / "feature_type_region_counts.png",
    )
    save_boxplot(
        per_region,
        feature_order,
        "mean_cross_entropy_bits",
        "Mean cross-entropy per region (bits)",
        "Mean cross-entropy by feature type",
        output_dir / "mean_cross_entropy_by_feature_type.png",
        None,
        2.0,
        rng,
    )
    save_boxplot(
        per_region,
        feature_order,
        "accuracy",
        "Reconstruction accuracy per region",
        "Reconstruction accuracy by feature type",
        output_dir / "accuracy_by_feature_type.png",
        (0.0, 1.0),
        None,
        rng,
    )


if __name__ == "__main__":
    main()
