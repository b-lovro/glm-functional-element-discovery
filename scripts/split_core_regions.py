from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


GENERATED_LABELS = ("Core1", "Core2")
DEFAULT_CORE1_FRACTION = 2 / 3


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_split_region(row, label: str, start: int, end: int) -> dict:
    region = row._asdict()
    region["region_id"] = f"{row.region_id}:{label}"
    region["start"] = start
    region["end"] = end
    region["length"] = end - start
    region["label"] = label
    region["feature_type"] = label
    return region


def split_core_regions(
    prepared_dir: Path,
    output_dir: Path,
    core_feature_type: str = "Core",
    core1_fraction: float = DEFAULT_CORE1_FRACTION,
) -> tuple[int, int, int]:
    if not 0 < core1_fraction < 1:
        raise ValueError("core1_fraction must be greater than 0 and less than 1")

    records_path = prepared_dir / "records.parquet"
    regions_path = prepared_dir / "regions.parquet"
    records = pd.read_parquet(records_path, engine="pyarrow")
    regions = pd.read_parquet(regions_path, engine="pyarrow")

    generated_mask = (
        regions["label"].isin(GENERATED_LABELS)
        & regions["feature_type"].isin(GENERATED_LABELS)
        & regions["region_id"].str.endswith(GENERATED_LABELS, na=False)
    )
    regions = regions.loc[~generated_mask].copy()

    core_regions = regions.loc[regions["feature_type"] == core_feature_type]
    if core_regions.empty:
        raise ValueError(f"No regions found with feature_type={core_feature_type!r}")

    additions = []
    for row in core_regions.itertuples(index=False):
        start = int(row.start)
        end = int(row.end)
        length = end - start
        if length != int(row.length):
            raise ValueError(
                f"Region {row.region_id} has length={row.length}, "
                f"but end-start={length}"
            )

        split_offset = round(length * core1_fraction)
        split_offset = min(max(1, split_offset), length - 1)
        core1_end = start + split_offset
        if core1_end > start:
            additions.append(_make_split_region(row, "Core1", start, core1_end))

        if end > core1_end:
            additions.append(_make_split_region(row, "Core2", core1_end, end))

    additions_frame = pd.DataFrame(additions, columns=regions.columns)
    augmented_regions = pd.concat(
        [regions, additions_frame],
        ignore_index=True,
    ).sort_values(
        ["record_id", "start", "end", "source_row", "region_id"],
        kind="stable",
    )

    duplicated_region_ids = augmented_regions["region_id"].duplicated()
    if duplicated_region_ids.any():
        duplicates = augmented_regions.loc[duplicated_region_ids, "region_id"].tolist()
        raise ValueError(f"Duplicate region_id values after splitting: {duplicates[:5]}")

    region_counts = augmented_regions.groupby("record_id").size()
    records = records.copy()
    records["region_count"] = (
        records["record_id"].map(region_counts).fillna(0).astype("int64")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(output_dir / "records.parquet", engine="pyarrow", index=False)
    augmented_regions.to_parquet(
        output_dir / "regions.parquet",
        engine="pyarrow",
        index=False,
    )

    core1_count = sum(1 for region in additions if region["label"] == "Core1")
    core2_count = sum(1 for region in additions if region["label"] == "Core2")
    return len(core_regions), core1_count, core2_count


def main() -> None:
    repository_root = _repository_root()
    parser = ArgumentParser(
        description=(
            "Add proportional Core1/Core2 split annotations to a prepared ribosome dataset. "
            "Original Core annotations are kept unchanged."
        )
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=repository_root / "data" / "prepared" / "ribosome",
        help="Directory containing records.parquet and regions.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "data" / "prepared" / "ribosome_core_split",
        help="Directory where the augmented prepared dataset will be written.",
    )
    parser.add_argument(
        "--core-feature-type",
        default="Core",
        help="Feature type to split.",
    )
    parser.add_argument(
        "--core1-fraction",
        type=float,
        default=DEFAULT_CORE1_FRACTION,
        help="Fraction of each Core region assigned to Core1.",
    )
    args = parser.parse_args()

    core_count, core1_count, core2_count = split_core_regions(
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        core_feature_type=args.core_feature_type,
        core1_fraction=args.core1_fraction,
    )
    print(f"Read Core regions: {core_count}")
    print(f"Added Core1 regions: {core1_count}")
    print(f"Added Core2 regions: {core2_count}")
    print(f"Core1 fraction: {args.core1_fraction:.6g}")
    print(f"Wrote augmented dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
