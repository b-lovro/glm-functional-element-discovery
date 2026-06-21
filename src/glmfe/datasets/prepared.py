from pathlib import Path

import pandas as pd


_RECORD_COLUMNS = [
    "record_id",
    "species",
    "fasta_file",
    "sequence",
    "sequence_length",
    "a_count",
    "c_count",
    "g_count",
    "t_count",
    "gc_fraction",
    "region_count",
]
_REGION_COLUMNS = [
    "region_id",
    "record_id",
    "species",
    "source_row",
    "annotation_file",
    "start",
    "end",
    "length",
    "label",
    "feature_type",
]


def load_prepared_dataset(
    prepared_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_parquet(
        prepared_dir / "records.parquet",
        engine="pyarrow",
    )
    regions = pd.read_parquet(
        prepared_dir / "regions.parquet",
        engine="pyarrow",
    )

    missing_record_columns = set(_RECORD_COLUMNS) - set(records.columns)
    if missing_record_columns:
        raise ValueError(
            f"Missing records columns: {sorted(missing_record_columns)}"
        )
    missing_region_columns = set(_REGION_COLUMNS) - set(regions.columns)
    if missing_region_columns:
        raise ValueError(
            f"Missing regions columns: {sorted(missing_region_columns)}"
        )
    if records["record_id"].duplicated().any():
        raise ValueError("records.record_id must be unique")

    record_ids = set(records["record_id"])
    unknown_record_ids = set(regions["record_id"]) - record_ids
    if unknown_record_ids:
        raise ValueError(
            f"Unknown regions.record_id values: {sorted(unknown_record_ids)}"
        )

    for row in records.itertuples(index=False):
        if not isinstance(row.sequence, str):
            raise ValueError(f"Sequence is not a string for record {row.record_id}")
        invalid_bases = sorted(set(row.sequence) - set("ACGT"))
        if invalid_bases:
            raise ValueError(
                f"Invalid bases for record {row.record_id}: {''.join(invalid_bases)}"
            )

    sequence_lengths = records.set_index("record_id")["sequence_length"]
    for row in regions.itertuples(index=False):
        sequence_length = int(sequence_lengths.loc[row.record_id])
        if not 0 <= row.start < row.end <= sequence_length:
            raise ValueError(
                f"Invalid region {row.region_id}: [{row.start}, {row.end}) "
                f"for sequence length {sequence_length}"
            )

    return records, regions
