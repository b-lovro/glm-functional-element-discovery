from pathlib import Path
import numpy as np
import pandas as pd


RECORD_COLUMNS = [
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
REGION_COLUMNS = [
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


def read_single_fasta(path):
    lines = path.read_text().splitlines()
    sequence = "".join(lines[1:]).upper().replace("U", "T")
    if not sequence:
        raise ValueError(f"FASTA sequence is empty: {path}")
    invalid_bases = sorted(set(sequence) - set("ACGT"))
    if invalid_bases:
        raise ValueError(f"Invalid bases in {path}: {''.join(invalid_bases)}")
    return sequence


def prepare_dataset(input_dir, output_dir):
    annotation_paths = sorted(input_dir.glob("*_cleaned_matches.csv"))

    records = []
    regions = []
    suffix = "_cleaned_matches.csv"

    for annotation_path in annotation_paths:
        species = annotation_path.name.removesuffix(suffix)
        fasta_path = input_dir / f"{species}.fasta"

        sequence = read_single_fasta(fasta_path)
        sequence_length = len(sequence)
        annotations = pd.read_csv(annotation_path)
        required_columns = {"start", "end", "label", "type"}
        missing_columns = required_columns - set(annotations.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {annotation_path}: {sorted(missing_columns)}"
            )

        for column in ("start", "end"):
            values = pd.to_numeric(annotations[column], errors="coerce")
            annotations[column] = values.astype("int64")

        for source_row, row in annotations.iterrows():
            start = int(row["start"])
            end = int(row["end"]) + 1
            regions.append(
                {
                    "region_id": f"{species}:{source_row}",
                    "record_id": species,
                    "species": species,
                    "source_row": source_row,
                    "annotation_file": annotation_path.name,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "label": row["label"],
                    "feature_type": row["type"],
                }
            )

        a_count = sequence.count("A")
        c_count = sequence.count("C")
        g_count = sequence.count("G")
        t_count = sequence.count("T")
        records.append(
            {
                "record_id": species,
                "species": species,
                "fasta_file": fasta_path.name,
                "sequence": sequence,
                "sequence_length": sequence_length,
                "a_count": a_count,
                "c_count": c_count,
                "g_count": g_count,
                "t_count": t_count,
                "gc_fraction": (g_count + c_count) / sequence_length,
                "region_count": len(annotations),
            }
        )

    records_frame = pd.DataFrame(records, columns=RECORD_COLUMNS).sort_values(
        "record_id", kind="stable"
    )
    regions_frame = pd.DataFrame(regions, columns=REGION_COLUMNS).sort_values(
        ["record_id", "start", "end", "source_row"], kind="stable"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.parquet"
    regions_path = output_dir / "regions.parquet"
    records_frame.to_parquet(records_path, engine="pyarrow", index=False)
    regions_frame.to_parquet(regions_path, engine="pyarrow", index=False)
    return records_path, regions_path, len(records_frame), len(regions_frame)


def main():
    repository_root = Path(__file__).resolve().parents[1]
    input_dir = repository_root / "data" / "raw" / "ribosome"
    output_dir = repository_root / "data" / "prepared" / "ribosome"
    records_path, regions_path, record_count, region_count = prepare_dataset(
        input_dir, output_dir
    )
    print(f"Prepared {record_count} records and {region_count} regions")
    print(f"records: {records_path}")
    print(f"regions: {regions_path}")


if __name__ == "__main__":
    main()
