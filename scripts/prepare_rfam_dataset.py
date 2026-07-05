from pathlib import Path

import pandas as pd


def parse_stockholm(path: Path):
    """
    Parse an Rfam Stockholm file.

    Returns
    -------
    list[dict]
        Each entry contains:
            accession
            family_name
            sequence_id
            sequence
    """

    sequences = []

    accession = None
    family_name = None

    with path.open(
        "r",
        encoding="latin-1",
        ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#=GF AC"):
                accession = line.split()[-1]
                continue

            if line.startswith("#=GF ID"):
                family_name = line.split()[-1]
                continue

            if line.startswith("#"):
                continue

            if line == "//":
                accession = None
                family_name = None
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            sequence_id, sequence = parts

            sequences.append(
                {
                    "accession": accession,
                    "family_name": family_name,
                    "sequence_id": sequence_id,
                    "sequence": sequence,
                }
            )

    return sequences

def clean_sequence(sequence: str) -> str | None:
    """
    Convert an aligned Rfam sequence into a DNA sequence suitable for RiNALMo.

    Returns None if the sequence contains unsupported characters.
    """

    sequence = sequence.upper()

    # Remove Stockholm alignment gaps
    sequence = sequence.replace("-", "")
    sequence = sequence.replace(".", "")

    # Convert RNA to DNA
    sequence = sequence.replace("U", "T")

    allowed = set("ACGT")

    if set(sequence) - allowed:
        return None

    return sequence


def main():
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

    repository_root = Path(__file__).resolve().parents[1]

    stockholm_path = (
        repository_root
        / "data"
        / "raw"
        / "Rfam.seed"
        / "Rfam.seed"
    )

    sequences = parse_stockholm(stockholm_path)

    print(f"Parsed {len(sequences)} aligned sequences\n")

    seen_sequences = set()
    valid_records = []

    invalid = 0
    duplicates = 0
    too_long = 0

    for entry in sequences:
        cleaned = clean_sequence(entry["sequence"])

        if cleaned is None:
            invalid += 1
            continue

        if len(cleaned) > 1022:
            too_long += 1
            continue

        if cleaned in seen_sequences:
            duplicates += 1
            continue

        seen_sequences.add(cleaned)

        entry["cleaned_sequence"] = cleaned
        valid_records.append(entry)

    print(f"Total aligned sequences : {len(sequences)}")
    print(f"Invalid sequences       : {invalid}")
    print(f"Too long (>1022)        : {too_long}")
    print(f"Duplicate sequences     : {duplicates}")
    print(f"Final benchmark         : {len(valid_records)}")

    records = []
    regions = []

    for entry in valid_records:
        sequence = entry["cleaned_sequence"]

        record_id = f"{entry['accession']}:{entry['sequence_id']}"

        sequence_length = len(sequence)

        a = sequence.count("A")
        c = sequence.count("C")
        g = sequence.count("G")
        t = sequence.count("T")

        records.append(
            {
                "record_id": record_id,
                "species": entry["sequence_id"],
                "fasta_file": "Rfam.seed",
                "sequence": sequence,
                "sequence_length": sequence_length,
                "a_count": a,
                "c_count": c,
                "g_count": g,
                "t_count": t,
                "gc_fraction": (g + c) / sequence_length,
                "region_count": 1,
            }
        )

        regions.append(
            {
                "region_id": record_id + ":0",
                "record_id": record_id,
                "species": entry["sequence_id"],
                "source_row": 0,
                "annotation_file": "Rfam.seed",
                "start": 0,
                "end": sequence_length,
                "length": sequence_length,
                "label": entry["family_name"],
                "feature_type": "rfam",
            }
        )

    records_frame = pd.DataFrame(records, columns=RECORD_COLUMNS)
    regions_frame = pd.DataFrame(regions, columns=REGION_COLUMNS)

    output_dir = (
        repository_root
        / "data"
        / "prepared"
        / "rfam"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    records_frame.to_parquet(
        output_dir / "records.parquet",
        engine="pyarrow",
        index=False,
    )

    regions_frame.to_parquet(
        output_dir / "regions.parquet",
        engine="pyarrow",
        index=False,
    )

    print("\nDone!")
    print(f"records : {len(records_frame)}")
    print(f"regions : {len(regions_frame)}")
    print(f"Saved to {output_dir}")

    print("\nExample:")

    example = valid_records[0]

    print(example["sequence"])
    print(example["cleaned_sequence"])


if __name__ == "__main__":
    main()