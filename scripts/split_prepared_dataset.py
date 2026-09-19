"""
Split a prepared GLM-FE dataset into smaller chunks.

Each chunk contains a valid records.parquet and regions.parquet pair,
allowing evaluation jobs to be run independently.
"""

from pathlib import Path

import pandas as pd


SOURCE = Path("data/prepared/rfam")
DESTINATION = Path("data/prepared")
N_CHUNKS = 20


def main():

    records = pd.read_parquet(SOURCE / "records.parquet")
    regions = pd.read_parquet(SOURCE / "regions.parquet")

    chunk_size = (len(records) + N_CHUNKS - 1) // N_CHUNKS

    print(f"Total records : {len(records)}")
    print(f"Chunk size    : {chunk_size}")

    for chunk in range(N_CHUNKS):

        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, len(records))

        chunk_records = records.iloc[start:end].copy()

        record_ids = set(chunk_records["record_id"])

        chunk_regions = regions[
            regions["record_id"].isin(record_ids)
        ].copy()

        out_dir = DESTINATION / f"rfam_part{chunk+1:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        chunk_records.to_parquet(
            out_dir / "records.parquet",
            index=False,
        )

        chunk_regions.to_parquet(
            out_dir / "regions.parquet",
            index=False,
        )

        print(
            f"{out_dir.name}: "
            f"{len(chunk_records):6d} records, "
            f"{len(chunk_regions):6d} regions"
        )


if __name__ == "__main__":
    main()