#!/usr/bin/env python
"""Merge multiple partial evaluation run directories into a single unified run directory.

Useful when long-running evaluation tasks (such as whole-sequence dependency maps
and block scores) are partitioned across multiple cluster jobs due to walltime limits.
This script combines:
  - `dependency_maps/maps/<record_id>/*.npz` (via symlinks or copies)
  - `dependency_maps/map_index.parquet`
  - `block_scores/per_span.parquet` (if present)
  - `block_scores/per_map.parquet` (if present)
  - `manifest.json` (if present)

Usage:
  python scripts/merge_run_parts.py \\
    --parts outputs/runs/rinalmo_p1 outputs/runs/rinalmo_p2 outputs/runs/rinalmo_p3 outputs/runs/rinalmo_p4 \\
    --output-run outputs/runs/rinalmo_tfmodisco_pre_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

import pandas as pd


def merge_run_parts(
    parts: Sequence[Path],
    output_run: Path,
    copy_files: bool = False,
    overwrite: bool = True,
) -> None:
    """Merge partial run directories into a unified output run directory."""
    if len(parts) < 2:
        raise ValueError(
            f"Expected at least 2 run parts to merge, got {len(parts)}"
        )

    for p in parts:
        if not p.is_dir():
            raise FileNotFoundError(f"Run part directory not found: {p}")

    output_run.mkdir(parents=True, exist_ok=overwrite)
    print(f"Merging {len(parts)} run parts into: {output_run}")

    # 1. Dependency Maps
    output_dep_dir = output_run / "dependency_maps"
    output_maps_dir = output_dep_dir / "maps"
    output_maps_dir.mkdir(parents=True, exist_ok=True)

    map_indices = []
    total_record_dirs = 0

    for part in parts:
        part_dep_dir = part / "dependency_maps"
        part_maps_dir = part_dep_dir / "maps"
        part_index_file = part_dep_dir / "map_index.parquet"

        if not part_index_file.exists():
            raise FileNotFoundError(
                f"Missing map_index.parquet in partition: {part_dep_dir}"
            )

        df_idx = pd.read_parquet(part_index_file)
        df_idx["run_id"] = output_run.name
        map_indices.append(df_idx)

        if part_maps_dir.is_dir():
            for rec_dir in part_maps_dir.iterdir():
                if not rec_dir.is_dir():
                    continue

                dest_rec_dir = output_maps_dir / rec_dir.name
                if dest_rec_dir.exists():
                    if overwrite:
                        if dest_rec_dir.is_symlink():
                            dest_rec_dir.unlink()
                        else:
                            shutil.rmtree(dest_rec_dir)
                    else:
                        raise FileExistsError(
                            f"Record map directory already exists: {dest_rec_dir}"
                        )

                if copy_files:
                    shutil.copytree(rec_dir, dest_rec_dir)
                else:
                    dest_rec_dir.symlink_to(rec_dir.resolve())
                total_record_dirs += 1

    unified_map_index = pd.concat(map_indices, ignore_index=True)
    unified_map_index_path = output_dep_dir / "map_index.parquet"
    unified_map_index.to_parquet(unified_map_index_path, index=False)
    print(
        f"  [dependency_maps] Merged {total_record_dirs} record map folder(s) "
        f"and {len(unified_map_index)} maps -> {unified_map_index_path}"
    )

    # 2. Block Scores (if present in parts)
    has_block_scores = any((p / "block_scores").is_dir() for p in parts)
    if has_block_scores:
        output_block_dir = output_run / "block_scores"
        output_block_dir.mkdir(parents=True, exist_ok=True)

        spans_list = []
        maps_list = []

        for part in parts:
            part_span_file = part / "block_scores" / "per_span.parquet"
            part_map_file = part / "block_scores" / "per_map.parquet"

            if not part_span_file.exists() or not part_map_file.exists():
                raise FileNotFoundError(
                    f"Partition {part} is missing block_scores parquet files."
                )

            df_spans = pd.read_parquet(part_span_file)
            df_spans["run_id"] = output_run.name
            spans_list.append(df_spans)

            df_maps = pd.read_parquet(part_map_file)
            df_maps["run_id"] = output_run.name
            maps_list.append(df_maps)

        unified_spans = pd.concat(spans_list, ignore_index=True)
        unified_maps = pd.concat(maps_list, ignore_index=True)

        unified_spans_path = output_block_dir / "per_span.parquet"
        unified_maps_path = output_block_dir / "per_map.parquet"
        unified_spans.to_parquet(unified_spans_path, index=False)
        unified_maps.to_parquet(unified_maps_path, index=False)
        print(
            f"  [block_scores] Merged {len(unified_spans)} spans and "
            f"{len(unified_maps)} map summaries -> {output_block_dir}"
        )

    # 3. Manifest (if present in parts[0])
    manifest_file_p0 = parts[0] / "manifest.json"
    if manifest_file_p0.exists():
        with manifest_file_p0.open("r") as f:
            manifest = json.load(f)

        manifest["run_id"] = output_run.name
        manifest["dependency_map_count"] = len(unified_map_index)
        if "record_count" in manifest:
            manifest["record_count"] = int(unified_map_index["record_id"].nunique())
        if has_block_scores:
            manifest["block_score_span_count"] = len(unified_spans)
            manifest["block_score_map_count"] = len(unified_maps)

        output_manifest_path = output_run / "manifest.json"
        with output_manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  [manifest] Wrote unified manifest -> {output_manifest_path}")

    # 4. Resolved config (if present in parts[0])
    resolved_config_p0 = parts[0] / "config.resolved.yaml"
    if resolved_config_p0.exists():
        output_resolved_path = output_run / "config.resolved.yaml"
        shutil.copy2(resolved_config_p0, output_resolved_path)

    print("DONE: Run parts merged successfully.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        required=True,
        type=Path,
        help="List of paths to partial run directories to merge.",
    )
    parser.add_argument(
        "--output-run",
        required=True,
        type=Path,
        help="Destination directory for the unified run outputs.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy .npz map folders instead of creating symlinks (default: False).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        default=True,
        help="Do not overwrite existing destination files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_run_parts(
        parts=args.parts,
        output_run=args.output_run,
        copy_files=args.copy,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
