"""Run one configured reconstruction evaluation and write its result tables.

The script loads a prepared dataset and model from YAML configuration, evaluates
all unique bases covered by annotated regions, aggregates those predictions per
region, and writes the run manifest and CSV tables to a dedicated folder.
"""

import json
import sys
from pathlib import Path

import yaml


def main():
    # Require one explicit run configuration so every output is reproducible.
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python scripts/run_reconstruction.py <run_config.yaml>"
        )

    # Resolve project imports and all configured relative paths from the repo root.
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))

    from glmfe.datasets.prepared import load_prepared_dataset
    from glmfe.seq_models.rinalmo import load_rinalmo_model
    from glmfe.seq_models.random import RandomSequenceModel
    from glmfe.tasks.reconstruction import run_reconstruction

    run_config_path = repository_root / sys.argv[1]
    with run_config_path.open() as handle:
        run_config = yaml.safe_load(handle)

    # Load and validate the canonical records and half-open annotation intervals.
    dataset = run_config["dataset"]
    prepared_dir = repository_root / dataset["prepared_dir"]
    records, regions = load_prepared_dataset(prepared_dir)

    # Construct the model adapter
    adapter = run_config["model"]["adapter"]
    weights_path = None
    if adapter == "rinalmo":
        weights_path = repository_root / run_config["model"]["weights"]
        model = load_rinalmo_model(
            run_config["model"]["size"],
            weights_path,
            run_config["model"]["device"],
        )
    elif adapter == "random":
        model = RandomSequenceModel()
    else:
        raise ValueError(
            f"Unsupported model adapter: {adapter}"
        )
    
    outputs_root = repository_root / run_config["outputs_root"]

    # Reserve the output directory before expensive inference and never overwrite
    # an existing run with the same run_id.
    run_dir = outputs_root / run_config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    reconstruction_dir = run_dir / "reconstruction"
    reconstruction_dir.mkdir(exist_ok=False)

    # Reconstruct each unique annotated base once, then aggregate those results
    # back into every annotation region that contains the base.
    checkpoint_tag = weights_path.stem if weights_path else None
    per_base, per_region = run_reconstruction(
        records,
        regions,
        model,
        run_config["reconstruction"]["context_length"],
        run_config["reconstruction"]["batch_size"],
        run_config["run_id"],
        dataset["dataset_id"],
        checkpoint_tag,
    )

    # Write the compact run manifest directly from the resolved configuration.
    manifest = {
        "run_id": run_config["run_id"],
        "dataset_id": dataset["dataset_id"],
        "model": run_config["model"],
        "reconstruction": run_config["reconstruction"],
        "records_path": str(
            Path(dataset["prepared_dir"]) / "records.parquet"
        ),
        "regions_path": str(
            Path(dataset["prepared_dir"]) / "regions.parquet"
        ),
        "record_count": len(records),
        "region_count": len(regions),
        "unique_target_count": len(per_base),
    }
    with (run_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    with (run_dir / "config.resolved.yaml").open("w") as handle:
        yaml.safe_dump(
            run_config,
            handle,
            sort_keys=False,
        )

    per_base.to_csv(reconstruction_dir / "per_base.csv", index=False)
    per_region.to_csv(reconstruction_dir / "per_region.csv", index=False)

    print(f"Run: {run_config['run_id']}")
    print(f"Records: {len(records)}")
    print(f"Regions: {len(regions)}")
    print(f"Unique annotated bases: {len(per_base)}")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
