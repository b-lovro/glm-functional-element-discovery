"""Run configured evaluation tasks and write their result tables.

The script loads a prepared dataset and model from YAML configuration, dispatches
the requested tasks, and writes the run manifest and task outputs to a dedicated
folder.
"""

import json
import sys
from pathlib import Path

import yaml

SUPPORTED_TASKS = {"reconstruction", "dependency_maps"}

def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python scripts/run_evaluation.py <run_config.yaml>"
        )

    # Resolve project imports and all configured relative paths from the repo root.
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    dependency_map_src = repository_root / "external" / "dependency_map" / "src"
    sys.path.insert(0, str(dependency_map_src))

    from glmfe.datasets.prepared import load_prepared_dataset
    from glmfe.seq_models.rinalmo import load_rinalmo_model
    from glmfe.seq_models.random import RandomSequenceModel
    from glmfe.tasks.reconstruction import run_reconstruction
    from glmfe.tasks.dependency_maps import run_dependency_maps

    run_config_path = repository_root / sys.argv[1]
    with run_config_path.open() as handle:
        run_config = yaml.safe_load(handle)

    run_tasks = run_config["run_tasks"]

    # Load and validate the canonical records and half-open annotation intervals.
    dataset = run_config["dataset"]
    prepared_dir = repository_root / dataset["prepared_dir"]
    records, regions = load_prepared_dataset(prepared_dir)

    # Construct the model adapter
    adapter = run_config["model"]["adapter"]
    weights_path = None
    model_config = run_config["model"][adapter]
    if adapter == "rinalmo":
        model = load_rinalmo_model(
            model_size=model_config["size"],
            weights_path=repository_root / model_config["weights"],
            device=model_config["device"],
        )
    elif adapter == "random":
        model = RandomSequenceModel(
            seed=model_config["seed"], 
            max_context_length=model_config["max_context_length"],
            )
    else:
        raise ValueError(f"Unsupported model adapter: {adapter}")

    outputs_root = repository_root / run_config["outputs_root"]

    # Reserve the output directory before expensive inference and never overwrite
    # an existing run with the same run_id unless overwrite=true in config file.
    overwrite = bool(run_config["overwrite"])
    run_dir = outputs_root / run_config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=overwrite)
    checkpoint_tag = weights_path.stem if weights_path else None
    task_results = {}
    for task in run_tasks:
        if task == "reconstruction":
            reconstruction_dir = run_dir / "reconstruction"
            reconstruction_dir.mkdir(exist_ok=overwrite)

            # Reconstruct each unique annotated base once, then aggregate those
            # results back into every annotation region that contains the base.
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
            per_base.to_csv(reconstruction_dir / "per_base.csv", index=False)
            per_region.to_csv(reconstruction_dir / "per_region.csv", index=False)
            task_results["reconstruction"] = per_base
        elif task == "dependency_maps":
            map_index = run_dependency_maps(
                records,
                model,
                run_config["dependency_maps"],
                run_dir,
                run_config["run_id"],
                dataset["dataset_id"],
                model.model_id,
                overwrite,
            )
            task_results["dependency_maps"] = map_index

    # Write the compact run manifest directly from the resolved configuration.
    manifest = {
        "run_id": run_config["run_id"],
        "dataset_id": dataset["dataset_id"],
        "model": run_config["model"],
        "run_tasks": run_tasks,
        "records_path": str(
            Path(dataset["prepared_dir"]) / "records.parquet"
        ),
        "regions_path": str(
            Path(dataset["prepared_dir"]) / "regions.parquet"
        ),
        "record_count": len(records),
        "region_count": len(regions),
    }
    if "reconstruction" in run_tasks:
        manifest["reconstruction"] = run_config["reconstruction"]
        manifest["unique_target_count"] = len(task_results["reconstruction"])
    if "dependency_maps" in run_tasks:
        manifest["dependency_maps"] = run_config["dependency_maps"]
        manifest["dependency_map_count"] = len(task_results["dependency_maps"])
        
    with (run_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    with (run_dir / "config.resolved.yaml").open("w") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)

    print(f"Run: {run_config['run_id']}")
    print(f"Records: {len(records)}")
    print(f"Regions: {len(regions)}")
    if "reconstruction" in run_tasks:
        print("Unique annotated bases: " f"{len(task_results['reconstruction'])}")
    if "dependency_maps" in run_tasks:
        print("Dependency maps: " f"{len(task_results['dependency_maps'])}")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
