"""Run configured evaluation tasks and write their result tables.

The script loads the prepared dataset and model for inference tasks, dispatches
the requested tasks, and writes the run manifest and task outputs to a dedicated
folder.
"""

import json
import sys
from pathlib import Path

import yaml

SUPPORTED_TASKS = {"reconstruction", "dependency_maps", "block_scores"}
MODEL_TASKS = {"reconstruction", "dependency_maps"}


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python scripts/run_evaluation.py <run_config.yaml>"
        )

    # Resolve project imports and all configured relative paths from the repo root.
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    dependency_map_sources = (
        repository_root / "external" / "dependency_map" / "src",
        repository_root / "external" / "dependency-map" / "src",
    )
    for dependency_map_src in dependency_map_sources:
        if dependency_map_src.is_dir():
            sys.path.insert(0, str(dependency_map_src))

    run_config_path = repository_root / sys.argv[1]
    with run_config_path.open() as handle:
        run_config = yaml.safe_load(handle)

    run_tasks = run_config["run_tasks"]
    unsupported_tasks = sorted(set(run_tasks) - SUPPORTED_TASKS)
    if unsupported_tasks:
        raise ValueError(
            "Unsupported task(s) in run_tasks: "
            + ", ".join(unsupported_tasks)
        )

    outputs_root = repository_root / run_config["outputs_root"]
    overwrite = bool(run_config["overwrite"])
    run_dir = outputs_root / run_config["run_id"]
    model_tasks = [task for task in run_tasks if task in MODEL_TASKS]
    postprocess_only = len(model_tasks) == 0
    if postprocess_only:
        if not run_dir.is_dir():
            raise ValueError(
                "block_scores-only runs require an existing run directory "
                f"with dependency maps: {run_dir}"
            )
    else:
        # Reserve the output directory before expensive inference and never
        # overwrite an existing run with the same run_id unless overwrite=true
        # in the config file.
        run_dir.mkdir(parents=True, exist_ok=overwrite)

    dataset = None
    records = None
    regions = None
    model = None
    checkpoint_tag = None
    if not postprocess_only:
        from glmfe.datasets.prepared import load_prepared_dataset

        # Load and validate the canonical records and half-open annotation intervals.
        dataset = run_config["dataset"]
        prepared_dir = repository_root / dataset["prepared_dir"]
        records, regions = load_prepared_dataset(prepared_dir)

        if dataset["dataset_id"] == "rinalmo_test":
            split_files = list(prepared_dir.glob("splits_seed*.json"))
            if not split_files:
                raise FileNotFoundError(f"No splits_seed*.json file found in {prepared_dir}")
            
            split_file = split_files[0]
            with split_file.open() as handle:
                splits = json.load(handle)
                
            if "test" not in splits:
                raise ValueError(f"'test' split missing from {split_file}")
                
            test_indices = splits["test"]
            records = records.iloc[test_indices].copy()
            valid_record_ids = set(records["record_id"])
            regions = regions[regions["record_id"].isin(valid_record_ids)].copy()

        # Construct the model adapter.
        adapter = run_config["model"]["adapter"]
        weights_path = None
        model_config = run_config["model"][adapter]
        if adapter == "rinalmo":
            from glmfe.seq_models.rinalmo import load_rinalmo_model

            model = load_rinalmo_model(
                model_size=model_config["size"],
                weights_path=repository_root / model_config["weights"],
                device=model_config["device"],
                lora_weights_path=repository_root / model_config["lora_weights"] if "lora_weights" in model_config else None,
            )
        elif adapter == "evo2":
            from glmfe.seq_models.evo2 import load_evo2_model

            cache_dir = model_config["cache_dir"] if "cache_dir" in model_config else None
            model = load_evo2_model(
                model_name=model_config["model_name"],
                device=model_config["device"],
                cache_dir=repository_root / cache_dir if cache_dir else None,
            )
        elif adapter == "random":
            from glmfe.seq_models.random import RandomSequenceModel

            model = RandomSequenceModel(
                seed=model_config["seed"],
                max_context_length=model_config["max_context_length"],
            )
        else:
            raise ValueError(f"Unsupported model adapter: {adapter}")
        checkpoint_tag = weights_path.stem if weights_path else None

    task_results = {}
    for task in run_tasks:
        if task == "reconstruction":
            from glmfe.tasks.reconstruction import run_reconstruction

            reconstruction_dir = run_dir / "reconstruction"
            reconstruction_dir.mkdir(exist_ok=overwrite)
            plot_results = bool(run_config["reconstruction"]["plot_results"])

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
            if plot_results:
                from glmfe.tasks.plots import plot_reconstruction_results

                plot_reconstruction_results(per_region, reconstruction_dir / "plots")
            task_results["reconstruction"] = per_base
        elif task == "dependency_maps":
            from glmfe.tasks.dependency_maps import run_dependency_maps

            map_index = run_dependency_maps(
                records,
                regions,
                model,
                run_config["dependency_maps"],
                run_dir,
                run_config["run_id"],
                dataset["dataset_id"],
                model.model_id,
                overwrite,
            )
            task_results["dependency_maps"] = map_index
        elif task == "block_scores":
            from glmfe.tasks.block_scores import run_block_scores

            plot_results = bool(run_config["block_scores"]["plot_results"])
            if "dependency_maps" in task_results:
                map_index = task_results["dependency_maps"]
            else:
                import pandas as pd

                map_index_path = (
                    run_dir / "dependency_maps" / "map_index.parquet"
                )
                if not map_index_path.exists():
                    raise ValueError(
                        "block_scores requires dependency_maps earlier in "
                        "run_tasks or an existing "
                        "dependency_maps/map_index.parquet in the run "
                        f"directory: {run_dir}"
                    )
                map_index = pd.read_parquet(
                    map_index_path,
                    engine="pyarrow",
                )
                task_results["dependency_maps"] = map_index
            per_span, per_map = run_block_scores(
                map_index,
                run_config["block_scores"],
                run_dir,
                overwrite,
            )
            if plot_results:
                from glmfe.tasks.plots import plot_block_score_results

                plot_block_score_results(
                    per_span,
                    per_map,
                    map_index,
                    run_dir,
                )
            task_results["block_scores"] = (per_span, per_map)

    # Write the compact run manifest directly from the resolved configuration.
    manifest_path = run_dir / "manifest.json"
    if postprocess_only and manifest_path.exists():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        manifest_tasks = list(manifest["run_tasks"])
        if "block_scores" not in manifest_tasks:
            manifest_tasks.append("block_scores")
        manifest["run_tasks"] = manifest_tasks
    else:
        manifest = {
            "run_id": run_config["run_id"],
            "run_tasks": run_tasks,
        }
        if dataset is not None:
            manifest["dataset_id"] = dataset["dataset_id"]
            manifest["records_path"] = str(
                Path(dataset["prepared_dir"]) / "records.parquet"
            )
            manifest["regions_path"] = str(
                Path(dataset["prepared_dir"]) / "regions.parquet"
            )
            manifest["record_count"] = len(records)
            manifest["region_count"] = len(regions)
        if model is not None:
            manifest["model"] = run_config["model"]
    if "reconstruction" in run_tasks:
        manifest["reconstruction"] = run_config["reconstruction"]
        manifest["unique_target_count"] = len(task_results["reconstruction"])
    if "dependency_maps" in run_tasks:
        manifest["dependency_maps"] = run_config["dependency_maps"]
        manifest["dependency_map_count"] = len(task_results["dependency_maps"])
    if "block_scores" in run_tasks:
        per_span, per_map = task_results["block_scores"]
        manifest["block_scores"] = run_config["block_scores"]
        manifest["block_score_span_count"] = len(per_span)
        manifest["block_score_map_count"] = len(per_map)

    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    resolved_config_path = run_dir / "config.resolved.yaml"
    if not postprocess_only or overwrite or not resolved_config_path.exists():
        with resolved_config_path.open("w") as handle:
            yaml.safe_dump(run_config, handle, sort_keys=False)

    print(f"Run: {run_config['run_id']}")
    if records is not None:
        print(f"Records: {len(records)}")
    if regions is not None:
        print(f"Regions: {len(regions)}")
    if "reconstruction" in run_tasks:
        print("Unique annotated bases: " f"{len(task_results['reconstruction'])}")
    if "dependency_maps" in run_tasks:
        print("Dependency maps: " f"{len(task_results['dependency_maps'])}")
    if "block_scores" in run_tasks:
        per_span, per_map = task_results["block_scores"]
        print(
            "Block scores: "
            f"{len(per_span)} spans, {len(per_map)} maps"
        )
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
