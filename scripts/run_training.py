"""Run configured pretraining and write checkpoint outputs.

The script loads the prepared dataset and model, dispatches
the training task, and logs the results to wandb.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Configure deterministic algorithms if requested in future, but these might
    # slow down training, so we just set the seeds for now to be safe.
    # torch.use_deterministic_algorithms(True)


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/run_training.py <training_config.yaml>")

    # Resolve project imports and all configured relative paths from the repo root.
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))

    run_config_path = repository_root / sys.argv[1]
    with run_config_path.open() as handle:
        run_config = yaml.safe_load(handle)

    # Fail fast: Access strictly by key
    training_config = run_config["training"]
    seed = training_config["seed"]
    set_seed(seed)

    outputs_root = repository_root / run_config["outputs_root"]
    overwrite = bool(run_config["overwrite"])
    resume = bool(training_config["resume"])
    run_dir = outputs_root / run_config["run_id"]

    run_dir.mkdir(parents=True, exist_ok=overwrite or resume)

    from glmfe.datasets.prepared import load_prepared_dataset

    # Load and validate the canonical records and half-open annotation intervals.
    dataset_config = run_config["dataset"]
    prepared_dir = repository_root / dataset_config["prepared_dir"]
    records, regions = load_prepared_dataset(prepared_dir)

    # Construct the model adapter.
    adapter = run_config["model"]["adapter"]
    model_config = run_config["model"][adapter]
    
    if adapter == "rinalmo":
        from glmfe.seq_models.rinalmo import load_rinalmo_model

        model = load_rinalmo_model(
            model_size=model_config["size"],
            weights_path=repository_root / model_config["weights"],
            device=model_config["device"],
        )
    elif adapter == "evo":
        from glmfe.seq_models.evo import load_evo_model

        cache_dir = model_config["cache_dir"] if "cache_dir" in model_config else None
        model = load_evo_model(
            model_name=model_config["model_name"],
            device=model_config["device"],
            cache_dir=repository_root / cache_dir if cache_dir else None,
        )
    elif adapter == "mock_rinalmo":
        # CPU stand-in for local development; see mock_rinalmo.py.
        from glmfe.seq_models.mock_rinalmo import load_mock_rinalmo_model

        model = load_mock_rinalmo_model(
            model_size=model_config["size"],
            device=model_config["device"] if "device" in model_config else "cpu",
            seed=model_config["seed"] if "seed" in model_config else 0,
        )
    else:
        raise ValueError(f"Unsupported model adapter for training: {adapter}")

    from glmfe.tasks.pretraining import run_pretraining

    run_pretraining(
        records=records,
        model=model,
        config=run_config,
        run_dir=run_dir,
        prepared_dir=prepared_dir,
    )

    # Write the compact run manifest directly from the resolved configuration.
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_config["run_id"],
        "dataset_id": dataset_config["dataset_id"],
        "model": run_config["model"],
        "training": training_config,
        "record_count": len(records),
    }

    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    resolved_config_path = run_dir / "config.resolved.yaml"
    with resolved_config_path.open("w") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)

    print(f"Training Run: {run_config['run_id']} completed.")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
