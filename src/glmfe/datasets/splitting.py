import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_records(
    records: pd.DataFrame,
    split_config: dict,
    seed: int,
    prepared_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Split complete sequences in train/val/test and return all splits."""
    
    splits_cache_path = prepared_dir / f"splits_seed{seed}.json"
    
    if splits_cache_path.exists():
        with splits_cache_path.open() as handle:
            splits_indices = json.load(handle)
        
        splits = {}
        for split_name, indices in splits_indices.items():
            splits[split_name] = records.iloc[indices]
        print(f"Loaded dataset splits from cache: {splits_cache_path}")
        return splits
        
    print(f"Generating new dataset splits and caching to: {splits_cache_path}")
    
    # Shuffle integer positions securely to track exact row slices
    n = len(records)
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n).tolist()
    
    train_frac = split_config["train"]
    val_frac = split_config["val"]
    test_frac = split_config["test"]
    
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0):
        raise ValueError(f"Train/Val/Test fractions must sum to 1.0, got {total_frac}")

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    
    splits_indices = {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test": shuffled_indices[val_end:],
    }
    
    if "overfit" in split_config:
        overfit_frac = split_config["overfit"]
        overfit_end = max(1, int(n * overfit_frac))
        splits_indices["overfit"] = shuffled_indices[:overfit_end]
        
    with splits_cache_path.open("w") as handle:
        json.dump(splits_indices, handle)
        
    splits = {}
    for split_name, indices in splits_indices.items():
        splits[split_name] = records.iloc[indices]
        
    return splits
