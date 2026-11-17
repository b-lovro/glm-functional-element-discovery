#!/usr/bin/env python3
"""Shared utilities, taxonomy constants, and run defaults for TF-MoDISco analysis scripts."""

from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_RUN_DIR: Path = Path("outputs/tfmodisco_runs/all_seq_n_24")
DEFAULT_INPUT_DIR: Path = Path("outputs/tfmodisco_inputs/all_seq")
DEFAULT_RECORDS_PARQUET: Path = Path("data/prepared/ribosome/records.parquet")
DEFAULT_REGIONS_PARQUET: Path = Path("data/prepared/ribosome/regions.parquet")

# Phylogenetically grouped 17 Mammalian Taxonomic Orders (Superorder -> Order)
MAMMALIAN_ORDER_PHYLOGENY: List[Tuple[str, str]] = [
    # Euarchontoglires
    ("Euarchontoglires", "Primates"),
    ("Euarchontoglires", "Rodentia"),
    ("Euarchontoglires", "Lagomorpha"),
    # Laurasiatheria
    ("Laurasiatheria", "Artiodactyla"),
    ("Laurasiatheria", "Perissodactyla"),
    ("Laurasiatheria", "Carnivora"),
    ("Laurasiatheria", "Pholidota"),
    ("Laurasiatheria", "Chiroptera"),
    ("Laurasiatheria", "Eulipotyphla"),
    # Afrotheria
    ("Afrotheria", "Proboscidea"),
    ("Afrotheria", "Sirenia"),
    ("Afrotheria", "Hyracoidea"),
    ("Afrotheria", "Macroscelidea"),
    # Xenarthra
    ("Xenarthra", "Cingulata"),
    ("Xenarthra", "Pilosa"),
    # Marsupialia
    ("Marsupialia", "Didelphimorphia"),
    ("Marsupialia", "Dasyuromorphia"),
]

ORDER_TO_SUPERORDER: Dict[str, str] = {
    order_name: superorder for superorder, order_name in MAMMALIAN_ORDER_PHYLOGENY
}

COMP_LABELS: List[str] = [
    "Core_Promoter_TSS",
    "Upstream_Promoter",
    "5prime_ETS",
    "ITS1",
    "ITS2",
    "Terminator",
    "Distal_IGS",
]


def ensure_repo_root_on_path() -> Path:
    """Ensure the repository root is present on sys.path for `scripts.*` imports."""
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return REPO_ROOT


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dynamic_device() -> torch.device:
    """Allocate PyTorch compute device dynamically across CUDA, MPS, and CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pattern_sort_key(name: str) -> Tuple[int, int]:
    """Return `(is_negative, pattern_index)` sort key for dotted or short pattern names."""
    text = str(name)
    is_neg = 1 if (text.startswith("neg_patterns.") or text.startswith("neg_")) else 0
    parts = text.split(".")[-1].split("_")
    return (is_neg, int(parts[-1]))


def format_pattern_display_name(name: str) -> str:
    """Format a TF-MoDISco pattern identifier cleanly, preserving a `neg_` prefix if negative."""
    text = str(name)
    if text.startswith("neg_patterns."):
        return f"neg_{text.split('.')[-1]}"
    if text.startswith("pos_patterns."):
        return text.split(".", 1)[1]
    return text


def extract_tax_order(record_id: str, strict: bool = True) -> str:
    """Extract taxonomic order from a `Class-Order-Species-Accession-Morph` record_id."""
    parts = str(record_id).split("-")
    if len(parts) < 2:
        raise ValueError(
            f"Unexpected record_id format (expected '-' separated fields): {record_id!r}"
        )
    tax_order = parts[1].strip()
    if strict and tax_order not in ORDER_TO_SUPERORDER:
        raise ValueError(f"Unknown mammalian order {tax_order!r} in record_id {record_id!r}")
    return tax_order


def compute_shannon_ic_bits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute per-position Shannon Information Content in bits (0.0 to 2.0) along the last axis."""
    p_safe = np.clip(probs, eps, 1.0)
    p_safe = p_safe / np.sum(p_safe, axis=-1, keepdims=True)
    entropy = -np.sum(p_safe * np.log2(p_safe), axis=-1)
    return np.clip(2.0 - entropy, 0.0, 2.0)
