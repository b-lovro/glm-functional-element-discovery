"""TF-MoDISco (modisco-lite) motif discovery from genomic-LM attributions.

Pipeline::

    records/regions + model
        -> fixed-length windows centred on each annotated region
        -> one-hot sequences        (N, 4, L)
        -> per-base attribution track (N, 4, L)   [via compute_attributions]
        -> `modisco motifs` (CLI)   -> results.h5
        -> `modisco report` (CLI)   -> report/

We call the modisco-lite **command line** interface (``modisco motifs`` /
``modisco report``) rather than its Python API: the CLI is the documented,
version-stable surface and consumes plain ``.npy`` arrays in ``(N, 4, L)``
layout (the CLI transposes to ``(N, L, 4)`` internally).

IMPORTANT — attribution source is an open design choice.
TF-MoDISco needs per-base *contribution* scores, but a masked-reconstruction /
dependency genomic LM does not emit those directly. The conversion lives behind
a single hook, :func:`compute_attributions`, with a clearly-labelled placeholder
default (deviation-from-uniform of the masked-reconstruction distribution). Swap
that implementation once the score definition is settled.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from glmfe.seq_models.base import BaseSequenceModel

# One-hot channel order shared with the reconstruction task.
_BASES = np.array(list("ACGT"))
_BASE_TO_INDEX = {base: index for index, base in enumerate("ACGT")}


def _fixed_window(sequence: str, center: int, window_length: int) -> str:
    """Return a `window_length` slice centred on `center`, 'N'-padded at edges.

    Padding keeps every example the same length L, which `modisco motifs`
    requires. Padding bases are 'N' so they one-hot to all-zeros and are
    excluded from the attribution track.
    """
    half = window_length // 2
    start = center - half
    end = start + window_length

    left_pad = max(0, -start)
    right_pad = max(0, end - len(sequence))
    core = sequence[max(0, start):min(len(sequence), end)]
    window = "N" * left_pad + core + "N" * right_pad
    # Guard against off-by-one from odd window lengths / clamping.
    return window[:window_length].ljust(window_length, "N")


def _one_hot(sequence: str) -> np.ndarray:
    """One-hot encode to shape (4, L); non-ACGT positions are all-zero."""
    one_hot = np.zeros((4, len(sequence)), dtype=np.float32)
    for position, base in enumerate(sequence):
        index = _BASE_TO_INDEX.get(base)
        if index is not None:
            one_hot[index, position] = 1.0
    return one_hot


def compute_attributions(
    model: BaseSequenceModel,
    windows: list[str],
    batch_size: int,
    method: str = "reconstruction",
) -> np.ndarray:
    """ATTRIBUTION HOOK — turn model outputs into a (N, 4, L) contribution track.

    The exact score definition is intentionally not fixed yet. The default
    ``"reconstruction"`` method is a documented placeholder: for each position it
    masks that base, reads the model's A/C/G/T distribution, and uses the
    deviation from the uniform 0.25 baseline as the per-base contribution.
    Positions that are not canonical A/C/G/T contribute zero.
    """
    if method != "reconstruction":
        raise ValueError(
            f"Unknown attribution method {method!r}; "
            "only 'reconstruction' is implemented (this is the hook to extend)."
        )

    attributions = np.zeros((len(windows), 4, len(windows[0])), dtype=np.float32)
    for window_index, window in enumerate(
        tqdm(windows, desc="MoDISco attributions", unit="window")
    ):
        window_length = len(window)
        # Mask every position of this (equal-length) window in one call.
        probabilities = model.predict_masked_base_probabilities(
            [window] * window_length,
            list(range(window_length)),
            batch_size,
        )  # (L, 4) in A/C/G/T order
        track = (probabilities - 0.25).astype(np.float32).T  # (4, L)
        # Zero out padded / non-ACGT positions so they cannot seed motifs.
        canonical = np.array([base in _BASE_TO_INDEX for base in window])
        track[:, ~canonical] = 0.0
        attributions[window_index] = track
    return attributions


def build_modisco_inputs(
    records: pd.DataFrame,
    regions: pd.DataFrame,
    model: BaseSequenceModel,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build modisco inputs without touching the modisco CLI (unit-testable).

    Returns ``(one_hot, attributions, window_index)`` where the arrays are shaped
    ``(N, 4, L)`` and ``window_index`` records the provenance of each window.
    """
    window_length = int(config["window_length"])
    if window_length <= 0:
        raise ValueError("modisco.window_length must be positive")
    if window_length > model.max_context_length:
        raise ValueError(
            f"modisco.window_length {window_length} exceeds model "
            f"max_context_length {model.max_context_length}"
        )
    batch_size = int(config["batch_size"])

    record_sequences = records.set_index("record_id")["sequence"]
    windows: list[str] = []
    index_rows = []
    for region in regions.itertuples(index=False):
        sequence = record_sequences.loc[region.record_id]
        center = (int(region.start) + int(region.end)) // 2
        window = _fixed_window(sequence, center, window_length)
        windows.append(window)
        index_rows.append(
            {
                "region_id": region.region_id,
                "record_id": region.record_id,
                "label": region.label,
                "region_start": int(region.start),
                "region_end": int(region.end),
                "window_center": center,
            }
        )

    if not windows:
        raise ValueError("No regions available to build modisco windows")

    one_hot = np.stack([_one_hot(window) for window in windows]).astype(np.float32)
    attributions = compute_attributions(
        model,
        windows,
        batch_size,
        method=config["attribution_method"] if "attribution_method" in config else "reconstruction",
    )
    window_index = pd.DataFrame(index_rows)
    return one_hot, attributions, window_index


def _run_cli(command: list[str]) -> None:
    """Run a modisco CLI command, surfacing a helpful error if it is missing."""
    if shutil.which(command[0]) is None:
        raise FileNotFoundError(
            f"'{command[0]}' CLI not found. Install it with "
            "`pip install modisco-lite` (added to requirements.txt)."
        )
    subprocess.run(command, check=True)


def run_modisco(
    records: pd.DataFrame,
    regions: pd.DataFrame,
    model: BaseSequenceModel,
    config: dict,
    run_dir: Path,
    overwrite: bool,
) -> dict:
    """Full task: build inputs, save arrays, then run `modisco motifs`/`report`."""
    output_dir = run_dir / "modisco"
    output_dir.mkdir(parents=True, exist_ok=overwrite)

    one_hot, attributions, window_index = build_modisco_inputs(
        records, regions, model, config
    )

    ohe_path = output_dir / "one_hot.npy"
    attr_path = output_dir / "attributions.npy"
    results_path = output_dir / "modisco_results.h5"
    np.save(ohe_path, one_hot)
    np.save(attr_path, attributions)
    window_index.to_csv(output_dir / "window_index.csv", index=False)

    window_length = int(config["window_length"])
    max_seqlets = int(config["max_seqlets_per_metacluster"])
    allow_missing_cli = bool(config["allow_missing_cli"]) if "allow_missing_cli" in config else False

    summary = {
        "window_count": int(one_hot.shape[0]),
        "window_length": window_length,
        "one_hot_path": str(ohe_path),
        "attributions_path": str(attr_path),
        "results_path": None,
        "report_dir": None,
        "cli_ran": False,
    }

    if shutil.which("modisco") is None and allow_missing_cli:
        print(
            "[modisco] 'modisco' CLI not found; wrote inputs only "
            f"({one_hot.shape} arrays). Install modisco-lite on the cluster to "
            "run motif discovery."
        )
        return summary

    _run_cli([
        "modisco", "motifs",
        "-s", str(ohe_path),
        "-a", str(attr_path),
        "-n", str(max_seqlets),
        "-o", str(results_path),
        "-w", str(window_length),
    ])
    summary["results_path"] = str(results_path)
    summary["cli_ran"] = True

    if "run_report" not in config or bool(config["run_report"]):
        report_dir = output_dir / "report"
        report_dir.mkdir(exist_ok=True)
        _run_cli([
            "modisco", "report",
            "-i", str(results_path),
            "-o", str(report_dir),
            "-s", str(report_dir),
        ])
        summary["report_dir"] = str(report_dir)

    return summary
