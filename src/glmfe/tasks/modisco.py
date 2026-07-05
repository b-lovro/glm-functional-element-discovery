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

Attribution scores are derived in one hook, :func:`compute_attributions`. The
default ``reconstruction_ic`` treats the model's masked A/C/G/T marginal as a
soft in-silico-mutagenesis estimate, centres it, and weights it by information
content — chosen because TF-MoDISco builds motifs (CWMs) directly from the
per-base vector and localises seqlets from the present base's score, so the LM's
masked marginal maps naturally onto what the algorithm expects. A simpler
``reconstruction_uniform`` (``p - 0.25``) is kept for comparison. See the hook's
docstring for the exact TF-MoDISco mechanics this mirrors.
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


# Attribution methods understood by :func:`compute_attributions`.
_ATTRIBUTION_METHODS = ("reconstruction_ic", "reconstruction_uniform")
_LOG2_EPS = 1e-12


def _masked_marginals(
    model: BaseSequenceModel,
    window: str,
    batch_size: int,
) -> np.ndarray:
    """Return the model's masked A/C/G/T marginal at every position, shape (L, 4).

    Masks each position of the (equal-length) window in turn and reads the
    reconstruction distribution. This is a soft in-silico-mutagenesis estimate:
    ``p_i(b)`` approximates the effect of placing base ``b`` at position ``i``.
    """
    window_length = len(window)
    return model.predict_masked_base_probabilities(
        [window] * window_length,
        list(range(window_length)),
        batch_size,
    ).astype(np.float32)  # (L, 4) in A/C/G/T order


def compute_attributions(
    model: BaseSequenceModel,
    windows: list[str],
    batch_size: int,
    method: str = "reconstruction_ic",
) -> np.ndarray:
    """ATTRIBUTION HOOK — turn model outputs into a (N, 4, L) *hypothetical* track.

    TF-MoDISco (see ``modiscolite/tfmodisco.py``) consumes this as
    ``hypothetical_contribs``: it forms the actual contribution
    ``one_hot * hypothetical``, localises seqlets from that summed over bases
    (so only the *present* base's score localises motifs), auto-calibrates a
    Laplacian null over the score distribution, and builds each motif's CWM from
    the per-base vector. The two implications: background positions should score
    ~0, and the per-position 4-vector should look like a PWM column.

    Methods (config ``modisco.attribution_method``):

    - ``reconstruction_ic`` (default, recommended): mask each position, read the
      A/C/G/T marginal ``p_i``, centre it (``p_i - mean_b p_i`` so background
      ~0 and the vector sums to zero), and scale by information content
      ``IC_i = 2 - H(p_i)`` (bits) so confident/conserved positions dominate and
      uncertain ones collapse toward zero. The present base scores high exactly
      when the model confidently predicts the actual base, i.e. constraint.
    - ``reconstruction_uniform``: the simpler ``p_i - 0.25`` deviation-from-
      uniform baseline, kept for comparison.

    Positions that are not canonical A/C/G/T (e.g. 'N' padding) contribute zero.
    """
    if method not in _ATTRIBUTION_METHODS:
        raise ValueError(
            f"Unknown attribution method {method!r}; "
            f"expected one of {_ATTRIBUTION_METHODS}."
        )

    attributions = np.zeros((len(windows), 4, len(windows[0])), dtype=np.float32)
    for window_index, window in enumerate(
        tqdm(windows, desc="MoDISco attributions", unit="window")
    ):
        probabilities = _masked_marginals(model, window, batch_size)  # (L, 4)

        if method == "reconstruction_uniform":
            hypothetical = probabilities - 0.25  # (L, 4)
        else:  # reconstruction_ic
            centered = probabilities - probabilities.mean(axis=1, keepdims=True)
            entropy = -np.sum(
                probabilities * np.log2(probabilities + _LOG2_EPS), axis=1
            )  # (L,), bits in [0, 2]
            information_content = np.maximum(0.0, 2.0 - entropy)  # (L,)
            hypothetical = information_content[:, None] * centered  # (L, 4)

        track = hypothetical.astype(np.float32).T  # (4, L)
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
        method=config["attribution_method"] if "attribution_method" in config else "reconstruction_ic",
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
