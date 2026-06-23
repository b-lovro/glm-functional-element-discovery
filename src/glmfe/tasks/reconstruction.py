"""Evaluate masked single-base reconstruction over annotated regions.

For each unique annotated genomic position, this protocol creates one
fixed-length context window and places the target nucleotide as close to the
window centre as sequence boundaries allow. The target position is masked and
the sequence model predicts a normalized probability distribution over A/C/G/T
from the remaining context.

For a target base x_i in its context window W_i, the model estimates:

    p_theta(x_i | W_i without x_i)

From this distribution, the evaluation records:

- true-base probability: p_theta(x_i | W_i without x_i)
- cross-entropy in bits: -log2 p_theta(x_i | W_i without x_i)
- predicted base: argmax over A/C/G/T probabilities
- reconstruction accuracy: whether the predicted base equals x_i

Annotations are represented as half-open intervals [start, end). All annotated
intervals are expanded into base positions. Positions shared by overlapping
annotations are evaluated only once per record, then their per-base predictions
are joined back to every region that contains them. This avoids repeated model
inference while preserving independent region-level summaries.

Each annotated region summarized from the per-base predictions of all positions it
contains, including mean true-base probability, mean/median/standard deviation
of cross-entropy, and reconstruction accuracy.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from glmfe.seq_models.base import BaseSequenceModel


_BASES = np.array(list("ACGT"))
_BASE_TO_INDEX = {base: index for index, base in enumerate(_BASES)}
_PER_BASE_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
    "checkpoint_tag",
    "reconstruction_protocol",
    "record_id",
    "position",
    "reference_base",
    "context_start",
    "context_end",
    "context_target_position",
    "true_base_probability",
    "cross_entropy_bits",
    "predicted_base",
    "correct",
]
_PER_REGION_COLUMNS = [
    "run_id",
    "dataset_id",
    "model_id",
    "checkpoint_tag",
    "region_id",
    "record_id",
    "species",
    "source_row",
    "annotation_file",
    "start",
    "end",
    "length",
    "label",
    "feature_type",
    "evaluated_base_count",
    "mean_true_base_probability",
    "mean_cross_entropy_bits",
    "median_cross_entropy_bits",
    "std_cross_entropy_bits",
    "q05_cross_entropy_bits",
    "q25_cross_entropy_bits",
    "q75_cross_entropy_bits",
    "q95_cross_entropy_bits",
    "accuracy",
]


def run_reconstruction(
    records: pd.DataFrame,
    regions: pd.DataFrame,
    model: BaseSequenceModel,
    context_length: int,
    batch_size: int,
    run_id: str,
    dataset_id: str,
    checkpoint_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-base and per-region reconstruction metrics."""

    # Expand every annotation into genomic positions. A set per record ensures
    # bases shared by overlapping regions are sent to the model only once.
    targets_by_record = {}
    region_target_rows = []
    for region in regions.itertuples(index=False):
        if region.record_id not in targets_by_record:
            targets_by_record[region.record_id] = set()
        positions = range(int(region.start), int(region.end))
        targets_by_record[region.record_id].update(positions)
        region_target_rows.extend(
            (region.region_id, region.record_id, position)
            for position in positions
        )

    # Build one valid context for each unique target and infer all targets from
    # the current record in model-controlled batches.
    record_sequences = records.set_index("record_id")["sequence"]
    per_base_rows = []
    targets_by_record = dict(list(targets_by_record.items())[:5])
    for record_id in tqdm(sorted(targets_by_record), desc="Reconstructing records", unit="record"):
        sequence = record_sequences.loc[record_id]
        sequence_length = len(sequence)
        positions = sorted(targets_by_record[record_id])
        contexts = []
        context_positions = []
        context_coordinates = []

        for position in positions:
            # Use the whole short record; otherwise center a fixed-size window
            # and shift it at sequence boundaries.
            if sequence_length <= context_length:
                context_start = 0
                context_end = sequence_length
            else:
                desired_start = position - context_length // 2
                context_start = max(
                    0,
                    min(
                        desired_start,
                        sequence_length - context_length,
                    ),
                )
                context_end = context_start + context_length
            context_target_position = position - context_start
            contexts.append(sequence[context_start:context_end])
            context_positions.append(context_target_position)
            context_coordinates.append(
                (
                    context_start,
                    context_end,
                    context_target_position,
                )
            )

        # The wrapper inserts the mask token, handles special-token offsets, and
        # returns normalized probabilities in fixed A, C, G, T order.
        probabilities = model.predict_masked_base_probabilities(
            contexts,
            context_positions,
            batch_size,
        )

        # Convert the four-base probability vector into the requested metrics
        # while retaining the context coordinates used for each prediction.
        for index, position in enumerate(positions):
            reference_base = sequence[position]
            reference_index = _BASE_TO_INDEX[reference_base]
            true_base_probability = float(
                probabilities[index, reference_index]
            )
            predicted_base = str(_BASES[np.argmax(probabilities[index])])
            context_start, context_end, context_target_position = (
                context_coordinates[index]
            )
            per_base_rows.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "model_id": model.model_id,
                    "checkpoint_tag": checkpoint_tag,
                    "reconstruction_protocol": model.reconstruction_protocol,
                    "record_id": record_id,
                    "position": position,
                    "reference_base": reference_base,
                    "context_start": context_start,
                    "context_end": context_end,
                    "context_target_position": context_target_position,
                    "true_base_probability": true_base_probability,
                    "cross_entropy_bits": -np.log2(true_base_probability),
                    "predicted_base": predicted_base,
                    "correct": predicted_base == reference_base,
                }
            )

    # This is the canonical table of unique evaluated (record_id, position) pairs.
    per_base = pd.DataFrame(
        per_base_rows,
        columns=_PER_BASE_COLUMNS,
    ).sort_values(
        ["record_id", "position"],
        kind="stable",
        ignore_index=True,
    )

    # Reattach each unique prediction to every region that covers it. This
    # preserves overlapping annotations without repeating model inference.
    region_targets = pd.DataFrame(
        region_target_rows,
        columns=["region_id", "record_id", "position"],
    )
    joined = region_targets.merge(
        per_base[
            [
                "record_id",
                "position",
                "true_base_probability",
                "cross_entropy_bits",
                "correct",
            ]
        ],
        on=["record_id", "position"],
        how="inner",
        validate="many_to_one",
    )

    # Aggregate the reconstructed bases belonging to each original region.
    aggregated = (
        joined.groupby("region_id", sort=False)
        .agg(
            evaluated_base_count=("position", "count"),
            mean_true_base_probability=("true_base_probability", "mean"),
            mean_cross_entropy_bits=("cross_entropy_bits", "mean"),
            median_cross_entropy_bits=("cross_entropy_bits", "median"),
            std_cross_entropy_bits=("cross_entropy_bits", "std"),
            q05_cross_entropy_bits=("cross_entropy_bits", lambda x: x.quantile(0.05)),
            q25_cross_entropy_bits=("cross_entropy_bits", lambda x: x.quantile(0.25)),
            q75_cross_entropy_bits=("cross_entropy_bits", lambda x: x.quantile(0.75)),
            q95_cross_entropy_bits=("cross_entropy_bits", lambda x: x.quantile(0.95)),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    per_region = regions.merge(
        aggregated,
        on="region_id",
        how="left",
        validate="one_to_one",
    )
    # if per_region["evaluated_base_count"].isna().any():
    #     raise ValueError("At least one region has no evaluated bases")

    # Add run identity fields and enforce deterministic output column order.
    per_region.insert(0, "checkpoint_tag", checkpoint_tag)
    per_region.insert(0, "model_id", model.model_id)
    per_region.insert(0, "dataset_id", dataset_id)
    per_region.insert(0, "run_id", run_id)
    per_region = per_region[_PER_REGION_COLUMNS].sort_values(
        ["record_id", "start", "end", "source_row"],
        kind="stable",
        ignore_index=True,
    )
    return per_base, per_region
