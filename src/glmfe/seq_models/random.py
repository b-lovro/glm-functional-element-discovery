from __future__ import annotations

import numpy as np

from glmfe.seq_models.base import BaseSequenceModel


class RandomSequenceModel(BaseSequenceModel):
    def __init__(self, seed: int, max_context_length: int) -> None:
        self.model_id = "random"
        self.reconstruction_protocol = "random"
        self.max_context_length = max_context_length
        self.rng = np.random.default_rng(seed)

    def predict_masked_base_probabilities(
        self,
        sequences: list[str],
        target_positions: list[int],
        batch_size: int,
    ) -> np.ndarray:
        if len(sequences) != len(target_positions):
            raise ValueError("sequences and target_positions must have equal length")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for sequence, target_position in zip(sequences, target_positions):
            if target_position < 0 or target_position >= len(sequence):
                raise ValueError("target position is outside its sequence")

        probabilities = self.rng.random((len(sequences), 4))
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def dependency_tokenize(
        self,
        sequence: str,
        mask_position: int | None,
    ) -> object:
        sequence = sequence.replace("U", "T")
        if len(sequence) > self.max_context_length:
            raise ValueError( f"Context length {len(sequence)} exceeds " f"{self.max_context_length}")
        if set(sequence) - set("ACGT"):
            raise ValueError("Random model contexts must contain only A/C/G/T/U")
        if mask_position is not None:
            if not 0 <= mask_position < len(sequence):
                raise ValueError(
                    f"Invalid mask position {mask_position} "
                    f"for sequence length {len(sequence)}"
                )
            sequence = (sequence[:mask_position] + "_" + sequence[mask_position + 1 :])
        return sequence

    def dependency_forward(
        self,
        tokenized_sequences: list[object],
        batch_size: int,
    ) -> np.ndarray:
        sequence_lengths = {
            len(sequence) for sequence in tokenized_sequences if isinstance(sequence, str)
            }
        if len(sequence_lengths) != 1:
            raise ValueError("Random dependency inputs must have equal sequence lengths")
        sequence_length = sequence_lengths.pop()

        logits = []
        for batch_start in range(0, len(tokenized_sequences), batch_size):
            batch_end = min(len(tokenized_sequences), batch_start + batch_size)
            logits.append(self.rng.random((batch_end - batch_start, sequence_length, 4)))
        return np.concatenate(logits, axis=0)
