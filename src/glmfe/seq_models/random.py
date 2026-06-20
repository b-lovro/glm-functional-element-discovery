from __future__ import annotations

import numpy as np

from glmfe.seq_models.base import BaseSequenceModel


class RandomSequenceModel(BaseSequenceModel):
    def __init__(self) -> None:
        self.model_id = "random"
        self.reconstruction_protocol = "random"
        self.max_context_length = 1022
        self.rng = np.random.default_rng(44)

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

        # for sequence, target_position in zip(sequences, target_positions):
        #     print(target_position)
        #     print(len(sequence))
        #     if target_position < 0 or target_position >= len(sequence):
        #         print(target_position)
        #         print(len(sequence))
        #         raise ValueError("target position is outside its sequence")

        probabilities = self.rng.random((len(sequences), 4))
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities