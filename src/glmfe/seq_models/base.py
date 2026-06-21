from abc import ABC, abstractmethod

import numpy as np


class BaseSequenceModel(ABC):
    model_id: str
    reconstruction_protocol: str
    max_context_length: int

    @abstractmethod
    def predict_masked_base_probabilities(
        self,
        sequences: list[str],
        target_positions: list[int],
        batch_size: int,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def dependency_tokenize(
        self,
        sequence: str,
        mask_position: int | None,
    ) -> object:
        ...

    @abstractmethod
    def dependency_forward(
        self,
        tokenized_sequences: list[object],
        batch_size: int,
    ) -> np.ndarray:
        ...
