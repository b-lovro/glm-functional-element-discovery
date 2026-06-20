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
