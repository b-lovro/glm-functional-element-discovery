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

    @abstractmethod
    def prepare_for_training(self, lora_config: dict) -> None:
        ...

    @abstractmethod
    def get_trainable_parameters(self) -> filter:
        ...

    @abstractmethod
    def compute_pretraining_loss(
        self, 
        sequences: list[str],
        is_start: list[bool] | None = None,
        is_end: list[bool] | None = None,
    ) -> object: # Returns torch.Tensor but we use object to avoid importing torch in base.py
        ...

