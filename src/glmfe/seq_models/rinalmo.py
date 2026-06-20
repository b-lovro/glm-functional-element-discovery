from pathlib import Path

import numpy as np
import torch
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo

from glmfe.seq_models.base import BaseSequenceModel


class RiNALMoSequenceModel(BaseSequenceModel):
    def __init__(
        self,
        model: RiNALMo,
        alphabet: Alphabet,
        model_size: str,
        device: torch.device,
    ):
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self.model_id = f"rinalmo-{model_size}"
        self.reconstruction_protocol = "masked_single_base"
        self.max_context_length = 1022
        self.nucleotide_token_indices = torch.tensor(
            [alphabet.tkn_to_idx[base] for base in "ACGT"],
            dtype=torch.int64,
            device=device,
        )

    def predict_masked_base_probabilities(
        self,
        sequences: list[str],
        target_positions: list[int],
        batch_size: int,
    ) -> np.ndarray:
        if len(sequences) != len(target_positions):
            raise ValueError(
                "sequences and target_positions must have the same length"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not sequences:
            return np.empty((0, 4), dtype=np.float32)

        probabilities = []
        for batch_start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[batch_start : batch_start + batch_size]
            batch_positions = target_positions[
                batch_start : batch_start + batch_size
            ]
            for sequence, target_position in zip(
                batch_sequences,
                batch_positions,
                strict=True,
            ):
                if len(sequence) > self.max_context_length:
                    raise ValueError(
                        f"Context length {len(sequence)} exceeds "
                        f"{self.max_context_length}"
                    )
                if set(sequence) - set("ACGT"):
                    raise ValueError("RiNALMo contexts must contain only A/C/G/T")
                if not 0 <= target_position < len(sequence):
                    raise ValueError(
                        f"Invalid context target position {target_position} "
                        f"for context length {len(sequence)}"
                    )

            tokens = torch.tensor(
                self.alphabet.batch_tokenize(batch_sequences),
                dtype=torch.int64,
                device=self.device,
            )
            token_positions = torch.tensor(
                batch_positions,
                dtype=torch.int64,
                device=self.device,
            ) + 1
            batch_indices = torch.arange(
                len(batch_sequences),
                dtype=torch.int64,
                device=self.device,
            )
            tokens[batch_indices, token_positions] = self.alphabet.mask_idx

            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(tokens)["logits"]
            masked_logits = logits[
                batch_indices,
                token_positions,
            ][:, self.nucleotide_token_indices]
            batch_probabilities = torch.softmax(
                masked_logits.float(),
                dim=-1,
            )
            probabilities.append(batch_probabilities.cpu().numpy())

        return np.concatenate(probabilities, axis=0)


def load_rinalmo_model(
    model_size: str,
    weights_path: Path,
    device: str,
) -> RiNALMoSequenceModel:
    if model_size != "micro":
        raise ValueError(f"Unsupported RiNALMo model size: {model_size}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing RiNALMo weights: {weights_path}")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for configured device {device}")

    config = model_config(model_size)
    alphabet = Alphabet(**config["alphabet"])
    model = RiNALMo(config)
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("RiNALMo checkpoint must contain a direct state dictionary")
    model.load_state_dict(state_dict)
    model = model.to(torch_device)
    model.eval()
    return RiNALMoSequenceModel(
        model,
        alphabet,
        model_size,
        torch_device,
    )
