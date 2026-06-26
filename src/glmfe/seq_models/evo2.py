# TODO:
# Current implementation performs one forward pass per target.
# Future optimization should batch prefixes of similar lengths.

from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np
import torch

from glmfe.seq_models.base import BaseSequenceModel


class Evo2SequenceModel(BaseSequenceModel):
    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.device = torch.device(device)

        from evo2 import Evo2

        self.evo = Evo2(model_name)
        self.evo.model.eval()

        self.model_id = model_name
        self.reconstruction_protocol = "autoregressive_next_base"
        self.dependency_autoregressive = True
        self.max_context_length = 1048576

        self.nucleotide_token_ids = torch.tensor(
            [65, 67, 71, 84],  # A C G T
            dtype=torch.long,
            device=self.device,
        )

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        return nullcontext()

    def predict_masked_base_probabilities(
        self,
        sequences: list[str],
        target_positions: list[int],
        batch_size: int,
    ) -> np.ndarray:

        if len(sequences) != len(target_positions):
            raise ValueError(
                "sequences and target_positions must have same length"
            )

        all_probs = []

        for sequence, target_position in zip(
            sequences,
            target_positions,
            strict=True,
        ):

            if target_position == 0:
                probs = np.full(4, 0.25, dtype=np.float32)
                all_probs.append(probs)
                continue

            prefix = sequence[:target_position]

            token_ids = torch.tensor(
                self.evo.tokenizer.tokenize(prefix),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                with self._autocast_context():
                    outputs, _ = self.evo(token_ids)

            logits = outputs[0]

            next_token_logits = logits[0, -1]

            nucleotide_logits = next_token_logits[
                self.nucleotide_token_ids
            ]

            nucleotide_probs = torch.softmax(
                nucleotide_logits.float(),
                dim=-1,
            )

            all_probs.append(
                nucleotide_probs.cpu().numpy()
            )

        return np.stack(all_probs)

    def dependency_tokenize(
        self,
        sequence: str,
        mask_position: int | None,
    ) -> object:
        sequence = sequence.replace("U", "T")
        if len(sequence) > self.max_context_length:
            raise ValueError(
                f"Context length {len(sequence)} exceeds "
                f"{self.max_context_length}"
            )
        if set(sequence) - set("ACGT"):
            raise ValueError("Evo2 contexts must contain only A/C/G/T/U")
        if mask_position is not None:
            raise ValueError(
                "Evo2 dependency maps use autoregressive substitution "
                "scoring and do not support masked inputs"
            )

        return torch.tensor(
            self.evo.tokenizer.tokenize(sequence),
            dtype=torch.long,
        )

    def dependency_forward(
        self,
        tokenized_sequences: list[object],
        batch_size: int,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not tokenized_sequences:
            return np.empty((0, 0, 4), dtype=np.float32)
        if not all(
            isinstance(tokens, torch.Tensor) and tokens.ndim == 1
            for tokens in tokenized_sequences
        ):
            raise TypeError("Evo2 dependency inputs must be 1D token tensors")

        token_lengths = {
            int(tokens.numel())
            for tokens in tokenized_sequences
            if isinstance(tokens, torch.Tensor)
        }
        if len(token_lengths) != 1:
            raise ValueError(
                "Evo2 dependency inputs must have equal token lengths"
            )
        sequence_length = token_lengths.pop()

        output_logits = []
        for batch_start in range(0, len(tokenized_sequences), batch_size):
            batch_tokens = tokenized_sequences[
                batch_start : batch_start + batch_size
            ]
            tokens = torch.stack(
                [token.to(self.device) for token in batch_tokens]
            )
            with torch.inference_mode():
                with self._autocast_context():
                    outputs, _ = self.evo(tokens)
            logits = outputs[0]
            nucleotide_token_ids = self.nucleotide_token_ids.to(logits.device)

            nucleotide_logits = torch.empty(
                (
                    logits.shape[0],
                    sequence_length,
                    4,
                ),
                dtype=torch.float32,
                device=logits.device,
            )
            nucleotide_logits[:, 0, :] = 0.0
            nucleotide_logits[:, 1:, :] = logits[
                :, :-1, nucleotide_token_ids
            ].float()
            output_logits.append(nucleotide_logits.cpu().numpy())

        result = np.concatenate(output_logits, axis=0)
        expected_shape = (
            len(tokenized_sequences),
            sequence_length,
            4,
        )
        if result.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected Evo2 dependency logits shape "
                f"{result.shape}; expected {expected_shape}"
            )
        return result

    def prepare_for_training(self, lora_config: dict) -> None:
        raise NotImplementedError("Training not yet implemented for Evo2")

    def get_trainable_parameters(self) -> filter:
        raise NotImplementedError("Training not yet implemented for Evo2")

    def compute_pretraining_loss(
        self, 
        sequences: list[str],
        is_start: list[bool] | None = None,
        is_end: list[bool] | None = None,
    ) -> object:
        raise NotImplementedError("Evo2 pretraining is not implemented yet.")



def load_evo2_model(
    model_name: str,
    device: str,
    cache_dir: Path | None = None,
) -> Evo2SequenceModel:
    if cache_dir is not None:
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_CACHE"] = str(cache_dir)

    if (
        device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"CUDA unavailable for device {device}"
        )

    return Evo2SequenceModel(
        model_name=model_name,
        device=device,
    )
