from pathlib import Path

import numpy as np
import torch

from evo import Evo

from glmfe.seq_models.base import BaseSequenceModel


class EvoSequenceModel(BaseSequenceModel):
    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.device = torch.device(device)

        self.evo = Evo(model_name, device=str(self.device))
        self.model = self.evo.model
        self.model.eval()

        self.tokenizer = self.evo.tokenizer

        self.model_id = model_name
        self.reconstruction_protocol = "autoregressive_next_base"

        # Evo 8k checkpoint
        self.max_context_length = 8192

        self.nucleotide_token_ids = torch.tensor(
            [65, 67, 71, 84],
            dtype=torch.long,
            device=self.device,
        )

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

        probabilities = []

        for sequence, target_position in zip(
            sequences,
            target_positions,
            strict=True,
        ):

            if target_position == 0:
                probabilities.append(
                    np.full(4, 0.25, dtype=np.float32)
                )
                continue

            prefix = sequence[:target_position]

            token_ids = torch.tensor(
                self.tokenizer.tokenize(prefix),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    logits, _ = self.model(token_ids)

            next_logits = logits[0, -1]

            nucleotide_logits = next_logits[
                self.nucleotide_token_ids
            ]

            probs = torch.softmax(
                nucleotide_logits.float(),
                dim=-1,
            )

            probabilities.append(
                probs.cpu().numpy()
            )

        return np.stack(probabilities)
    # TODO:
    # Implement these methods once Evo reconstruction is validated.
    # Current development plan:
    #   1. Reconstruction (current)
    #   2. Dependency maps
    #   3. Pretraining / LoRA

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
            raise ValueError(
                "Evo contexts must contain only A/C/G/T/U"
            )

        if mask_position is not None:
            raise ValueError(
                "Evo dependency maps use autoregressive scoring "
                "and do not support masked inputs"
            )

        return torch.tensor(
            self.tokenizer.tokenize(sequence),
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
            raise TypeError("Evo1 dependency inputs must be 1D token tensors")

        token_lengths = {
            int(tokens.numel())
            for tokens in tokenized_sequences
            if isinstance(tokens, torch.Tensor)
        }
        if len(token_lengths) != 1:
            raise ValueError(
                "Evo1 dependency inputs must have equal token lengths"
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
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                ):
                    outputs=self.model(tokens)
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
                f"Unexpected Evo1 dependency logits shape "
                f"{result.shape}; expected {expected_shape}"
            )
        return result
        

    def prepare_for_training(
        self,
        lora_config: dict,
    ) -> None:
        raise NotImplementedError(
            "LoRA training is not yet implemented for Evo."
        )

    def get_trainable_parameters(
        self,
    ) -> filter:
        raise NotImplementedError(
            "LoRA training is not yet implemented for Evo."
        )

    def compute_pretraining_loss(
        self,
        sequences: list[str],
        is_start: list[bool] | None = None,
        is_end: list[bool] | None = None,
    ) -> object:
        raise NotImplementedError(
            "Pretraining loss is not yet implemented for Evo."
        )


def load_evo_model(
    model_name: str,
    device: str,
) -> EvoSequenceModel:

    if (
        device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"CUDA unavailable for {device}"
        )

    return EvoSequenceModel(
        model_name=model_name,
        device=device,
    )