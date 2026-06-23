# TODO:
# Current implementation performs one forward pass per target.
# Future optimization should batch prefixes of similar lengths.

import numpy as np
import torch
from evo2 import Evo2

from glmfe.seq_models.base import BaseSequenceModel


class Evo2SequenceModel(BaseSequenceModel):
    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.device = torch.device(device)

        self.evo = Evo2(model_name)
        self.evo.model.eval()

        self.model_id = model_name
        self.reconstruction_protocol = "autoregressive_next_base"
        self.max_context_length = 1048576

        self.nucleotide_token_ids = torch.tensor(
            [65, 67, 71, 84],  # A C G T
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
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
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


def load_evo2_model(
    model_name: str,
    device: str,
) -> Evo2SequenceModel:

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