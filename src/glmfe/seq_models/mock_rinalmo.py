"""CPU mock of the RiNALMo adapter for local development.

The real RiNALMo model needs the `rinalmo` package, flash-attn, CUDA and a
GPU with real weights (see `setup/setup_rinalmo_env.sh`). None of that runs on
a laptop. This module provides a tiny, randomly-initialised stand-in that:

* implements the exact `BaseSequenceModel` interface RiNALMo does (in fact it
  reuses `RiNALMoSequenceModel` for reconstruction / dependency / loss logic),
* runs on CPU with only `torch` + `numpy` (already in `requirements.txt`),
* needs no downloaded weights and no `rinalmo`/flash-attn install.

Outputs are meaningless — the point is to exercise the *code paths*
(reconstruction, dependency maps, the pretraining loop, checkpointing) so you
can catch bugs locally before pushing to the cluster where the real model runs.

Wire it in with `model.adapter: mock_rinalmo` (see
`configs/runs/mock_rinalmo_reco_test.yaml`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from glmfe.seq_models.rinalmo import RiNALMoSequenceModel

# Vocabulary mirroring the tokens the RiNALMo adapter relies on. The special
# tokens come first so `<cls>` sits at index 0 (prepended) and the base tokens
# cover every character validated by the real adapter.
_SPECIAL_TOKENS = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
_BASE_TOKENS = list("ACGTNRYKMSWBDHVI-")
_VOCAB = _SPECIAL_TOKENS + _BASE_TOKENS

# Per-size hidden width, purely to loosely resemble the real size ordering.
_MODEL_SIZES = {
    "micro": 32,
    "mega": 64,
}


class MockAlphabet:
    """Minimal stand-in for `rinalmo.data.alphabet.Alphabet`.

    Only exposes the attributes/methods `RiNALMoSequenceModel` uses:
    `tkn_to_idx`, `mask_idx`, `pad_idx` and `batch_tokenize`.
    """

    def __init__(self) -> None:
        self.tkn_to_idx = {token: index for index, token in enumerate(_VOCAB)}
        self.idx_to_tkn = {index: token for token, index in self.tkn_to_idx.items()}
        self.cls_idx = self.tkn_to_idx["<cls>"]
        self.eos_idx = self.tkn_to_idx["<eos>"]
        self.unk_idx = self.tkn_to_idx["<unk>"]
        self.mask_idx = self.tkn_to_idx["<mask>"]
        self.pad_idx = self.tkn_to_idx["<pad>"]

    @property
    def vocab_size(self) -> int:
        return len(_VOCAB)

    def batch_tokenize(self, sequences: list[str]) -> list[list[int]]:
        """Return `<cls> ... <eos>` token-id lists, one per sequence.

        Matches the real adapter's assumption that tokenization adds one
        leading and one trailing special token (hence the `+1`/`1:-1`/`-2`
        offsets in `RiNALMoSequenceModel`).
        """
        tokenized = []
        for sequence in sequences:
            body = [self.tkn_to_idx.get(char, self.unk_idx) for char in sequence]
            tokenized.append([self.cls_idx] + body + [self.eos_idx])
        return tokenized


class MockRiNALMo(nn.Module):
    """Tiny embedding + linear head returning RiNALMo-shaped logits."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.embedding(tokens)
        hidden = torch.tanh(self.encoder(hidden))
        # Match the real model's return contract: {"logits": (B, L, vocab)}.
        return {"logits": self.lm_head(hidden)}


class MockRiNALMoSequenceModel(RiNALMoSequenceModel):
    """RiNALMo adapter backed by the mock model.

    Inherits reconstruction / dependency / pretraining-loss logic unchanged and
    only overrides the LoRA/PEFT training hooks so local training needs no
    `peft` install: every parameter is simply left trainable.
    """

    def prepare_for_training(self, lora_config: dict) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self.model.train()
        trainable = sum(p.numel() for p in self.get_trainable_parameters())
        print(
            f"[mock_rinalmo] LoRA skipped; training all "
            f"{trainable / 1e6:.3f} M mock parameters"
        )

    def get_trainable_parameters(self) -> filter:
        return filter(lambda p: p.requires_grad, self.model.parameters())


def load_mock_rinalmo_model(
    model_size: str,
    device: str = "cpu",
    seed: int = 0,
) -> MockRiNALMoSequenceModel:
    """Construct a CPU mock that behaves like `load_rinalmo_model`."""
    if model_size not in _MODEL_SIZES:
        raise ValueError(
            f"Unknown mock RiNALMo size {model_size!r}; "
            f"expected one of {sorted(_MODEL_SIZES)}"
        )

    torch_device = torch.device(device)
    torch.manual_seed(seed)

    alphabet = MockAlphabet()
    model = MockRiNALMo(alphabet.vocab_size, _MODEL_SIZES[model_size])
    model = model.to(torch_device)
    model.eval()

    return MockRiNALMoSequenceModel(
        model,
        alphabet,
        model_size,
        torch_device,
    )
