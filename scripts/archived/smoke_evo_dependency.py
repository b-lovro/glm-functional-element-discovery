import torch

from glmfe.seq_models.evo import load_evo_model

print("=" * 60)
print("Loading Evo...")
print("=" * 60)

model = load_evo_model(
    model_name="evo-1-8k-base",
    device="cuda:0",
)

sequence = "ACGTACGTACGT"

print("\nTokenizing...")

tokens = model.dependency_tokenize(
    sequence,
    mask_position=None,
)

print("Token tensor shape:", tokens.shape)

print("\nForward pass...")

outputs = model.dependency_forward(
    [tokens],
    batch_size=1,
)

print("Output shape:", outputs.shape)

print("Expected shape:")
print("(1,", len(sequence), ",4)")

print("\nSuccess!")