import os
from pathlib import Path

# Cache EVO2 weights inside project workspace
models_dir = Path(__file__).resolve().parents[1] / "models" / "evo2"
os.environ["HF_HOME"] = str(models_dir)
os.environ["HF_HUB_CACHE"] = str(models_dir)

import torch

# Strict import checks: no monkey patches
import flash_attn
import flash_attn_2_cuda
import vortex.ops.attn_interface

from evo2 import Evo2


def main():
    model_name = "evo2_7b"
    max_len = 512

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq = "ACGT" * (max_len // 4)

    print("=" * 80)
    print("EVO2 smoke test")
    print("=" * 80)
    print("torch:", torch.__version__)
    print("torch cuda:", getattr(torch.version, "cuda", "N/A"))
    print("cuda available:", torch.cuda.is_available())
    print("flash_attn:", flash_attn.__version__)
    print("device:", device)
    print("model:", model_name)
    print("sequence length:", len(seq))
    print("sequence preview:", seq[:100])

    print("\nBuilding model...")
    model = Evo2(model_name)
    model.model.eval()

    print("Tokenizing...")
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    print("tokens shape:", tuple(input_ids.shape))

    print("Running forward pass...")
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, _ = model(input_ids)
        else:
            outputs, _ = model(input_ids)

    logits = outputs[0]

    print("\nResults:")
    print("logits:", logits)
    print("shape:", tuple(logits.shape))
    print("\nDone.")


if __name__ == "__main__":
    main()