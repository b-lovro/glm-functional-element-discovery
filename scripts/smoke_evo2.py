import os
from pathlib import Path

# Set HuggingFace cache directory to download weights into the workspace folder
models_dir = Path(__file__).resolve().parents[1] / "models" / "evo2"
os.environ["HF_HOME"] = str(models_dir)
os.environ["HF_HUB_CACHE"] = str(models_dir)

import torch
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
    print("torch cuda:", getattr(torch.version, 'cuda', 'N/A'))
    print("cuda available:", torch.cuda.is_available())
    print("device:", device)
    print("model:", model_name)
    print("sequence length:", len(seq))
    print("sequence preview:", seq[:100])

    print("\nBuilding model...")
    # Evo2 will automatically download and cache weights from HuggingFace
    model = Evo2(model_name)
    model = model.to(device)
    model.eval()

    print("Tokenizing...")
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).to(device)
    print("tokens shape:", tuple(input_ids.shape))

    print("Running forward pass...")
    with torch.no_grad():
        if device.type == "cuda":
            # 7B model can run in bfloat16 without Transformer Engine
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, _ = model(input_ids)
        else:
            outputs, _ = model(input_ids)

    logits = outputs[0]

    print("\nResults:")
    print("outputs shape (batch, length, vocab):", tuple(outputs.shape))
    print("logits shape (length, vocab):", tuple(logits.shape))
    print("logits dtype:", logits.dtype)
            
    print("\nDone.")

if __name__ == "__main__":
    main()
