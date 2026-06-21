import os
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
models_dir = PROJECT_DIR / "models" / "evo2"
os.environ["HF_HOME"] = str(models_dir)
os.environ["HF_HUB_CACHE"] = str(models_dir)

import torch
import torch.nn.functional as F
from evo2 import Evo2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evo2_7b")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    print("=" * 80)
    print("EVO2 frozen-forward + probe-backward smoke test")
    print("=" * 80)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("device:", torch.cuda.get_device_name(0))
    print("seq_len:", args.seq_len)
    print("steps:", args.steps)

    print("\nLoading frozen EVO2...")
    evo = Evo2(args.model)
    evo.model.eval()

    # Tiny trainable probe on top of frozen EVO2 logits.
    # This tests backward/optimizer on A40, but NOT EVO2 weight training.
    probe = torch.nn.Linear(512, 512).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    seq = ("ACGT" * ((args.seq_len + 3) // 4))[: args.seq_len]
    tokens = torch.tensor(
        evo.tokenizer.tokenize(seq),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    print("tokens:", tuple(tokens.shape))

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        # EVO2 local wrapper is inference/no_grad; keep it frozen intentionally.
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, _ = evo(tokens)
                frozen_logits = outputs[0].detach()

        # Train small probe to predict next token from frozen EVO2 logits.
        probe_logits = probe(frozen_logits)

        loss = F.cross_entropy(
            probe_logits[:, :-1, :].float().reshape(-1, probe_logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )

        print(f"step {step}: loss = {loss.item():.6f}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        print(f"step {step}: grad_norm = {float(grad_norm):.6f}")

        opt.step()

        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"step {step}: peak cuda memory = {peak_gb:.2f} GB")

    print("\nProbe backward smoke test OK.")
    print("Note: this tested backward through the probe only, not EVO2 weights.")


if __name__ == "__main__":
    main()