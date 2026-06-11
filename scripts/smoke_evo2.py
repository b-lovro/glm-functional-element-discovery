import os
from pathlib import Path

# Set HuggingFace cache directory to download weights into the workspace folder
models_dir = Path(__file__).resolve().parents[1] / "models" / "evo2"
os.environ["HF_HOME"] = str(models_dir)
os.environ["HF_HUB_CACHE"] = str(models_dir)

import torch

# --- Monkey Patch for vtx compatibility with PyTorch 2.1.0 ---
# The newer 'vtx' package uses a PyTorch 2.4 feature called `add_safe_globals`.
# PyTorch 2.1.0 doesn't have this, which causes a crash during checkpoint loading.
# We inject a harmless dummy function here so vtx can safely skip it!
if not hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals = lambda *args, **kwargs: None
# PyTorch 2.1's `weights_only=True` is too strict and crashes on EVO2 checkpoints.
# We intercept torch.load and force it to False since we trust the weights.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import vortex.ops.attn_interface
_old_fwd = vortex.ops.attn_interface.flash_attn_gpu.fwd
def _patched_fwd(*args, **kwargs):
    new_args = list(args)
    # vtx args: 0:q, 1:k, 2:v, 3:out, 4:alibi, 5:drop, 6:scale, 7:causal, 8:win_l, 9:win_r, 10:softcap, 11:ret_sm, 12:gen
        
    # flash-attn==2.3.2 expects exactly 11 args (no softcap, no alibi_slopes).
    # We must remove softcap (index 10) and alibi_slopes (index 4).
    # We pop the higher index first so it doesn't shift the lower index!
    if len(new_args) == 13:
        new_args.pop(10) # Remove softcap
        new_args.pop(4)  # Remove alibi_slopes
            
    # Call the C++ backend
    res = _old_fwd(*new_args, **kwargs)

    # flash-attn==2.3.2 returns 8 elements: (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state)
    # vtx expects exactly 4 return values: (out, softmax_lse, S_dmask, rng_state)
    # These are at indexes 0, 5, 6, and 7!
    if isinstance(res, (list, tuple)) and len(res) >= 8:
        return (res[0], res[5], res[6], res[7])
            
    return res
        
vortex.ops.attn_interface.flash_attn_gpu.fwd = _patched_fwd

# This patching will not hinder performance and the two arguments we are stripping out are completely inert
# NEED TO PATCH ALSO BACKWARD FUNCTION .bwd
# -----------------------------------------------------------

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
            # 7B model can run in bfloat16 without Transformer Engine
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, _ = model(input_ids)
        else:
            outputs, _ = model(input_ids)

    logits = outputs[0]

    print("\nResults:")

    print('Logits: ', logits)
    print('Shape (batch, length, vocab): ', logits.shape)
            
    print("\nDone.")

if __name__ == "__main__":
    main()
