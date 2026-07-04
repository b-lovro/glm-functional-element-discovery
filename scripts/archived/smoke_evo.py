import torch
from evo import Evo


def print_gpu_memory(stage: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[{stage}] GPU allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB",
            flush=True,
        )


print("=" * 60, flush=True)
print("Starting Evo smoke test", flush=True)
print("=" * 60, flush=True)

print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

assert torch.cuda.is_available(), "CUDA is not available!"

device = "cuda:0"

print(f"Using device: {device}", flush=True)
print_gpu_memory("startup")

print("\n[1] Constructing Evo model...", flush=True)

model = Evo(
    "evo-1-8k-base",
    device=device,
)

print("[2] Evo model constructed successfully!", flush=True)
print_gpu_memory("after model load")

model.model.eval()

print("[3] Tokenizing sequence...", flush=True)

sequence = "ACGTACGT"

tokens = model.tokenizer.tokenize(sequence)

print(f"Sequence: {sequence}", flush=True)
print(f"Tokens: {tokens}", flush=True)

input_ids = torch.tensor(
    tokens,
    dtype=torch.long,
    device=device,
).unsqueeze(0)

print(f"Input tensor shape: {input_ids.shape}", flush=True)
print_gpu_memory("after tokenization")

print("[4] Starting forward pass...", flush=True)

with torch.no_grad():
    outputs = model.model(input_ids)

print("[5] Forward pass finished!", flush=True)
print_gpu_memory("after forward")

print("=" * 60, flush=True)
print("Output inspection", flush=True)
print("=" * 60, flush=True)

print(f"Returned object type: {type(outputs)}", flush=True)

if isinstance(outputs, tuple):
    print(f"Tuple length: {len(outputs)}", flush=True)

    for i, obj in enumerate(outputs):
        if torch.is_tensor(obj):
            print(
                f"Output {i}: tensor shape={obj.shape}, "
                f"dtype={obj.dtype}",
                flush=True,
            )
        else:
            print(
                f"Output {i}: {type(obj)}",
                flush=True,
            )
else:
    if torch.is_tensor(outputs):
        print(
            f"Tensor shape: {outputs.shape}",
            flush=True,
        )
    else:
        print(outputs, flush=True)

print("=" * 60, flush=True)
print("Smoke test completed successfully!", flush=True)
print("=" * 60, flush=True)