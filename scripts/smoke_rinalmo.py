import argparse
import sys
from pathlib import Path

import torch


repo_root = Path(__file__).resolve().parents[1]
external_rinalmo = repo_root / "external" / "RiNALMo"
if external_rinalmo.exists():
    sys.path.insert(0, str(external_rinalmo))

from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo


parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=Path, default=repo_root / "weights" / "rinalmo_micro_pretrained.pt")
parser.add_argument("--config", type=str, default="micro", choices=["nano", "micro", "mega", "giga"])
parser.add_argument("--max-len", type=int, default=512)
args = parser.parse_args()

weights_path = args.weights.expanduser().resolve()
if not weights_path.exists():
    raise FileNotFoundError(f"Missing RiNALMo weights: {weights_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq = "AUGGCUACGUAGCUAGCUAGCUAGCUAGCUA"[: args.max_len]

print("=" * 80)
print("RiNALMo smoke test")
print("=" * 80)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", device)
print("config:", args.config)
print("weights:", weights_path)
print("sequence length:", len(seq))
print("sequence preview:", seq[:100])

print("\nBuilding model...")
config = model_config(args.config)
print("flash attention:", config.model.transformer.use_flash_attn)
alphabet = Alphabet(**config["alphabet"])
model = RiNALMo(config)

print("Loading local weights...")
state_dict = torch.load(weights_path, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
if isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Tokenizing...")
tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=device)
print("tokens shape:", tuple(tokens.shape))

print("Running forward pass...")
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        outputs = model(tokens)

representations = outputs["representation"]
pooled = representations.mean(dim=1)

print("output keys:", list(outputs.keys()))
print("representation shape:", tuple(representations.shape))
print("pooled embedding shape:", tuple(pooled.shape))
print("representation dtype:", representations.dtype)
print("\nDone.")
