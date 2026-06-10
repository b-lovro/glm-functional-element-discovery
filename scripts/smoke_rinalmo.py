from pathlib import Path

import torch

from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo


weights_path = Path(__file__).resolve().parents[1] / "models" / "rinalmo" / "rinalmo_micro_pretrained.pt"
config_name = "micro"
max_len = 512

if not weights_path.exists():
    raise FileNotFoundError(f"Missing RiNALMo weights: {weights_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq = "AUGGCUACGUAGCUAGCUAGCUAGCUAGCUA"[:max_len]

print("=" * 80)
print("RiNALMo smoke test")
print("=" * 80)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", device)
print("config:", config_name)
print("weights:", weights_path)
print("sequence length:", len(seq))
print("sequence preview:", seq[:100])

print("\nBuilding model...")
config = model_config(config_name)
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
