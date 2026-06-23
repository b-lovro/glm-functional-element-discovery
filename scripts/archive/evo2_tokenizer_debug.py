# scripts/archive/evo2_tokenizer_debug.py

import os
from pathlib import Path

models_dir = Path(__file__).resolve().parents[1] / "models" / "evo2"
os.environ["HF_HOME"] = str(models_dir)
os.environ["HF_HUB_CACHE"] = str(models_dir)

from evo2 import Evo2

model = Evo2("evo2_7b")

print("A:", model.tokenizer.tokenize("A"))
print("C:", model.tokenizer.tokenize("C"))
print("G:", model.tokenizer.tokenize("G"))
print("T:", model.tokenizer.tokenize("T"))

print("ACGT:", model.tokenizer.tokenize("ACGT"))

print("Tokenizer type:", type(model.tokenizer))
print("Tokenizer attrs:", dir(model.tokenizer))
