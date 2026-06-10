```bash
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="group6-rinalmo"
PROJECT_DIR="$HOME/Documents/Faks/semester_2/ML4RG/Project/glm-functional-element-discovery"
RINALMO_DIR="$PROJECT_DIR/external/RiNALMo"

echo "Project dir:  $PROJECT_DIR"
echo "RiNALMo dir:  $RINALMO_DIR"
echo "Conda env:    $ENV_NAME"

cd "$RINALMO_DIR"

echo
echo "Creating environment_no_flash.yml..."
cp environment.yml environment_no_flash.yml

python - <<'PY'
from pathlib import Path

path = Path("environment_no_flash.yml")
lines = path.read_text().splitlines()

new_lines = []
for line in lines:
    if "flash-attn" in line:
        print("Removing:", line)
        continue
    new_lines.append(line)

path.write_text("\n".join(new_lines) + "\n")
PY

echo
echo "Creating conda environment: $ENV_NAME"
conda env create --name "$ENV_NAME" -f environment_no_flash.yml

echo
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo
echo "Checking PyTorch before CUDA compiler install..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

echo
echo "Installing CUDA compiler inside conda env..."
conda install -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo
echo "Checking nvcc..."
which nvcc
nvcc --version
echo "CUDA_HOME=$CUDA_HOME"

echo
echo "Installing flash-attn..."
pip install flash-attn==2.3.2 --no-build-isolation

echo
echo "Installing RiNALMo editable..."
cd "$PROJECT_DIR"
pip install -e external/RiNALMo

echo
echo "Final verification..."
python - <<'PY'
import torch
import flash_attn
from rinalmo.pretrained import get_pretrained_model

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__)

model, alphabet = get_pretrained_model(model_name="micro-v1")
print("RiNALMo micro loaded OK")
PY

echo
echo "DONE. Environment installed: $ENV_NAME"
```
