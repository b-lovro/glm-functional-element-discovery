#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="group6-rinalmo"
# if you are locally change these two to you actual paths
ENV_PREFIX="/opt/modules/i12g/anaconda/envs/$ENV_NAME"
PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"
RINALMO_DIR="$PROJECT_DIR/external/RiNALMo"

cd "$PROJECT_DIR"
mkdir -p external

[ -d "$RINALMO_DIR" ] || git clone git@github.com:lbcb-sci/RiNALMo.git "$RINALMO_DIR"

cd "$RINALMO_DIR"
cp environment.yml environment_no_flash.yml
grep -v "flash-attn" environment_no_flash.yml > tmp.yml && mv tmp.yml environment_no_flash.yml

CONDA_CHANNEL_PRIORITY=flexible conda env create \
  --prefix="$ENV_PREFIX" \
  -f environment_no_flash.yml

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

conda install -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

which nvcc
nvcc --version

pip install flash-attn==2.3.2 --no-build-isolation

cd "$PROJECT_DIR"
pip install -e external/RiNALMo

python -c "import torch, flash_attn; from rinalmo.pretrained import get_pretrained_model; get_pretrained_model(model_name='micro-v1'); print('RiNALMo OK')"

echo "DONE: $ENV_NAME"