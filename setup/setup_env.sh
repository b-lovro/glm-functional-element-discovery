#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="group6-unified"
# if you are locally change these two to you actual paths
ENV_PREFIX="/opt/modules/i12g/anaconda/envs/$ENV_NAME"
PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"

EVO2_DIR="$PROJECT_DIR/external/EVO2"
RINALMO_DIR="$PROJECT_DIR/external/RiNALMo"

cd "$PROJECT_DIR"
mkdir -p external

[ -d "$EVO2_DIR" ] || git clone https://github.com/arcinstitute/evo2.git "$EVO2_DIR"
[ -d "$RINALMO_DIR" ] || git clone https://github.com/lbcb-sci/RiNALMo.git "$RINALMO_DIR"

cd "$RINALMO_DIR"

cp environment.yml environment_no_flash.yml
grep -v "flash-attn" environment_no_flash.yml > tmp.yml && mv tmp.yml environment_no_flash.yml

CONDA_CHANNEL_PRIORITY=flexible conda env create \
  --prefix="$ENV_PREFIX" \
  -f environment_no_flash.yml

conda install --prefix "$ENV_PREFIX" -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 -y

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"

"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir flash-attn==2.3.2 --no-build-isolation

"$ENV_PREFIX/bin/python" -m pip install "huggingface_hub[cli]"

cd "$PROJECT_DIR"

# Install RiNALMo first
"$ENV_PREFIX/bin/python" -m pip install -e external/RiNALMo

# Install EVO2
"$ENV_PREFIX/bin/python" -m pip install -e external/EVO2

echo "DONE: Unified environment created at $ENV_PREFIX"
