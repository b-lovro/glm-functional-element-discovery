#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="group6-evo2"
# if you are locally change these two to you actual paths
ENV_PREFIX="/opt/modules/i12g/anaconda/envs/$ENV_NAME"
PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"
EVO2_DIR="$PROJECT_DIR/external/EVO2"

cd "$PROJECT_DIR"
mkdir -p external

[ -d "$EVO2_DIR" ] || git clone https://github.com/arcinstitute/evo2.git "$EVO2_DIR"

conda create --prefix "$ENV_PREFIX" python=3.11 -y

# A compatible PyTorch must be installed before flash attention
"$ENV_PREFIX/bin/python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$ENV_PREFIX/bin/python" -m pip install flash-attn==2.8.0.post2 --no-build-isolation
"$ENV_PREFIX/bin/python" -m pip install "huggingface_hub[cli]"

cd "$PROJECT_DIR"
"$ENV_PREFIX/bin/python" -m pip install -e external/EVO2

# Verify installation
"$ENV_PREFIX/bin/python" -m evo2.test.test_evo2_generation --model_name evo2_7b

echo "DONE: $ENV_PREFIX"
