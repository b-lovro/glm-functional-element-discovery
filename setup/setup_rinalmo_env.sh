#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="group6-rinalmo"
# if you are locally change these two to you actual paths
ENV_PREFIX="/opt/modules/i12g/anaconda/envs/$ENV_NAME"
PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"
RINALMO_DIR="$PROJECT_DIR/external/RiNALMo"
DEPENDENCY_MAP_DIR="$PROJECT_DIR/external/dependency_map"

cd "$PROJECT_DIR"
mkdir -p external

[ -d "$RINALMO_DIR" ] || git clone git@github.com:lbcb-sci/RiNALMo.git "$RINALMO_DIR"
[ -d "$DEPENDENCY_MAP_DIR" ] || git clone https://github.com/Turakar/dependency-map.git "$DEPENDENCY_MAP_DIR"

cd "$RINALMO_DIR"
cp environment.yml environment_no_flash.yml
grep -v "flash-attn" environment_no_flash.yml > tmp.yml && mv tmp.yml environment_no_flash.yml

CONDA_CHANNEL_PRIORITY=flexible conda env create \
  --prefix="$ENV_PREFIX" \
  -f environment_no_flash.yml

conda install --prefix "$ENV_PREFIX" "matplotlib<=3.8.4" -c nvidia \
  cuda-nvcc=11.8 cuda-cudart-dev=11.8 -y

CUDA_HOME="$ENV_PREFIX" PATH="$ENV_PREFIX/bin:$PATH" 
"$ENV_PREFIX/bin/python" -m pip install flash-attn==2.3.2 --no-build-isolation

cd "$PROJECT_DIR"
"$ENV_PREFIX/bin/python" -m pip install -e external/RiNALMo

# Patch dependency_map's pyproject.toml to avoid numpy dependency conflict with scikit-learn
sed -i 's/"numpy>=2"/"numpy<2"/g' external/dependency_map/pyproject.toml
"$ENV_PREFIX/bin/python" -m pip install -e external/dependency_map
"$ENV_PREFIX/bin/python" -m pip install kaleido
"$ENV_PREFIX/bin/plotly_get_chrome"

# Install project-specific requirements with pinned versions for compatibility
"$ENV_PREFIX/bin/python" -m pip install -r requirements.txt

echo "DONE: $ENV_PREFIX"