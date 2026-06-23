#!/usr/bin/env bash
#SBATCH --job-name=evo2_smoke
#SBATCH --output=logs/evo2_smoke_%j.out
#SBATCH --error=logs/evo2_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"
ENV="/opt/modules/i12g/anaconda/envs/group6-evo2-rebuild"

cd "$PROJECT_DIR"

"$ENV/bin/python" scripts/archive/smoke_evo2.py