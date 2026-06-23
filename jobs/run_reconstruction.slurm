#!/usr/bin/env bash
#SBATCH --job-name=run_reco
#SBATCH --output=logs/run_reco_%j.out
#SBATCH --error=logs/run_reco_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"
ENV="/opt/modules/i12g/anaconda/envs/group6-rinalmo"

cd "$PROJECT_DIR"

"$ENV/bin/python" scripts/run_reconstruction.py configs/runs/rinalmo-v1.yaml