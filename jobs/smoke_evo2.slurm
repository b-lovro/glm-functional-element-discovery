#!/usr/bin/env bash
#SBATCH --job-name=smoke_evo2
#SBATCH --output=logs/smoke_evo2_%j.out
#SBATCH --error=logs/smoke_evo2_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_DIR="/data/ceph/hdd/project/node_07/ml4rg_students/2026/project06/glm-functional-element-discovery"
ENV="/opt/modules/i12g/anaconda/envs/group6-evo2-rebuild"

cd "$PROJECT_DIR"

"$ENV/bin/python" scripts/smoke_evo2.py