#!/bin/bash
#SBATCH --job-name=rinalmo_recon_acc
#SBATCH --output=outputs/logs/rinalmo_recon_acc_%j.log
#SBATCH --error=outputs/logs/rinalmo_recon_acc_%j.err
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

set -e

# To run locally without SLURM: bash scripts/run_rinalmo_reconstruction_accuracy_slurm.sh
# To submit via SLURM: sbatch scripts/run_rinalmo_reconstruction_accuracy_slurm.sh
# Progress bar in: outputs/logs/rinalmo_recon_acc_%j.err

# Change these parameters as needed
INPUT_DIR="data/raw/ribosome"
OUTPUT_DIR="outputs/ribosome/reconstruction_accuracy"
TEST_NAME="test"
MODEL_NAME="mega" 
BATCH_SIZE=32
CONTEXT_WINDOW=1000
STRIDE=250

echo "Starting background GPU monitoring..."
GPU_LOG="outputs/logs/gpu_util_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv > "$GPU_LOG"
while true; do 
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader >> "$GPU_LOG"
    sleep 30
done &
GPU_MONITOR_PID=$!

echo "Running reconstruction accuracy script..."
python scripts/rinalmo_reconstruction_accuracy.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_name "$TEST_NAME" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --context_window "$CONTEXT_WINDOW" \
    --stride "$STRIDE"

echo "Stopping GPU monitor..."
kill $GPU_MONITOR_PID

echo "Done!"
