#!/bin/bash
#SBATCH --job-name=rinalmo_dependency_map
#SBATCH --output=outputs/logs/rinalmo_dependency_map_%j.log
#SBATCH --error=outputs/logs/rinalmo_dependency_map_%j.err
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

set -e

# To run locally without SLURM: bash scripts/run_dependency_map_slurm.sh
# To submit via SLURM: sbatch scripts/run_dependency_map_slurm.sh
# Progress bar in: outputs/logs/rinalmo_dependency_map_%j.err

# Change these parameters as needed
INPUT_FILE="data/raw/ribosome/Mammalia-Artiodactyla-Addax nasomaculatus-GCA_044231825-morph3.fasta"
OUTPUT_DIR="outputs/ribosome/dependency_maps"
MAX_SEQUENCES=1
MODEL_NAME="mega"
BATCH_SIZE=16
#MODEL_PATH="models/rinalmo/rinalmo_micro_pretrained.pt"
SUBSET_START=30100
SUBSET_END=30200
TEST_NAME="test"

# To run on crass_phages dataset:
# INPUT_FILE="data/raw/crass_phages/Files for crAss-like phage genomic feature analysis/GPD_sequences.fa"
# OUTPUT_DIR="outputs/crass_phages/dependency_maps"

echo "Starting background GPU monitoring..."
# This loops in the background every 30 seconds and logs GPU stats to a CSV file
GPU_LOG="outputs/logs/rinalmo_dependency_map_${SLURM_JOB_ID}_gpu_util.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv > "$GPU_LOG"
while true; do 
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader >> "$GPU_LOG"
    sleep 30
done &
GPU_MONITOR_PID=$!

echo "Running dependency map script..."
python scripts/rinalmo_dependency_map.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_sequences "$MAX_SEQUENCES" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --test_name "$TEST_NAME" \
    --subset_start "$SUBSET_START" \
    --subset_end "$SUBSET_END"

# Stop the GPU monitor once the python script finishes
echo "Stopping GPU monitor..."
kill $GPU_MONITOR_PID

echo "Done!"
