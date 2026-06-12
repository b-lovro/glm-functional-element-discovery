#!/bin/bash
#SBATCH --job-name=rinalmo_dependency_map
#SBATCH --output=outputs/logs/rinalmo_dependency_map_%j.log
#SBATCH --error=outputs/logs/rinalmo_dependency_map_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

set -e

# To run locally without SLURM: bash scripts/run_dependency_map_slurm.sh
# To submit via SLURM: sbatch scripts/run_dependency_map_slurm.sh

# Change these parameters as needed
INPUT_FILE="data/raw/ribosome/Mammalia-Artiodactyla-Addax nasomaculatus-GCA_044231825-morph3.fasta"
OUTPUT_DIR="outputs/ribosome/dependency_maps"
MAX_SEQUENCES=1
MODEL_NAME="micro"
BATCH_SIZE=4
#MODEL_PATH="models/rinalmo/rinalmo_micro_pretrained.pt"
SUBSET_START=10
SUBSET_END=18
TEST_NAME="test"

# To run on crass_phages dataset:
# INPUT_FILE="data/raw/crass_phages/Files for crAss-like phage genomic feature analysis/GPD_sequences.fa"
# OUTPUT_DIR="outputs/crass_phages/dependency_maps"

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

echo "Done!"
