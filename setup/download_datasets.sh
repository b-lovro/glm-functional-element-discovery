#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RIBO_DIR="${ROOT_DIR}/data/raw/ribosome"
CRASS_DIR="${ROOT_DIR}/data/raw/crass_phages"
WEIGHTS_DIR="${ROOT_DIR}/models/rinalmo"

RIBO_URL="https://www.dropbox.com/scl/fi/18de0etvaqimgtattqwvl/MammalianRibosomalDNA.zip?rlkey=eja6nrap2nxq0jwj1rjsyr1cd&dl=1"
CRASS_URL="https://datashare.biochem.mpg.de/s/RQP37ow3RdkGddc/download"
RINALMO_MICRO_URL="https://zenodo.org/records/15043668/files/rinalmo_micro_pretrained.pt"
RINALMO_MEGA_URL="https://zenodo.org/records/15043668/files/rinalmo_mega_pretrained.pt"

download() {
    local url="$1"
    local out="$2"

    if [ -f "$out" ]; then
        echo "Exists, skipping: $out"
    else
        wget -O "$out" "$url"
    fi
}

mkdir -p "$RIBO_DIR" "$CRASS_DIR" "$WEIGHTS_DIR"

echo "Ribosome dataset"
download "$RIBO_URL" "$RIBO_DIR/MammalianRibosomalDNA.zip"
unzip -n "$RIBO_DIR/MammalianRibosomalDNA.zip" -d "$RIBO_DIR"

echo "crAss-like phage dataset"
download "$CRASS_URL" "$CRASS_DIR/crass_dataset.zip"
unzip -n "$CRASS_DIR/crass_dataset.zip" -d "$CRASS_DIR"

echo "RiNALMo weights"
download "$RINALMO_MICRO_URL" "$WEIGHTS_DIR/rinalmo_micro_pretrained.pt"
download "$RINALMO_MEGA_URL" "$WEIGHTS_DIR/rinalmo_mega_pretrained.pt"

echo "EVO2 weights"
export HF_HOME="${ROOT_DIR}/models/evo2"
export HF_HUB_CACHE="${ROOT_DIR}/models/evo2"

echo "Downloading EVO2 7B model via Hugging Face CLI..."
hf download arcinstitute/evo2_7b

echo "Done."