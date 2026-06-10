#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"

RIBOSOME_DIR="${RAW_DIR}/ribosome"
CRASS_DIR="${RAW_DIR}/crass_phages"

RIBOSOME_URL="https://www.dropbox.com/scl/fi/18de0etvaqimgtattqwvl/MammalianRibosomalDNA.zip?rlkey=eja6nrap2nxq0jwj1rjsyr1cd&dl=1"
CRASS_URL="https://datashare.biochem.mpg.de/s/RQP37ow3RdkGddc/download"

mkdir -p "${RIBOSOME_DIR}"
mkdir -p "${CRASS_DIR}"

echo "Downloading ribosome dataset..."
cd "${RIBOSOME_DIR}"

if [ ! -f MammalianRibosomalDNA.zip ]; then
    wget -O MammalianRibosomalDNA.zip "${RIBOSOME_URL}"
else
    echo "MammalianRibosomalDNA.zip already exists, skipping download."
fi

echo "Unzipping ribosome dataset..."
unzip -n MammalianRibosomalDNA.zip

echo "Downloading crAss-like phage dataset..."
cd "${CRASS_DIR}"

if [ ! -f crass_dataset.zip ]; then
    wget -O crass_dataset.zip "${CRASS_URL}"
else
    echo "crass_dataset.zip already exists, skipping download."
fi

echo "Unzipping crAss-like phage dataset..."
unzip -n crass_dataset.zip

echo "Done."
