#!/bin/sh

set -e  # Exit immediately if any command fails

echo "Preparing Kururu dataset..."

# Create necessary folder structure if missing
mkdir -p raw
mkdir -p models
mkdir -p data/train/images
mkdir -p data/train/labels
mkdir -p data/test/images

# Run Python script
python prepare_dataset.py \
  --excel_file "raw/Pixels — Rostro-cloacal length (20250404).xlsx" \
  --input_folders "/run/media/alex/DATA-1/Data/Ingrid Ribeiro/TESTING" "/run/media/alex/DATA-1/Data/Ingrid Ribeiro/TRAINING" \
  --output_folder "data"

echo ""
echo "✅ Dataset ready!"

