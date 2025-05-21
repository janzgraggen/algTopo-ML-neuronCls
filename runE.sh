#!/bin/bash

# Check for two arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <configname>"
  echo "Example: $0 configname (only name not path, no .yaml)"
  exit 1
fi

CONFIG_DIR="src/configs/$1.yaml"

# Run Extraction
python run_MultiFeatureExtract.py \
    "$CONFIG_DIR"


