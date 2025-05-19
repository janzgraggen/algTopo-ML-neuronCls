#!/bin/bash

# Check for two arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <features_dir_name> <results_dir_name>"
  echo "Example: $0 features:kanari18_laylab features:kanari18_laylab_results"
  exit 1
fi

FEATURES_DIR="output/features/$1/"
RESULTS_DIR="output/results/$2"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=".:$PYTHONPATH"

# Run training
morphoclass --verbose train \
  --features-dir "$FEATURES_DIR" \
  --model-config src/configs/model-multiconcat.yaml \
  --splitter-config src/configs/splitter-stratified-k-fold.yaml \
  --checkpoint-dir "$RESULTS_DIR"

# Run evaluation
morphoclass --verbose evaluate performance \
  "$RESULTS_DIR/checkpoint.chk" \
  "$RESULTS_DIR/eval.html"
