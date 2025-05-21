#!/bin/bash

# Check for at least one argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 -axon <axon_dir> -apical <apical_dir> -basal <basal_dir> -out <output_dir>"
    echo "Example: $0 -axon axon_dir -apical apical_dir -basal basal_dir -out output_dir"
    exit 1
fi

# Run the Python script with the provided arguments
python run_NeutriteFeatureConcat.py "$@"
