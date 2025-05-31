#!/bin/bash

set -e  # Exit immediately on any error

# Check if the morpho environment already exists
if conda info --envs | grep -q "^morpho "; then
    echo "Conda environment 'morpho' already exists. Skipping creation."
    exit 0
fi

# Create conda environment for morphoclass
conda create --yes --name morpho python=3.11

# Activate the environment (this only works in interactive shells)
# So instead we use `conda run` for scripting
conda run -n morpho conda install --yes -c conda-forge dionysus
conda run -n morpho pip install -r requirements.txt
conda run -n morpho pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

# Clone and install morphoclass
git clone git@github.com:lidakanari/morphoclass.git
cd morphoclass
conda run -n morpho ./install.sh
cd ..

# Cleanup
rm -rf morphoclass

echo "Morphoclass setup completed."
