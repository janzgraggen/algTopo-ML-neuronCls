# Check if the morpho environment already exists
if conda info --envs | grep -q "^morpho "; then
    echo "Conda environment 'morpho' already exists. Skipping creation."
    exit 0
fi

# create conda environment for morphoclass
conda create --name morpho python=3.11
conda activate morpho

#install morphoclass prerequisites
conda install -c conda-forge dionysus
pip install -r conda_setup_requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

# install mophoroclass
git clone  git@github.com:lidakanari/morphoclass.git
cd morphoclass
./install.sh
cd ..

# delete the morphoclass repo
rm -rf morphoclass

