# algTopo-ML-neuronCls

Repo for the Master Semester project: 
  - Using algebraic topology and ML to cluster and classify neurons. 
  - Spring semester 2025
  - Project by Jan Zgraggen, supervised by Lida Kanari 
  
Note: The tda_toolbox folder contains files from [this repo](https://github.com/Eagleseb/tda_toolbox) by Sébastien Morand. I modified them slightly to be up to date with newer versions of python and some libraries. 

### Requirements

Due to issues with versions and on Mac architecture use : 
```bash
./requirements_setup.sh
```

which essentially does the following: 

```bash
# Check if the morpho environment already exists
if conda info --envs | grep -q "^morpho "; then
    echo "Conda environment 'morpho' already exists. Skipping creation."
    exit 0
fi
```
```bash
# create conda environment for morphoclass
conda create --name morpho python=3.11
conda activate morpho
```
```bash
#install morphoclass prerequisites
conda install -c conda-forge dionysus
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```
```bash
# install mophoroclass
git clone  git@github.com:lidakanari/morphoclass.git
cd morphoclass
./install.sh
cd ..
```
```bash
# delete the morphoclass repo
rm -rf morphoclass
```

# Usage of tdm methods (run_tdm.py)
This python file reproduces some results from [Kanari et al. 2019](https://academic.oup.com/cercor/article/29/4/1719/5304727).


## Dataset Preparation
The the Dataset used is Reconstructions.zip which is found [here](https://zenodo.org/record/5909613#.YygtAmxBw5k)

For reproduction of the result place the dataset in the `assets/` directory. 
Depending on the structure of the dataset place it in one of the subdir's:
- datasets_flat
- datasets_structured_label
- datasets_structured_layer

where the last one is the most commonly used for this project and expects:
```
Dataset/Layer/Label/file 
```
to be placed in **assets/datasets_structured_layer/**

## Configuration

### Configuration Parameters
Before running the script, adjust the following configuration parameters to suit your data and experiment:

```python
DATAPATH = "Data/"  # Path to the dataset
LAYER = "L3"        # Layer type (e.g., "L3")
NEURITE_TYPE = "apical_dendrite"  # Neurite type (e.g., "apical_dendrite")
PH_F = "radial_distances"  # Persistence homology function (e.g., "radial_distances")
VECTORIZATION = ["persistence_image", "wasserstein", "bottleneck", "sliced_wasserstein", "landscape"]  # List of vectorization methods

# Persistence image parameters
FLATTEN = True  # Flatten the image for vectorization

# Sliced Wasserstein parameters
M_SW = 20  # Number of slices for sliced Wasserstein

# Landscape parameters
K_LS = 1   # Number of landscapes
M_LS = 1   # Resolution for landscapes

# Classifier and Cross-validation settings
CLS = sklearn.svm.SVC()  # Classifier (e.g., Support Vector Classifier)
CV = sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Cross-validation settings

GRID = sklearn.model_selection.GridSearchCV(  # GridSearch for hyperparameter tuning
    param_grid={
        'C': [1, 10, 100, 1000],
        'kernel': ['linear', 'rbf'],
        'class_weight': ['balanced', None],
        'gamma': [0.001, 0.0001],
    },
    estimator=CLS,
    cv=CV,
)
```

### Loading Data

The `load_data` function is used to load the dataset and apply persistence image and vectorization.

```python
labels, pers_images = load_data(
    datapath=DATAPATH,
    types=LAYER,
    neurite_type=NEURITE_TYPE,
    pers_hom_function=PH_F,
    vectorization=VECTORIZATION,
    flatten=FLATTEN,
    M=M_SW,
    k=K_LS,
    m=M_LS
)
```

### Training the Model

The `skTrainer` class is used to train the model with different vectorization methods. You can specify the classifier (e.g., SVC, DecisionTree) and the cross-validation method.

```python
trainer = skTrainer(
    data=pers_images,
    labels=labels,
    cls=CLS,
    crosvalidate=CV,
    gridsearch=GRID
)

# Loop through vectorization methods and train the model
for vector_method in VECTORIZATION:
    trainer.train_crossvalidation(vectorization_select=vector_method)
```

### Hyperparameter Tuning with Grid Search

You can perform hyperparameter tuning by calling the `train_gridsearch` method:

```python
trainer.train_gridsearch()
```

This will search for the best hyperparameters for the SVM classifier using GridSearchCV.

## Notes

- You can now set up the experiment choosing `LAYER`, `Neutrite Type`, `Function for the TMD persistent homology algo`
and the `Vectorization`
- You can also easily switch between different classifiers (e.g., `DecisionTreeClassifier`, `QuadraticDiscriminantAnalysis`) by changing the `CLS` parameter.

# Usage of graph methods (run_graph.py)
This python file reproduces some results from [Kanari et al. 2024](https://www.biorxiv.org/content/10.1101/2024.09.13.612635v1).




### Using morphoclass via CLI
First the CLI directory contains shell scripts for
 - feature extraction
 - training
 - evaluation

of different model configurations. The commands for the such look like the following for feature extraction: 
```bash
morphoclass extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    image-tmd-rd \
    output/extract-features/pc-L5/apical/image-tmd-rd/ --force
```
for training: 
```bash
morphoclass train \
    --features-dir output/extract-features/pc-L5/apical/image-tmd-rd/ \
    --model-config CLI/configs/model-decision-tree.yaml \
    --splitter-config CLI/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-image-tmd-rd-decision-tree/
```
and for evaluation: 
```bash
morphoclass evaluate performance \
    output/pc-L5-apical-image-tmd-rd-decision-tree/checkpoint.chk \
    output/evaluation_report.html
```

The .sh can be used to reproduce some results. 

### Creating more flexible features

There is only an selection of features supported via comand line. 
Therefore I implemented more general feature extraction based on morphoclass's soucecode that supports all extractions which are supported by the package. 

#### Config: 
Similar to above: 
```bash
PATH = "assets/datasets_structured_layer/kanari18_laylab"
LAYER = "L5"
TYPES = ""  # or ["L5_TPC:A", ...]
NEURITE_TYPE = "apical_dendrite"
```

#### Available features: 
```bash
FEATURE_EXTRACTOR = transforms.Compose([
    # feature extraction. 
    
    # GLOBAL:
    # transforms.ExtractNumberLeaves(),
    # transforms.ExtractNumberBranchPoints(),
    # transforms.ExtractMaximalApicalPathLength(),
    # transforms.TotalPathLength(),
    # transforms.AverageBranchOrder(),
    # transforms.AverageRadius(),

    # EDGE:
    # transforms.ExtractEdgeIndex(),
    # transforms.ExtractDistanceWeights(),

    # NODE:
    # transforms.ExtractBranchingAngles(),
    # transforms.ExtractConstFeature(),
    # transforms.ExtractCoordinates(),
    # transforms.ExtractDiameters(),
    # transforms.ExtractIsBranching(),
    # transforms.ExtractIsIntermediate(),
    # transforms.ExtractIsLeaf(),
    # transforms.ExtractIsRoot(),
    # transforms.ExtractPathDistances(),
    #transforms.ExtractRadialDistances(),
    # transforms.ExtractVerticalDistances()
])
```
#### Data loading, scaling writing and fetching is supported via: 
loading: 
```bash
load_graph(**kwargs)
```

scaling: 
```bash
scale_graph(**kwargs)
```

writing: 
```bash
write_featuers(**kwargs)
```
fetching: 
```bash
load_features(**kwargs)
```


