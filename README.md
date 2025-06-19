# Using Algebraic Topology and Graph Neural Networks in Multi Embedding Fusion Models for classifying neuronal morphologies
> Master Semester Project (Spring 2025):

> Project By Jan Zgraggen, Supervised by Lida Kanari

> *Keywords:* TMD, TDA, Morphology Classification, Embedding Fusion

**Code Attributions:**
- The `src/tda_toolbox` folder is adapted from [`tda_toolbox`](https://github.com/Eagleseb/tda_toolbox) by Sébastien Morand with minor updates for Python compatibility.

- Some methods are adapted or inspired form[`morphoclass`](https://github.com/BlueBrain/morphoclass) by BlueBrain. Furthermore the Deep learning Training loop is entirely accessed via CLI tool of [`morphoclass`](https://github.com/BlueBrain/morphoclass) .

---

## 🗂️ Table of Contents
1. [Overview](#-overview)
2. [Requirements](#-requirements)
3. [Dataset Preparation](#-dataset-preparation)
4. [Classification with Traditional Models](#-classification-with-traditional-models)
    - [Configuration](#configuration)
    - [Execution](#execution-run-experiment)
5. [Feature Extraction for DeepLearning Models](#-feature-extraction-for-deeplearning-models)
    - [Configuration](#configuration-1)
    - [Execution](#execution-run-experiment-1)
6. [Classification using ManNet (GNN) or MultiConcat (MEFN)](#-classification-using-mannet-gnn-or-multiconcat-mefn)
    - [Configuration](#configuration-2)
    - [Execution](#execution-run-experiment-2)
7. [Remarks](#-remarks)

---

## ⚙️ Requirements
### Environment Setup:
Set up using:

```bash
./runSETUP.sh
```

This script:

* Checks for an existing `morpho` conda environment
* Creates the environment with Python 3.11
* Installs TDA and PyTorch dependencies
* Clones and installs `morphoclass`
* Cleans up the cloned directory

<details>
<summary>Click to expand manual steps</summary>

```bash
conda create --name morpho python=3.11
conda activate morpho
conda install -c conda-forge dionysus
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

git clone git@github.com:lidakanari/morphoclass.git
cd morphoclass && ./install.sh && cd ..
rm -rf morphoclass
```

</details>
---


## 🤖 Dataset Preparation

Download the dataset: E.g from [here](https://zenodo.org/record/5909613#.YygtAmxBw5k) to access the the datased from [Kanari et a. (2022)](https://www.biorxiv.org/content/10.1101/2020.04.15.040410v1).

Place it inside `assets/` under one of the following:

```
assets/
├── datasets_flat/
├── datasets_structured_label/
└── datasets_structured_layer/  ← Recommended
```

Expected format for structured data (!!! please resturcture if not provided like this):

```
assets/datasets_structured_layer/Layer/Label/file
```

---

## 🤖 Classification with Traditional Models
This Reproduces *Experiments 1* of the Project. They essentialy consist of 
reproducing results from  [Kanari et al. (2019)](https://academic.oup.com/cercor/article/29/4/1719/5304727).

### Configuration
Update the GLOBAL  parameters in `TrainTraditional.py`

<details>
<summary>CONFIG EXAMPLE (Click to expand)</summary>

```bash
    DATAPATH = "Data/" 
    LAYER = "L5"
    NEURITE_TYPE = "apical_dendrite"
    PH_F = "radial_distances"
    VECTORIZATION = ["persistence_image", "wasserstein", "landscape"]
    FLATTEN = True
    M_SW = 20
    K_LS = 1
    M_LS = 1
    CLS = sklearn.svm.SVC()
    CV = sklearn.model_selection.StratifiedKFold(**args**)
    GRID = sklearn.model_selection.GridSearchCV( **args**)
```
</details>


### Execution (Run Experiment):
You can run the Experiment using the correspoding shellscript as follows:
```
./runT.sh
```
---

#### explained functionality of the py script

- Data Loading:

    ```python
    labels, pers_images = load_data(...)
    ```

- Training the Model:

    ```python
    trainer = skTrainer(data=pers_images, labels=labels, ...)
    for vector_method in VECTORIZATION:
        trainer.train_crossvalidation(vectorization_select=vector_method)
    ```

- Grid Search:

    ```python
    trainer.train_gridsearch()
    ```



## 🤖 Feature Extraction for DeepLearning Models 
The DeapLearning Models used in this project require to load already extracted features form files. The Pipeline For feature extraction and saving them to files is described here: 
### Configuration: 
Modify or Create a Feature extraction file located at `src/configs/extract-*`

Examples for accepded values can be found e.g in `src/configs/extract-k18`. Note that this Feature Extraction step requires the data to be organized as given in the DataPreparation section. 


<details>
<summary>CONFIG EXAMPLE (Click to expand)</summary>

```bash
PATH: "assets/datasets_structured_layer/kanari18_laylab"
OUT_PATH: "output/features/k18/"
LAYER: "L5"
TYPES: ["L5_TPC:A","L5_TPC:B","L5_TPC:C","L5_UPC"]
NEURITE_TYPE: "apical_dendrite"

ADD_VEC: True
ADD_MORPH: True

FEATURE_EXTRACTOR:
#GLOBAL:
  # - ExtractNumberLeaves
  # - ExtractNumberBranchPoints
  # - ExtractMaximalApicalPathLength
  # - TotalPathLength
  # - AverageBranchOrder
  # - AverageRadius
#EDGE:
  # - ExtractDistanceWeights
#NODE:
  # - ExtractBranchingAngles
  # - ExtractCoordinates
  # - ExtractDiameters
  # - ExtractIsBranching
  # - ExtractIsIntermediate
  # - ExtractIsLeaf
  # - ExtractIsRoot
  # - ExtractPathDistances
  - ExtractRadialDistances
  # - ExtractVerticalDistances

CONFIG_MORPHOMETRICS: "src/configs/feature-morphometrics.yaml"
NORMALIZE: true

PH_F: "radial_distances"
VECTORIZATION:
  - persistence_image
  - wasserstein
  - bottleneck
  - sliced_wasserstein
  - landscape
FLATTEN: false
M_SW: 20
K_LS: 5
M_LS: 150

```
</details>

### Execution (Run Experiment):
Use the shellscriptwrapper for the Morphoclass CLI to extract the features: 
```
./runE.sh configname
```

#### explained functionality:
- `runE.sh` calls `run_MultiFeatureExtract.py`

- `run_MultiFeatureExtract.py` allows:
    - Arbitrary Graph, TDA and morphometric feature extraction
    - Combination and normalization

- Feature Handling via `MorphologyDatasetManager`

    - Loading
        - `mdm = MorphologyDatasetManager(...)`

    - Adding  Features
        - `mdm.add_vecotized_pd(...)` 
        - `mdm.add_morphometrics(...)`

    - FileIO (save and read)
        - `mdm.write_features(...)`
        - `mdm_from_dir = MorphologyDatasetManager.from_features_dir(path)`


## 🤖 Classification using ManNet (GNN) or MultiConcat (MEFN):
(MEFN: Multi Embedding Fusion Network)

Running Experiments with the default configuration of `ManNet` Reproduces *Experiment 2* of the Project. They essentialy consist of reproducing results from  [Kanari et al. (2024)](https://www.biorxiv.org/content/10.1101/2024.09.13.612635v1).

Running Experiments with `MultiConcat` reproduces *Experiment 3* which is exploring Multi Embedding Fusion for morphology Classification

### Configuration:
Configure the respective config file `model-gnn.yaml` or `model-multiconcat.yaml` located at: `src/configs/`.

<details>
<summary> ManNet CONFIG EXAMPLE (Click to expand)</summary>

```bash
batch_size: 2
model_class: morphoclass.models.ManNet
model_params:
  edge_weight_idx: null
  flow: target_to_source
  lambda_max: 3.0
  n_features: 1
  normalization: sym
  pool_name: avg
n_epochs: 20
optimizer_class: torch.optim.Adam
optimizer_params:
  lr: 0.005

```
</details>


<details>
<summary> MultiConcat CONFIG EXAMPLE (Click to expand)</summary>

```bash
batch_size: 2
model_class: src.models.MultiConcat.MultiConcat
model_params:
  # General: 
  n_classes: 4
  bn: False
  dropout: 0
  embedding_dropout: 0
  embedding_dim: 32
  cls_hidden_dim: 128
  linear_hidden_dim: 16
  cheb_conv_hidden_dim: 128
  normalize_emb_weights: False
  normalize_emb_temp: 1

  # GNN:
  flow: target_to_source
  lambda_max: 3.0
  n_node_features: 1
  normalization: sym
  pool_name: avg

  # embeddings: Vectorization/Morphometrics/gnn:
  embeddings:
    - gnn
    - persistence_image
    - wasserstein
    - bottleneck
    - sliced_wasserstein
    - landscape    
    - morphometrics
oversampling: False
n_epochs: 20
optimizer_class: torch.optim.Adam
optimizer_params:
  lr: 0.001


```
</details>



> Note: Do not change the filename in order to not interfere with the execution wrapper.

### Execution (Run Experiment):

In order to run the experiment with ManNet run the wrapper: 
```
./runGNN.sh features_dir_name   results_dir_name
```
Idem. for Multiconcat: 
```
./runMC.sh features_dir_name   results_dir_name
```
> Note: This process uses extracted features from `output/features/features_dir_name` and saves the results in `output/results/results_dir_name`.

#### explained functionality of the shell wrapers: 
- `runMC.sh` and `runCNN.sh` essentially do:
    - Train with morphoclass:
        ```bash
        morphoclass --verbose train \
                --features-dir output/features/features_dir_name \
                --model-config src/configs/model-modelname.yaml \
                --splitter-config src/configs/splitter-stratified-k-fold.yaml \
                --checkpoint-dir output/results/results_dir_name
        ```

    - Evaluate performance with morphoclass:
        ```bash
        morphoclass --verbose evaluate performance \
                output/results/results_dir_name/checkpoint.chk \
                output/results/results_dir_name/eval.html
        ```



## 🧵 Remarks

### Fixing Segmentation Faults (PyTorch CPU)

💥 Issue: `ReLU()` (or other PyTorch functions) can crash on macOS/Linux due to threading in OpenMP/MKL

🛠️ Solution: Limit threads (handled via runMODEL.sh wraper)

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Then safely run:

```bash
morphoclass train --features-dir ... --model-config model-gnn.yaml ...
```

### Scripts for using different morphoclass models for the entire pipeline:

These are indicative and dont work as provided, paths and configurations have to be rechecked.
<details>
<summary> GNN script (Click to expand)</summary>

```bash
morphoclass --verbose extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    graph-rd\
    output/extract-features/pc-L5/apical/graph-rd/ --force

morphoclass --verbose train \
    --features-dir output/extract-features/pc-L5/apical/graph-rd/ \
    --model-config src/configs/model-gnn.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-graph-rd-gnn/

morphoclass --verbose evaluate performance \
    output/pc-L5-apical-graph-rd-gnn/checkpoint.chk \
    output/evaluation_report_gnn.html
```
</details>


<details>
<summary> XGB (Click to expand)</summary>

```bash
  morphoclass --verbose extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    image-tmd-rd \
    output/extract-features/pc-L5/apical/image-tmd-rd/ --force

morphoclass --verbose train \
    --features-dir output/extract-features/pc-L5/apical/image-tmd-rd/ \
    --model-config src/configs/archive/model-xgb.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-image-tmd-rd-xgb/

morphoclass --verbose evaluate performance \
    output/pc-L5-apical-image-tmd-rd-xgb/checkpoint.chk \
    output/evaluation_report_xgb.html

```
</details>

<details>
<summary> DecisionTree (Click to expand)</summary>

```bash
  morphoclass --verbose extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    image-tmd-rd \
    output/extract-features/pc-L5/apical/image-tmd-rd/ --force

morphoclass --verbose train \
    --features-dir output/extract-features/pc-L5/apical/image-tmd-rd/ \
    --model-config src/configs/archive/model-decision-tree.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-image-tmd-rd-decision-tree/

morphoclass --verbose evaluate performance \
    output/pc-L5-apical-image-tmd-rd-decision-tree/checkpoint.chk \
    output/evaluation_report.html

```
</details>

<details>
<summary> PersLay (Click to expand)</summary>

```bash
  morphoclass --verbose train \
    --features-dir output/myown_features/ \
    --model-config src/configs/archive/model-perslay.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/perslay_myfeatures_check/

morphoclass --verbose evaluate performance \
    output/perslay_myfeatures_check/checkpoint.chk \
    output/evaluation_report.html

```
</details>
<details>
<summary> CNN (Click to expand)</summary>

```bash
  morphoclass --verbose extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    image-tmd-rd \
    output/extract-features/pc-L5/apical/image-tmd-rd/ --force

morphoclass --verbose train \
    --features-dir output/extract-features/pc-L5/apical/image-tmd-rd/ \
    --model-config src/configs/archive/model-cnn.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-image-tmd-rd-cnn/

morphoclass --verbose evaluate performance \
    output/pc-L5-apical-image-tmd-rd-cnn/checkpoint.chk \
    output/evaluation_report2.html

```
</details>


