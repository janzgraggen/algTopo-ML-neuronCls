# 🧠 algTopo-ML-neuronCls
> Master Semester Project (Spring 2025):

> **Clustering and Classifying Neurons using Algebraic Topology and Machine Learning**

> Project By Jan Zgraggen, Supervised by Lida Kanari

---

## 📁 Overview

This repository contains tools and scripts for:

* Traditional ML classification using topological data analysis (TDA)
* Deep learning-based neuron classification with **morphoclass**
* Custom multimodal feature extraction and training

📎 *Note:* The `src/tda_toolbox` folder is adapted from [this repo](https://github.com/Eagleseb/tda_toolbox) by Sébastien Morand with minor updates for Python compatibility.

📎 *Note:* Some methods are adapted or inspired form[`morphoclass`](https://github.com/BlueBrain/morphoclass) by BlueBrain. Furthermore the Deep learning Training loop is entirely accessed via CLI tool of `morphoclass`.

---

## 🗂️ Table of Contents

1. [Requirements](#-requirements)
2. [Classification with Traditional Models](#-classification-with-traditional-models)

   * [Dataset Setup](#dataset-preparation)
   * [Configuration](#configuration)
   * [Training & Evaluation](#training-the-model)
3. [Classification with Deep Learning](#-classification-using-deeplearning)

   * [Using morphoclass CLI](#21-using-morphoclass-via-cli)
   * [Multimodal Pipeline](#22-creating-and-training-with-multimodal-data)
4. [Technical Notes & Fixes](#-remarks)

---

## ⚙️ Requirements
### Env
Set up using:

```bash
./requirements_setup.sh
```

This script:

* Checks for an existing `morpho` conda environment
* Creates the environment with Python 3.11
* Installs TDA and PyTorch dependencies
* Clones and installs `morphoclass`
* Cleans up the cloned directory

<details>
<summary>🔧 Click to expand manual steps</summary>

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


### Dataset Preparation

Download the dataset [here (Reconstructions.zip)](https://zenodo.org/record/5909613#.YygtAmxBw5k)
Place it inside `assets/` under one of the following:

```
assets/
├── datasets_flat/
├── datasets_structured_label/
└── datasets_structured_layer/  ← Recommended
```

Expected format for structured data:

```
assets/datasets_structured_layer/Layer/Label/file
```

---

## 🧪 Classification with Traditional Models


### Configuration

Update the following parameters in `run_TrainTdmML.py`:

```python
DATAPATH = "Data/"
LAYER = "L3"
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

---

### Loading and Training

**Data Loading:**

```python
labels, pers_images = load_data(...)
```

**Training the Model:**

```python
trainer = skTrainer(data=pers_images, labels=labels, ...)
for vector_method in VECTORIZATION:
    trainer.train_crossvalidation(vectorization_select=vector_method)
```

**Grid Search:**

```python
trainer.train_gridsearch()
```

---

## 🤖 Classification Using DeepLearning

### 2.1 Using morphoclass via CLI

Shell scripts under `CLI/` automate:

* Feature extraction
* Training
* Evaluation

**Examples:**

```bash
# Feature extraction
morphoclass extract-features input.csv apical image-tmd-rd output/dir --force

# Training
morphoclass train --features-dir ... --model-config ... --splitter-config ... --checkpoint-dir ...

# Evaluation
morphoclass evaluate performance model.chk output/report.html
```

---

### 2.2 Creating and Training with Multimodal Data

#### 2.2.1 Feature Extraction (Custom)

`run_MultiFeatureExtract.py` allows:

* Arbitrary Graph, TDA and morphometric feature extraction
* Combination and normalization

**Supported Parameters:**

```python
PATH = "assets/datasets_structured_layer/kanari18_laylab"
OUT_PATH = "output/multiconcat_features_full"
LAYER = "L5"
TYPES = ""  # or ["L5_TPC:A", ...]
NEURITE_TYPE = "apical_dendrite"
FEATURE_EXTRACTOR = transforms.Compose([**transforms**])
CONFIG_MORPHOMETRICS = "CLI/configs/feature-morphometrics.yaml"
NORMALIZE = True # normalize the features
PH_F = "radial_distances"   
VECTORIZATION = ["persistence_image", "wasserstein","bottleneck","sliced_wasserstein", "landscape"] 
FLATTEN = False # flatten the image
M_SW = 20 #  sliced_wasserstein: number of slices
K_LS = 10 #  landscape: number of landscapes
M_LS = 5 #  landscape:resolution
```

---

#### 2.2.2 Feature Handling via `MorphologyDatasetManager`

```python
# Load
mdm = MorphologyDatasetManager(...)

# Add features
mdm.add_vecotized_pd(...)
mdm.add_morphometrics(...)

# Save or read
mdm.write_features(...)
mdm_from_dir = MorphologyDatasetManager.from_features_dir(path)
```

---

#### 2.2.3 Model Training
Export Pythonpath to make custom models findable by the morphoclass CLI
```bash
export PYTHONPATH=".:$PYTHONPATH"
```

Train custom models with saved multimodal features via morphoclass CLI:

```bash
morphoclass --verbose train \
    --features-dir output/saved_custom_features/ \
    --model-config src/configs/model-customModel.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/customModel_checkpoint/

```

📎 *Note:* shell scripts providing the terminal commands for the experiments presented in the report, as well as tests are found in `scripts`. Paste commands to terminal for execution. 

---

## 🧵 Remarks

### Fixing Segmentation Faults (PyTorch CPU)

💥 Issue: `ReLU()` (or other PyTorch functions) can crash on macOS/Linux due to threading in OpenMP/MKL

🛠️ Solution: Limit threads

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Then safely run:

```bash
morphoclass train --features-dir ... --model-config model-gnn.yaml ...
```

---

