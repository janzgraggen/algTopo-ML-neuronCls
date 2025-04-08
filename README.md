# algTopo-ML-neuronCls

Repo for the Master Semester project: 
  - Using algebraic topology and ML to cluster and classify neurons. 
  - Spring semester 2025
  - Project by Jan Zgraggen, supervised by Lida Kanari 
  
Requirements can be found in `requirements.txt`

## Directory Structure


```
/project-directory
  /Data            # Directory containing the dataset
  /src             # Source code directory
    /data.py       # Data loading functions
    /train.py      # Training functions (skTrainer)
  /run_kanari19.py # Reproduction of paper results
  README.md        # This file
```

## Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

# Usage of run_kanari19.py
This reproduces the results from [Kanari et al., 2019](https://academic.oup.com/cercor/article/29/4/1719/5304727).


## Dataset Preparation
And download the Dataset Reconstructions.zip which is found [here](https://zenodo.org/record/5909613#.YygtAmxBw5k)

Place your dataset in the `Data/` directory. The folder should look like the followning
```
├── Data
│   ├── L1_DAC
│   ├── L1_HAC
│   ├── L2_TPC:B
│   ├── L3_TPC:A
│   ├── L4_BP
│   ├── L4_BTC
│   ├── L4_CHC
│   ├── L4_DBC
│   ├── L5_UPC
│   ├── L6_BPC
│   └── L6_UPC
```
(only showing a subset)
to ensure the data is formatted correctly and compatible with the `load_data()` function in the `src/data.py` file.

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

### Full Example

Here’s an example script that combines all the steps:

```python
import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import sklearn.discriminant_analysis

from src.data import load_data
from src.train import skTrainer

# Configuration
DATAPATH = "Data/"
LAYER = "L3"
NEURITE_TYPE = "apical_dendrite"
PH_F = "radial_distances"
VECTORIZATION = ["persistence_image", "wasserstein", "bottleneck", "sliced_wasserstein", "landscape"]

FLATTEN = True
M_SW = 20
K_LS = 1
M_LS = 1

# Classifier and Cross-validation settings
CLS = sklearn.svm.SVC()
CV = sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
GRID = sklearn.model_selection.GridSearchCV(
    param_grid={
        'C': [1, 10, 100, 1000],
        'kernel': ['linear', 'rbf'],
        'class_weight': ['balanced', None],
        'gamma': [0.001, 0.0001],
    },
    estimator=CLS,
    cv=CV,
)

# Load data
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

# Train model
trainer = skTrainer(
    data=pers_images,
    labels=labels,
    cls=CLS,
    crosvalidate=CV,
    gridsearch=GRID
)

# Train with different vectorization methods
for vector_method in VECTORIZATION:
    trainer.train_crossvalidation(vectorization_select=vector_method)

# Perform grid search to tune hyperparameters
trainer.train_gridsearch()
```

## Notes

- You can now set up the experiment choosing `LAYER`, `Neutrite Type`, `Function for the TMD persistent homology algo`
and the `Vectorization`
- You can also easily switch between different classifiers (e.g., `DecisionTreeClassifier`, `QuadraticDiscriminantAnalysis`) by changing the `CLS` parameter.
```
