import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import sklearn.discriminant_analysis

from src.data.load_tdm import load_tmd
from src.train.train_tdm import skTrainer
import pickle
import numpy as np

## ------------------------ DEFINE EXPERIMENTS --------------------------------
EXP_MULTILAYERTRAIN = False # if True, load multiple layers
EXP_L5_CLASS_DISTINGUISH = True
EXP_L5_SUBCLASS = True

## ------------------------ DATA LOAD PARAMS --------------------------------
#DATAPATH = "assets/datasets_structured_layer/kanari18_laylab" ## alt DATAPATH = "assets/datasets_structured_layer/pyramidal-cells"   
DATAPATH = "assets/datasets_structured_layer/pyramidal-cells"   
LAYER = "L5"
#TYPES = ["L5_TPC:A", "L5_TPC:C", "L5_UPC"] # alt TYPES = ["TPC_A","TPC_C","UPC"]  # or str or list of neuron types to load
if EXP_MULTILAYERTRAIN:
    TYPES = {}
    TYPES["L2"] = ["L2_TPC:A", "L2_TPC:B", "L2_IPC"] # or str or list of neuron types to load
    TYPES["L3"] = ["L3_TPC:A", "L3_TPC:C"] # or str or list of neuron types to load
    TYPES["L4"] = ["L4_TPC", "L4_SSC", "L4_UPC"] # or str or list of neuron types to load
    TYPES["L5"] = ["L5_TPC:A", "L5_TPC:B","L5_TPC:C","L5_UPC"] # or str or list of neuron types to load
if EXP_L5_CLASS_DISTINGUISH: 
    TYPES = ["TPC_all","UPC"] 
if EXP_L5_SUBCLASS:
    TYPES = ["TPC_AB","TPC_C","UPC"]
NEURITE_TYPE = "apical_dendrite"
PH_F = "radial_distances"   
VECTORIZATION = ["persistence_image", "wasserstein","bottleneck","sliced_wasserstein", "landscape"] # or "landscape" or "bottleneck" or "wasserstein" or "slice_wasserstein"

# ------------------------ Vectirization Params --------------------------------
#persistence_image:
FLATTEN = True # flatten the image
# sliced_wasserstein:
M_SW = 20 # number of slices
# landscape:
K_LS = 5 # number of landscapes
M_LS = 150 # resolution

# ------------------------ CLASSIFIER PARAMS --------------------------------
CLS = sklearn.svm.SVC() #or other classifiers like sklearn.tree.DecisionTreeClassifier() or sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
CV = sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # or int 
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
                                    
# ------------------------ Data loading --------------------------------
labels, vectorized_pds = load_tmd(
    datapath=DATAPATH,
    layer=LAYER,
    types=TYPES,
    neurite_type= NEURITE_TYPE,
    pers_hom_function= PH_F,
    vectorization= VECTORIZATION,
    flatten=FLATTEN,
    M= M_SW,
    k = K_LS,
    m = M_LS
)

### ------------------------ Training dataset --------------------------------

trainer = skTrainer(
    data=vectorized_pds,
    labels=labels,
    cls=CLS,
    crosvalidate=CV,
    gridsearch=GRID
)

### --------- EXPERIMENT: Training loop on L5: Distinguish TPC and UPC ----------
if EXP_L5_CLASS_DISTINGUISH or EXP_L5_SUBCLASS:
    best_avg_acc = trainer.train_gridsearch()
    for  vec in VECTORIZATION:
        print(f"Best cross validated accuracy with distance {vec}: {best_avg_acc[vec]}")
    avg_over_dist = np.mean(list(best_avg_acc.values()))
    print(f"Best average accuracy over distances = {avg_over_dist}")


## ------------------------ Showcase of use of vectorization select ––––––––––––––––----------
# in a Loop:
# for vec in VECTORIZATION:
#     trainer.train_gridsearch(vectorization_select= vec)
#     trainer.train_crossvalidation(vectorization_select= vec)
# OR directly:
#trainer.train_crossvalidation(vectorization_select= "persistence_image")




### ------------------------ Training loop on multiple layers ----------------------
# Load the layer_trainers dictionary from the pickle file
if EXP_MULTILAYERTRAIN:
    try: 
        with open("output/layer_trainers.pkl", "rb") as f:
            layer_trainers = pickle.load(f)

        # Verify the loaded data
        for layer, trainer in layer_trainers.items():
            print(f"Loaded trainer for layer: {layer}")
    except FileNotFoundError:
        print("output/layer_trainers.pkl not found. Creating a new one.")
        # If the file doesn't exist, create a new layer_trainers dictionary

        layer_trainers = {}
        for layer in ["L2", "L3", "L4", "L5"]:
            
            labels, vectorized_pds = load_tmd(
            datapath=DATAPATH,
            layer=layer,
            types=TYPES[layer],
            neurite_type= NEURITE_TYPE,
            pers_hom_function= PH_F,
            vectorization= VECTORIZATION,
            flatten=FLATTEN,
            M= M_SW,
            k = K_LS,
            m = M_LS
            )

            layer_trainers[layer]  = skTrainer(
            data=vectorized_pds,
            labels=labels,
            cls=CLS,
            crosvalidate=CV,
            gridsearch=GRID
        )

        # Save the layer_trainers dictionary to a file
        with open("output/layer_trainers.pkl", "wb") as f:
            pickle.dump(layer_trainers, f)


    for layer, trainer in layer_trainers.items():
        print(f"\n================================")
        print(f"================================")
        print(layer)
        for vec in VECTORIZATION:
            print(vec)
            trainer.train_gridsearch(vectorization_select= vec)
