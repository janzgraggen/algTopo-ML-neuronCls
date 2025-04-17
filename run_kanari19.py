import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import sklearn.discriminant_analysis

from src.data.load_tdm import load_data
from src.train import skTrainer


## ------------------------ DATA LOAD PARAMS --------------------------------
DATAPATH = "Data/"
LAYER = "L3"
NEURITE_TYPE = "apical_dendrite"
PH_F = "radial_distances"   
VECTORIZATION = ["persistence_image", "wasserstein","bottleneck","sliced_wasserstein", "landscape"] # or "landscape" or "bottleneck" or "wasserstein" or "slice_wasserstein"

# ------------------------ Vectirization Params --------------------------------
#persistence_image:
FLATTEN = True # flatten the image
# sliced_wasserstein:
M_SW = 20 # number of slices
# landscape:
K_LS = 1 # number of landscapes
M_LS = 1 # resolution

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
labels, pers_images = load_data(
    datapath=DATAPATH,
    types=LAYER,
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
    data=pers_images,
    labels=labels,
    cls=CLS,
    crosvalidate=CV,
    gridsearch=GRID
)

# for i in range(len(VECTORIZATION)):
#     trainer.train_crossvalidation(vectorization_select= VECTORIZATION[i])
    
#trainer.train_crossvalidation(vectorization_select= "persistence_image")
#trainer.train_gridsearch()

layer_trainers = {}
for layer in ["L2", "L3", "L4", "L5"]:
    
    labels, pers_images = load_data(
    datapath=DATAPATH,
    types=layer,
    neurite_type= NEURITE_TYPE,
    pers_hom_function= PH_F,
    vectorization= VECTORIZATION,
    flatten=FLATTEN,
    M= M_SW,
    k = K_LS,
    m = M_LS
    )

    layer_trainers[layer]  = skTrainer(
    data=pers_images,
    labels=labels,
    cls=CLS,
    crosvalidate=CV,
    gridsearch=GRID
)

for key, trainer in layer_trainers.values():
    print(f"\n================================")
    print(f"================================")
    print(key)
    for vec in VECTORIZATION:
        print(vec)
        trainer.train_crossvalidation(vectorization_select= vec)
