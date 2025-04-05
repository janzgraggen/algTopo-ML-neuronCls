import numpy as np
import sklearn.calibration
import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import sklearn 

from data import load_data
from train import train_crossvalidation, skTrainer

DATAPATH = "../Data/"
LAYER = "L2"
NEURITE_TYPE = "apical_dendrite"


#classifier
CLS = sklearn.svm.SVC()
#crossvalidation strategy

# grid search over classifier and cv strategy

PH_F = "radial_distances"                                          #or sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
                                            # or try -> as in paper suposedly: sklearn.svm.LinearSVC
K_FOLDS = 3


"""
Distance Functions for Mat of persistence diags -> classify based on array -> pw distance relations are feature row per idx

"""

# ------------------------ Data loading --------------------------------
labels, pers_images = load_data(
    datapath=DATAPATH,
    types=LAYER,
    neurite_type= NEURITE_TYPE,
    pers_hom_function= PH_F,
    flatten=True
)

### ------------------------ Training dataset --------------------------------

trainer = skTrainer(
    data=pers_images,
    labels=labels,
)

trainer.train_gridsearch()



train_crossvalidation(
    labels=labels,
    pers_images=pers_images,
    sk_clf=CLS,
    n_splits=K_FOLDS
)

