import numpy as np
import sklearn.tree
import sklearn 

from data import load_data
from train import train_crossvalidation

DATAPATH = "../Data/"
LAYER = "L2"
NEURITE_TYPE = "apical_dendrite"
CLS = sklearn.tree.DecisionTreeClassifier() # alt: sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis() 
                                            #or sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
                                            # or try -> as in paper suposedly: sklearn.svm.LinearSVC
K_FOLDS = 3


"""
Distance Functions for Mat of persistence diags -> classify based on array -> pw distance relations are feature row per idx

"""

# ------------------------ Data loading --------------------------------
labels, pers_images, pers_diagrams = load_data(
    datapath=DATAPATH,
    types=LAYER,
    neurite_type= NEURITE_TYPE,
    flatten=True
)

### ------------------------ Training dataset --------------------------------

train_crossvalidation(
    labels=labels,
    pers_images=pers_images,
    sk_clf=CLS,
    n_splits=K_FOLDS
)

