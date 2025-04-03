import numpy as np
import sklearn.calibration
import sklearn.svm
import sklearn.tree
import sklearn 

from data import load_data
from train import train_crossvalidation

DATAPATH = "../Data/"
LAYER = "L2"
NEURITE_TYPE = "apical_dendrite"
CLS = sklearn.svm.LinearSVC(
    C=1.0,
    loss="squared_hinge",
    penalty="l2",
    dual=True,
    tol=0.001,
    multi_class="ovr",
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=2000,
) # alt: sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis() 
CLS = sklearn.tree.DecisionTreeClassifier(
    max_depth=5,  # Limits the depth of the tree to prevent overfitting
    class_weight="balanced",  # Ensures the classifier accounts for class imbalance
)
CLS = sklearn.svm.SVC(
    C=3.5,
    gamma=0.001,
    kernel="rbf",

)
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

train_crossvalidation(
    labels=labels,
    pers_images=pers_images,
    sk_clf=CLS,
    n_splits=K_FOLDS
)

