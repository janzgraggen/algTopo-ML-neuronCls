import importlib
import numpy as np
import sklearn.tree
import sklearn 

from data import load_data, split_data
from train import train_sklearn_classifier




datapath = "../Data/"
types = ["L2_IPC", "L2_TPC:A"]
neurite_type = "apical_dendrite"


# ------------------------ Data loading --------------------------------
labels, pers_images = load_data(
    datapath=datapath,
    types=types,
    neurite_type=neurite_type,
)

X_train, X_test, y_train, y_test = split_data(
    labels=labels,
    pers_images=pers_images,
    train_size=0.8
)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# ------------------------ Training dataset --------------------------------
X_train = [i.flatten() for i in X_train]
X_test = [i.flatten() for i in X_test]


# with use of tht train.py file
classifier = sklearn.tree.DecisionTreeClassifier() # alt: sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis() or sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
clf = train_sklearn_classifier("tree","DecisionTreeClassifier",  X_train, y_train) 

#direct 
cls = sklearn.tree.DecisionTreeClassifier()
cls.set_params()
cls.fit(X_train, y_train)

# ------------------------ Prediction --------------------------------
#direct
print(cls.predict(X_test))
print(y_test)
print(cls.score(X_test, y_test))


# print(predict(clf, X_test[0]))
# print(y_test[0])
  
# print(predict(clf, X_test[1]))
# print(y_test[1])



