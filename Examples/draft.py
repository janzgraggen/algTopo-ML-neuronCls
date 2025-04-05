import importlib
import numpy as np
def split_data(labels, pers_images, train_size=0.8):
    """
    Split the data into training and testing sets.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(pers_images, labels, train_size=train_size)
    return X_train, X_test, y_train, y_test

def cross_validation(labels, pers_images, n_splits=5):
    """
    Split the data into training and testing sets.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits)
    
    for train_index, test_index in kf.split(pers_images):
        X_train, X_test = pers_images[train_index], pers_images[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        yield X_train, X_test, y_train, y_test
    




list_of_modules = ["discriminant_analysis", "discriminant_analysis", "tree"]
list_of_classifiers = [
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "DecisionTreeClassifier",
]
def train_sklearn_classifier(mod, classifier, data, labels, **kwargs):
    """
WRAPER
    Trains the classifier from mod of sklearn with data and targets.

    Returns a fited classifier.
    """
    clas_mod = importlib.import_module("sklearn." + mod)
    clf = getattr(clas_mod, classifier)()
    clf.set_params(**kwargs)

    clf.fit(data, labels)

    return clf
def predict(clf, data):
    """
WRAPER
    Predict label for data for the trained classifier clf.

    Returns the index of the predicted class for each datapoint in data.
    """
    predict_label = clf.predict([data])

    return predict_label[0]

