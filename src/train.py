import numpy as np
import importlib
from data import cross_validation
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate





def train_crossvalidation(labels, pers_images, sk_clf, n_splits=5):
    """
    Train the classifier using cross-validation and calculate accuracies.
    """
    #cross-validation doc:
    #       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    score = cross_validate(estimator=sk_clf, X=pers_images ,y=labels, scoring= ["accuracy"],cv= n_splits)
    # The scoring metric strings:
    #       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-api-overview
    accuracies = score["test_accuracy"]
    avg_accuracy = np.mean(accuracies)

    print(f"Average accuracy: {avg_accuracy:.2f}")
    return avg_accuracy
    
    