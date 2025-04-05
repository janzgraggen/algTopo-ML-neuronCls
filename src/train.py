import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate ,GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from collections.abc import Iterable
from sklearn.model_selection import BaseCrossValidator



def train_sklearn_classifier(labels,pers_image, sk_clf, train_test_ratio):
    """
    Train a sklearn classifier with the given data and labels.
    """

    # Split the data into training and testing sets
    split_index = int(len(pers_image) * train_test_ratio)
    X_train, X_test = pers_image[:split_index], pers_image[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    # Train the classifier
    sk_clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = sk_clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return sk_clf



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
    

class skTrainer(): 
    def __init__(
        self, 
        data: any | list[any],
        labels,
        crosvalidate: any | None = None,
        gridsearch: GridSearchCV | None = None,
        cls: any | None = None,
        ):
        ## SET DEFAULTS
        
        if not cls:
            self.cls = cls
        else:
            self.cls = SVC()

        if not crosvalidate:
            self.cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        else:
            self.cv = crosvalidate

        if not gridsearch:
            paramgrid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced', None]},
            ]
            self.gs = GridSearchCV(self.cls,paramgrid,cv=self.cv)
        else:
            self.gs = gridsearch

        self.X = [data] if type(data) is not list else data
        self.y = labels
        
  
    def train_crossvalidation(self):
        """
        Train the classifier using cross-validation and calculate accuracies.
        """
        avg_accuracies = []
        for data in self.X: 
            #cross-validation doc:
            #       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
            score = cross_validate(estimator=self.cls, X=data ,y=self.y, scoring= ["accuracy"],cv= self.cv)
            # The scoring metric strings:
            #       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-api-overview
            accuracies = score["test_accuracy"]
            avg_accuracies.append(np.mean(accuracies))

        print(f"Average accuracy: {avg_accuracies:.2f}")
        return avg_accuracies
    
    def train_gridsearch(self, return_estimator: bool = False):
        """
        Train the classifier using cross-validation and calculate accuracies.
        """
        best_average_accuracies = []
        best_estimators = []
        for data in self.X:
            #cross-validation doc:
            #       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
            GS =  GridSearchCV(estimator=self.gs, X=data ,y=self.y, scoring= ["accuracy"],cv= self.cv).best_estimator_
            best_estimators.append(GS.best_estimator_)
            best_average_accuracies.append(GS.best_score_)
            

        print(f"Best parameters: {GS.best_params_}")
        print(f"Average accuracy: {score:.2f}")
        return best_average_accuracies, best_estimators if  return_estimator else best_average_accuracies
        
    
