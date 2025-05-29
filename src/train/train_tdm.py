import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate ,GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from typing import Union

def train_sklearn_classifier(labels,pers_image, sk_clf, train_test_ratio):
    """
    Train a sklearn classifier with the given data and labels.
    -> here standalone, however should be implemented in the skTrainer class
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
    
def train_crossvalidation(labels, data_dict, sk_clf, n_splits=5):
        """
        Train the classifier using cross-validation and calculate accuracies.
        -> here standalone, however in mainly used in the skTrainer class
        """
        avg_accuracies = {}
        print("================================")
        print(f"Training a {sk_clf.__class__.__name__}-Classifier with CrossValidation...")
        print("––––")
        for name, data in data_dict.items(): 
            #cross-validation doc:
            #       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
            score = cross_validate(estimator=sk_clf, X=data ,y=labels, scoring= "accuracy",cv= n_splits)
            # The scoring metric strings:
            #       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-api-overview
            accuracies = score['test_score']
            avg_accuracies[name] = np.mean(accuracies)
            print(f"Average accuracy of Data = {name}: {avg_accuracies[name]:.6f}")
            print("––––")
        return avg_accuracies

class skTrainer(): 
    def __init__(
        self, 
        data,#: dict[str, np.ndarray]
        labels,
        cls: Union[any, None] = SVC(),
        crosvalidate: Union[any, None] = None,
        gridsearch: Union[GridSearchCV, None] = None
    ):
        ## SET DEFAULTS
        self.cls = cls

        if not crosvalidate:
            self.cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        else:
            self.cv = crosvalidate

        if not gridsearch:
            paramgrid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced', None]},
            ]
            self.gs = GridSearchCV(self.cls,paramgrid,cv=self.cv, scoring= "accuracy")
        else:
            self.gs = gridsearch

        self.X = data
        self.y = labels

    def train_crossvalidation(self, vectorization_select = False):
        """
        Train the classifier using cross-validation and calculate accuracies.
        Parameters
        ----------
        vectorization_select : str or list of str, optional
            The vectorization method(s) to use for training. If False, all methods are used.
            Default is False.
        Returns
        -------
        avg_accuracies : dict
            A dictionary containing the average accuracies for each vectorization method.
        -------
        
        """
        avg_accuracies = {}
        print("\n================================")
        print(f"Training a {self.cls.__class__.__name__}-Classifier with CrossValidation...")
        print("––––")

        if vectorization_select:
            if type(vectorization_select) == str:
                data_dict = {vectorization_select: self.X[vectorization_select]}
            else: data_dict = {k: self.X[k] for k in vectorization_select}
        else:
            data_dict = self.X

        print(f"Data dict: {data_dict.keys()}")
        for name in data_dict: 
            data = data_dict[name]
            #cross-validation doc:
            #       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
            score = cross_validate(estimator=self.cls, X=data ,y=self.y, scoring= "accuracy",cv= self.cv)
            # The scoring metric strings:
            #       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-api-overview
            accuracies = score['test_score']
            avg_accuracies[name] = np.mean(accuracies)
            print(f"Average accuracy of Data = {name}: {avg_accuracies[name]:.6f}")
            print("––––")
        return avg_accuracies
    
    def train_gridsearch(self,vectorization_select = False, return_estimator: bool = False):
        """
        Train the classifier using Gridsearch and cross-validation and calculate accuracies.
        Parameters
        ----------
        vectorization_select : str or list of str, optional
            The vectorization method(s) to use for training. If False, all methods are used.
            Default is False.
        return_estimator : bool, optional
            If True, return the best estimator for each vectorization method.
            Default is False.
        Returns
        -------
        best_average_accuracies : dict
            A dictionary containing the best average accuracies for each vectorization method.
        best_estimators : dict
            A dictionary containing the best estimators for each vectorization method.
        -------
        """
        best_average_accuracies = {}
        best_estimators = {}
        print("\n=================================")
        print(f"Training a {self.cls.__class__.__name__}-Classifier with GridSearchCV...")
        print("––––")
        if vectorization_select:
            if type(vectorization_select) == str:
                data_dict = {vectorization_select: self.X[vectorization_select]}
            else: data_dict = {k: self.X[k] for k in vectorization_select}
        else:
            data_dict = self.X
            for vec in data_dict:
                data = data_dict[vec]
                #grid search doc:
                #       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
                self.gs.fit(np.array(data), np.array(self.y))
                best_estimators[vec] = self.gs.best_estimator_
                best_average_accuracies[vec] = self.gs.best_score_
                print(f"Average accuracy of Data = {vec}: {self.gs.best_score_:.6f}")
                print(f"Best parameters for training Data ={vec}: {self.gs.best_params_}")
                print("––––")
        
        #return best_average_accuracies, best_estimators if  return_estimator else best_average_accuracies
        return best_average_accuracies
        
    
