import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate 


def train_sklearn_classifier(labels,pers_image, sk_clf, train_test_ratio):
    """
    Train a sklearn classifier with the given data and labels.
    """
    # Flatten the persistence images
    pers_image = [i.flatten() for i in pers_image]

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
    
    