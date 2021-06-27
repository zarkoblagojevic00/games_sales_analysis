import numpy as np
import copy
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as log_reg


from logistic_regression import LogisticRegression


class OneVsRestClassifier:

    def __init__(self, model):
        self.model = model
        self.classifiers = []
        self.probability_matrix = None

    def fit(self, x, y):
        unique_classes = np.unique(y)
        for unique_class in unique_classes:
            model = copy.deepcopy(self.model)
            model_wrapper = OneVsRestClassifier._OneVsRestModelWrapper(model)
            model_wrapper.fit_model(x, y, predicted_class=unique_class)
            self.classifiers.append(model_wrapper)

    def predict(self, x):
        self.__create_probability_matrix(x)
        return self.__predict_classes()

    def __create_probability_matrix(self, x):
        n_rows = x.shape[0]
        n_cols = len(self.classifiers)
        self.probability_matrix = np.empty(shape=(n_rows, n_cols))
        for idx, classifier in enumerate(self.classifiers):
            self.probability_matrix[:, idx] = classifier.predict(x)[:, 0]

    def __predict_classes(self):
        return np.apply_along_axis(self.__predict_class, 1, self.probability_matrix)

    def __predict_class(self, row):
        idx = row.argmax()
        predicted_class = self.classifiers[idx].predicted_class
        return predicted_class

    class _OneVsRestModelWrapper:
        def __init__(self, model):
            self.model = model
            self.predicted_class = None

        def fit_model(self, x, y, predicted_class):
            self.predicted_class = predicted_class
            target = (y == self.predicted_class).astype(int).reshape(-1, 1)
            self.model.fit(x, target)

        def predict(self, x):
            return self.model.predict_proba(x)


# Test
def compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = OneVsRestClassifier(LogisticRegression(epochs=250))
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    print(classification_report(y_test, clf_predictions))

    clf = log_reg(multi_class='ovr')
    clf.fit(X_train, y_train.ravel())
    clf_predictions = clf.predict(X_test)
    print(classification_report(y_test, clf_predictions))


iris = datasets.load_iris()
compare(iris.data, iris.target)
