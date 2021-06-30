from pandas import DataFrame
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from classifiers.logistic_regression import LogisticRegression as MyLogReg
from classifiers.one_vs_rest import OneVsRestClassifier
from classifiers.naive_bayes import NaiveBayes

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class ClassificationTester:

    def __init__(self, x, y, norm=True):
        self.x_train, self.y_train, self.x_test, self.y_test = self.__train_test_split(x, y, norm)
        self.classifiers = {
            'My Logistic Regression': OneVsRestClassifier(MyLogReg(epochs=250)),
            'SK Logistic Regression': LogisticRegression(multi_class='ovr'),
            'My Gaussian Naive Bayes': NaiveBayes(),
            'SK Gaussian Naive Bayes': GaussianNB(),
            'SVM': svm.SVC(kernel='linear'),
            'Random Forest': RandomForestClassifier(random_state=1, bootstrap=True)
        }

    def compare_results(self, show_pred=False):
        for name, classifier in self.classifiers.items():
            self.__show_results(name, classifier, show_pred)

    def __train_test_split(self, x, y, norm):
        if norm:
            x = (x - x.mean()) / x.std()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)

        # my classifiers expect numpy arrays
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return x_train, y_train, x_test, y_test

        return x_train.to_numpy(), y_train.values, x_test.to_numpy(), y_test.values

    def __show_results(self, name, classifier, show_pred):
        classifier.fit(self.x_train, self.y_train)
        pred = classifier.predict(self.x_test)
        true = self.y_test

        print(f'\n----- {name} Classification results -----')
        if show_pred:
            result = DataFrame({'True': true, 'Predicted': pred})
            print(result)
            print(50 * '-')

        accuracy = sum(true == pred) / len(true)
        print(f'Accuracy: {accuracy}')
        print(50 * '-')
        print('Classification report:\n', classification_report(true, pred, zero_division=0))
        print(50 * '-')
