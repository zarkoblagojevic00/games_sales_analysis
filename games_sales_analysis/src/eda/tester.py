from pandas import DataFrame
from numpy import ndarray

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from classifiers.logistic_regression import LogisticRegression as MyLogReg
from classifiers.one_vs_rest import OneVsRestClassifier
from classifiers.naive_bayes import NaiveBayes

from sklearn.metrics import classification_report
from src.eda.train_test_splitter import BasicTrainTestSplitter, LeaveOneOutTrainTestSplitter


class ClassificationTester:

    def __init__(self, x, y, norm=True, tts_strat='basic'):
        self.train_test_splits = []
        self.classifiers = {
            'My Logistic Regression': OneVsRestClassifier(MyLogReg(epochs=250)),
            'SK Logistic Regression': LogisticRegression(multi_class='ovr'),
            'My Gaussian Naive Bayes': NaiveBayes(),
            'SK Gaussian Naive Bayes': GaussianNB(),
            'SVM': svm.SVC(kernel='linear'),
            'Random Forest': RandomForestClassifier(random_state=1, bootstrap=True)
        }

        # my classifiers expect numpy arrays
        x = x if isinstance(x, ndarray) else x.to_numpy()
        y = y if isinstance(y, ndarray) else y.values

        if norm:
            x = (x - x.mean()) / x.std()

        self.tts_strat = tts_strat
        self.__train_test_split(x, y)

    def compare_results(self, show_pred=False):
        for name, classifier in self.classifiers.items():
            self.__show_results(name, classifier, show_pred)

    def __train_test_split(self, x, y):
        if self.tts_strat == 'basic':
            self.train_test_splits = BasicTrainTestSplitter(x, y).train_test_split()
        elif self.tts_strat == 'leave_one_out':
            self.train_test_splits = LeaveOneOutTrainTestSplitter(x, y).train_test_split()

    def __show_results(self, name, classifier, show_pred):
        if self.tts_strat == 'basic':
            self.__show_basic_report(name, classifier, show_pred)
        elif self.tts_strat == 'leave_one_out':
            self.__show_leave_one_out_report(name, classifier)

    def __show_basic_report(self, name, classifier, show_pred):
        for train_test_split in self.train_test_splits:
            x_train, x_test, y_train, y_test = train_test_split
            classifier.fit(x_train, y_train)
            pred = classifier.predict(x_test)
            true = y_test

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

    def __show_leave_one_out_report(self, name, classifier):
        n_correct_predictions = 0
        n_total_predictions = len(self.train_test_splits)

        for train_test_split in self.train_test_splits:
            x_train, x_test, y_train, y_test = train_test_split
            classifier.fit(x_train, y_train)
            pred = classifier.predict(x_test)
            true = y_test
            if true == pred:
                n_correct_predictions += 1

        accuracy = n_correct_predictions / n_total_predictions
        print(f'\n----- {name} Leave-one-out Cross Validation results -----')
        print(50 * '-')
        print(f'Accuracy: {accuracy}')
        print(50 * '-')
