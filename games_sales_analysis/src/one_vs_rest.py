import numpy as np
import copy


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
            model_wrapper.fit(x, y, predicted_class=unique_class)
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
        # using np.ma because apply_along_axis from np trims strings to same length
        return np.ma.apply_along_axis(self.__predict_class, 1, self.probability_matrix)

    def __predict_class(self, row):
        max_probability_idx = row.argmax()
        predicted_class = self.classifiers[max_probability_idx].predicted_class
        return predicted_class

    class _OneVsRestModelWrapper:
        def __init__(self, model):
            self.model = model
            self.predicted_class = None

        def fit(self, x, y, predicted_class):
            self.predicted_class = predicted_class
            target = (y == self.predicted_class).astype(int).reshape(-1, 1)
            self.model.fit(x, target)

        def predict(self, x):
            return self.model.predict_proba(x)
