import numpy as np
import copy

from probability_matrix import ProbabilityMatrixPredictor


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
        predictor = ProbabilityMatrixPredictor(self.classifiers)
        return predictor.predict(x)

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
