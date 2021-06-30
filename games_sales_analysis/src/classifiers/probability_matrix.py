import numpy as np


class ProbabilityMatrixPredictor:

    def __init__(self, probability_predictors):
        self.probability_predictors = probability_predictors
        self.probability_matrix = None

    def predict(self, x):
        self.__create_probability_matrix(x)
        return self.__predict_classes()

    def __create_probability_matrix(self, x):
        n_rows = x.shape[0]
        n_cols = len(self.probability_predictors)
        self.probability_matrix = np.empty(shape=(n_rows, n_cols))
        for idx, predictor in enumerate(self.probability_predictors):
            self.probability_matrix[:, idx] = predictor.predict(x)[:, 0]

    def __predict_classes(self):
        # using np.ma because apply_along_axis from np trims strings to same length
        return np.ma.apply_along_axis(self.__predict_class, 1, self.probability_matrix)

    def __predict_class(self, row):
        max_probability_idx = np.nanargmax(row)
        predicted_class = self.probability_predictors[max_probability_idx].predicted_class
        return predicted_class
