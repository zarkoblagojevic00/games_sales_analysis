import numpy as np

from probability_matrix import ProbabilityMatrixPredictor


class NaiveBayes:

    def __init__(self):
        self.separated = {}
        self.predictors = []

    def fit(self, x, y):
        self.__init_predictors(x, y.flatten())

    def predict(self, x):
        predictor = ProbabilityMatrixPredictor(self.predictors)
        return predictor.predict(x)

    def __init_predictors(self, x, y):
        self.__separate_samples_by_class(x, y)
        total_samples = x.shape[0]
        for target_class in self.separated:
            predictor = self.__create_predictor(target_class, total_samples)
            self.predictors.append(predictor)

    def __separate_samples_by_class(self, x, y):
        for sample, target_class in zip(x, y):
            if target_class not in self.separated:
                self.separated[target_class] = []
            self.separated[target_class].append(sample)

    def __create_predictor(self, target_class, total_samples):
        class_samples = self.separated[target_class]
        total_class_samples = len(class_samples)
        class_probability = total_class_samples / total_samples
        predictor = NaiveBayes._NaiveBayesClassPredictor(target_class,
                                                           class_samples,
                                                           class_probability)
        return predictor

    class _NaiveBayesClassPredictor:

        def __init__(self, predicted_class, class_samples, class_probability):
            self.predicted_class = predicted_class
            self.class_samples = class_samples
            self.class_probability = class_probability
            self.col_means = np.mean(class_samples, axis=0)
            if len(class_samples) > 1:
                self.col_stds = np.std(class_samples, axis=0)
            else:
                self.col_stds = [1] * len(class_samples[0])

            print()

        # P(class|sample) ~ P(sample|class) * P(class) (omitting division with P(sample))
        # P(class|sample) ~ П(datum[i]|class) * P(class) (hence the naive part)
        # P(datum[i]|class) will be approximated with gaussian pdf
        # because we are assuming that each column of data has gaussian distribution
        def predict(self, samples):
            n_rows = samples.shape[0]
            quasi_probabilities = np.empty(shape=(n_rows, 1))
            for idx, sample in enumerate(samples):
                quasi_probabilities[idx, 0] = self.__predict_quasi_probability(sample)
            return quasi_probabilities

        # applying log scaling to avoid float underflow
        # log transforms П into sum
        def __predict_quasi_probability(self, sample):
            quasi_probability = np.log(self.class_probability)
            for idx, datum in enumerate(sample):
                quasi_probability += np.log(self.__calculate_gaussian_likelihood(idx, datum))
            return quasi_probability

        def __calculate_gaussian_likelihood(self, idx, x):
            mean = self.col_means[idx]
            std = self.col_stds[idx]

            # value of gaussian pdf
            exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
            return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
