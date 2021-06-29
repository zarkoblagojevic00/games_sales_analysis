import numpy as np

class NaiveBayes:

    def __init__(self):
        self.separated = {}
        self.descriptors = {}

    def fit(self, x, y):
        self.__init_descriptors(x, y.flatten())

    def predict(self, x):
        probabilities = self.predict_proba(x)
        return max(probabilities, key=probabilities.get)

    def predict_proba(self, x):
        probabilities = {}
        for target_class, descriptor in self.descriptors.items():
            probabilities[target_class] = descriptor.predict_quasi_probability(x)

        return probabilities

    def __init_descriptors(self, x, y):
        self.__separate_samples_by_class(x, y)
        total_samples = x.shape[0]
        for target_class in self.separated:
            self.descriptors[target_class] = self.__create_descriptor(target_class, total_samples)

    def __separate_samples_by_class(self, x, y):
        for sample, target_class in zip(x, y):
            if target_class not in self.separated:
                self.separated[target_class] = []
            self.separated[target_class].append(sample)

    def __create_descriptor(self, target_class, total_samples):
        class_samples = self.separated[target_class]
        total_class_samples = len(class_samples)
        class_probability = total_class_samples / total_samples
        descriptor = NaiveBayes._NaiveBayesClassDescriptor(target_class,
                                                           class_samples,
                                                           class_probability)
        return descriptor

    class _NaiveBayesClassDescriptor:

        def __init__(self, predicted_class, class_samples, class_probability):
            self.predicted_class = predicted_class
            self.class_samples = class_samples
            self.class_probability = class_probability
            self.col_means = np.mean(class_samples, axis=0)
            self.col_stds = np.std(class_samples, axis=0)

        # P(class|sample) ~ P(sample|class) * P(class) (omitting division with P(sample))
        # P(class|sample) ~ П(datum[i]|class) * P(class) (hence the naive part)
        # datum[i]|class will be approximated with gaussian pdf
        # because we are assuming that each column of data has gaussian distribution
        def predict_quasi_probability(self, sample):
            quasi_probability = self.class_probability
            for idx, datum, in enumerate(sample):
                quasi_probability += self.__calculate_gaussian_likelihood(idx, datum)

            return quasi_probability

        def __calculate_gaussian_likelihood(self, idx, x):
            mean = self.col_means[idx]
            std = self.col_stds[idx]

            # value of gaussian pdf
            exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
            return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


x = np.array([[2.7810836, 2.550537003],
              [1.465489372, 2.362125076],
              [3.396561688, 4.400293529],
              [1.38807019, 1.850220317],
              [3.06407232, 3.005305973],
              [7.627531214, 2.759262235],
              [5.332441248, 2.088626775],
              [6.922596716, 1.77106367],
              [8.675418651, -0.242068655],
              [7.673756466, 3.508563011]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)

x_train = x[:7]
y_train = y[:7]
x_test = x[7]
y_test = y[7]

nb = NaiveBayes()
nb.fit(x_train, y_train)
print(nb.predict_proba(x_test))
pred = nb.predict(x_test)

correct = pred == y_test
num_of_correct = sum(correct)
num_of_incorrect = len(y_test) - num_of_correct

print(f'Correct: {num_of_correct}   Incorrect: {num_of_incorrect}')
result = num_of_correct / len(y_test)
print(f'Result={result}')