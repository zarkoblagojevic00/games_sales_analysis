import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.coefficients = []  # column_vector

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def predict(self, x):
        return self.__predict_probability(x) > self.threshold

    def predict_proba(self, x):
        x_biased = np.insert(x, 0, 1, axis=1)
        return self.__predict_probability(x_biased)

    def fit(self, x, y, verbose=False):
        cost_history = self.__do_gradient_descent(x, y)

        if verbose:
            self.__show_details(cost_history)

    def __do_gradient_descent(self, x, y):
        x_biased = np.insert(x, 0, 1, axis=1)
        self.coefficients = np.random.normal(loc=0, scale=0.01, size=(x_biased.shape[1], 1))

        cost_history = []
        for epoch in range(self.epochs):
            cost_history.append(self.__calculate_cost(x_biased, y))
            self.__update_coefficients(x_biased, y)

        return cost_history

    # Logistic Regression cannot be optimized by Sum of squared errors (SSE)
    # because sigmoid function introduces non-linearity (cost function is not convex)
    # Function used for calculating cost has the following form: {note that h(x) = predict(x)}
    # when y = 1 => cost(h(x), y) = -log(h(x))
    # when y = 0 => cost(h(x), y) = -log(1 - h(x))
    # written as one equation :
    #                   cost(h(x), y) = -y * log(h(x)) - (1-y) * log(1 - h(x))
    #
    # COST = 1/m * sum[cost(h(x), y)] (sum over m samples, m = num of samples)
    def __calculate_cost(self, x, y):
        y_t = y.transpose()
        predicted = self.__predict_probability(x)
        sample_size = len(y)
        cost = (-y_t @ np.log(predicted) - (1 - y_t) @ np.log(1 - predicted)) / sample_size
        return cost[0]

    def __update_coefficients(self, x, y):
        gradient = self.__calculate_gradient(x, y)
        self.coefficients -= self.learning_rate * gradient

    # Gradiant calculated by formula: 1/m * sum[(h(x)-y) * x] (over m samples)
    def __calculate_gradient(self, x, y):
        errors = self.__calculate_errors(x, y)
        sample_size = len(y)
        return x.transpose() @ errors / sample_size

    def __calculate_errors(self, x, y):
        return self.__predict_probability(x) - y

    def __predict_probability(self, x):
        linear_predict = x @ self.coefficients
        return LogisticRegression.__sigmoid(linear_predict)

    def __show_details(self, sse_history):
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch: 3d}   SSE: {sse_history[epoch]}')
        plt.plot(range(self.epochs), sse_history)
        plt.title('Cost function over epochs')
        plt.xlabel(f'Epochs({self.epochs})')
        plt.ylabel('Cost')
        plt.show()

