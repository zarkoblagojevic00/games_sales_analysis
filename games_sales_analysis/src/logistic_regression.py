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
        x_biased = np.insert(x, 0, 1, axis=1)
        return self.__predict_probability(x_biased) > self.threshold

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

# Test
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
x_test = x[7:]
y_test = y[7:]

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train, verbose=True)
pred = log_reg.predict(x_test)

correct = pred == y_test
num_of_correct = sum(correct)
num_of_incorrect = len(y_test) - num_of_correct

print(f'Correct: {num_of_correct}   Incorrect: {num_of_incorrect}')
result = num_of_correct / len(y_test)
print(f'Result={result[0]}')