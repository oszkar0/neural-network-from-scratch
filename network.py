import numpy as np


class NeuralNetwork:
    def __init__(self, loss_func_derivative):
        self.loss_func_derivative = loss_func_derivative
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            for j in range(len(x_train)):
                # calculate output
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # adjust the weights and biases
                error = self.loss_func_derivative(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagate(error, learning_rate)

    def predict(self, x_input):
        # calculate and return output
        output = x_input

        for layer in self.layers:
            output = layer.forward_propagate(output)

        return output

    def test(self, x_test, y_test):
        successes = 0

        for i in range(len(x_test)):
            if np.argmax(self.predict(x_test[i])) == np.argmax(y_test[i]):
                successes += 1

        return successes / len(x_test)