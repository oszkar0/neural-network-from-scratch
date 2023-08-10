import numpy as np


class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagate(self, input_data):
        raise NotImplementedError

    def backward_propagate(self, output_error, learning_rate):
        raise NotImplementedError


class FullyConnectedLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        #  create weights and bias matrices, initially with random values
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagate(self, input_data):
        self.input = input_data
        # calculate and return output Y = X*W + B
        self.output = np.matmul(input_data, self.weights) + self.bias
        return self.output

    def backward_propagate(self, output_error, learning_rate):
        # dE/dX = dE/dY * W-transposed
        input_error = np.matmul(output_error, self.weights.T)
        # dE/dW = X-transposed * dE/dY
        weights_error = np.matmul(self.input.T, output_error)
        # dE/dB = dE/dY
        bias_error = output_error

        # W = W - learning_rate * dE/dW
        self.weights -= learning_rate * weights_error
        # B = B - learning_rate * dE/dB
        self.bias -= learning_rate * bias_error
        return input_error


class ActivationLayer(BaseLayer):
    def __init__(self, activation_func, activation_func_derivative):
        super.__init__()
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative

    def forward_propagate(self, input_data):
        self.input = input_data
        # Y = f(X)
        self.output = self.activation_func(self.input)
        return self.output

    def backward_propagate(self, output_error, learning_rate):
        # input_error = dE/dY * f'(X)
        return self.activation_func_derivative(self.input) * output_error
