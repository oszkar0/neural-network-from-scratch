import numpy as np
import layers
import network
import loss_functions
import activation_functions
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# reshape into two dimensional array
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
# convert number to different type
x_train = x_train.astype('float32')
# normalize
x_train /= 255
# encode output into vectors with ten elements
y_train = to_categorical(y_train)

# repeat actions for test data
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

test_network = network.NeuralNetwork(loss_functions.mse_loss_derivative)
test_network.add_layer(layers.FullyConnectedLayer(28*28, 100))
test_network.add_layer(layers.ActivationLayer(activation_functions.tanh, activation_functions.tanh_derivative))
test_network.add_layer(layers.FullyConnectedLayer(100, 50))
test_network.add_layer(layers.ActivationLayer(activation_functions.tanh, activation_functions.tanh_derivative))
test_network.add_layer(layers.FullyConnectedLayer(50, 10))
test_network.add_layer(layers.ActivationLayer(activation_functions.tanh, activation_functions.tanh_derivative))

# train network
test_network.fit(x_train[:5000], y_train[:5000], epochs=20, learning_rate=0.1)

# test
print(test_network.test(x_test[:1000], y_test[:1000]))

