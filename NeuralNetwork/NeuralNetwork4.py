import numpy as np
from math import e
from math import log

def create_network(hidden_layer_neurons, input_layer_neurons, output_layer_neurons):

    layers = len(hidden_layer_neurons) + 1  # +1 perchè bisogna includere il livello iniziale delle feature

    network = list()

    network.append(2 * np.random.random((hidden_layer_neurons[0], input_layer_neurons))-1)  # prima matrice di teta, tra il layer 1 e il 2

    for index, neurons in enumerate(hidden_layer_neurons[:-1]):
        start_layer_neurons = hidden_layer_neurons[index+1]
        end_layer_neurons = hidden_layer_neurons[index]

        network.append(2 * np.random.random(((start_layer_neurons, end_layer_neurons)))-1)

    network.append(2 * np.random.random((output_layer_neurons, hidden_layer_neurons[-1]))-1)  # prima matrice di teta, tra il layer 1 e il 2

    return network

class NeuralNetwork():

    def __init__(self, X, Y, neurons):

        self.Y = np.array(Y)
        self.X = np.array(X)
        self.layers = len(neurons)+1
        self.network = create_network(neurons, len(self.X[0]), len(self.Y[0]))
        self.bias = 2 * np.random.random((self.layers, len(self.X))) - 1

        for net in self.network:
            print(net.shape)
        input()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivation(self, x):
        return x*(1-x)

    def activation_value(self, layer_minus_one, previous_layer):

        # ---- aggiunta bias -----
        previous_a = np.r_[[self.bias[0]],previous_layer]

        theta_layer_matrix = self.network[layer_minus_one]

        z = theta_layer_matrix.dot(previous_a)
        a = self.sigmoid(z)
        return a

    def forward_propagation(self):

        activation_values = list()

        a = self.X.T

        #removed first a from the append

        for layer in range(len(self.network)):  # per tutti gli hidden layer + output layer di cui dobbiamo calcolare gli activation values

            a = self.activation_value(layer,a)
            activation_values.append(a)

        return activation_values

    def backward_propagation(self):

        activation_values = self.forward_propagation()

        deltas = list()
        print(activation_values[-1].shape)
        print(self.Y.shape)
        input()
        delta = activation_values[-1] - self.Y.T
        deltas.append(delta)

        for layer in range(self.layers -2,-1,-1):

            print(self.network[layer+1])
            print(self.network[layer+1].shape)
            print(delta.shape)
            delta = np.dot(self.network[layer+1].T,delta)
            print(delta.shape)
            print(activation_values[0].shape, activation_values[1].shape)
            input()
            delta = np.multiply(delta, np.multiply(activation_values[layer],1 - activation_values[layer]))
            print(delta)
        input()


    def regularization(self, _lambda):

        regularization = 0

        for matrix in self.network:

            squared = np.square(matrix)
            squared = np.sum(squared)
            regularization += squared

        return regularization * _lambda / 2*len(self.Y)

    def cost_function(self, _lambda = 0):

        a = self.forward_propagation()
        J = 0
        for row in range(len(self.Y)):

            for index, y in enumerate(self.Y[0]):

                if y == 1:

                    J += log(a[row][index])

                else:

                    J += log(1 - a[row][index])

        J = - J / len(self.Y)

        return J + self.regularization(_lambda)


    def print_network(self):

        for net in self.network:
            print('\n', net)



