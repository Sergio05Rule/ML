import numpy as np
from math import e
from math import log

def create_network(hidden_layer_neurons, input_layer_neurons, output_layer_neurons):

    layers = len(hidden_layer_neurons) + 1  # +1 perchè bisogna includere il livello iniziale delle feature

    network = list()

    network.append(np.ones((hidden_layer_neurons[0], input_layer_neurons)))  # prima matrice di teta, tra il layer 1 e il 2

    for index, neurons in enumerate(hidden_layer_neurons[:-1]):
        start_layer_neurons = hidden_layer_neurons[index+1]
        end_layer_neurons = hidden_layer_neurons[index]+1

        network.append(np.ones((start_layer_neurons, end_layer_neurons)))

    network.append(np.ones((output_layer_neurons, hidden_layer_neurons[-1]+1)))  # prima matrice di teta, tra il layer 1 e il 2

    return network

class NeuralNetwork():

    def __init__(self, X, Y, neurons):

        self.Y = np.array(Y)
        self.X = np.array(X)
        self.X = (np.c_[ np.ones((len(Y),1)) ,self.X])  #aggiunta x0
        self.network = create_network(neurons, len(self.X[0]), len(self.Y[0]))
        lenght = len(neurons)+1
        self.bias = np.ones((len(neurons)+1, len(self.X)))
        print('\n\nX VALUES',self.X,'\n\n')

        for net in self.network:
            print(net.shape)
        input()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_value(self, layer_minus_one, previous_layer):

        theta_layer_matrix = self.network[layer_minus_one]
        #z = list()
        #for theta in theta_layer_matrix:
        #    z.append(self.X.dot(theta.T))

        z = theta_layer_matrix.dot(previous_layer.T)

        print('\n THETA LAYER ',theta_layer_matrix)
        print('\n LAYER ',previous_layer.T)
        print('\n ZETA',z)
        input()

        a = self.sigmoid(z)
        print('\n A',a)

        # ---- aggiunta bias -----

        a = (np.c_[ np.ones((len(a[0]),1)) ,a.T])
        # in questa forma ho una matrice (lunghezza del dataset) X (numero di neuroni con bias)
        return a

    def forward_propagation(self):

        a = self.activation_value(0,self.X)
        for layer in range(1, len(self.network)):  # per tutti gli hidden layer + output layer di cui dobbiamo calcolare gli activation values

            a = self.activation_value(layer,a)

        # removing bias from output layer
        a = a.T[1:]
        return a.T

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



