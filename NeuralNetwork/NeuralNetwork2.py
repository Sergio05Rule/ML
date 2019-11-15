import numpy as np
from math import e

def create_network(hidden_layer_neurons, first_layer_neurons):

    layers = len(hidden_layer_neurons) + 1  # +1 perchè bisogna includere il livello iniziale delle feature

    network = list()

    network.append(np.ones((hidden_layer_neurons[0],first_layer_neurons)))  # prima matrice di teta, tra il layer 1 e il 2

    for index, neurons in enumerate(hidden_layer_neurons[:-1]):
        start_layer_neurons = hidden_layer_neurons[index+1]
        end_layer_neurons = hidden_layer_neurons[index]+1

        network.append(np.ones((start_layer_neurons, end_layer_neurons)))

    return network

class NeuralNetwork():

    def __init__(self, X, Y, neurons):

        self.Y = np.array(Y)
        self.X = np.array(X)
        self.X = (np.c_[ np.ones((len(Y),1)) ,self.X])  #aggiunta x0
        print('len', len(self.X[0]))
        self.network = create_network(neurons, len(self.X[0]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_value(self, layer_minus_one, previous_layer):

        theta_layer_matrix = self.network[layer_minus_one]
        #z = list()
        #for theta in theta_layer_matrix:
        #    z.append(self.X.dot(theta.T))

        z = theta_layer_matrix.dot(previous_layer.T)

        a = self.sigmoid(z)

        # ---- aggiunta bias -----

        a = (np.c_[ np.ones((len(a[0]),1)) ,a.T])  #aggiunta x0
        # in questa forma ho una matrice (lunghezza del dataset) X (numero di neuroni con bias)
        return a

    def forward_propagation(self):

        a = self.activation_value(0,self.X)
        for layer in range(1, len(self.network)):  # per tutti gli hidden layer + output layer di cui dobbiamo calcolare gli activation values

            a = self.activation_value(layer,a)

        # removing bias from output layer
        a = a.T[1:]
        return a.T

    def cost_function(self):

        a = self.forward_propagation()
        cost = 0

        for index in range(len(self.Y)):

            






