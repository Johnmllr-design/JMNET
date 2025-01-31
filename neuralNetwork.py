# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes the declaration, forward pass, backpropagation, and training loop.
# Date: [1/15/2025]

from typing import List
from utils import Node
import helpers as hp


class NeuralNetwork():

    def __init__(self, num_inputs, num_hidden_layers, layer_size):
        network = []
        for i in range(0, num_hidden_layers):
            new_Layer = []
            for i in range(0, layer_size):
                if i == 0:
                    new_Layer.append(Node(num_inputs))
                else:
                    new_Layer.append(Node(layer_size))
            network.append(new_Layer)
        network.append([Node(layer_size)])
        self.network = network

    def printNetwork(self):
        for i in range(0, len(self.network)):
            for node in self.network[i]:
                print("PRINTNET: node " + str(node) + "in hidden layer " + str(i) +
                      " with weights " + str(node.weights) + " and bias " + str(node.bias))
