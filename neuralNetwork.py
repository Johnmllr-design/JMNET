# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for the declaration of the 
# structure of the neural network
# Date: [1/15/2025]

from typing import List
from utils import Node
import helpers as hp
import random


class NeuralNetwork():

    def __init__(self, num_inputs, num_hidden_layers, layer_size):
        network = [[None]]
        for i in range(0, num_hidden_layers):
            curLayer = []
            if i == 0:
                for j in range(0, layer_size):
                    curLayer.append(Node(num_inputs))
            else:
                for j in range(0, layer_size):
                    curLayer.append(Node(layer_size))
            network.append(curLayer)
        network.append([Node(layer_size)])
        self.network = network

    def reset_network(self):
        for layer_index in range(1, len(self.network)):
            for node in self.network[layer_index]:
                for i in range(0, len(node.weights)):
                    node.weights[i] = random.uniform(1, 5)
                node.bias = 0.0



    def printNetwork(self):
        for i in range(0, len(self.network)):
            if self.network[i] == [None]:
                print("this is the input layer")
            else:
                for node in self.network[i]:
                    print("PRINTNET: node " + str(node) + "in hidden layer " + str(i) +
                        " with weights " + str(node.weights) + " and bias " + str(node.bias))
