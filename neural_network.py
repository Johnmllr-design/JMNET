# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for the declaration of the
# structure of the neural network
# Date: [1/15/2025]

from typing import List
from utils import Node


# class to hold the declaration of a neural network. It is essentially an array of arrays of
# nodes, with each array representing a layer of nodes. The current implementation has one output node,
# which passes forward the prediction of the ith input.
class NeuralNetwork:
    def __init__(
        self,
        num_inputs: int,
        num_hidden_layers: int,
        layer_size: int,
        activation_function: str,
    ):
        network = [[None]]

        for i in range(0, num_hidden_layers):
            curLayer = []
            if i == 0:
                # append layer_size nodes which each take num_inputs inpus
                for j in range(0, layer_size):
                    curLayer.append(Node(num_inputs, activation_function))
            else:
                # append layer_size nodes which each take num_inputs inputs
                for j in range(0, layer_size):
                    curLayer.append(Node(layer_size, activation_function))
            network.append(curLayer)

        # append one node for the output layer
        network.append([Node(layer_size, activation_function)])
        self.network = network
        self.activation_function = activation_function
