from typing import List
import random
import helpers as hp


# This python module is a collection of a variety of useful things, such as
# neurons used to build the architure
# of a neural network, including an input node, an output node, and a node class for
# the nodes in the hidden layer between the input and the output nodes


class Node:
    def __init__(self, numWeights: int, activation_function: str) -> None:
        self.weights = []
        self.bias = 0.0
        self.z = -1
        self.activation_function = activation_function
        for i in range(0, numWeights):
            randNum = random.uniform(1, 2)
            self.weights.append(randNum)

    def compute(self, input: List[float]):
        weighted_Input = hp.dot(self.weights, input) + self.bias
        self.z = weighted_Input
        output = 0.0
        if self.activation_function == "sigmoid":
            output = hp.sigmoid(weighted_Input)
        elif self.activation_function == "RELU":
            output = hp.ReLU(weighted_Input)
        return output
