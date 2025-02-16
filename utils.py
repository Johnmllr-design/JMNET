from typing import List
import random
import helpers as hp


# This python module is a collection of a variety of useful things, such as
# neurons used to build the architure
# of a neural network, including an input node, an output node, and a node class for
# the nodes in the hidden layer between the input and the output nodes


class Node:
    def __init__(self, numWeights: int) -> None:
        self.weights = []
        self.bias = 0.0
        for i in range(0, numWeights):
            randNum = random.uniform(1, 2)
            self.weights.append(randNum)

    def compute(self, input: List[float]):
        weighted_Input = hp.dot(self.weights, input) + self.bias
        output = hp.sigmoid(weighted_Input)
        return output
