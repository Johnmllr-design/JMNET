from typing import List
from activation import softplus
import random as rd
import helpers as hp


# This python module is a collection of a variety of neurons used to build the architure
# of a neural network, including an input node, an output node, and a node class for
# the nodes in the hidden layer between the input and the output nodes


class Node():

    def __init__(self, numWeights: int) -> None:
        self.weights = []
        self.bias = 0.0
        for i in range(0, numWeights):
            self.weights.append(rd.randrange(0, 10))

    def compute(self, input: List[float]):
        weighted_Input = hp.dot(self.weights, input) + self.bias
        return softplus.evaluate(weighted_Input)




        

         






    
        




    
