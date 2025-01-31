from NeuralNetwork import NeuralNetwork
from optimizer import optimizer
from typing import List
from utils import Node
from helpers import SSR
import numpy as np
import matplotlib.pyplot as plt


class TrainNetwork:
    def __init__(self):
        pass

    def input_verification(self, net: NeuralNetwork, training_data) -> bool:
        return len(net.network[0][0].weights) == len(training_data[0][0])

    def forward_pass(self, net: NeuralNetwork, observation: List[float]) -> float:
        current_input = observation
        all_layer_outputs = []
        for layer in net.network:
            layer_Outputs = []
            for node in layer:
                layer_Outputs.append(node.compute(current_input))
            all_layer_outputs.append(layer_Outputs)
            current_input = layer_Outputs
        return [current_input[0], all_layer_outputs]

    def train(self, network: NeuralNetwork, training_data: List, num_epochs: int) -> None:
        if self.input_verification(network, training_data):
            for i in range(0, num_epochs):
                training_inputs = [observation[0]
                                   for observation in training_data]
                observed = [observation[1] for observation in training_data]
                predictions = []
                layerOutputs = []
                for observation in training_inputs:
                    predictions.append(
                        self.forward_pass(network, observation)[0])
                    layerOutputs.append(
                        self.forward_pass(network, observation)[1])
                optimizer_object = optimizer()


nnObj = NeuralNetwork(1, 1, 2)
trainObj = TrainNetwork()
trainingData = [[[0.1], 0.025], [[0.55], 1], [[1], 0.01]]
nnObj.printNetwork()
trainObj.train(nnObj, trainingData, 2)
