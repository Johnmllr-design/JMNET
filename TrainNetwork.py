from NeuralNetwork import NeuralNetwork
from typing import List
from utils import Node
from helpers import SSR
import numpy as np
import matplotlib.pyplot as plt
from PredictionAnalysis import display_graph


class TrainNetwork:
    def __init__(self):
        pass

    def input_verification(self, net: NeuralNetwork, training_data) -> bool:
        return len(net.network[0][0].weights) == len(training_data[0][0])

    def forward_pass(self, net: NeuralNetwork, observation: List[float]) -> float:
        current_input = observation
        outputs = [observation]
        i = 0
        for layer in net.network:
            layer_Outputs = []
            for node in layer:
                layer_Outputs.append(node.compute(current_input))
            outputs.append(layer_Outputs)
            current_input = layer_Outputs
            i += 1
        return outputs


