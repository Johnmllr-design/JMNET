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
        first_layer = []
        for i in range(0, num_inputs):
            first_layer.append(Node(1))
        network.append(first_layer)
        for i in range(0, num_hidden_layers):
            new_Layer = []
            for i in range(0, layer_size):
                new_Layer.append(Node(len(network[-1])))
            network.append(new_Layer)
        network.append([Node(len(network[-1]))])
        self.network = network
            
    

    def forward_pass(self, observation: List[float]) -> float:
        current_input = observation
        all_layer_outputs = [observation]
        for layer in self.network:
            layer_Outputs = []
            for node in layer:
                layer_Outputs.append(node.compute(current_input))
            all_layer_outputs.append(layer_Outputs)
            current_input = layer_Outputs
        return [current_input[0], all_layer_outputs]
            


        
    
    #train based on Full-Batch Gradient Descent (Batch Gradient Descent),
    # that is, each epoch puts the full set of training data through the network
    def train(self, training_data) -> None:
        if self.input_verification(training_data):
            training_inputs = [observation[0] for observation in training_data]
            print("the training inputs are " + str(training_inputs))
            outputs = []
            layerOutputs = []
            for observation in training_inputs:
                outputs.append(self.forward_pass(observation)[0])
                layerOutputs.append(self.forward_pass(observation)[1])
            print(outputs)
            print("\n\n")
            print("the outputs of the activation functions at each layer are:")
            for l in layerOutputs:
                print(l)

    def input_verification(self, training_data) -> bool:
        return len(self.network[0][0].weights) == len(training_data[0][0])

    
    def printNetwork(self):
        for i in range(0, len(self.network)):
            for node in self.network[i]:
                print("at node " + str(node) + "in layer" + str(i) + " with weights " + str(node.weights))



net = NeuralNetwork(1, 1, 2) 
trainingdata = [[[1.9], 1.2], [[9.9], 1.9], [[1.1], 0.8]]
net.printNetwork()
net.train(trainingdata)
