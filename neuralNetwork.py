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
        first_Layer = True
        neural_Network = []
        for i in range(0, num_hidden_layers):
            current_Layer = []
            # if this is the first layer of the hidden layer, add nodes with only as many weights as 
            # there are inputs into the model
            if first_Layer:
                for j in range(0, layer_size):
                    newNode = Node(num_inputs)
                    current_Layer.append(newNode)
                first_Layer = False
            else:
                # if this is a layer after the first layer of the hidden layer, append layers_size nodes with
                # as many weights as there were nodes in the previous player
                for j in range(0, layer_size):
                    newNode = Node(layer_size)
                    current_Layer.append(newNode)
            neural_Network.append(current_Layer)
        #append an output neuron for a final numerical output inference
        neural_Network.append([Node(len(neural_Network[-1]))])
        #set the NeuralNetwork field of the neural network object to be this array of nodes
        self.NeuralNetwork = neural_Network

    def forward_pass(self, observation: List[float]) -> float:
        current_input = observation
        for layer in self.NeuralNetwork:
            layer_Outputs = []
            for node in layer:
                layer_Outputs.append(node.compute(current_input))
            print("output of layer " + str(layer) + " is " + str(layer_Outputs))
            current_input = layer_Outputs
        return current_input[0]
            


        
    
    #train based on Full-Batch Gradient Descent (Batch Gradient Descent),
    # that is, each epoch puts the full set of training data through the network
    def train(self, training_data, num_epochs) -> None:
        if self.input_verification(training_data):
            #perform num_epochs on the dataset
            training_outputs = [observation[1] for observation in training_data]
            for i in range(0, num_epochs):
                current_epoch_outputs = []
                for observation in training_data:
                    current_epoch_outputs.append(self.forward_pass(observation[0]))
                print("the prediction of epoch i is " + str(current_epoch_outputs) + " with the actual observationt ouputs being " + str(training_outputs))
                sum_of_squared_results = hp.SSR(current_epoch_outputs, training_outputs)
                print("the ssr for epoch " + str(i) + " is " + str(sum_of_squared_results))


    

    def input_verification(self, training_data) -> bool:
        return len(self.NeuralNetwork[0][0].weights) == len(training_data[0][0])
    

    def printNetwork(self):
        print("the input \'layer\' has " + str(len(self.NeuralNetwork[0][0].weights)) + " nodes, each for a different input parameter")
        for layer in self.NeuralNetwork:
            if len(layer) == 1:
                print("at output neuron with weight " + str(layer[0].weights))
            else:
                print(layer)



network = NeuralNetwork(1, 1, 2) 
trainingdata = [[[0.0], 1.2], [[0.5], 1.9], [[1.1], 0.8]]     
network.train(trainingdata, 1)   

