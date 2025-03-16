# main.py
# TrainNetwork
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes the declaration, forward pass, backpropagation, and training loop.
# Date: [1/15/2025]

from NeuralNetwork import NeuralNetwork
from typing import List
from helpers import mean_squared_error
import numpy as np
from sigmoid_optimizer import sigmoid_optimizer
from ReLU_optimizer import ReLU_optimizer
from helpers import plot


class TrainNetwork:
    def __init__(self):
        pass

    # input verification ensures that if a network was established to have x inputs,
    # the input provided will match that size. it uses the trick where the first layer of the 
    # neural network is [None], representing the "input layer" that simly passes on the features
    # to the hidden layer
    def input_verification(self, net: NeuralNetwork, training_data: List) -> bool:
        if len(net.network[1][0].weights) == len(training_data[0][0]):
            return True
        else:
            raise ValueError("input feature size doesn't match the model's architecture. Redefine your model")

    # forward pass is a function to pass the input through the network and save the outputs
    # of each node in each layer, stored as an 2D array of node outputs
    def forward_pass(self, net: NeuralNetwork, observation: List[float]):
        
        current_input = observation
        outputs = [observation]
        i = 0
        for index in range(1, len(net.network)):
            
            layer_Outputs = []
            
            for node in net.network[index]:
                layer_Outputs.append(node.compute(current_input))
            
            outputs.append(layer_Outputs)
            
            current_input = layer_Outputs
            i += 1
        
        return outputs
    
    
    #train takes the network object and the training data, and performs
    # 1000 epochs on the data to stochastically put each observation through the 
    # network, and then calls the appropriate backpropagation class to adjust weights

    def train(self, net: NeuralNetwork, trainingData: List, epochs: int):
        
        inputs = []
        labels = []
        optimizer = None


        if net.activation_function == "sigmoid":
            optimizer = sigmoid_optimizer()
        elif net.activation_function == "RELU":
            optimizer = ReLU_optimizer()

        # save the inputs and training labels for later plotting
        for input, label in trainingData:

            inputs.append(input)

            labels.append(label)
        
        # determine if the number of input features is correct for the model
        self.input_verification(net, trainingData)

        for i in range(0, epochs):

            outputs = [] # use an array to store the outputs of each input for the current epoch

            for input, label in trainingData:
           
                # forward pass the current input
                activations = self.forward_pass(net, input)

                outputs.append(activations[-1][0])

                # backpropagate the error with respect to each weight
                optimizer.backpropagate(net, activations, label)
            

            # if we are at the last epoch, plot the outputs of the function
            if i == epochs - 1:
                plot(inputs, labels, outputs)



            
        

        
        
        
    



    

