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

    def input_verification(self, net: NeuralNetwork, training_data) -> bool:
        return len(net.network[0][0].weights) == len(training_data[0][0])

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

    def train(self, net: NeuralNetwork, trainingData: List):
        
        epochs = 1000
        inputs = []
        labels = []
        optimizer = None


        if net.activation_function == "sigmoid":
            optimizer = sigmoid_optimizer()
        elif net.activation_function == "RELU":
            optimizer = ReLU_optimizer()

        # save the inputs and training labels for later plotting
        for input, label in trainingData:

            inputs.append(input[0])

            labels.append(label)

        for i in range(0, epochs):

            outputs = []
            inputs = []
            labels = []
            for input, label in trainingData:
           
                # forward pass the current input
                activations = self.forward_pass(net, input)

                outputs.append(activations[-1][0])
                inputs.append(input[0])
                labels.append(label)

                # backpropagate the error with respect to each weight
                optimizer.backpropagate(net, activations, label)
            

            # if we are at the last epoch, plot the outputs of the function
            if i == 999:
                plot(inputs, labels, outputs)



            
        

        
        
        
    



    

