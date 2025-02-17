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
import matplotlib.pyplot as plt
from helpers import SSR
from Optimizer import Optimizer


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
    
    
    def train(self, net: NeuralNetwork, trainingData):
        
        optimizer = Optimizer()
        epochs = 1000
        inputs = []
        labels = []
        
        for input, label in trainingData:

            inputs.append(input[0])

            labels.append(label)
        trained = False

        while not trained:

            for i in range(0, epochs):

                outputs = []
                inputs = []
                labels = []
                for input, label in trainingData:
           
                    activations = self.forward_pass(net, input)

                    outputs.append(activations[-1][0])
                    inputs.append(input[0])
                    labels.append(label)

                    optimizer.backpropagate(net, activations, label)
            
        
                import matplotlib.pyplot as plt

                if i == 999:
                    mse = mean_squared_error(labels, outputs)
                    print("the mse is " + str(mse))
                    if mse > 0.003:
                        net.reset_network()
                    else:
                        trained = True
                        # Plotting the first line
                        plt.plot(inputs, labels, label='labels')

                        # Plotting the second line
                        plt.plot(inputs, outputs, label='predictions')

                        # Adding labels and title
                        plt.xlabel('X-axis')
                        plt.ylabel('Y-axis')
                        plt.title('NeuralNetwork line fitting')

                        # Adding a legend
                        plt.legend()

                        # Displaying the graph
                        plt.show()

        
        
        
    



    


