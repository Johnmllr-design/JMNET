from NeuralNetwork import NeuralNetwork
from typing import List
from utils import Node
from helpers import SSR
import numpy as np
import matplotlib.pyplot as plt
from PredictionAnalysis import display_graph
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

        for i in range(0, epochs):

            outputs = []
            for input, label in trainingData:
           
                activations = self.forward_pass(net, input)

                outputs.append(activations[-1][0])

                optimizer.backpropagate(net, activations, label)
        
            import matplotlib.pyplot as plt

            if i == 999:
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
        
        
        
    



    


