from typing import List
from NeuralNetwork import NeuralNetwork
from helpers import ReLU_derivative


class ReLU_optimizer:
    def __init__(self):
        pass

    def backpropagate(self, networkObject: NeuralNetwork, activations: List[List[float]], true_Label: float):

        previous_deltas = []
        learning_rate = 0.01

        for layer_index in reversed(range(1, len(networkObject.network))):
            
            #first step, we must establish the deltas of the output layer

            if layer_index == len(networkObject.network) - 1:
                
                predicted_value = activations[layer_index][0]#the output of the last neuron

                output_node_delta =  (true_Label - predicted_value) * ReLU_derivative(networkObject.network[layer_index][0].z)

                for weight_index in range(0, len(networkObject.network[layer_index][0].weights)):
                   
                    previous_weight = networkObject.network[layer_index][0].weights[weight_index]
                    new_weight = previous_weight + (output_node_delta * learning_rate * activations[layer_index - 1][weight_index])
                    networkObject.network[layer_index][0].weights[weight_index] = new_weight
                
                #update the bias
                networkObject.network[layer_index][0].bias = networkObject.network[layer_index][0].bias +  (output_node_delta * learning_rate)
                
                previous_deltas.append(output_node_delta)

            else:
                new_deltas = []

                for node_index in range(0, len(networkObject.network[layer_index])):

                    error_signal = 0.0
                    for i in range(0, len(networkObject.network[layer_index + 1])):

                        error_signal += (networkObject.network[layer_index + 1][i].weights[node_index] * previous_deltas[i])
                    
                    current_delta = ReLU_derivative(networkObject.network[layer_index][node_index].z) * error_signal
                    
                    for weight_index in range(0, len(networkObject.network[layer_index][node_index].weights)):

                        old_weight = networkObject.network[layer_index][node_index].weights[weight_index]
                        new_weight = old_weight + (learning_rate * current_delta * activations[layer_index - 1][weight_index])
                        networkObject.network[layer_index][node_index].weights[weight_index] = new_weight
                    
                     #update the bias
                    networkObject.network[layer_index][node_index].bias = networkObject.network[layer_index][node_index].bias +  (current_delta * learning_rate)

                    new_deltas.append(current_delta)
                
                previous_deltas = new_deltas
           
        

            
    


