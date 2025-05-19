from typing import List
from neural_network import NeuralNetwork
from helpers import ReLU_derivative


# class sigmoid_optimizer to implement the backprop for sigmoid
class sigmoid_optimizer:
    def __init__(self):
        pass

    # backpropagate takes the network object, and the activations from each layer, and then uses these to calculate the
    # partial derivatives with respect to each weight. It has a set learning rate of 0.3
    def backpropagate(self, networkObject: NeuralNetwork, activations: List[List[float]], true_Label: float):
        
        previous_deltas = []
        learning_rate = 0.3

        for layer_index in reversed(range(1, len(networkObject.network))):
            # first step, we must establish the deltas of the output layer to begin the back-propagation process,
            # in which the delta can be used in conjugation with the input to the current node to then determine the
            # gradient to be added to the new weight

            predicted_value = activations[-1][0]  # the output of the last neuron

            if layer_index == len(networkObject.network) - 1:
                # the delta of the output node for the sigmoid function with respect to the output
                # node is 2 * label - prediction * (prediction * (1 - prediction))
                output_node_delta = (2 * (true_Label - predicted_value) * predicted_value * (1 - predicted_value))

                for weight_index in range(0, len(networkObject.network[layer_index][0].weights)):
                    # for each weight in the output node, use the delta to adjust the weight with the formula:
                    # new weight = old weight + (delta * learning rate * input from the previous layer to the current node)
                    previous_weight = networkObject.network[layer_index][0].weights[weight_index]
                    new_weight = previous_weight + (output_node_delta * learning_rate * activations[layer_index - 1][weight_index])
                    networkObject.network[layer_index][0].weights[weight_index] = new_weight

                # update the bias
                networkObject.network[layer_index][0].bias = networkObject.network[layer_index][0].bias + (output_node_delta * learning_rate)

                previous_deltas.append(output_node_delta)

            else:
                # if we are at a node in a hidden layer, we must now use a different formula to apply the chain rule to the
                # network to determine the gradients with respect to the weights on the neurons in the current hidden layer
                new_deltas = []

                for node_index in range(0, len(networkObject.network[layer_index])):
                    # firstly, use a for loop to determine the error signal, that is, sum the weight of each
                    # node in the next layer that connects it to the current node times it's previous delta.
                    error_signal = 0.0
                    for i in range(0, len(networkObject.network[layer_index + 1])):
                        error_signal += (networkObject.network[layer_index + 1][i].weights[node_index]* previous_deltas[i])
                        

                    # use the error signal to determine the current delta
                    current_delta = (activations[layer_index][node_index]* (1 - activations[layer_index][node_index])) * error_signal

                    # for each weight in the current node, use the current delta to update the weights of it's respective node
                    for weight_index in range(0, len(networkObject.network[layer_index][node_index].weights)):
                        old_weight = networkObject.network[layer_index][node_index].weights[weight_index]
                        new_weight = old_weight + (learning_rate * current_delta * activations[layer_index - 1][weight_index])
                
                        networkObject.network[layer_index][node_index].weights[weight_index] = new_weight

                    # update the bias
                    networkObject.network[layer_index][node_index].bias = networkObject.network[layer_index][node_index].bias + (current_delta * learning_rate)

                    new_deltas.append(current_delta)

                previous_deltas = new_deltas
