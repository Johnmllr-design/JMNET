# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes the declaration, forward pass, backpropagation, and training loop.
# Date: [1/15/2025]

from neurons import Node

class neuralNetwork():

    def buildNetwork(self, num_attributes, num_layers, layer_size):
        input_Layer = True
        neural_Network = []
        for i in range(0, num_layers):
            current_Layer = []
            if input_Layer:
                for i in range(0, layer_size):
                    newNode = Node(num_attributes)
                    current_Layer.append(newNode)
                input_Layer = False
            else:
                for i in range(0, layer_size):
                    newNode = Node(layer_size)
                    current_Layer.append(newNode)
            neural_Network.append(current_Layer)

        return neural_Network
        
obj = neuralNetwork()
network = obj.buildNetwork(num_attributes = 3, num_layers = 2, layer_size = 3)
for layer in network:
    for node in layer:
        print("this is node " + str(node) + " with weights " + str(node.weights))

