# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes demonstation code
# Date: [1/15/2025]

from TrainNetwork import TrainNetwork
from NeuralNetwork import NeuralNetwork
from datasets import renewable_energy_dataset
from datasets import cosin_curve
from datasets import d3_cosin_curve
from datasets import pseudo_random
from datasets import yoe_and_happ

    
print("this is the driver Python file for a John's neural network implementation.\n")
print("below is a sample demonstration of how the algorithm works to fit a line to a cosin curve. \n This is the only file that utilizes an external library, which is numpy for dataset generation\n")

# declaration of a neural network with 1hidden layers and 3 nodes in each layer
num_inputs = 2
num_layers = 2
num_nodes_per_layer = 15
activation_function = "sigmoid"
trainer = TrainNetwork() # declaration of a trainer for the network
network = NeuralNetwork(num_inputs, num_layers, num_nodes_per_layer, activation_function)


dataset = yoe_and_happ()

epochs = 250
# train the network "net" with the dataset
trainer.train(network, dataset, epochs)

