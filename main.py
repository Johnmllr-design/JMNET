# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes demonstation code
# Date: [1/15/2025]

from TrainNetwork import TrainNetwork
from NeuralNetwork import NeuralNetwork
import numpy as np

    
print("this is the driver Python file for a John's neural network implementation.\n")
print("below is a sample demonstration of how the algorithm works to fit a line to a cosin curve. \n This is the only file that utilizes an external library, which is numpy for dataset generation\n")

# declaration of a neural network with 1hidden layers and 3 nodes in each layer
num_inputs = 1
num_layers = 3
num_nodes_per_layer = 10
activation_function = "sigmoid"


net = NeuralNetwork(num_inputs, num_layers, num_nodes_per_layer, activation_function)
trainer = TrainNetwork()


# Generate input values (e.g., from 0 to 2*pi)
inputs = np.linspace(0, 10 * np.pi, 10000)  # 1000 points between 0 and 20*pi

# Compute cosine values
cosine_values = np.cos(inputs)

# Scale cosine values to be between 0 and 1
scaled_cosine_values = (cosine_values + 1) / 2

# Create the dataset in the form [[input], output]
dataset = [[[x], y] for x, y in zip(inputs, scaled_cosine_values)]
epochs = 20
# train the network "net" with the dataset
trainer.train(net, dataset, epochs)

