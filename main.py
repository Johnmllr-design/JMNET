# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes demonstation code
# Date: [1/15/2025]

from train_network import TrainNetwork
from neural_network import NeuralNetwork
from handle_data import sleep_data_three_observations
from helpers import plot

print("this is the driver Python file for John's neural network implementation.\n")
print("below is a sample demonstration of how the algorithm works to fit a line to a cosin curve. \n This is the only file that utilizes an external library, which is numpy for dataset generation\n")

# declaration of a neural network with 1hidden layers and 3 nodes in each layer
num_inputs = 3
num_layers = 2
num_nodes_per_layer = 10
activation_function = "sigmoid"
trainer = TrainNetwork() # declaration of a trainer for the network
network = NeuralNetwork(num_inputs, num_layers, num_nodes_per_layer, activation_function)
dataset =  sleep_data_three_observations()
epochs = 200


# train the network "net" with the dataset
metadata = trainer.train(network, dataset, epochs)

 # plot the metadata (inputs, outputs, labels) from model training
plot(metadata[0], metadata[1], metadata[2])                







