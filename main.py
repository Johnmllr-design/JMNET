# main.py
# Neural Network Implementation
# Author: [John Miller]
# Description: This is the main file for a self-implemented neural network from scratch.
# It includes demonstation code
# Date: [1/15/2025]

from TrainNetwork import TrainNetwork
from NeuralNetwork import NeuralNetwork

    
print("this is the driver Python file for a John's neural network implementation.\n")
print("below is a sample demonstration of how the algorithm works to fit a line to a Sin curve")

# declaration of a neural network with 2 hidden layers and 2 nodes in each layer
net = NeuralNetwork(1, 1, 4)
trainer = TrainNetwork()
training_data =[
    [[0.0], 0.0],
    [[0.12822827], 0.12787716],
    [[0.25645654], 0.25365458],
    [[0.38468481], 0.375267],
    [[0.51291309], 0.49071755],
    [[0.64114136], 0.59811053],
    [[0.76936963], 0.69568255],
    [[0.8975979], 0.78183148],
    [[1.02582617], 0.85514276],
    [[1.15405444], 0.91441262],
    [[1.28228272], 0.95866785],
    [[1.41051099], 0.98718178],
    [[1.53873926], 0.99948622],
    [[1.66696753], 0.99537911],
    [[1.7951958], 0.97492759],
    [[1.92342407], 0.93846887],
    [[2.05165235], 0.8865993],
    [[2.17988062], 0.82017239],
    [[2.30810889], 0.74027763],
    [[2.43633716], 0.64822835],
    [[2.56456543], 0.5455349],
    [[2.6927937], 0.43388374],
    [[2.82102197], 0.31510856],
    [[2.94925025], 0.19115863],
    [[3.07747852], 0.06407022]
]

trainer.train(net, training_data)

