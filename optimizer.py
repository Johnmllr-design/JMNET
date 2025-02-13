from typing import List
from NeuralNetwork import NeuralNetwork
from helpers import softplus_derivative
from TrainNetwork import TrainNetwork


class optimizer:
    def __init__(self):
        pass

    def backpropagate(self, networkObject: NeuralNetwork, activations: List[List[int]], trueLabel: float):

        previous_Deltas = []
        learning_rate = 0.5

        for index in reversed(range(0, len(networkObject.network))):

            if index == len(networkObject.network) - 1:
                
                networkPrediction = activations[-1][0]
                print("the predictio  is "  + str(networkPrediction) + " and true label is " + str(trueLabel))
                delta = (trueLabel - networkPrediction) * networkPrediction * (1 - networkPrediction)
                print("the new delta is " + str(delta))
                print("now using the delta to adjust node at " + str(index))
                for i in range(0, len(networkObject.network[index][0].weights)):
                    pass
                #     curWeight = networkObject.network[index][0].weights[i]
                    
                #     print("previous activation for weight " + str(i) + " in node " + str([index, 0]) + " is " + str(activations[index- 1][i]))
                #     print("the new weight is curWeight "  +str(curWeight) + " plus " + str(learning_rate) + " times " + str(delta) + " times "  + str(activations[index - 1][i]))
                #     print("i is " + str(i) + " and index - 1 is " + str(index - 1))
                #     newWeight = curWeight + (learning_rate * delta * activations[index - 1][i])
                #     networkObject.network[index][0].weights[i] = newWeight

                # networkObject.network[index][0].bias = networkObject.network[index][0].bias + (delta * learning_rate)
                # previous_Deltas = [delta]
            
            else:
                break
                print()
                new_deltas = []
                print("the deltas of layer " + str(index + 1) + " are " + str(previous_Deltas))
                for node in range(0, len(networkObject.network[index])):
                    
                    current_node_delta = 0.0
                    current_node_activation = activations[index][node]
                    
                    for previous_Delta_Index in range(0, len(previous_Deltas)):
                        
                        print("adding the connection to previous delta " + str(previous_Deltas[previous_Delta_Index]) + " times weight " + str(networkObject.network[index + 1][previous_Delta_Index].weights[node]) + " at node " + str([index + 1, previous_Delta_Index]))
                        
                        current_node_delta += previous_Deltas[previous_Delta_Index] * networkObject.network[index + 1][previous_Delta_Index].weights[node] * (current_node_activation * (1 - current_node_activation))
                    
                    for weight in range(0, len(networkObject.network[index][node].weights)):
                        old_weight = networkObject.network[index][node][weight]
                        new_weight = old_weight  + (learning_rate * activations[index - 1][weight] * current_node_delta)
                        networkObject.network[index][node].weights[weight] = new_weight
                    
                    new_deltas.append(current_node_delta)
                previous_Deltas = new_deltas




            
    

nnTest = NeuralNetwork(1, 1, 2)
op = optimizer()
nnTest.printNetwork()
Trainer = TrainNetwork()
testData = [[0.5], 1]
activations = Trainer.forward_pass(nnTest, [0.5])
print("activations are " + str(activations))
print()
op.backpropagate(nnTest, activations, 1)

