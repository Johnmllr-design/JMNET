from NeuralNetwork import NeuralNetwork
from typing import List



class gradientDescent():


    def gradient_descent(self, predicted, observed, model: NeuralNetwork, layer_outputs: List[List]):
        for i in range(0, len(model.network)):
            for j in range(0, len(model.network[i])):
                self.compute_weight_gradients(model, i, j, observed, predicted, layer_outputs)

    def compute_weight_gradients(self, model:NeuralNetwork, layer: int, layerIndex: int, observed: List[float], predicted: List[float], layerOutputs: List[float]):
        for weightIndex in range(0, len(model.network[layer][layerIndex].weights)):
            print("looking at the " + str(layerIndex) + " node in the " + str(layer) + "eth layer to optimize it's " + str(weightIndex) +"eth weight")
            cur_sum = 0.0
            oldWeight = model.network[layer][layerIndex].weights[weightIndex]
            print("it's previous weight is " + str(oldWeight))
            for i in range(0, len(predicted)):
                print("at observation " + str(i) + " we are finding the difference of " + str(observed[i]) + " and " + str(predicted[i]))
                cur_sum += -2 * (observed[i] - predicted[i]) * layerOutputs[i][layer][weightIndex]
            step_size = cur_sum * 0.1 # multiply the current sum with the learning rate to get the step size
            model.network[layer][layerIndex].weights[weightIndex] = oldWeight - step_size # subtract the step size to get the new weight

        
        

    
# Approach: take the gradient of the loss function with respect
# to each of the weights in each node

# dSSR with respect to the predicted = -2 (observed[i] - predicted[i])
# dPredicted with respect to Weight w = output of node connected to current node via edge correcponding to weight w
# so, dSSR with respect to weight w = -2 (observed[i] - predicted[i]) * output of node connected to current node via edge correcponding to weight 

