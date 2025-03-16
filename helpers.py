# an auxillary helpers class with short, mathematical functions
from math import log
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# linear algebraic dot product of two arrays
def dot(a1, a2) -> float:
    if len(a1) != len(a2):
        return -1
    else:
        sum = 0.0
        for i in range(0, len(a1)):
            sum += a1[i] * a2[i]
        return sum


# softplus function of a given input
def softplus(input: float) -> float:
    return log(1 + exp(input))


# softplus derivative function of a given input
def softplus_derivative(input: float):
    return 1 / 1 + exp(input)


# MSE of two arrays
def mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have the same length")

    total_error = 0
    for i in range(len(actual)):
        total_error += (actual[i] - predicted[i]) ** 2

    return total_error / len(actual)


def SSR(a1, a2) -> float:
    sum = 0.0
    for i in range(0, len(a1)):
        sum += (a1[i] - a2[i]) * (a1[i] - a2[i])
    return sum / len(a1)


def MSE(a1: float, a2: float) -> float:
    return 0.5 * (a1 * a2) * (a1 * a2)


def ReLU(input: float) -> float:
    return 0 if input < 0.0 else input


def ReLU_derivative(input: float) -> int:
    if input > 0:
        return 1
    return 0


def sigmoid(input: float) -> float:
    return 1 / (1 + exp(input * -1))


def sigmid_derivative(input: float):
    return sigmoid(input) * (1 - sigmoid(input))


def plot(inputs, labels, output):

    if len(inputs[0]) == 2:
        # Extracting columns from the data
        input1 = [arr[0] for arr in inputs]
        input2 = [arr[1] for arr in inputs]

        # Convert lists to numpy arrays for better handling
        input1 = np.array(input1)
        input2 = np.array(input2)
        output = np.array(output)
        labels = np.array(labels)

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plotting both output and label on the z-axis
        ax.scatter(input1, input2, output, c="blue", label="Output", s=50)
        ax.scatter(input1, input2, labels, c="red", label="Label", s=50)

        # Adding labels to the axes
        ax.set_xlabel("Temperature")
        ax.set_ylabel("humidity")
        ax.set(zlabel="probability of rainfall")

        # Adding a legend
        ax.legend()

        # Show the plot
        plt.show()

    else:
        print("the loss is " + str(mean_squared_error(labels, output)))

        # Plotting the first line
        plt.plot(inputs, labels, label="labels")

        # Plotting the second line
        plt.plot(inputs, output, label="predictions")

        # Adding labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("NeuralNetwork line fitting")

        # Adding a legend
        plt.legend()

        # Displaying the graph
        plt.show()
