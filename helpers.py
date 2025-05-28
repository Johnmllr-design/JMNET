# an auxillary helpers class with short, mathematical functions
from math import log
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

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


def plot(inputs, labels, predictions):
    if len(inputs[0]) == 3:
       
        inputs = np.array(inputs)
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        # Extract coordinates
        x = inputs[:, 0]  # bedtime
        y = inputs[:, 1]  # wakeup
        z = inputs[:, 2]  # activity level
        
        # Compute absolute error
        error = np.abs(predictions - labels)

        # Setup figure with 3 subplots
        fig = plt.figure(figsize=(18, 6))

        # Prediction plot
        ax1 = fig.add_subplot(131, projection='3d')
        p1 = ax1.scatter(x, y, z, c=predictions, cmap='viridis', s=60)
        ax1.set_title("Predicted Sleep Optimality")
        ax1.set_xlabel("Bedtime")
        ax1.set_ylabel("Wakeup")
        ax1.set_zlabel("Activity Level")
        cbar1 = plt.colorbar(p1, ax=ax1)
        cbar1.set_label("Prediction")

        # Label plot
        ax2 = fig.add_subplot(132, projection='3d')
        p2 = ax2.scatter(x, y, z, c=labels, cmap='plasma', s=60)
        ax2.set_title("Actual Sleep Optimality (Label)")
        ax2.set_xlabel("Bedtime")
        ax2.set_ylabel("Wakeup")
        ax2.set_zlabel("Activity Level")
        cbar2 = plt.colorbar(p2, ax=ax2)
        cbar2.set_label("Label")

        # Error plot
        ax3 = fig.add_subplot(133, projection='3d')
        p3 = ax3.scatter(x, y, z, c=error, cmap='coolwarm', s=60)
        ax3.set_title("Prediction Error (|Label - Prediction|)")
        ax3.set_xlabel("Bedtime")
        ax3.set_ylabel("Wakeup")
        ax3.set_zlabel("Activity Level")
        cbar3 = plt.colorbar(p3, ax=ax3)
        cbar3.set_label("Absolute Error")

        plt.tight_layout()
        plt.show()


    elif len(inputs[0]) == 2:
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
        ax.scatter(input1, input2, output, c="blue", label="Prediction", s=50)
        ax.scatter(input1, input2, labels, c="red", label="Label", s=50)

        # Adding labels to the axes
        ax.set_xlabel("steps")
        ax.set_ylabel("activity level")
        ax.set(zlabel= "sleep optimality")

        # Adding a legend
        ax.legend()

        # Show the plot
        plt.show()

    else:
        print("the loss is " + str(mean_squared_error(labels, predictions)))

        # Plotting the first line
        plt.plot(inputs, labels, label="labels")

        # Plotting the second line
        plt.plot(inputs, predictions, label="predictions")

        # Adding labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("NeuralNetwork line fitting")

        # Adding a legend
        plt.legend()

        # Displaying the graph
        plt.show()


# plot the losses of the program, epoch by epoch
def plot_losses(losses):
    plt.plot(losses) 

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()   

