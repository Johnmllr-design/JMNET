import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt

def display_graph(predicted, observed, label1="predicted", label2="observed", title="predicting data using a Neural Network", xlabel="X-axis", ylabel="Y-axis"):
    """
    Displays a graph of two arrays using matplotlib.

    Args:
        array1: The first array of numerical data.
        array2: The second array of numerical data.
        label1: Label for the first array (default: "Array 1").
        label2: Label for the second array (default: "Array 2").
        title: Title of the graph (default: "Graph of Two Arrays").
        xlabel: Label for the x-axis (default: "X-axis").
        ylabel: Label for the y-axis (default: "Y-axis").
    """
    plt.plot(predicted, label=label1)
    plt.plot(observed, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
