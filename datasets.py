import numpy as np
from typing import List


def renewable_energy_dataset() -> List:# Set a random seed for reproducibility
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    num_samples = 10

    # Input1: Size of the house (500 to 3000 square feet)
    input1 = np.random.uniform(500, 3000, num_samples)

        # Input2: Distance from the city center (0 to 20 miles)
    input2 = np.random.uniform(0, 20, num_samples)

    # Scale inputs to be between 0 and 1
    input1_scaled = (input1 - 500) / (3000 - 500)  # Scale size to [0, 1]
    input2_scaled = input2 / 20                    # Scale distance to [0, 1]

    # Define the plane equation: z = a*x + b*y + c
    # Let's choose coefficients for the plane
    a = 0.5  # Coefficient for size
    b = -0.5 # Coefficient for distance
    c = 0.5  # Constant term

    # Output: House price (scaled between 0 and 1)
    output = a * input1_scaled + b * input2_scaled + c

    # Normalize the output to ensure it lies between 0 and 1
    output = (output - np.min(output)) / (np.max(output) - np.min(output))

    # Combine into the desired format
    dataset = [[[input1_scaled[i], input2_scaled[i]], output[i]] for i in range(num_samples)]
    return dataset

def cosin_curve():
    # Generate input values (e.g., from 0 to 2*pi)
    inputs = np.linspace(0, 10 * np.pi, 10000)  # 1000 points between 0 and 20*pi

    # Compute cosine values
    cosine_values = np.cos(inputs)

    # Scale cosine values to be between 0 and 1
    scaled_cosine_values = (cosine_values + 1) / 2

    # Create the dataset in the form [[input], output]
    data =  [[[x], y] for x, y in zip(inputs, scaled_cosine_values)]
    return data

def d3_cosin_curve():
    # Generate input values (e.g., from 0 to 2*pi)
    inputs = np.linspace(0, 10 * np.pi, 10000)  # 1000 points between 0 and 20*pi

    # Compute cosine values
    cosine_values = np.cos(inputs)

    # Scale cosine values to be between 0 and 1
    scaled_cosine_values = (cosine_values + 1) / 2

    # Create the dataset in the form [[input], output]
    data =  [[[x, x], y] for x, y in zip(inputs, scaled_cosine_values)]
    return data

def pseudo_random():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    num_samples = 50

    # Input1: Random values between 0 and 1
    input1 = np.random.uniform(0, 1, num_samples)
    input1.sort()
    # Output: Random values between 0 and 1
    output = np.random.uniform(0, 1, num_samples)

    # Combine into the desired format
    dataset = [[[input1[i]], output[i]] for i in range(0, len(input1))]
    return dataset

def yoe_and_happ():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    num_samples = 50

    # Input1: Years of experience (scaled between 0 and 1)
    input1 = np.random.uniform(0, 1, num_samples)

    # Input2: Positivity score (scaled between 0 and 1)
    input2 = np.random.uniform(0, 1, num_samples)

    # Output: Job performance score (positively correlated with both inputs)
    #  Let's assume the output is a weighted sum of the inputs, with some noise
    output = 0.6 * input1 + 0.4 * input2  # Weighted sum
    output += np.random.normal(0, 0.05, num_samples)  # Add some noise
    output = np.clip(output, 0, 1)  # Ensure output is between 0 and 1
    # Combine into the desired format
    dataset = [[[input1[i], input2[i]], output[i]] for i in range(num_samples)]
    return dataset
