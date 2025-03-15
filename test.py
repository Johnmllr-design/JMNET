import matplotlib.pyplot as plt

# Example dataset: Each entry is [[input1, input2], output]
dataset = [
    [[0.2, 0.5], 0.3],
    [[0.8, 0.1], 0.6],
    [[0.4, 0.7], 0.5],
    [[0.9, 0.3], 0.7],
    [[0.1, 0.9], 0.2]
]

# Extract input pairs and output values without NumPy
inputs = [pair[0] for pair in dataset]  # Extracts [[input1, input2], ...]
outputs = [pair[1] for pair in dataset]  # Extracts [output1, output2, ...]

# Separate inputs into two lists: input1 and input2
input1 = [x[0] for x in inputs]
input2 = [x[1] for x in inputs]

# Print arrays
print("Input Pairs (Input1, Input2):")
print(inputs)
print("\nOutputs:")
print(outputs)

# Scatter plot of input pairs, color-coded by output values
scatter = plt.scatter(input1, input2, c=outputs, cmap='viridis', edgecolors='black')

# Labels and title
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("Dataset Visualization (Input1 vs Input2, Colored by Output)")
plt.colorbar(scatter, label="Output Value")  # Show output values as colors

# Show plot
plt.show()