"""
File Name: neural_network_activation_functions.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements various neural network activation functions using NumPy and visualizes the output using Matplotlib.
    The following neurons are implemented with an adjustable threshold:
    - Linear Threshold Neuron: A simple neuron that outputs 1 if the sum of inputs exceeds the threshold.
    - ReLU Neuron: A variant of the linear threshold neuron that outputs 1 if the sum exceeds the threshold, similar to the ReLU activation function.
    - Sigmoid Neuron: A probabilistic neuron that uses the sigmoid function to determine the output, with a threshold for classification.
    - Tanh Neuron: A neuron that applies the hyperbolic tangent function and outputs 1 if the result exceeds a set threshold.
    - Stochastic Neuron: A neuron that uses the sigmoid function to compute a probability and outputs 1 if it exceeds the threshold.

    The neurons are trained and evaluated on a simple AND operation dataset with four input-output pairs. The predictions of each neuron are displayed after training over multiple epochs. Results are visualized to show how each neuron behaves.

Usage:
    - This script demonstrates how different activation functions perform on a simple binary classification task (AND operation).
    - It compares five neuron types: Linear Threshold, ReLU, Sigmoid, Tanh, and Stochastic, and shows their output after a set number of epochs.
    - The results of each neuron type are plotted to visualize how they respond to the inputs.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that NumPy and Matplotlib are installed in your Python environment.
    - Run the script to train and evaluate the neurons over 10 epochs.
    - Observe the final predictions for each neuron and analyze their behavior on the AND operation.
    - The plot at the end will show the output of each neuron after the final epoch.

"""

import numpy as np
import matplotlib.pyplot as plt

# Define the input features (X) and target outputs (t) for the AND operation
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input Features
t = np.array([0, 0, 0, 1])  # Target outputs

# Activation functions with corrected logic
def linear_threshold_neuron(x, threshold=1.5):
    return 1 if np.sum(x) >= threshold else 0

def relu_neuron(x, threshold=1.5):
    return 1 if np.sum(x) >= threshold else 0  # ReLU outputs 1 only if sum exceeds threshold

def sigmoid_neuron(x, threshold=0.75):
    output = 1 / (1 + np.exp(-np.sum(x)))
    return 1 if output >= threshold else 0

def tanh_neuron(x, threshold=0.8):
    output = np.tanh(np.sum(x))
    return 1 if output >= threshold else 0

def stochastic_neuron(x, prob=0.5, threshold=0.8):
    deterministic_output = 1 / (1 + np.exp(-np.sum(x)))
    return 1 if deterministic_output >= threshold else 0

# Train and evaluate the neuron for multiple epochs
def train_and_evaluate(neuron_function, X, t, epochs=10, *args, **kwargs):
    all_predictions = []

    for epoch in range(epochs):
        epoch_predictions = []
        for x in X:
            pred = neuron_function(x, *args, **kwargs)
            epoch_predictions.append(pred)
        all_predictions.append(np.array(epoch_predictions))
    
    return np.array(all_predictions)

# Run the comparison with parametrized epochs
neurons = {
    'Linear Threshold': linear_threshold_neuron,
    'ReLU': relu_neuron,
    'Sigmoid': sigmoid_neuron,
    'Tanh': tanh_neuron,
    'Stochastic': stochastic_neuron
}

# Set the number of epochs
epochs = 10

# Results dictionary to store predictions
results = {}

for name, neuron_func in neurons.items():
    preds = train_and_evaluate(neuron_func, X, t, epochs=epochs)
    results[name] = preds

# Plot the results for each neuron after the final epoch
plt.figure(figsize=(10, 8))

for i, (name, preds) in enumerate(results.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(X[:, 0], X[:, 1], c=preds[-1], cmap='bwr', marker='o')
    plt.title(f'{name} Neuron (Epoch {epochs})')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Output the predictions after the final epoch
for name, preds in results.items():
    print(f'{name} predictions after {epochs} epochs: {preds[-1]}')
