"""
File Name: tanh_neuron_model.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a Tanh neuron model, designed for binary classification tasks. The Tanh (Hyperbolic Tangent) 
    activation function is used to compute the output of the neuron. The model learns to classify the inputs into two classes 
    by adjusting its weights and bias using gradient descent. The script trains the neuron on a small dataset and adjusts 
    the weights and bias to minimize the classification error.

    Key components:
    - Tanh function: A function that transforms the output of the weighted sum of inputs into a value between -1 and 1.
    - Training process: The neuron updates its weights and bias to minimize the error between predicted values and true class labels.
    - Binary classification task: The script uses a dataset of inputs (features) and corresponding binary target labels.

Usage:
    - This script demonstrates the application of the Tanh neuron for binary classification. It trains the neuron over a 
      specified number of epochs.
    - The neuron adjusts its weights and bias to minimize the classification error, using the Tanh function to compute the 
      output.
    - The updated weights and bias are printed at each epoch to show the progress of the training process.
    - At the end of training, the final weights and bias values are displayed.

Dependencies:
    - numpy

Instructions:
    - Ensure that NumPy is installed in your Python environment.
    - Run the script to train the Tanh neuron over the specified number of epochs.
    - The script will print the updated weights and bias after each training iteration.
    - At the end of training, the final weights and bias values are shown to summarize the training process.

"""

import numpy as np

# Hyperbolic Tangent (Tanh) function
def tanh(z):
    return np.tanh(z)

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 0, 0, 1])  # Target values

# Initial parameters
w_tanh = np.array([0.0, 0.0])  # Initial weights
b_tanh = 0.0  # Initial bias
learning_rate = 0.1  # Learning rate
epochs = 10  # Maximum number of epochs

# Training the Tanh Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_tanh, X[i]) + b_tanh

        # Apply Tanh activation
        y = tanh(z)

        # Compute error
        error = t[i] - y

        # Compute gradient and update weights and bias
        gradient = error * (1 - y**2)  # Derivative of tanh(z) is 1 - tanh^2(z)
        w_tanh += learning_rate * gradient * X[i]
        b_tanh += learning_rate * gradient
        print('w=',w_tanh,'b=',b_tanh)

# Final weights and bias after training with Tanh
print("Final weights:", w_tanh)
print("Final bias:", b_tanh)
