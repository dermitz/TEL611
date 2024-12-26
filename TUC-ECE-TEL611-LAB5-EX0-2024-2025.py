"""
File Name: perceptron_ltn.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements the Perceptron Learning Algorithm for a Linear Threshold Neuron (LTN). The perceptron model is 
    a binary classifier that updates its weights and bias based on classification errors during training. The Linear Threshold 
    Neuron (LTN) uses a threshold function to decide whether the output should be 1 or 0, based on the weighted sum of the 
    inputs compared to a threshold value.

    Key components:
    - Linear Threshold Neuron (LTN): A simple neuron model that outputs 1 if the weighted sum of inputs exceeds the 
      threshold, otherwise 0.
    - Perceptron Learning Algorithm: The weights and bias are updated iteratively based on the error between the predicted 
      output and the target output.
    - Binary classification task: The script trains the perceptron on a dataset of binary input-output pairs.

Usage:
    - This script demonstrates the Perceptron Learning Algorithm on a simple binary classification task. The model learns 
      to classify the input data over a series of epochs by adjusting its weights and bias.
    - The updated weights and bias are printed after each iteration.
    - At the end of training, the final weights and bias are displayed.

Dependencies:
    - numpy

Instructions:
    - Ensure that NumPy is installed in your Python environment.
    - Run the script to train the perceptron on the dataset for the specified number of epochs.
    - The script will print the updated weights and bias after each training iteration.
    - At the end of training, the final weights and bias values will be displayed.

"""

# Implementation of Perceptron Learning Algorithm for a Linear Threshold Neuron (LTN)

import numpy as np

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 0, 0, 1])  # Target values

# Initial parameters
w = np.array([0.0, 0.0])  # Initial weights
b = 0.0  # Initial bias
theta = 0.0  # Threshold
learning_rate = 1  # Learning rate
epochs = 10  # Maximum number of epochs

# Perceptron Learning Algorithm
for epoch in range(epochs):
    for i in range(len(X)):
        # Calculate the output
        z = np.dot(w, X[i]) + b
        y = 1 if z >= theta else 0
        
        # Update weights and bias if there is an error
        error = t[i] - y
        if error != 0:
            w += learning_rate * error * X[i]
            b += learning_rate * error
            print('w=',w,'b=',b)

# Final weights and bias
print('Final : w=', w,'and b=', b)
