"""
File Name: relu_perceptron_training.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a perceptron with ReLU activation function for binary classification tasks. The perceptron model 
    learns to classify inputs by adjusting its weights and bias using the gradient descent algorithm. The ReLU activation 
    function is used to produce an output of 0 if the weighted sum of the inputs is negative, and a positive value if the 
    weighted sum is positive.

    Key components:
    - ReLU function: A non-linear activation function that outputs the input if positive, and zero otherwise.
    - Training process: The perceptron updates its weights and bias to minimize classification error.
    - Binary classification task: The script uses a dataset of binary input-output pairs for training.

Usage:
    - This script demonstrates the application of a ReLU-based perceptron for binary classification. The perceptron is 
      trained on a simple dataset over a specified number of epochs.
    - The updated weights and bias are printed after each training iteration.
    - The final weights and bias after training are displayed at the end of the process.

Dependencies:
    - numpy

Instructions:
    - Ensure that NumPy is installed in your Python environment.
    - Run the script to train the perceptron on the dataset for the specified number of epochs.
    - The script will print the updated weights and bias after each iteration.
    - At the end of training, the final weights and bias values will be displayed.

"""

# Initialize environment for ReLU-based Perceptron Training

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
w_relu = np.array([0.0,0.0])  # Initial weights
b_relu = 1.0  # Initial bias
learning_rate = 1  # Learning rate
epochs =10  # Maximum number of epochs

# ReLU-based Training
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_relu, X[i]) + b_relu
        
        # Apply ReLU activation
        y = max(0, z)
        
        # Compute error
        error = t[i] - y
        
        # Update weights and bias
        if error != 0:
            w_relu += learning_rate * error * X[i]
            b_relu += learning_rate * error
            print('w=',w_relu,'b=',b_relu)
# Final weights and bias after training with ReLU
w_relu, b_relu
print("Final weights and bias after training with ReLU")
print('w=',w_relu,'b=',b_relu)
