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
