# Reinitialize environment for ReLU-based Perceptron Training

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