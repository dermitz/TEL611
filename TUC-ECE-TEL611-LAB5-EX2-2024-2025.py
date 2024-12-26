"""
File Name: sigmoid_neuron_model.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a Sigmoid neuron model for binary classification tasks. The Sigmoid activation function is used 
    to compute the output of the neuron. The model learns to classify inputs into two classes by adjusting its weights and 
    bias using gradient descent. The script trains the neuron on a simple dataset and adjusts the weights and bias to minimize 
    classification error.

    Key components:
    - Sigmoid function: A function that transforms the output of the weighted sum of inputs into a value between 0 and 1.
    - Training process: The neuron updates its weights and bias to minimize the error between predicted values and true class labels.
    - Binary classification task: The script uses a dataset with binary input-output pairs for training.

Usage:
    - This script demonstrates the application of a Sigmoid neuron for binary classification. The neuron is trained over a 
      specified number of epochs, and its weights and bias are updated based on the classification error.
    - The updated weights and bias are printed at each epoch to show the progress of training.
    - At the end of training, the final weights and bias values are displayed.

Dependencies:
    - numpy

Instructions:
    - Ensure that NumPy is installed in your Python environment.
    - Run the script to train the Sigmoid neuron on the dataset for the specified number of epochs.
    - The script will print the updated weights and bias after each training iteration.
    - The final weights and bias after training will be displayed at the end.

"""

# Sigmoid Neuron implementation
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 0, 0, 1])  # Target values

# Initial parameters
w_sigmoid = np.array([0.0, 0.0])  # Initial weights
b_sigmoid = 0.0  # Initial bias
learning_rate = 0.1  # Smaller learning rate for smooth convergence
epochs = 5  # Maximum number of epochs

# Training the Sigmoid Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_sigmoid, X[i]) + b_sigmoid
        
        # Apply Sigmoid activation
        y = sigmoid(z)
        
        # Compute error
        error = t[i] - y
        
        # Compute gradient and update weights and bias
        gradient = error * y * (1 - y)
        w_sigmoid += learning_rate * gradient * X[i]
        b_sigmoid += learning_rate * gradient
        print('w=',w_sigmoid,'b=',b_sigmoid)

# Final weights and bias after training with Sigmoid
w_sigmoid, b_sigmoid
print("Final weights and bias after training with Sigmoid")
print('w=',w_sigmoid,'b=',b_sigmoid)
