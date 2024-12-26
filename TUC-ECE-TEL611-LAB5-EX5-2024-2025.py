"""
File Name: stochastic_neuron_model.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a stochastic neuron model that is trained on a simple binary classification task. The neuron outputs
    a binary response based on a probabilistic decision using a sigmoid activation function. The following operations are included:
    - Sigmoid function: Computes the probability of firing based on the weighted sum of inputs.
    - Training: The neuron adjusts its weights and bias based on the error between predicted and target values using stochastic
      firing. The training process involves updating the weights and bias through a learning rule based on the error signal.
    - The task is a simple XOR-like classification problem with four input-output pairs, where the outputs are probabilistically
      determined by the neuron.

Usage:
    - This script demonstrates how a stochastic neuron model can be trained on a small dataset to learn a binary classification task.
    - The neuron is trained over a set number of epochs, and during each epoch, the weights and bias are updated.
    - The final weights and bias values after training are printed at the end of the script.

Dependencies:
    - numpy

Instructions:
    - Ensure that NumPy is installed in your Python environment.
    - Run the script to train the stochastic neuron over the specified number of epochs.
    - The script will print the updated weights and bias at each epoch, showing the progress of the learning process.
    - At the end of the training, the final weights and bias are displayed.

"""

#Stochastic Neuron Model
import numpy as np

# Sigmoid function for probability calculation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 1, 1, 0])  # Target values (example)

# Initial parameters
w_stochastic = np.array([0.0, 0.0])  # Initial weights
b_stochastic = 0.0  # Initial bias
learning_rate = 0.1  # Learning rate
epochs = 10  # Maximum number of epochs

# Training the Stochastic Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_stochastic, X[i]) + b_stochastic

        # Compute activation probability using sigmoid
        p = sigmoid(z)

        # Stochastic firing: output is 1 with probability p, else 0
        y = np.random.binomial(1, p)

        # Compute error
        error = t[i] - y

        # Update weights and bias
        w_stochastic += learning_rate * error * X[i]
        b_stochastic += learning_rate * error
        print('w=',w_stochastic,'b=',b_stochastic)

# Final weights and bias after training with the Stochastic Neuron
print("Final weights and bias after training with the Stochastic Neuron")
print("Final weights:", w_stochastic)
print("Final bias:", b_stochastic)
