"""
File Name: softmax_neuron_model.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a SoftMax neuron model, designed for multi-class classification. The SoftMax function is used to
    compute the output probabilities across multiple classes. The neuron learns to classify data points into one of several
    classes by adjusting its weights and bias using gradient descent. The script trains the model on a small dataset with
    one-hot encoded targets for multi-class classification.

    Key components:
    - SoftMax function: A function that transforms the output logits into a probability distribution over multiple classes.
    - Training process: The neuron adjusts its weights and bias based on the error between the predicted probabilities and the
      true class labels (using one-hot encoding) through a learning rule. The weights and biases are updated during each
      epoch.
    - Multi-class classification task: The script uses a dataset of inputs (features) and corresponding target class labels.

Usage:
    - This script demonstrates the application of the SoftMax neuron for multi-class classification. It trains the neuron over a
      specified number of epochs.
    - The neuron adjusts its weights and bias to minimize the classification error, using the SoftMax function to output
      class probabilities.
    - The updated weights and bias are printed at each epoch to show the progress of the training process.
    - At the end of training, the final weights and bias values are displayed.

Dependencies:
    - numpy

Instructions:
    - Ensure that NumPy is installed in your Python environment.
    - Run the script to train the SoftMax neuron over the specified number of epochs.
    - The script will print the updated weights and bias after each training iteration.
    - At the end of training, the final weights and bias values are shown to summarize the training process.

"""

import numpy as np

# SoftMax function
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z)

# Dataset (inputs and one-hot encoded targets for multi-class classification)
X = np.array([
    [1, 2],
    [1, 3],
    [0, 2],
    [2, 1]
])
t = np.array([
    [1, 0, 0],  # Class 1
    [0, 1, 0],  # Class 2
    [0, 0, 1],  # Class 3
    [1, 0, 0]   # Class 1
])

# Parameters
num_classes = 3
num_features = X.shape[1]
w_softmax = np.zeros((num_features, num_classes))  # Weights matrix
b_softmax = np.zeros(num_classes)  # Bias vector
learning_rate = 0.1
epochs = 10

# Training the SoftMax Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute logits (z)
        z = np.dot(X[i], w_softmax) + b_softmax

        # Apply SoftMax activation
        y = softmax(z)

        # Compute error
        error = t[i] - y

        # Update weights and bias
        w_softmax += learning_rate * np.outer(X[i], error)
        b_softmax += learning_rate * error
        print('w=',w_softmax,'b=',b_softmax)
# Final weights and bias after training with SoftMax
print("Final weights and bias after training with SoftMax")
print("Final weights:", w_softmax)
print("Final bias:", b_softmax)
