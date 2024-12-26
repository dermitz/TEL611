"""
File Name: perceptron_random_data.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the implementation of the Perceptron algorithm for binary classification using randomly 
    generated 2D data. The labels are based on a simple linear rule where the sum of the two features determines the 
    class. The Perceptron is trained on the data, and the final weights and bias are displayed. A plot is generated 
    to show the decision boundary learned by the Perceptron, along with the data points from both classes.

    Key Components:
    - Data Generation: Random 2D points are generated, and labels are assigned based on whether the sum of the two 
      features is greater than 1.
    - Perceptron Training: The Perceptron algorithm is implemented manually, where weights and bias are updated during 
      training to classify the data points correctly.
    - Decision Boundary: The learned decision boundary is plotted alongside the data points, allowing visualization of 
      the classifier's decision-making process.
    - Visualization: The script generates a plot that displays the training data points, decision boundary, and axes.

Usage:
    - This script is useful for understanding the Perceptron algorithm in the context of simple binary classification.
    - It shows the process of training a linear classifier on random data and visualizing the decision boundary.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib) are installed in your Python environment.
    - Run the script to train the Perceptron model on the random dataset and visualize the decision boundary.
    - The final weights and bias will be printed in the console, and a plot will be displayed showing the data points 
      and the decision boundary.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random training data
np.random.seed(42)  # For reproducibility
num_points = 100  # Number of data points
X = np.random.rand(num_points, 2)  # Random points in 2D space
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Labels based on a simple linear rule

# Initialize parameters
weights = np.zeros(X.shape[1])  # Initialize weights to zero
bias = 0  # Initialize bias to zero
learning_rate = 0.1  # Learning rate
epochs = 1000   # Number of epochs

# Training the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        # Calculate the output
        linear_output = np.dot(weights, X[i]) + bias
        y_pred = 1 if linear_output >= 0 else 0
        
        # Update parameters if the prediction is incorrect
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Final weights:", weights)
print("Final bias:", bias)

# Visualization of results
plt.figure(figsize=(8, 6))

# Plot data points
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i == 1 else "")

# Decision boundary calculation
x_values = np.linspace(0, 1, 100)
if weights[1] != 0:  # Check to avoid division by zero
    decision_boundary = -(weights[0] * x_values + bias) / weights[1]
    plt.plot(x_values, decision_boundary, color='green', label='Decision Boundary')
else:  # Special case: vertical line when weights[1] is 0
    plt.axvline(-bias / weights[0], color='green', label='Decision Boundary')

# Formatting the plot
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary with Random Data')
plt.legend()
plt.show()
