"""
File Name: perceptron_iris_classification.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the implementation of a Perceptron algorithm for binary classification using the 
    Iris dataset. The dataset is preprocessed by selecting two features and standardizing the data. The Perceptron 
    model is trained using the training set and tested on a separate test set. The accuracy of the model is evaluated 
    and printed. A visualization of the decision boundary is generated, showing the classification of the two Iris 
    species ('Setosa' and 'Versicolor').

    Key Components:
    - Dataset: The Iris dataset is loaded, with only the first two features and the first 150 samples (Setosa and 
      Versicolor classes) used for training and testing.
    - Preprocessing: The features are standardized using `StandardScaler` from scikit-learn.
    - Perceptron Training: The Perceptron algorithm is implemented manually to learn the decision boundary for classifying 
      the two Iris species.
    - Model Evaluation: The model's accuracy is computed based on its predictions on the test data.
    - Visualization: A scatter plot is created showing the training samples, and the decision boundary is plotted to 
      visualize the model's classification.

Usage:
    - This script can be used to demonstrate the Perceptron algorithm for binary classification, including model training, 
      evaluation, and visualization.
    - It is particularly useful for understanding the process of training a linear classifier and how it can be visualized.

Dependencies:
    - numpy
    - scikit-learn
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, scikit-learn, matplotlib) are installed in your Python environment.
    - Run the script to train the Perceptron model on the Iris dataset and evaluate its performance.
    - The accuracy of the model will be printed in the console, and a plot of the decision boundary will be displayed.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data[:150, :2]  # Use first 100 samples (setosa and versicolor) and first 2 features for visualization
y = iris.target[:150]    # Class labels: 0 = setosa, 1 = versicolor

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perceptron Algorithm
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.01
epochs = 20

# Training
for epoch in range(epochs):
    for i in range(len(X_train)):
        linear_output = np.dot(weights, X_train[i]) + bias
        y_pred = 1 if linear_output >= 0 else 0
        error = y_train[i] - y_pred
        weights += learning_rate * error * X_train[i]
        bias += learning_rate * error

# Evaluate the model
correct_predictions = 0
for i in range(len(X_test)):
    linear_output = np.dot(weights, X_test[i]) + bias
    y_pred = 1 if linear_output >= 0 else 0
    if y_pred == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print("Accuracy on test data:", accuracy)

# Visualization
plt.figure(figsize=(10, 6))

# Plot data points
for i in range(len(y_train)):
    if y_train[i] == 0:
        plt.scatter(X_train[i][0], X_train[i][1], color='red', label='Setosa' if i == 0 else "")
    else:
        plt.scatter(X_train[i][0], X_train[i][1], color='blue', label='Versicolor' if i == 0 else "")

# Plot decision boundary
x_values = np.linspace(-2, 2, 100)
decision_boundary = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, decision_boundary, color='green', label='Decision Boundary')

# Customize plot
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.legend()
plt.title("Perceptron Decision Boundary on Iris Dataset")
plt.grid()
plt.show()
