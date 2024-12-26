"""
File Name: linear_discriminant_analysis_2d.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a Linear Discriminant Analysis (LDA) classifier for a synthetic 2D dataset with two classes. 
    The data is generated from two multivariate normal distributions with different means and covariance matrices. 
    The script computes the LDA parameters, applies the classifier, and visualizes the decision boundary that separates the two classes.

    Key Components:
    - Data Generation: The script generates two classes of synthetic data using multivariate normal distributions. 
      Each class has different mean vectors and covariance matrices.
    - LDA Parameter Computation: The script calculates the means, covariance matrices, and computes the weight vector 
      and bias term for the Linear Discriminant Function.
    - Discriminant Function: The script defines the linear discriminant function that is used to classify data points 
      based on the computed weights and bias.
    - Classification: The script classifies data points by applying the discriminant function, and calculates the 
      classification accuracy.
    - Decision Boundary Visualization: The script visualizes the decision boundary by generating a mesh grid of points 
      and computing the decision value for each point. The resulting decision boundary is then plotted along with the 
      data points.
    - Accuracy Evaluation: The script computes and prints the accuracy of the classifier by comparing predicted and 
      actual class labels.

Usage:
    - This script demonstrates how to perform binary classification using Linear Discriminant Analysis (LDA) for 2D 
      feature spaces. It generates synthetic data, calculates the LDA parameters, and visualizes the decision boundary 
      separating the two classes.
    - The script can be adapted for more complex datasets with higher-dimensional feature spaces.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib) are installed in your Python environment.
    - Run the script to generate the synthetic dataset, compute the LDA parameters, classify the data points, 
      visualize the decision boundary, and compute the classification accuracy.

Notes:
    - This implementation assumes that both classes share the same covariance matrix (homoscedasticity).
    - The script generates a decision boundary in a 2D feature space, and can be extended for higher-dimensional spaces.
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the Dataset
np.random.seed(42)  # For reproducibility

# Generate data for class 1 (mean=[2, 2], covariance=[[1, 0.8], [0.8, 1]])
class_1 = np.random.multivariate_normal([2, 2], [[1, 0.8], [0.8, 1]], 100)

# Generate data for class 2 (mean=[7, 7], covariance=[[1, -0.6], [-0.6, 1]])
class_2 = np.random.multivariate_normal([7, 7], [[1, -0.6], [-0.6, 1]], 100)

# Combine the data and create labels (0 for class 1, 1 for class 2)
X = np.vstack([class_1, class_2])
y = np.array([0] * 100 + [1] * 100)

# Step 2: Compute Linear Discriminant Function Parameters
# Calculate means of each class
mean_1 = np.mean(class_1, axis=0)
mean_2 = np.mean(class_2, axis=0)

# Calculate the shared covariance matrix (assuming equal covariance)
cov_1 = np.cov(class_1.T)
cov_2 = np.cov(class_2.T)
cov_shared = (cov_1 + cov_2) / 2

# Compute the weights and bias
w = np.linalg.inv(cov_shared) @ (mean_2 - mean_1)  # Weight vector
b = -0.5 * (mean_2.T @ np.linalg.inv(cov_shared) @ mean_2 - mean_1.T @ np.linalg.inv(cov_shared) @ mean_1)

# Step 3: Define the Discriminant Function and Classifier
# Define the discriminant function
def discriminant_function(x):
    return np.dot(w, x) + b

# Define the classifier
def classify(x):
    return 1 if discriminant_function(x) > 0 else 0

# Apply the classification to all data points
y_pred = np.array([classify(point) for point in X])

# Step 4: Evaluate the Classification Accuracy
accuracy = np.mean(y_pred == y)
print(f'Classification Accuracy: {accuracy:.2f}')

# Step 5: Visualize the Decision Boundary
# Create a mesh grid for visualization
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

# Calculate the decision boundary
Z = np.array([discriminant_function(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=0, cmap='coolwarm', alpha=0.3)
plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
plt.scatter(class_2[:, 0], class_2[:, 1], color='red', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary with Linear Discriminant Function')
plt.legend()
plt.show()
