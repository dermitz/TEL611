"""
File Name: linear_discriminant_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the implementation of Linear Discriminant Analysis (LDA) for a synthetic dataset with 
    three classes. The script generates data from three different multivariate normal distributions, each representing 
    a class. The LDA approach involves calculating class statistics (means, covariance), defining the discriminant 
    function, classifying data points, and visualizing the decision regions for classification.

    Key Components:
    - Data Generation: The script generates three classes of synthetic data points using multivariate normal distributions 
      with different means and a shared covariance matrix.
    - Class Statistics: The script computes the means, shared covariance matrix, and inverse of the covariance matrix. 
      It also assumes equal priors for all classes.
    - Discriminant Function: The script defines the discriminant function for classifying data points based on the LDA model.
    - Classification: The script classifies all data points by evaluating the discriminant functions for each class and 
      assigning the class with the highest score.
    - Decision Region Visualization: The script visualizes the decision regions in the feature space and shows how the 
      classifier distinguishes between the different classes.
    - Accuracy Evaluation: The script calculates and prints the accuracy of the model by comparing predicted and actual class labels.

Usage:
    - This script demonstrates how to implement Linear Discriminant Analysis for multi-class classification problems.
    - It shows the process of calculating the decision boundaries and evaluating the classifier's performance on a synthetic dataset.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib) are installed in your Python environment.
    - Run the script to generate the synthetic dataset, classify the data points using LDA, visualize the decision boundaries, 
      and compute the accuracy of the model.

Notes:
    - This implementation assumes that the classes follow Gaussian distributions with a shared covariance matrix.
    - The script visualizes the decision boundaries for a 2D feature space and can be extended for higher dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Dataset
np.random.seed(42)

# Define class means
mean_1 = [2, 2]
mean_2 = [6, 6]
mean_3 = [10, 2]

# Shared covariance matrix
cov_matrix = [[1, 0.5], [0.5, 1]]

# Generate data
class_1 = np.random.multivariate_normal(mean_1, cov_matrix, 100)
class_2 = np.random.multivariate_normal(mean_2, cov_matrix, 100)
class_3 = np.random.multivariate_normal(mean_3, cov_matrix, 100)

# Combine into a dataset
X = np.vstack([class_1, class_2, class_3])
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

# Step 2: Compute Class Statistics
means = [np.mean(class_1, axis=0), np.mean(class_2, axis=0), np.mean(class_3, axis=0)]
cov_matrix_shared = np.cov(X, rowvar=False)  # Shared covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix_shared)  # Inverse of covariance matrix
priors = [1/3, 1/3, 1/3]  # Equal priors for all classes

# Step 3: Define Linear Discriminant Function
def discriminant_function(x, mean, inv_cov, prior):
    return (
        -0.5 * np.dot(np.dot((x - mean), inv_cov), (x - mean).T)
        + np.log(prior)
    )

# Step 4: Classify Data
def classify(x, means, inv_cov_matrix, priors):
    scores = [
        discriminant_function(x, mean, inv_cov_matrix, prior)
        for mean, prior in zip(means, priors)
    ]
    return np.argmax(scores)

# Classify all points in the dataset
y_pred = np.array([classify(x, means, inv_cov_matrix, priors) for x in X])

# Step 5: Visualize the Results
def plot_decision_regions(X, y, means, inv_cov_matrix, priors):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_labels = np.array([classify(point, means, inv_cov_matrix, priors) for point in grid_points])
    grid_labels = grid_labels.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_labels, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title('Decision Regions (Linear Discriminant Functions)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_regions(X, y, means, inv_cov_matrix, priors)

# Step 6: Evaluate Accuracy
accuracy = np.mean(y == y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
