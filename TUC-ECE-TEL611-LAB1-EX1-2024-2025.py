"""
File Name: pca_data_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the application of Principal Component Analysis (PCA) to a randomly generated dataset.
    PCA is a dimensionality reduction technique that transforms data into a new coordinate system defined by the directions of maximum variance (principal components).
    This script reduces the dimensionality of a 3-dimensional dataset to 2 dimensions for visualization and provides insights into the explained variance.

Key Components:
    1. Data Generation: A synthetic dataset of 300 samples with 3 features is generated using random values with different standard deviations and means.
    2. Data Preprocessing: The dataset is standardized using `StandardScaler` to ensure that each feature has a mean of 0 and a standard deviation of 1.
    3. PCA Application: PCA is applied to reduce the data from 3 features to 2 principal components for visualization.
    4. Visualization: A scatter plot of the PCA-reduced data is created to help visualize the structure of the data in the new 2D space.
    5. Variance Analysis: The explained variance ratio for each principal component is calculated to understand how much of the original variance is retained.

Usage:
    - The script is ideal for illustrating the power of PCA in dimensionality reduction.
    - It is useful for students learning about feature reduction and the importance of standardizing data before applying PCA.

Dependencies:
    - numpy
    - matplotlib
    - sklearn

Instructions:
    - Run the script to generate the random data, apply PCA, and visualize the transformed data in a 2D scatter plot.
    - The script will also display the explained variance ratio, helping you understand the effectiveness of the dimensionality reduction.

Notes:
    - PCA is sensitive to the scale of the data, which is why it is important to standardize the features before applying PCA.
    - The explained variance ratio shows how much of the data's original variance is retained after applying PCA, which helps assess the quality of the dimensionality reduction.

Questions for Students:
    1. Why is it important to standardize the data before applying PCA?
    2. What do the scatter plot and explained variance tell us about the structure of the data?
    3. Try changing the random data generation parameters. How does it affect the PCA results?
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Generate random data
np.random.seed(42)  # Ensure reproducibility
n_samples = 300
n_features = 3

data = np.random.randn(n_samples, n_features) * [5, 10, 2] + [10, 20, -5]

# Show the first 5 samples of the generated data
print("Original Data (First 5 Samples):")
print(data[:5, :])

# Step 2: Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Step 3: Apply PCA
n_components = 2
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data_standardized)

# Show the first 5 samples after PCA transformation
print("\nPCA-Reduced Data (First 5 Samples):")
print(data_pca[:5, :])

# Step 4: Visualize the PCA-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', alpha=0.7, edgecolor='k')
plt.title("PCA Transformation (2D View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

# Step 5: Analyze the Explained Variance
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratios:", explained_variance)
print(f"Total Variance Explained: {explained_variance.sum() * 100:.2f}%")
