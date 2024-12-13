# Exercise: PCA with Randomly Generated Data

# Objective:
# Perform PCA on randomly generated data to understand dimensionality reduction and feature extraction.

# Instructions:
# 1. Generate a random dataset with 3 features and 300 samples.
# 2. Standardize the dataset.
# 3. Perform PCA and reduce the dataset to 2 principal components.
# 4. Visualize the results using a scatter plot.

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

# Questions for Students:
# 1. Why is it important to standardize the data before applying PCA?
# 2. What do the scatter plot and explained variance tell us about the structure of the data?
# 3. Try changing the random data generation parameters. How does it affect the PCA results?