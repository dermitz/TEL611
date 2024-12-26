"""
File Name: pca_iris_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script applies Principal Component Analysis (PCA) to the Iris dataset for dimensionality reduction.
    PCA is used to reduce the four-dimensional feature space (sepal length, sepal width, petal length, and petal width) of the Iris dataset to two dimensions for visualization. 
    The script also evaluates the explained variance of the principal components.

Key Components:
    1. Data Loading: The Iris dataset is loaded, which consists of 150 samples with four features.
    2. Data Normalization: The features are scaled using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
    3. PCA Application: PCA is applied to reduce the dimensionality of the dataset from 4 to 2 principal components.
    4. Visualization: A scatter plot is created to visualize the dataset in the reduced 2D space, with color-coding based on the class labels.
    5. Explained Variance Analysis: The variance explained by each principal component is printed to assess the amount of information retained after reduction.

Usage:
    - This script is useful for students and practitioners learning about PCA and its application to real-world datasets.
    - It demonstrates how PCA can reduce the dimensionality of data while retaining important information.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - sklearn

Instructions:
    - Run the script to load the Iris dataset, standardize the features, apply PCA, and visualize the results in a 2D scatter plot.
    - The script will also print the explained variance ratio, helping you assess the effectiveness of PCA.

Notes:
    - PCA is sensitive to the scale of the data, which is why feature scaling is an important preprocessing step.
    - The explained variance ratio indicates how much of the original data's variance is retained in each principal component.

Questions for Students:
    1. What is the importance of standardizing the features before applying PCA?
    2. How does PCA help in visualizing high-dimensional data?
    3. What do the explained variance ratios tell us about the principal components in the dataset?
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
# Loading the Iris dataset
iris = load_iris()
X = iris.data # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Classes (Iris Setosa, Versicolor, Virginica)
# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Applying PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X) # Print Features (sepal length, sepal width, petal length, petal width)
print(X_scaled) # Print scaled Features
print(y) #Print IRIS Classes
# Visualizing the results
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset in 2 Dimensions')
plt.colorbar(label='Classes (0: Setosa, 1: Versicolor, 2: Virginica)')
plt.show()
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by PCA: {explained_variance}')
