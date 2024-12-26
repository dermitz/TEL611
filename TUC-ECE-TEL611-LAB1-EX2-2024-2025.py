"""
File Name: lda_iris_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script applies Linear Discriminant Analysis (LDA) on the Iris dataset to reduce its dimensionality for visualization.
    The dataset consists of four features related to three species of Iris plants.
    LDA is used to project the dataset onto two components, which are then plotted for visualization.

Key Components:
    1. Data Loading: The Iris dataset is loaded from sklearn's datasets module.
    2. Data Preprocessing: The feature data is standardized using `StandardScaler` to ensure the LDA algorithm performs optimally.
    3. LDA Application: Linear Discriminant Analysis is applied to reduce the dimensionality from four features to two.
    4. Visualization: The results of the LDA transformation are visualized using a scatter plot where different Iris species are represented by different colors and markers.

Usage:
    - The script demonstrates the application of LDA for dimensionality reduction in a supervised classification problem.
    - It is suitable for visualizing high-dimensional data in lower dimensions, aiding in the understanding of class separability.

Dependencies:
    - numpy
    - matplotlib
    - sklearn

Instructions:
    - Run the script to load the Iris dataset, apply LDA, and visualize the results in a 2D plot.
    - The plot will display the transformed data with points representing different species of Iris, allowing you to observe the separability between classes.

Notes:
    - LDA is particularly useful for classification problems with multiple classes, as it seeks to find a lower-dimensional representation that maximizes class separability.
    - In this script, the feature values are standardized before applying LDA, as LDA is sensitive to the scale of the data.

"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (three species of Iris)
y = iris.target  # Labels (three species of Iris)
# Standardize the feature values (LDA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply LDA for dimensionality reduction (reduce to 2 components for visualization)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
# Plot the results of LDA
plt.figure(figsize=(8, 6))
# Plot the LDA-transformed data points
colors = ['red', 'green', 'blue']
markers = ['s', 'x', 'o']
for label, color, marker in zip(np.unique(y), colors, markers):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1],
                color=color, label=iris.target_names[label], marker=marker)
plt.title("LDA Projection of Iris Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend(loc='best')
plt.grid(True)
plt.show()
