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
