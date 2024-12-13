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
