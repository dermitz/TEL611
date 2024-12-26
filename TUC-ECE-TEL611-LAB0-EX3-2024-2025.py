"""
File Name: iris_data_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script performs basic data analysis and visualization on the Iris dataset. It loads the dataset, prints the first few rows, and visualizes the relationships between two features (sepal length and sepal width) using a scatter plot. The visualization also includes a legend indicating the species of the Iris flowers.

Key Components:
    1. **Data Loading**: The Iris dataset is loaded using the `load_iris()` function from Scikit-Learn.
    2. **Data Inspection**: The first few rows of the dataset are displayed, along with the target values and feature names.
    3. **Data Visualization**: A scatter plot is created to visualize the relationship between sepal length and sepal width, with color coding for the species of Iris.
    4. **Manual Legend Creation**: A custom legend is created to represent the three Iris species (Setosa, Versicolor, and Virginica) in the scatter plot.

Usage:
    - Run the script in a Python environment to perform basic data analysis and visualization on the Iris dataset.
    - The script will output the first few rows of the dataset, class names, feature names, and display a scatter plot of the data.

Dependencies:
    - matplotlib
    - sklearn

Instructions:
    - Run the script in a Python environment.
    - The Iris dataset will be loaded and printed with the target values.
    - A scatter plot showing the relationship between sepal length and sepal width will be displayed with a color-coded legend indicating the species.

Notes:
    - This script demonstrates basic data inspection and visualization using the Iris dataset, which is a classic dataset in machine learning.
    - The dataset contains features such as sepal length, sepal width, petal length, and petal width, which are used to classify Iris species.


"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Load the Iris dataset
iris = load_iris()
# Print the first few rows of the dataset
print("== Iris Dataset Head ==")
print("Sepal Length, Sepal Width, Petal Length, Petal Width, Target")
for i in range(5):  # Display first 5 rows
    print(f"{iris.data[i]} , {iris.target[i]}")
# Print class names and feature names
print("\n== Class Names ==")
print(iris.target_names)
print("\n== Feature Names ==")
print(iris.feature_names)
# Visualize the dataset with a scatter plot
plt.figure(figsize=(8, 6))
# Create the scatter plot
scatter = plt.scatter(
    iris.data[:, 0],  # Sepal length
    iris.data[:, 1],  # Sepal width
    c=iris.target,    # Color by species
    cmap='viridis',
    edgecolor='k',
    s=100
)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset Scatter Plot')
# Manually add the legend for the species
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='setosa'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='versicolor'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='virginica')
], title='Species')
plt.show()
