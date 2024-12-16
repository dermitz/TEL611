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
