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
