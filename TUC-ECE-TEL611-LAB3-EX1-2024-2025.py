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
