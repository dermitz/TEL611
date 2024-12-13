import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Generate Sample Data for Three Classes
np.random.seed(42)  # For reproducibility

# Generate data for Class A
class_A = np.random.multivariate_normal([2, 2], [[1, 0.2], [0.2, 1]], 50)

# Generate data for Class B
class_B = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], 50)

# Generate data for Class C
class_C = np.random.multivariate_normal([8, 2], [[1, 0.3], [0.3, 1]], 50)

# Combine the data and create labels
X = np.vstack([class_A, class_B, class_C])  # Feature matrix
y = np.array([0] * 50 + [1] * 50 + [2] * 50)  # Labels (0 for Class A, 1 for Class B, 2 for Class C)

# Step 2: Visualize the Dataset
plt.figure(figsize=(8, 6))
plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', label='Class B')
plt.scatter(class_C[:, 0], class_C[:, 1], color='green', label='Class C')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dataset: Three Classes')
plt.legend()
plt.show()

# Step 3: Train an SVM with a Linear Kernel
svm_linear = SVC(kernel='linear', C=1, decision_function_shape='ovo')  # One-vs-One for multi-class
svm_linear.fit(X, y)

# Step 4: Evaluate the Linear SVM
y_pred_linear = svm_linear.predict(X)
accuracy_linear = accuracy_score(y, y_pred_linear)
print(f"Accuracy with Linear Kernel: {accuracy_linear:.2f}")

# Step 5: Visualize the Decision Boundary for Linear Kernel
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X, y, svm_linear, "Linear SVM Decision Boundary")

# Step 6: Train an SVM with an RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5, decision_function_shape='ovo')  # One-vs-One for multi-class
svm_rbf.fit(X, y)

# Step 7: Evaluate the RBF SVM
y_pred_rbf = svm_rbf.predict(X)
accuracy_rbf = accuracy_score(y, y_pred_rbf)
print(f"Accuracy with RBF Kernel: {accuracy_rbf:.2f}")

# Step 8: Visualize the Decision Boundary for RBF Kernel
plot_decision_boundary(X, y, svm_rbf, "RBF SVM Decision Boundary")