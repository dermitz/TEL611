"""
File Name: svm_linear_classification.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the use of a Support Vector Machine (SVM) with a linear kernel for binary classification in 
    a two-dimensional space. The dataset consists of two classes, which are generated using multivariate normal 
    distributions with different means and covariance matrices. The SVM model is trained to classify the two classes, 
    and the decision boundary (the optimal hyperplane) is visualized along with the data points.

    Key Components:
    - Data Generation: The script generates two classes of data points in 2D space, with each class having 50 samples. 
      The data points are drawn from a multivariate normal distribution with different means and covariance matrices.
    - SVM Model: A Support Vector Machine with a linear kernel is used to classify the data. The model is trained on the 
      generated dataset.
    - Decision Boundary: After training the SVM model, the decision boundary (the hyperplane separating the two classes) 
      is computed using the model's coefficients. The decision boundary is then plotted alongside the data points.
    - Visualization: The script generates a scatter plot to visualize the two classes and the decision boundary, highlighting 
      the optimal linear discriminant function learned by the SVM.

Usage:
    - This script is useful for demonstrating how SVM can be applied to a simple binary classification task in a 2D space.
    - It shows how to visualize the decision boundary of a linear SVM and how to use it for separating two classes.

Dependencies:
    - numpy
    - matplotlib
    - scikit-learn

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib, scikit-learn) are installed in your Python environment.
    - Run the script to generate the synthetic dataset, train the linear SVM model, and visualize the results.
    - The script will display the dataset, the decision boundary, and provide an optimal separating hyperplane for classification.

Notes:
    - This script demonstrates the case where classes are linearly separable, which allows the SVM to learn a simple 
      linear boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate sample data for two classes
np.random.seed()  # For reproducibility
class_A = np.random.multivariate_normal([2, 2], [[1, 0.2], [0.2, 1]], 50)
class_B = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], 50)

# Combine data and create labels
X = np.vstack((class_A, class_B))  # Combine features of both classes
y = np.hstack((np.zeros(class_A.shape[0]), np.ones(class_B.shape[0])))  # Labels: 0 for Class A, 1 for Class B
print(X)
print(y)
# Fit a linear Support Vector Machine (SVM) for classification
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# Extract the coefficients of the decision boundary from the SVM
w1, w2 = svm_model.coef_[0]
b = svm_model.intercept_[0]
#print('w1=',w1)
#print('w2=',w2)
#print('b=',b)

# Generate the decision boundary
x = np.linspace(0, 7, 100)
y_boundary = -(w1 * x + b) / w2  # Decision boundary
#print(x)
#print(y_boundary)

# Plot the data points and the SVM decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', label='Class A', alpha=0.7)
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', label='Class B', alpha=0.7)
plt.plot(x, y_boundary, color='black', label='Decision Boundary (SVM)', linewidth=2)

# Add labels and legend
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('2D Space with Optimal Linear Discriminant Function (SVM)')
plt.legend()
plt.grid(True)
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.show()
