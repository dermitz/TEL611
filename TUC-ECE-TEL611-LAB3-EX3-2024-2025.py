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
