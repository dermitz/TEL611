import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset: Revised Loan Approval Problem
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])  # Loan decision: at least two conditions met

# Create MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(3, 2), activation='relu', max_iter=1000, random_state=42)

# Train the model
mlp.fit(X, y)

# Predict the outcomes
y_pred = mlp.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print("Model Accuracy:", accuracy)

# Print predictions
print("Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} | True Output: {y[i]} | Predicted Output: {y_pred[i]}")

# Visualize predictions
colors = ['red' if yi == 0 else 'blue' for yi in y]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# True values
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, label='True Values', s=100, edgecolors='k')

# Predicted values
pred_colors = ['red' if yi == 0 else 'blue' for yi in y_pred]
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred_colors, marker='x', label='Predicted Values', s=100)

ax.set_xlabel('Employment (x1)')
ax.set_ylabel('Credit History (x2)')
ax.set_zlabel('Debt (x3)')
ax.set_title('Loan Approval Decision Using MLP')
ax.legend()
plt.show()
