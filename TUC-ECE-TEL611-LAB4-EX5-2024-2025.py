import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 1, 1, 0])  # Outputs (XOR)

# Visualize the data
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Problem')
plt.legend()
plt.grid()
plt.show()

# Create a Multi-Layer Perceptron (MLP) neural network
mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', max_iter=1000, random_state=42)

# Train the model
mlp.fit(X, y)

# Make predictions
y_pred = mlp.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print("Model accuracy:", accuracy)

# Display predictions
for i in range(len(X)):
    print(f"Input: {X[i]} | Expected Output: {y[i]} | Predicted Output: {y_pred[i]}")
