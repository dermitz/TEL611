"""
File Name: loan_approval_mlp.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the use of a Multi-layer Perceptron (MLP) classifier to predict loan approval decisions based 
    on three features: Employment, Credit History, and Debt. The model uses the ReLU activation function in the hidden layers 
    and evaluates the performance using accuracy score. The dataset includes binary outcomes (0 for not approved, 1 for approved), 
    and the MLP is trained to classify whether at least two out of the three conditions (Employment, Credit History, Debt) 
    are met for loan approval.

    Key components:
    - Dataset: A binary classification dataset representing loan approval based on three input features.
    - MLPClassifier: A neural network classifier that uses multiple hidden layers and ReLU activation to model the relationship 
      between input features and the loan approval decision.
    - Evaluation: The model's accuracy is evaluated using the accuracy score from scikit-learn's metrics module.
    - Visualization: The input data and predictions are visualized in a 3D scatter plot, with different colors representing true 
      and predicted outcomes.

Usage:
    - This script can be used to predict loan approval decisions based on input features such as Employment, Credit History, 
      and Debt.
    - The MLPClassifier is trained on the dataset, and the model's performance is evaluated and printed.
    - The true and predicted outcomes are plotted in a 3D space for visual inspection of the model's predictions.

Dependencies:
    - numpy
    - sklearn
    - matplotlib

Instructions:
    - Ensure that NumPy, scikit-learn, and Matplotlib are installed in your Python environment.
    - Run the script to train the MLP model on the dataset.
    - The script will output the accuracy score of the model, along with the true and predicted loan approval decisions.
    - A 3D scatter plot will be displayed, showing both the true and predicted values for visual comparison.

"""

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
